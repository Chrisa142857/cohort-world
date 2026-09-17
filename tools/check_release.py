#!/usr/bin/env python3
"""Release gate: run before every commit and before every push or data upload.

  python tools/check_release.py code [--root .]            files git would track (git ls-files + staged/untracked-allowed)
  python tools/check_release.py data <file-or-dir> ...     data files headed for the dataset host
  python tools/check_release.py selftest                   proves each check can fail

Exit 0 = nothing found. Exit 1 = findings, one per line, `file:line: rule: excerpt`.

What it refuses, and why each rule exists:
  * file names that mark private or individual-level data: `priv/`, `*private*`, `*_subject*`
    (labels.k = 1 items), `*_private_n*`, `PhenoCal_*_full*` (its per-(disease, horizon) means include
    test-set outcomes, so it must not ship as a scoring file);
  * absolute paths into the lab file systems and the author's user name, which leak where private
    data lives, break every clone, and deanonymise a double-blind submission. A `release-gate:
    allow-file` build script does NOT get a pass on these two: its waiver covers only naming a
    private store or a host, which is the provenance being recorded;
  * names of the private item and count stores (`cohortworld_private`, `private_counts`);
  * any tracked file over --max-mb (data belongs on the dataset host, not in git);
  * in data mode: any JSONL row whose `labels.k` is below --min-k, or that has no `labels.k` when the
    file is an item file.
Patterns are arguments with derived defaults; nothing here is specific to one checkout.
"""
import argparse, json, os, re, subprocess, sys

NAME_RULES = [
    ("private-dir", re.compile(r"(^|/)priv(/|$)")),
    ("private-name", re.compile(r"private", re.I)),
    ("individual-level", re.compile(r"_subject")),
    ("full-panel", re.compile(r"PhenoCal[^/]*_full")),
]
CONTENT_RULES = [
    ("abs-path", re.compile(r"/(work|work_fast|ram|export_home)/USERS/")),
    ("abs-path", re.compile(r"/export_home/|/tmp/ziquanw")),
    ("user-name", re.compile(r"ziquanw|ziquan_wei|wzq10101", re.I)),
    ("private-store", re.compile(r"cohortworld_private|private_counts|_private_n\b")),
    ("host-name", re.compile(r"\b(hummer|yukon|frontier)\b")),
    # a value only the user can decide (licence, dataset id) is written as this marker until decided
    ("unfilled", re.compile(r"RELEASE-TODO")),
]
# A line may carry `release-gate: allow` when naming a private store is the point of the line (the
# build-only loader, a test asserting a path is refused). A comment does not move the environment's
# semantic hash, which is computed on code with comments stripped.
ALLOW = re.compile(r"release-gate:\s*allow")
# A build script may declare `release-gate: allow-file` in its first 5 lines. Only honoured under build/:
# those scripts ran against the private UKB extract and are shipped for provenance, so the NAME of a
# private table they read is shown as it was rather than rewritten.
ALLOW_FILE = re.compile(r"release-gate:\s*allow-file")
ALLOW_FILE_DIRS = ("build/",)
# ...but the waiver is for provenance, not a blanket exemption. Naming the private store is the fact
# being recorded; an absolute lab path or the author's user name is not. Those identify a machine and
# a person, they break every clone, and under double-blind review they deanonymise the submission --
# no provenance argument reaches them. This list is why a whole-file waiver is safe to offer at all.
ALLOW_FILE_WAIVES = ("private-store", "host-name")
# The gate names every pattern it refuses, so its own text is exempt from the content rules.
SELF = "tools/check_release.py"
TEXT_EXT = {".py", ".sh", ".toml", ".md", ".txt", ".json", ".yaml", ".yml", ".cfg", ".ini", ".patch", ".tsv", ".cff", ""}


def tracked_files(root):
    """What a commit would contain: tracked files plus untracked files the .gitignore allows."""
    out = subprocess.run(["git", "-C", root, "ls-files", "--cached", "--others", "--exclude-standard"],
                         capture_output=True, text=True, check=True).stdout.split("\n")
    return [f for f in out if f]


def check_names(paths):
    for p in paths:
        for rule, rx in NAME_RULES:
            if rx.search(p):
                yield f"{p}:0: {rule}: file name"


def check_content(root, paths, max_mb):
    for p in paths:
        if p == SELF:
            continue
        full = os.path.join(root, p)
        if not os.path.isfile(full):
            continue
        size = os.path.getsize(full)
        if size > max_mb * 1024 * 1024:
            yield f"{p}:0: too-large: {size / 1048576:.1f} MB > {max_mb} MB"
            continue
        if os.path.splitext(p)[1] not in TEXT_EXT:
            continue
        try:
            lines = open(full, encoding="utf-8").read().splitlines()
        except UnicodeDecodeError:
            continue
        waived = (ALLOW_FILE_WAIVES if p.startswith(ALLOW_FILE_DIRS)
                  and any(ALLOW_FILE.search(l) for l in lines[:5]) else ())
        for i, line in enumerate(lines, 1):
            if ALLOW.search(line):
                continue
            for rule, rx in CONTENT_RULES:
                if rule in waived:
                    continue
                if rx.search(line):
                    yield f"{p}:{i}: {rule}: {line.strip()[:120]}"


# In data files the host rule matches only a host-like use (a URL, an ssh target, host:port). A model's
# own text is released verbatim, and "frontier" as an English word is not a machine name.
DATA_CONTENT_RULES = [(r, rx) for r, rx in CONTENT_RULES if r != "host-name"] + [
    ("host-name", re.compile(r"(?:https?://|ssh |@|//)(?:hummer|yukon|frontier)\b|\b(?:hummer|yukon|frontier):[0-9]"))]
# Exact counts that must never leave the private store in any data file: a cohort's member count, and
# the event / at-risk / person counts behind a rate. Released files carry bands and rates instead.
# (CohortCal's own `n` / `n_events` pass: its items meet UK Biobank's minimum of 5 by construction.)
EXACT_COUNT_KEY = re.compile(r'"(members|n_event|n_at_risk|n_person_event|index_points)"\s*:')


def check_data(paths, min_k):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += [os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs]
        else:
            files.append(p)
    yield from check_names(files)
    yield from check_content("", [f for f in files if f.endswith((".md", ".txt", ".cff", ".tsv"))], 5)
    for f in files:
        if f.endswith((".json", ".jsonl")):
            # model outputs and traces carry metadata as well as answers: a lab path, a host name or a
            # run id in a config block would ship with them, so the content rules apply here too.
            with open(f) as fh:
                for i, line in enumerate(fh, 1):
                    for rule, rx in DATA_CONTENT_RULES:
                        m = rx.search(line)
                        if m:
                            yield f"{f}:{i}: {rule}: ...{line[max(0, m.start() - 40):m.start() + 60].strip()}..."
                            break
        if f.endswith(".json"):
            m = EXACT_COUNT_KEY.search(open(f).read())
            if m:
                yield f"{f}:0: exact-count: key {m.group(1)}"
            continue
        if not f.endswith(".jsonl"):
            continue
        item_file = os.path.basename(f).startswith(("t1_", "t2_", "t3_"))
        with open(f) as fh:
            for i, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                m = EXACT_COUNT_KEY.search(line)
                if m:
                    yield f"{f}:{i}: exact-count: key {m.group(1)}"
                if '"labels"' not in line and not item_file:
                    continue                          # no labels.k to check; skip the parse
                row = json.loads(line)
                k = (row.get("labels") or {}).get("k")
                if k is None:
                    if item_file:
                        yield f"{f}:{i}: missing-k: item row has no labels.k"
                    continue
                if k < min_k:
                    yield f"{f}:{i}: small-cell: labels.k={k} < {min_k}"


def selftest():
    """Every rule must be able to fire; a gate that cannot fail certifies nothing."""
    import tempfile
    fails = []
    names = ["official/priv/x.json", "CohortWorld_v3_C2_test_private_n.jsonl", "t3_rate_unseen_subject.jsonl",
             "PhenoCal_C2_full.jsonl", "gen/ok.py"]
    got = {r.split(": ")[1] for r in check_names(names)}
    for rule in ("private-dir", "private-name", "individual-level", "full-panel"):
        if rule not in got:
            fails.append(f"name rule {rule} did not fire")
    if list(check_names(["gen/score_ct_auroc.py"])):
        fails.append("name rules fired on a clean name")
    with tempfile.TemporaryDirectory() as d:
        bad = os.path.join(d, "bad.py")
        open(bad, "w").write('P = "/work/USERS/someone/x"\nQ = "cohortworld_private"\nH = "ssh hummer"\n')
        got = {r.split(": ")[1] for r in check_content(d, ["bad.py"], 5)}
        for rule in ("abs-path", "private-store", "host-name"):
            if rule not in got:
                fails.append(f"content rule {rule} did not fire")
        allowed = os.path.join(d, "allowed.py")
        open(allowed, "w").write('f = "x_private_counts.jsonl"  # release-gate: allow (build-only loader)\n')
        if list(check_content(d, ["allowed.py"], 5)):
            fails.append("allow marker did not suppress its line")
        os.makedirs(os.path.join(d, "build"), exist_ok=True); os.makedirs(os.path.join(d, "gen"), exist_ok=True)
        body = '# release-gate: allow-file (provenance)\nP = "cohortworld_private"\nH = "ssh hummer"\n'
        open(os.path.join(d, "build", "b.py"), "w").write(body)
        open(os.path.join(d, "gen", "b.py"), "w").write(body)
        if list(check_content(d, ["build/b.py"], 5)):
            fails.append("allow-file under build/ did not suppress the rules it waives")
        if not list(check_content(d, ["gen/b.py"], 5)):
            fails.append("allow-file was honoured outside build/")
        # the waiver is narrow: provenance never licenses a lab path or a user name, so a build
        # script carrying one is still refused. Without this case the waiver could widen unnoticed.
        leaky = '# release-gate: allow-file (provenance)\nP = "/work/USERS/someone/x"\nU = "ziquanw"\n'
        open(os.path.join(d, "build", "leaky.py"), "w").write(leaky)
        got = {r.split(": ")[1] for r in check_content(d, ["build/leaky.py"], 5)}
        for rule in ("abs-path", "user-name"):
            if rule not in got:
                fails.append(f"allow-file wrongly suppressed {rule} under build/")
        good = os.path.join(d, "good.py")
        open(good, "w").write('P = os.environ.get("CW_ITEMS")\n')
        if list(check_content(d, ["good.py"], 5)):
            fails.append("content rules fired on clean code")
        big = os.path.join(d, "big.txt")
        open(big, "wb").write(b"x" * (2 * 1024 * 1024))
        if not any("too-large" in r for r in check_content(d, ["big.txt"], 1)):
            fails.append("size rule did not fire")
        items = os.path.join(d, "t3_rate_unseen.jsonl")
        open(items, "w").write(json.dumps({"labels": {"k": 10}}) + "\n" + json.dumps({"labels": {"k": 1}}) + "\n"
                               + json.dumps({"labels": {}}) + "\n")
        got = list(check_data([items], 10))
        if not any("small-cell" in r for r in got) or not any("missing-k" in r for r in got):
            fails.append(f"data rules did not fire: {got}")
        if any(":1:" in r for r in got):
            fails.append("data rule fired on a k=10 row")
        side = os.path.join(d, "PhenoCal_C1_axes_sidecar.jsonl")
        open(side, "w").write(json.dumps({"prompt_prefix": "x", "members": 12}) + "\n")
        marg = os.path.join(d, "marginal.json")
        open(marg, "w").write(json.dumps({"rates": {"A": {"1": {"n_event": 3, "rate": 0.1}}}}))
        ok = os.path.join(d, "PhenoCal_C2_test.jsonl")
        open(ok, "w").write(json.dumps({"metadata": {"n": 40, "n_events": 6}}) + "\n")
        got = list(check_data([side, marg], 10))
        if sum("exact-count" in r for r in got) != 2:
            fails.append(f"exact-count rule did not fire on members / n_event: {got}")
        if list(check_data([ok], 10)):
            fails.append("exact-count rule fired on CohortCal's n / n_events")
        card = os.path.join(d, "README.md")
        open(card, "w").write("license: RELEASE-TODO\n")
        if not any("unfilled" in r for r in check_data([card], 10)):
            fails.append("unfilled marker in a dataset card did not fire")
        traces = os.path.join(d, "t3_rate_traces.jsonl")
        open(traces, "w").write(
            json.dumps({"labels": {"k": 10}, "run": {"name": "run23"}, "text": "0.8%"}) + "\n"
            + json.dumps({"labels": {"k": 10}, "client": {"base_url": "http://hummer:8000/v1"}}) + "\n"
            + json.dumps({"labels": {"k": 10}, "id": "/work/USERS/someone/x"}) + "\n")
        got = {r.split(": ")[1] for r in check_data([traces], 10)}
        for rule in ("host-name", "abs-path"):
            if rule not in got:
                fails.append(f"data-mode content rule {rule} did not fire inside a record")
        clean = os.path.join(d, "t3_rate_clean.jsonl")
        open(clean, "w").write(json.dumps({"labels": {"k": 10}, "text": "the rate is 0.8%"}) + "\n"
                               + json.dumps({"labels": {"k": 10}, "text": "boundary edge frontier horizon"}) + "\n")
        if list(check_data([clean], 10)):
            fails.append("data-mode content rules fired on a clean record")
    print("SELFTEST " + ("FAIL\n  " + "\n  ".join(fails) if fails else "PASS: every rule fires on a bad input and not on a good one"))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("code"); c.add_argument("--root", default="."); c.add_argument("--max-mb", type=float, default=5.0)
    dd = sub.add_parser("data"); dd.add_argument("paths", nargs="+"); dd.add_argument("--min-k", type=int, default=10)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    if a.mode == "code":
        files = tracked_files(a.root)
        found = list(check_names(files)) + list(check_content(a.root, files, a.max_mb))
        n = len(files)
    else:
        found = list(check_data(a.paths, a.min_k))
        n = len(a.paths)
    for f in found:
        print(f)
    print(f"{a.mode}: {len(found)} finding(s) over {n} path(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
