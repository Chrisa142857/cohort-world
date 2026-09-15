#!/usr/bin/env python
"""T2 (delta) ceiling ladder: is there a gap between the agent and a no-read floor?

CPU only. Reads the frozen environment, changes nothing in it.

T2's reward (cohortworld_env.reward.delta_reward) is
    ALPHA * R + (1 - CRPS(report, y)),   y in {down: 0, indistinguishable: 0.5, up: 1}
    R = 1[decision == truth] - 1[truth == T2_PRIOR_CLASS]
so the per-item maximum is 2.25 on an up/down item and 1.00 on an indistinguishable one -- the
ceiling is NOT a constant and any T2 mean must be read against the class balance of its own subset.
That is why every rung here is scored on the same item subset as the rung it is compared to.

The ladder:
  rung0  always-hedge / majority class / no-read heuristics (prompt fields only, cross-fitted)
  rung1  oracle (a point mass on the truth)
  rung2  the agent (rescored traces from a run directory)

Run:  .venv/bin/python gen/t2_ceiling.py --selfcheck
      .venv/bin/python gen/t2_ceiling.py --out <rundir>
"""
import argparse, collections, json, os, random, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cohortworld_env import reward as R
from cohortworld_env.lattice import _shared, _band_lo, _atrisk, PANEL

ITEMS = R.CONSTANTS()["items"]
HORIZ = [1, 2, 3, 5, 10]            # gen/build_items.py HORIZ; asserted against the manifest below
BOOT, SEED = 1000, 42

# ponytail: the three published disease tables are byte-identical across versions (same md5), so
# one read of C2's serves all three -- as gen/build_items.py already does.
DIS = {g: json.load(open(f"{PANEL}/CohortWorld_C2_diseases_g{g}.json"))["diseases"] for g in (1, 2, 3)}
DIDX = {g: {c: i for i, c in enumerate(DIS[g])} for g in (1, 2, 3)}


# --------------------------------------------------------------------------- items

def load(split):
    with open(f"{ITEMS}/t2_delta_{split}.jsonl") as f:
        return [dict(json.loads(l), _idx=i) for i, l in enumerate(f)]


def is_served(d):
    """The answerability filter DeltaTaskset.load applies, replicated so the served subset can be
    named without a GPU. Serving is not cosmetic: an item the environment cannot answer is not a
    hard item, and scoring a policy on it measures nothing."""
    start = d["_audit"]["from"]
    lat = _shared(start.split(":", 1)[0])
    return not (lat is None or d.get("cross_version")
                or start not in lat.states or d["_audit"]["to"] not in lat.states
                or d["disease"] not in lat.didx)


# --------------------------------------------------------------------------- separability

def wilson(x, n, z=1.96):
    """Copied from gen/build_edges_pooled.py rather than imported: that module's import pulls the
    whole edge build. Identity with it is asserted in --selfcheck."""
    if n <= 0:
        return 0.0, 1.0
    import math
    c = (x + z * z / 2) / (n + z * z)
    h = z / (n + z * z) * math.sqrt(max(0.0, x * (n - x) / n) + z * z / 4)
    return max(0.0, c - h), min(1.0, c + h)


def separated(p1, n1, p2, n2):
    if n1 <= 0 or n2 <= 0:
        return False
    lo1, up1 = wilson(p1 * n1, n1)
    lo2, up2 = wilson(p2 * n2, n2)
    return up1 < lo2 or up2 < lo1


def _rate(state, hi, di, g):
    m = state.get(f"rates_g{g}", state.get("rates"))
    row = m[str(hi)] if isinstance(m, dict) else m[hi]
    v = row.get(str(di)) if isinstance(row, dict) else (row[di] if di < len(row) else None)
    return None if v is None else float(v)


def separability(d, denom="atrisk"):
    """Recompute the Wilson test for one item from the published cells.

    `denom="atrisk"` reproduces the flag the label was built from (at-risk band lower edge).
    `denom="cohort"` is the WRONG, anti-conservative cohort-size denominator that the at-risk band
    replaces. `denom="inflated"` is 10x the at-risk count, and it is the POSITIVE CONTROL: it
    must separate strictly more items, or this function is not reading the denominator at all and
    its "0 unsupported labels" would be worth nothing.
    """
    a = d["_audit"]
    v1, v2 = a["from"].split(":", 1)[0], a["to"].split(":", 1)[0]
    l1, l2 = _shared(v1), _shared(v2)
    if l1 is None or l2 is None or a["from"] not in l1.states or a["to"] not in l2.states:
        return None
    g, hi = int(d["granularity"]), HORIZ.index(int(d["horizon"]))
    di = DIDX[g].get(d["disease"])
    if di is None:
        return None
    s1, s2 = l1.states[a["from"]], l2.states[a["to"]]
    p1, p2 = _rate(s1, hi, di, g), _rate(s2, hi, di, g)
    if p1 is None or p2 is None:
        return None
    if denom == "cohort":
        n1, n2 = _band_lo(s1.get("n_band", "")), _band_lo(s2.get("n_band", ""))
    else:
        b1, b2 = _atrisk(v1, a["from"], g, hi, di), _atrisk(v2, a["to"], g, hi, di)
        n1 = _band_lo(b1) if isinstance(b1, str) and b1[:1].isdigit() else 0
        n2 = _band_lo(b2) if isinstance(b2, str) and b2[:1].isdigit() else 0
        if denom == "inflated":
            n1, n2 = n1 * 10, n2 * 10
    return {"p1": p1, "p2": p2, "n1": n1, "n2": n2, "sep": separated(p1, n1, p2, n2),
            "sign": "up" if p2 > p1 else "down" if p2 < p1 else "flat"}


# --------------------------------------------------------------------------- scoring

def score(report, direction):
    """The frozen reward, applied to a report the caller already holds.

    Not a re-derivation: --selfcheck asserts it reproduces `info._cw_score.reward` on every archived
    T2 episode, which is the only evidence that this formula is the environment's.
    """
    y = R._T2_Y[direction]
    said = R.t2_decision(report)
    r = (1.0 if said == direction else 0.0) - (1.0 if R.T2_PRIOR_CLASS == direction else 0.0)
    return R.ALPHA * r + (1.0 - R.crps(report, y))


PM = {"up": [1.0] * 5, "down": [0.0] * 5, "indistinguishable": [0.5] * 5}


def boot(xs, n=BOOT, seed=SEED):
    """Mean with a percentile bootstrap CI. Returns (mean, lo, hi, n)."""
    if not xs:
        return (float("nan"),) * 3 + (0,)
    rng = random.Random(seed)
    k = len(xs)
    ms = sorted(sum(rng.choices(xs, k=k)) / k for _ in range(n))
    return statistics.fmean(xs), ms[int(.025 * n)], ms[int(.975 * n)], k


# --------------------------------------------------------------------------- rung0 policies

def loo_majority(items, key):
    """Leave-one-out best constant within a stratum of PROMPT-VISIBLE fields.

    In-sample it is the ceiling of "guess from the question alone"; cross-fitted it is what that
    prior actually buys on a held-out item. Both are reported, because the gap between them is the
    difference between a heuristic and a memorised item set.
    """
    tot = collections.Counter(i["answer_direction"] for i in items)
    per = collections.defaultdict(collections.Counter)
    for i in items:
        per[key(i)][i["answer_direction"]] += 1
    out = []
    for i in items:
        c = collections.Counter(per[key(i)])
        c[i["answer_direction"]] -= 1
        if not any(c.values()):
            c = collections.Counter(tot)
            c[i["answer_direction"]] -= 1
        out.append(max(sorted(c), key=lambda k_: c[k_]))
    return out


def age_only(items):
    """The single strongest no-read rule in the set: on an `age_band shift`, older is up and younger
    is down; everything else hedges. Split out from `semantic` because 67% of the unseen items ARE
    age_band shifts, so this one rule carries almost all of the prior-knowledge score."""
    out = []
    for d in items:
        if d["axis"] == "age_band" and d["op"] == "shift":
            lat = _shared(d["_audit"]["from"].split(":", 1)[0])
            st = lat.states.get(d["_audit"]["from"]) if lat else None
            a = _band_lo(str(st["axes"]["age_band"])) if st else 0
            b = _band_lo(str(d["code"]))
            out.append("up" if b > a else "down" if b < a else "indistinguishable")
        else:
            out.append("indistinguishable")
    return out


def sign_reports(items):
    """The report the ENVIRONMENT itself writes for an agent that observes, moves, observes, and
    never states a conclusion (`reward.delta_report`, AUTO_OBSERVED): a single value, the sign of
    the difference between two readings. It is the measurement the T2 prompt prescribes, so it is
    the ceiling of "did the experiment, drew no inference" -- and it can never say
    `indistinguishable`, because the two published rates are never exactly equal on these items."""
    out = []
    for d in items:
        s = separability(d)
        out.append([1.0] if s and s["sign"] == "up" else [0.0] if s and s["sign"] == "down"
                   else [0.5])
    return out


def semantic(items):
    """A clinician's prior, applied with no rate ever read: a comorbidity or a drug added to the
    record raises risk; moving to an older age band raises it, younger lowers it; a sex switch or a
    change of coding resolution has no signed prior, so it hedges."""
    out = []
    for d in items:
        ax, op = d["axis"], d["op"]
        if ax in ("icd", "opcs", "bnf") and op in ("add", "switch"):
            out.append("up")
        elif ax == "age_band" and op == "shift":
            lat = _shared(d["_audit"]["from"].split(":", 1)[0])
            st = lat.states.get(d["_audit"]["from"]) if lat else None
            frm = str(st["axes"]["age_band"]) if st else ""
            a, b = _band_lo(frm), _band_lo(str(d["code"]))
            out.append("up" if b > a else "down" if b < a else "indistinguishable")
        else:
            out.append("indistinguishable")
    return out


# --------------------------------------------------------------------------- rung2: the agent

def agent_rows(trace_file):
    """(idx, direction, reward) per episode, plus the metrics needed to verify `score`."""
    out = []
    for line in open(trace_file):
        rec = json.loads(line)
        for t in rec.get("traces", []):
            s = (t.get("info") or {}).get("_cw_score")
            if s is None:
                continue
            out.append({"idx": rec["task"]["data"]["idx"],
                        "direction": rec["task"]["data"]["answer_direction"],
                        "reward": float(s["reward"]), "crps_loss": float(s["crps_loss"]),
                        "R": float(s["R"]), "correct": float(s["correct"]),
                        "level": float(s["report_level"]), "ok": bool(t.get("ok"))})
    return out


def find_t2_runs(root):
    """Every run directory holding scored cohortworld-delta traces, newest first. A fuller T2 eval
    landing while this runs is picked up without editing anything."""
    hits = []
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            if not (f.startswith("traces") and f.endswith(".jsonl")):
                continue
            p = os.path.join(dp, f)
            try:
                head = json.loads(open(p).readline())
            except Exception:
                continue
            if head.get("env", {}).get("id") == "cohortworld-delta":
                hits.append(p)
    return sorted(hits, reverse=True)


# --------------------------------------------------------------------------- selfcheck

def selfcheck():
    # 1. the reward's shape, from its own constants
    assert score(PM["up"], "up") == 2.25, score(PM["up"], "up")
    assert score(PM["down"], "down") == 2.25
    assert score(PM["indistinguishable"], "indistinguishable") == 1.0
    assert score(PM["indistinguishable"], "up") == 0.5
    assert score(PM["up"], "indistinguishable") == -0.75
    assert score(PM["up"], "down") == 0.0
    # 2. the formula IS the environment's, on every archived T2 episode
    n = 0
    for p in find_t2_runs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       "experiments")):
        for r in agent_rows(p):
            assert abs(r["reward"] - (R.ALPHA * r["R"] + 1.0 - r["crps_loss"])) < 1e-9, r
            n += 1
    assert n >= 12, f"only {n} archived T2 episodes found; the cross-check is not exercised"
    # 3. wilson matches the edge builder's, including the case that motivated the at-risk fix
    assert separated(0.0, 30, 0.5, 30) and not separated(0.10, 10, 0.20, 10)
    # 4. the bootstrap brackets its own point estimate and narrows with n
    m, lo, hi, k = boot([0.0, 1.0] * 30)
    assert lo < m < hi and k == 60
    m2, lo2, hi2, _ = boot([0.0, 1.0] * 300)
    assert (hi2 - lo2) < (hi - lo)
    # 5. the LOO fit cannot see its own label
    xs = [{"answer_direction": "up", "axis": "a", "op": "b"}] + \
         [{"answer_direction": "down", "axis": "a", "op": "b"}] * 3
    assert loo_majority(xs, lambda i: (i["axis"], i["op"]))[0] == "down"
    # 6. horizons are the ones the index assumes
    assert _shared("C1").horizons == HORIZ, _shared("C1").horizons
    print(f"selfcheck OK ({n} archived T2 episodes cross-checked)")


# --------------------------------------------------------------------------- report

def ladder(items, agent=None, label=""):
    lines, res = [], {}
    truth = [d["answer_direction"] for d in items]
    bal = collections.Counter(truth)
    n = len(items)

    def add(name, decisions, reports=None):
        reps = reports or [PM[x] for x in decisions]
        rs = [score(r, t) for r, t in zip(reps, truth)]
        acc = [1.0 if x == t else 0.0 for x, t in zip(decisions, truth)] if decisions else None
        m, lo, hi, k = boot(rs)
        a = boot(acc) if acc else None
        res[name] = {"reward": [m, lo, hi], "n": k,
                     "accuracy": [a[0], a[1], a[2]] if a else None}
        lines.append(f"  {name:<34s} reward {m:+.4f} [{lo:+.4f}, {hi:+.4f}]" +
                     (f"   acc {a[0]:.4f} [{a[1]:.4f}, {a[2]:.4f}]" if a else "") + f"   n={k}")
        return rs

    lines.append(f"\n=== {label}  n={n} ===")
    lines.append("  class balance: " + ", ".join(f"{k} {v} ({100*v/n:.1f}%)"
                                                 for k, v in bal.most_common()))
    res["class_balance"] = dict(bal)
    maj = bal.most_common(1)[0][0]

    add("rung0 always-hedge", ["indistinguishable"] * n)
    add(f"rung0 majority ({maj})", [maj] * n)
    add("rung0 no-read LOO (axis,op)", loo_majority(items, lambda i: (i["axis"], i["op"])))
    add("rung0 no-read LOO (ax,op,hor,gran)",
        loo_majority(items, lambda i: (i["axis"], i["op"], i["horizon"], i["granularity"])))
    add("rung0 no-read insample (axis,op)",
        _insample(items, lambda i: (i["axis"], i["op"])))
    add("rung0 no-read semantic prior", semantic(items))
    add("rung0 no-read AGE RULE only", age_only(items))
    # Strictly ZERO tool calls: `code` is the target age band and it is written in the prompt, so
    # this rung reads nothing at all -- not even the cohort it is standing in.
    add("rung0 ZERO-READ LOO (axis,op,code)",
        loo_majority(items, lambda i: (i["axis"], i["op"], i["code"])))
    sr = sign_reports(items)
    add("rung1a two-reading sign (auto)", [R.t2_decision(r) for r in sr], sr)
    ora = add("rung1b oracle", truth)

    # The question the ladder exists to answer: how much of the oracle's lead over a no-read floor
    # survives once the floor is prior knowledge rather than a constant. Paired, on these items.
    lines.append("  headroom (paired, same items):")
    agev = [score(PM[x], t) for x, t in zip(age_only(items), truth)]
    zerov = [score(PM[x], t) for x, t in
             zip(loo_majority(items, lambda i: (i["axis"], i["op"], i["code"])), truth)]
    signv = [score(r, t) for r, t in zip(sr, truth)]
    hedgev = [score(PM["indistinguishable"], t) for t in truth]
    for nm, a_, b_ in (("oracle - always-hedge", ora, hedgev),
                       ("oracle - AGE RULE", ora, agev),
                       ("oracle - ZERO-READ LOO", ora, zerov),
                       ("two-reading sign - AGE RULE", signv, agev),
                       ("two-reading sign - ZERO-READ LOO", signv, zerov)):
        m, lo, hi, _ = boot([x - y for x, y in zip(a_, b_)])
        res[f"headroom {nm}"] = [m, lo, hi]
        lines.append(f"    {nm:<34s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    if agent:
        by = {a["idx"]: a for a in agent}
        sub = [(d, by[d["_idx"]]) for d in items if d["_idx"] in by]
        if sub:
            it2 = [d for d, _ in sub]
            rs = [a["reward"] for _, a in sub]
            m, lo, hi, k = boot(rs)
            acc = boot([a["correct"] for _, a in sub])
            lines.append(f"  {'rung2 agent':<34s} reward {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                         f"   acc {acc[0]:.4f} [{acc[1]:.4f}, {acc[2]:.4f}]   n={k}")
            res["rung2 agent"] = {"reward": [m, lo, hi], "n": k,
                                  "accuracy": list(acc[:3])}
            # every comparison recomputed on the agent's OWN items, paired
            t2 = [d["answer_direction"] for d in it2]
            srs = sign_reports(it2)
            for nm, dec, rep in (("always-hedge", ["indistinguishable"] * len(it2), None),
                                 ("no-read LOO (axis,op)",
                                  loo_majority(it2, lambda i: (i["axis"], i["op"])), None),
                                 ("no-read AGE RULE only", age_only(it2), None),
                                 ("ZERO-READ LOO (axis,op,code)",
                                  loo_majority(it2, lambda i: (i["axis"], i["op"], i["code"])),
                                  None),
                                 ("semantic prior", semantic(it2), None),
                                 ("two-reading sign", [R.t2_decision(r) for r in srs], srs),
                                 ("oracle", t2, None)):
                reps = rep or [PM[x] for x in dec]
                ref = [score(r, t) for r, t in zip(reps, t2)]
                dm, dlo, dhi, _ = boot([a - b for a, b in zip(rs, ref)])
                res[f"paired agent - {nm}"] = [dm, dlo, dhi]
                lines.append(f"    paired  agent - {nm:<26s} {dm:+.4f} "
                             f"[{dlo:+.4f}, {dhi:+.4f}]")
    res["_oracle_mean"] = statistics.fmean(ora)
    return lines, res


def _insample(items, key):
    per = collections.defaultdict(collections.Counter)
    for i in items:
        per[key(i)][i["answer_direction"]] += 1
    return [max(sorted(per[key(i)]), key=lambda k_: per[key(i)][k_]) for i in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--split", default="unseen")
    ap.add_argument("--out", help="run directory from gen/newrun.sh")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    items = load(a.split)
    served = [d for d in items if is_served(d)]
    out = [f"T2 CEILING LADDER  env {R.CONSTANTS()['env_version']} "
           f"spec {R.REWARD_SPEC_VERSION} alpha {R.ALPHA}",
           f"items {ITEMS}", f"split={a.split}: {len(items)} items, {len(served)} served "
           f"({100*len(served)/len(items):.1f}%), {len(items)-len(served)} unanswerable"]
    res = {"env_version": R.CONSTANTS()["env_version"], "spec": R.REWARD_SPEC_VERSION,
           "alpha": R.ALPHA, "items": ITEMS, "split": a.split,
           "n_items": len(items), "n_served": len(served)}

    # ---- item-set characterisation: is the label a thing the environment can support? ----
    out.append("\n=== separability recheck (Wilson, 95%) ===")
    agree = collections.Counter()
    badlabel, ctrl_sep, mine_sep, infl_sep = [], 0, 0, 0
    for d in items:
        s = separability(d)
        c = separability(d, denom="cohort")
        infl_sep += bool((separability(d, denom="inflated") or {}).get("sep"))
        lab_sep = d["answer_direction"] != "indistinguishable"
        if s is None:
            agree["uncomputable"] += 1
            continue
        mine_sep += s["sep"]
        ctrl_sep += bool(c and c["sep"])
        agree["agree" if s["sep"] == lab_sep else "disagree"] += 1
        if lab_sep and not s["sep"]:
            badlabel.append(d["_idx"])
        d["_sep"], d["_sign"] = s["sep"], s["sign"]
    nc = sum(agree.values()) - agree["uncomputable"]
    out.append(f"  recomputed separable: {mine_sep}/{len(items)} ({100*mine_sep/len(items):.1f}%)")
    out.append(f"  label agrees with recomputation: {agree['agree']}/{nc}"
               f" ({100*agree['agree']/max(1,nc):.1f}%), disagree {agree['disagree']},"
               f" uncomputable {agree['uncomputable']}")
    out.append(f"  items labelled up/down whose intervals OVERLAP: {len(badlabel)}"
               f" ({100*len(badlabel)/len(items):.1f}%)  <- labels the environment cannot support")
    out.append(f"  control A (wrong, cohort-total denominator): {ctrl_sep} separable")
    out.append(f"  control B (POSITIVE, 10x at-risk denominator):  {infl_sep} separable"
               f" -- the test reads its denominator iff {infl_sep} > {mine_sep}")
    assert infl_sep > mine_sep, ("the separability test does not respond to its denominator; its "
                                 "'0 unsupported labels' is not evidence of anything")
    res.update(n_separable_recomputed=mine_sep, control_inflated_separable=infl_sep, label_agree=agree["agree"],
               label_disagree=agree["disagree"], uncomputable=agree["uncomputable"],
               unsupported_labels=len(badlabel), control_separable=ctrl_sep)

    # does the published sign alone give the answer away, without any interval reasoning?
    signacc = sum(1 for d in items if d.get("_sign") == d["answer_direction"]) / len(items)
    out.append(f"  sign-of-difference alone (reads BOTH rates, no interval): "
               f"acc {signacc:.4f} -- an agent that reads two cells and takes the sign")
    res["sign_only_accuracy"] = signacc

    # ---- per-stratum table: which strata carry the set, and is each one guessable ----
    out.append("\n=== strata of the FULL split (axis, op) ===")
    out.append(f"  {'stratum':<26s} {'n':>4s}  {'up':>4s} {'down':>4s} {'ind':>4s}  "
               f"{'semantic':>8s} {'sign':>8s}")
    sem, sr = semantic(items), sign_reports(items)
    by = collections.defaultdict(list)
    for d, sm, s_ in zip(items, sem, sr):
        by[(d["axis"], d["op"])].append((d, sm, R.t2_decision(s_)))
    strata = {}
    for k in sorted(by, key=lambda k_: -len(by[k_])):
        rows = by[k]
        c = collections.Counter(d["answer_direction"] for d, _, _ in rows)
        sa = sum(1 for d, m, _ in rows if m == d["answer_direction"]) / len(rows)
        ga = sum(1 for d, _, g_ in rows if g_ == d["answer_direction"]) / len(rows)
        out.append(f"  {k[0] + ' ' + k[1]:<26s} {len(rows):>4d}  {c['up']:>4d} {c['down']:>4d} "
                   f"{c['indistinguishable']:>4d}  {sa:>8.3f} {ga:>8.3f}")
        strata[f"{k[0]} {k[1]}"] = {"n": len(rows), "balance": dict(c),
                                    "semantic_acc": sa, "sign_acc": ga}
    res["strata"] = strata

    agent = None
    runs = find_t2_runs(os.path.join(root, "experiments"))
    out.append("\n=== rung2 sources ===")
    for p in runs:
        rows = agent_rows(p)
        out.append(f"  {len(rows):4d} episodes  {os.path.relpath(p, root)}")
    if runs:
        agent = agent_rows(runs[0])
        res["agent_trace"] = os.path.relpath(runs[0], root)

    for label, sub, ag in ((f"{a.split} ALL", items, None),
                           (f"{a.split} SERVED", served, None),
                           ("train (agent's own split)", load("train"), agent)):
        if ag is not None:
            sub = [d for d in sub if is_served(d)]
        lines, r = ladder(sub, ag, label)
        out += lines
        res[label] = r

    txt = "\n".join(out)
    print(txt)
    if a.out:
        m = res[f"{a.split} SERVED"]["rung0 always-hedge"]["reward"][0]
        o = res[f"{a.split} SERVED"]["_oracle_mean"]
        base = (f"t2_ceiling_{a.split}_hedge{m:.4f}_oracle{o:.4f}_"
                f"spec{R.REWARD_SPEC_VERSION}_seed{SEED}")
        os.makedirs(f"{a.out}/artifacts", exist_ok=True)
        open(f"{a.out}/artifacts/{base}.txt", "w").write(txt + "\n")
        json.dump(res, open(f"{a.out}/artifacts/{base}.json", "w"), indent=1)
        print(f"\nwrote {a.out}/artifacts/{base}.{{txt,json}}")


if __name__ == "__main__":
    main()
