"""Correct-scoring audit for an archived arms run.

Every number here is produced by the ENVIRONMENT'S OWN reward code (`cohortworld_env.reward`) run
over the archived trace, never by a restatement of it in gen/. That is not tidiness: this file used
to carry its own copy of the answer-extraction rule, gen/results_table.py carried a second and
gen/score_direct.py a third, and the three disagreed -- on the same reply one read a mean of 3.5 and
another 0.0264. `ArchivedTrace` is the whole bridge that makes the reuse possible.

NO_ANSWER / NO_ANSWER_LOSS are gone (reward-spec 6.5). There is no unanswered episode any more: the
report ladder always emits one, and what used to be "unanswered" is now the `report_level` an
episode resolved to. Anything here that says "answered" means LEVEL 1-2, i.e. the agent stated it.
"""
import collections, glob, json, os, statistics as st, sys

from cohortworld_env import reward
from cohortworld_env.reward import well_formed
from cohortworld_env.taskset import _crps          # 1 - CRPS; the orientation gen/ was written to
from cohortworld_phenocal.taskset import _base_rates


class ArchivedTrace:
    """One line of a traces.jsonl, shaped so `cohortworld_env.reward` can score it.

    `reward.messages()` walks `trace.messages`; a trace read back from JSON is a dict of `nodes`.
    Six lines here mean the rescoring path IS the scoring path (reward-spec 1.3) rather than a
    lookalike of it -- which is the only way a number computed from an archive can be quoted beside
    a number computed during a run.
    """
    __slots__ = ("messages", "last_reply", "raw")

    def __init__(self, tr):
        self.raw = tr
        self.messages = [nd.get("message", {}) or {} for nd in tr.get("nodes", [])]
        replies = [m.get("content") for m in self.messages
                   if m.get("role") == "assistant" and m.get("content")]
        self.last_reply = replies[-1] if replies else ""


def tool_msgs(tr):
    for nd in tr.get("nodes", []):
        m = nd.get("message", {})
        if m.get("role") == "tool":
            c = m.get("content")
            yield c if isinstance(c, str) else json.dumps(c)


def tool_calls(tr):
    """(tool_name, content) per tool result. The name is what `cw_move` hits are counted off."""
    for nd in tr.get("nodes", []):
        m = nd.get("message", {})
        if m.get("role") == "tool":
            c = m.get("content")
            yield m.get("name"), (c if isinstance(c, str) else json.dumps(c))


def nav_hits(tr):
    """(move_hit, move_miss, look) -- navigation success counted off `cw_move` ONLY.

    The published "navigation hit rate" (91.8 / 86.6 / 85.4 / 82.5%) counted every tool result
    carrying `"hit": true`, and `cw_look` returns that unconditionally. So the metric rose whenever
    an arm looked more often, and the real decline was ~3x what was reported: on the same files the
    true cw_move rate is 90.7 -> 76.3 (arm A) where the old definition read 93.2 -> 86.6. A hit rate
    whose numerator includes an operation that cannot miss is not a hit rate.
    """
    hit = miss = look = 0
    for name, c in tool_calls(tr):
        if name == "cw_look":
            look += 1
        elif name == "cw_move":
            if '"hit": true' in c:
                hit += 1
            elif '"hit": false' in c:
                miss += 1
    return hit, miss, look


def assistant_text(tr):
    out = []
    for nd in tr.get("nodes", []):
        m = nd.get("message", {})
        if m.get("role") == "assistant":
            c = m.get("content")
            out.append(c if isinstance(c, str) else json.dumps(c))
    return out


_REF_FALLBACK = []


def ref_base(d):
    """The reference forecast `b` for one item, from the PINNED table (reward-spec 6.3).

    `source` is the item's own: PhenoCal items carry `variant`, internal items do not and are
    `"internal"`. Reading it off the item is also what stops this file crashing on the internal
    traces, which it did -- `KeyError: 'variant'` -- on the task that is the actual RL target.

    Falls back to the legacy per-variant table only while `official/reference_rates_*.json` has not
    been built yet, and says so once. The legacy table has the N8 hole (no entry -> no reference at
    all for 104/304 internal unseen items); the pinned cascade closes it, so a number computed under
    the fallback is not the number the reward will produce.
    """
    src = d.get("variant") or "internal"
    try:
        return reward.base_rate(src, d["disease"], int(d["horizon"]))
    except FileNotFoundError:
        if not _REF_FALLBACK:
            _REF_FALLBACK.append(1)
            print("  [ref] official/reference_rates_*.json not built -- falling back to the legacy"
                  " per-variant table. Internal items with no train key have NO reference under it;"
                  " build the pinned one before quoting a skill score.")
        if src == "internal":
            return None
        return _base_rates(src).get((d["disease"], d["horizon"]))
    except KeyError:
        return None


def stated_report(tr):
    """(values, source, malformed) for the report the agent STATED -- ladder levels 1-2 only.

    Delegates to `cohortworld_env.reward`: `commit_payload` + `answer_line` + `parse_values` +
    `well_formed`, i.e. exactly the code the reward runs. There is no separate rule in gen/ any more
    (D2 / N11 / N14). `well_formed` is `MIN_REPORT` values, every one inside [0, 1]; a report with
    any value outside is malformed AS A WHOLE and rejected -- no drop, no clamp, no percent rescale.

    Use `full_report` when you want what the reward actually scores: this stops at the stated
    levels, so it can return nothing, while the reward's ladder always emits a report.
    """
    t = tr if isinstance(tr, ArchivedTrace) else ArchivedTrace(tr)
    malformed = False
    payload = reward.commit_payload(t)
    if payload is not None and payload.get("rates") is not None:
        vals = reward.parse_values(payload["rates"])
        if well_formed(vals):
            return vals, "stated_commit", False
        malformed = True
    stated = reward.answer_line(reward.last_reply(t))
    if stated is not None:
        vals = reward.parse_values(stated)
        if well_formed(vals):
            return vals, "stated_text", malformed
        malformed = malformed or bool(vals)
    return [], None, malformed


def full_report(tr, disease, horizon):
    """(values, level_name, malformed) -- the report the REWARD scores, all four ladder levels.

    Never empty: an episode that stated nothing is scored on the last rate it read, and one that
    read nothing on `FLOOR_REPORT`. That is D4 -- 5.8-18.3% of rollouts used to collapse onto one
    constant worth 1.008 reward units against a 0.098 sd of the informative signal, so "navigated,
    read six cells, format broke" was indistinguishable from "did nothing".
    """
    t = tr if isinstance(tr, ArchivedTrace) else ArchivedTrace(tr)
    vals, level, malformed = reward.rate_report(t, disease, int(horizon))
    return vals, reward.LEVEL_NAMES[level], bool(malformed)


def committed_rates(tr):
    """The stated report's values, or []. One rule, five callers -- see `stated_report`."""
    return stated_report(tr)[0]


def raw_committed_rates(tr):
    """The committed list WITHOUT the [0,1] filter.

    `committed_rates` applies the same range filter the environment does, which is right for
    reproducing what was scored but useless for asking whether anything was dropped -- the filtered
    list can never contain an out-of-range value, so a test on it always says "none". The scoring
    path silently drops out-of-range entries, so a committed [1,2,3,4,5] meaning 1%-5% is scored as
    a point mass at 100%. Use this to find those rollouts.
    """
    # Read through `reward.commit_payload`, not off the tool messages. The `committed` stop fires on
    # the assistant's tool CALL, so the commit's RESULT never becomes a trace node -- verifiers holds
    # it until the next model request and there is none -- and a scan of tool messages returns [] for
    # every such rollout, silently. Verified on one run: 10 of 12 rollouts state a commit and a
    # tool-message scan found none of them.
    payload = reward.commit_payload(tr if isinstance(tr, ArchivedTrace) else ArchivedTrace(tr))
    if payload is None or payload.get("rates") is None:
        return []
    return reward.parse_values(payload["rates"])          # parse_values does NO range filtering


def bands_of(tr):
    """Every at_risk value served to this rollout, in order."""
    out = []
    for t in tool_msgs(tr):
        if '"at_risk"' not in t:
            continue
        try:
            o = json.loads(t[t.index("{"):t.rindex("}") + 1])
        except Exception:
            continue
        if "at_risk" in o:
            out.append(o["at_risk"])
    return out


def band_mid(b):
    """Numeric midpoint of a band label, for a correlation. None if not a range."""
    if not isinstance(b, str):
        return None
    s = b.replace("+", "-").strip()
    parts = [p for p in s.split("-") if p.strip().isdigit()]
    if not parts:
        return None
    ns = [int(p) for p in parts]
    return sum(ns) / len(ns)


def analyse(path, label):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = len(rows)
    print(f"\n{'='*78}\n{label}   n={n}/120   file={path}\n{'='*78}")
    if not n:
        print("  no rollouts yet")
        return None

    stop = collections.Counter()
    cats = collections.Counter()
    per = []
    n10 = tot_tool = hit = miss = look = toolerr = n_malformed = 0
    levels = collections.Counter()
    nullmetric = []

    for r in rows:
        tr = r["traces"][0]
        d = r["task"]["data"]
        m = tr.get("metrics") or {}
        sc = tr.get("stop_condition")
        stop[sc] += 1
        cl = m.get("crps_loss")
        errs = tr.get("errors") or []
        if any("ToolsetError" in json.dumps(e) for e in errs) or "ToolsetError" in json.dumps(errs):
            toolerr += 1

        if not isinstance(cl, (int, float)) and sc not in ("error",):
            nullmetric.append((d["idx"], sc))


        for t in tool_msgs(tr):
            tot_tool += 1
            if '"bound": "n<10"' in t:
                n10 += 1
        h, ms, lk = nav_hits(tr)
        hit += h
        miss += ms
        look += lk

        bs = bands_of(tr)
        # RESCORE, do not read. `metrics.crps_loss` was written by whatever _crps was current when
        # the rollout ran; older traces carry the clamped, n*n-biased estimator, so
        # reading it silently mixes two scoring rules into one table. Recompute from the committed
        # values with the _crps imported above, and report how far the stored value drifted.
        stated, src, malformed = stated_report(tr)
        vals, level, _ = full_report(tr, d["disease"], d["horizon"])
        levels[level] += 1
        if malformed:
            n_malformed += 1
        base = ref_base(d)
        refl = 1.0 - _crps([base], d["true_rate"]) if base is not None else None
        # Every episode has a report now (reward-spec 1.1), so `cl` is always a real loss and the
        # NO_ANSWER sentinel has no referent. `answered` means the agent STATED it (level 1-2), and
        # the two headlines reward-spec 6.6 requires -- all episodes and stated-only -- are just
        # this list and its `answered` subset.
        cl_new = 1.0 - _crps(vals, d["true_rate"])
        answered = bool(stated)
        if not answered:
            if sc == "error":
                cats["error"] += 1
            elif sc == "context_length":
                cats["context_length"] += 1
            elif sc == "out_of_budget":
                cats["out_of_budget"] += 1
            elif m.get("truncated") == 1.0:
                cats["truncated"] += 1
            else:
                cats[f"finished_no_answer({sc})"] += 1
        per.append(dict(idx=d["idx"], stop=sc, cl=cl_new, cl_stored=cl, answered=answered,
                        level=level, src=src, malformed=malformed,
                        raw=raw_committed_rates(tr),
                        truth=d["true_rate"], base=base, vals=vals, bands=bs,
                        atext=" ".join(assistant_text(tr)),
                        ref=refl, ref_m=m.get("crps_ref_loss")))

    # ---- completion / stop table
    print(f"  stop_condition: {dict(stop)}")
    print(f"  report_level (reward-spec 6.6): {dict(levels)}"
          f"   malformed reports rejected: {n_malformed}")
    print(f"  NOT STATED by the agent {sum(cats.values())}/{n} = {sum(cats.values())/n*100:.1f}%"
          f" (scored on the ladder, not on a sentinel)  ->  {dict(cats)}")
    if nullmetric:
        print(f"  !! NULL-METRIC with normal stop: {len(nullmetric)} -> {nullmetric[:6]}")
    if toolerr:
        print(f"  ToolsetError rollouts: {toolerr}")

    # ---- crps
    issued = [p["cl"] for p in per]
    ans = [p for p in per if p["answered"]]
    refs = [p["ref"] for p in per if isinstance(p["ref"], (int, float))]
    mref = sum(refs) / len(refs) if refs else None
    if mref:
        print(f"  base-rate reference loss (computed, n={len(refs)}): {mref:.4f}")
    drift = [abs(p["cl"] - p["cl_stored"]) for p in per
             if isinstance(p["cl_stored"], (int, float)) and p["answered"]]
    if drift and max(drift) > 1e-6:
        print(f"  RESCORED: stored crps_loss differs on {sum(1 for x in drift if x>1e-6)}/{len(drift)}"
              f" answered rollouts, max |delta| {max(drift):.4f}, mean {sum(drift)/len(drift):+.4f}"
              f"  (traces predate the estimator fix)")
    # Skill is a RATIO OF TWO MEANS and both must be taken over the SAME items. Averaging the model
    # over all n while averaging the reference over only the items that have a base rate is the
    # N8 defect in tooling form: the internal reference is built from the train split and the unseen
    # split is held-out chapters, so 104/304 unseen items have no reference at all. `skill_on`
    # therefore drops any item without one from BOTH sides and prints the n it used.
    def skill_on(rows, tag):
        common = [p for p in rows if isinstance(p["ref"], (int, float))]
        if not common:
            return
        mm = sum(p["cl"] for p in common) / len(common)
        rr = sum(p["ref"] for p in common) / len(common)
        drop = len(rows) - len(common)
        print(f"    skill vs base-rate ref ({tag}, common n={len(common)}"
              f"{f', {drop} dropped for no reference' if drop else ''}): {1-mm/rr:+.4f}")

    print(f"\n  crps ALL EPISODES (n={n}, every one scored on its ladder report): {sum(issued)/n:.4f}")
    skill_on(per, "issued")
    if ans:
        ca = [p["cl"] for p in ans]
        print(f"  crps STATED-ONLY subset (n={len(ans)}): {sum(ca)/len(ca):.4f}"
              f"   median {st.median(ca):.4f}")
        skill_on(ans, "answered")
        ae = [abs(st.median(p["vals"]) - p["truth"]) for p in ans if p["vals"]]
        if ae:
            print(f"  median |predicted(median) - truth| (n={len(ae)}): {st.median(ae):.5f}")

    # ---- n<10 / navigation
    print(f"\n  tool returns {tot_tool}   bound n<10 {n10}"
          f" ({n10/tot_tool*100:.2f}%)" if tot_tool else "  no tool returns")
    if hit + miss:
        print(f"  navigation hit rate (cw_move only): {hit}/{hit+miss} = {hit/(hit+miss)*100:.1f}%"
              f"   [+{look} cw_look calls, which always report hit and are NOT navigation]")

    # ---- at_risk usage
    allb = [b for p in per for b in p["bands"]]
    cb = collections.Counter(map(str, allb))
    print(f"\n  at_risk served: {len(allb)} values over {sum(1 for p in per if p['bands'])} rollouts")
    for b, c in cb.most_common():
        print(f"      {b:<34} {c:>5}  {c/len(allb)*100:>5.1f}%")

    served = [p for p in per if p["bands"]]
    ref_kw = sum(1 for p in served
                 if any(str(b) in p["atext"] for b in p["bands"])
                 or "at risk" in p["atext"].lower() or "at-risk" in p["atext"].lower())
    if served:
        print(f"  rollouts referencing at_risk in later reasoning: {ref_kw}/{len(served)}"
              f" = {ref_kw/len(served)*100:.0f}%")

    # ---- band vs prediction behaviour
    rowsb = []
    for p in ans:
        if not p["vals"] or p["base"] is None:
            continue
        mids = [band_mid(b) for b in p["bands"]]
        mids = [x for x in mids if x is not None]
        if not mids:
            continue
        mid = st.median(mids)
        med = st.median(p["vals"])
        spread = max(p["vals"]) - min(p["vals"])
        # How far the point estimate sits from the base rate.
        #
        # Report the UNDIVIDED distance. The earlier version divided by the base rate, and that
        # scaling manufactures the very correlation this is meant to detect: band_mid is
        # anti-correlated with the base rate (spearman -0.18 to -0.26 in all three arms), so
        # dividing by base inflates the statistic exactly where band_mid is large. Under the
        # divided form arm B read spearman +0.328 and looked like it was shrinking on at_risk;
        # undivided the signal disappears (A -0.245, B +0.057, C -0.106), and the placebo arm shows
        # the same inflated +0.376, which no clinician-prompt story can explain.
        shrink = abs(med - p["base"]) if p["base"] else None
        rowsb.append((mid, spread, shrink, med))
    if len(rowsb) >= 8:
        print(f"\n  band-midpoint vs behaviour (n={len(rowsb)} answered rollouts with a numeric band)")
        qs = sorted(rowsb)
        half = len(qs) // 2
        for nm, grp in (("low  at_risk", qs[:half]), ("high at_risk", qs[len(qs)-half:])):
            sp = [g[1] for g in grp]
            sh = [g[2] for g in grp if g[2] is not None]
            print(f"      {nm}: n={len(grp)} midpoint med {st.median([g[0] for g in grp]):.1f}"
                  f"  spread med {st.median(sp):.5f}"
                  f"  |pred-base| med {st.median(sh):.5f}" if sh else "")
        try:
            import statistics as _s
            xs = [g[0] for g in rowsb]
            for nm, j in (("spread", 1), ("|pred-base|", 2)):
                ys = [g[j] for g in rowsb if g[j] is not None]
                xx = [g[0] for g in rowsb if g[j] is not None]
                if len(ys) > 3 and _s.pstdev(xx) and _s.pstdev(ys):
                    print(f"      pearson r(band_mid, {nm}) = {_s.correlation(xx, ys):+.3f}  n={len(ys)}")
        except Exception as e:
            print(f"      corr unavailable: {e}")
    else:
        print(f"\n  band-vs-behaviour: only {len(rowsb)} usable rollouts, not reported")

    return per


HI = 0.15   # the stratum boundary; see `strata`'s docstring


def strata(per, label):
    """Report split by true rate, because the aggregate is decided by a minority of items.

    23% of PhenoCal_120 items have a true rate above 0.15 and they carry 60-67% of the whole loss.
    Scored against the read-nothing reference on the SAME items -- which carries the same scale
    exposure, so this is not the arithmetic of larger truths giving larger errors -- the model is
    2.0-2.5x worse on them, while on the other 77% arm B actually BEATS that reference. An
    unstratified headline reports the 23% and calls it the benchmark.

    The subset is identifiable before the answer: its median base rate is 0.2257 against 0.0542,
    r(base rate, is-high) = +0.661, and a single base-rate threshold separates them at 87% against
    a 76% majority baseline. So the agent can see which regime it is in.
    """
    rows = [p for p in per if p["answered"] and p["vals"] and p["base"] is not None
            and p["raw"] and max(p["raw"]) <= 1.0]
    if len(rows) < 20:
        print(f"  [{label}] strata: only {len(rows)} clean answered rollouts, not reported")
        return
    oor = sum(1 for p in per if p["raw"] and max(p["raw"]) > 1.0)
    # rollouts that answered on the ANSWER line rather than through the commit tool have no raw
    # list, so their scale cannot be checked; they are excluded too and counted separately rather
    # than folded into the out-of-range count.
    noraw = sum(1 for p in per if p["answered"] and p["vals"] and not p["raw"])
    print(f"\n  [{label}] BY TRUE RATE (excluded: {oor} out-of-range commits,"
          f" {noraw} answered without a commit-tool list)")
    print(f"      {'subset':>14} {'n':>4} {'model':>8} {'ref':>8} {'skill':>8} {'under>2x':>9}")
    for name, sub in ((f"truth > {HI}", [p for p in rows if p["truth"] > HI]),
                      (f"truth <= {HI}", [p for p in rows if p["truth"] <= HI])):
        if not sub:
            continue
        m = sum(1.0 - _crps(p["vals"], p["truth"]) for p in sub) / len(sub)
        r = sum(1.0 - _crps([p["base"]*x for x in (0.5, 0.75, 1.0, 1.25, 1.5)], p["truth"])
                for p in sub) / len(sub)
        und = sum(1 for p in sub if st.median(p["vals"]) < p["truth"] * 0.5)
        print(f"      {name:>14} {len(sub):>4} {m:>8.4f} {r:>8.4f} {1-m/r:>+8.3f}"
              f" {und:>4} ({und/len(sub)*100:>3.0f}%)")


def floor_check(per, label):
    """The trivial policy: base rate +/-50% as five values, reading nothing."""
    if not per:
        return
    L, R = [], []
    for p in per:
        if p["base"] is None:
            continue
        b = p["base"]
        pred = [b * 0.5, b * 0.75, b, b * 1.25, b * 1.5]
        L.append(1.0 - _crps(pred, p["truth"]))
        R.append(p["ref"])
    if L:
        print(f"  [{label}] trivial base-rate+/-50% policy: crps={sum(L)/len(L):.4f}"
              f"  skill vs ref = {1-(sum(L)/len(L))/(sum(R)/len(R)):+.4f}  n={len(L)}")


def main():
    pats = sys.argv[1:]
    allper = {}
    for pat in pats:
        for f in sorted(glob.glob(pat), key=os.path.getmtime):
            dn = os.path.dirname(f)
            lab = os.path.basename(f) if "artifacts" in dn else os.path.basename(dn)[-8:]
            allper[lab] = analyse(f, lab)
    print(f"\n{'='*78}\nTRIVIAL-POLICY FLOOR\n{'='*78}")
    for lab, per in allper.items():
        floor_check(per, lab)
    print(f"\n{'='*78}\nSTRATIFIED BY TRUE RATE\n{'='*78}")
    for lab, per in allper.items():
        if per:
            strata(per, lab)


if __name__ == "__main__":
    main()
