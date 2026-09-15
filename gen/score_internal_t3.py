#!/usr/bin/env python3
"""AUROC and calibration slope for an INTERNAL CohortWorld T3 run.

`gen/score_cohortworld_phenocal.py` cannot do this: its per-(disease, horizon) floor is built from
PhenoCal's base rates, which are indexed by PhenoCal's disease vocabulary, so an internal item whose
disease is a lattice code asserts out -- `AssertionError: no floor for ('Q', 5)`. The internal half needs
the internal reference, and the internal reference already exists and is already what the REWARD used:
`cohortworld_env.reward.base_rate(source, disease, horizon)`, called by cs_grpo._read as "the judge's own
binarising threshold, called rather than restated". This script calls the same function. Nothing here
restates a threshold, a report ladder or a metric.

  * the report is read through `arm_audit.full_report` == `reward.rate_report` -- the exact ladder the
    reward scored, the same one score_cohortworld_phenocal uses, so the external and internal cells are
    read by one object rather than two copies of an intention;
  * point = the report's median, the statistic the board's R uses;
  * label = truth > base_rate, which is `contribution_metric`'s own `y = (t > f)`;
  * score = point / base_rate, which is `contribution_metric`'s own `s = p / f` -- a ratio, because the
    items span two orders of magnitude of base rate and a raw point would rank the common diseases first;
  * slope = polyfit(truth, point, 1)[0], the board's orientation (score_cohortworld_phenocal:10).

  gen/score_internal_t3.py --traces <traces.jsonl> --label NAME [--source default] [--boot 2000]
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cohortworld_env"))
from arm_audit import full_report                     # noqa: E402
from binary_metrics import evaluate                   # noqa: E402
from cohortworld_env import reward as R               # noqa: E402


def _why(tr):
    """The failure class of an errored rollout, short enough to be a report_level key."""
    es = tr.get("errors") or []
    if not es:
        return str(tr.get("stop_condition") or "unknown")
    m = str(es[0].get("message") or "")
    return "uv" if "prepare uv script" in m else (es[0].get("type") or "error")


def rows(path, source):
    """(truth, base, point, report_level, ok, idx) per record, skipping errored rollouts.

    idx is appended LAST so every existing r[0..4] index keeps its meaning. It exists so rows can be
    restricted to the exact items another arm completed, which is what makes a like-for-like
    comparison possible when one arm was stopped early."""
    out = []
    for line in open(path, errors="ignore"):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        d = (rec.get("task") or {}).get("data") or {}
        truth, dis, hor = d.get("true_rate"), d.get("disease"), d.get("horizon")
        if truth is None or not dis or hor is None:
            continue
        trs = rec.get("traces") or []
        if not trs:
            continue
        # An episode whose HARNESS never started is not a prediction. `full_report` is the reward's
        # ladder and by design never returns empty (D4: a rollout that stated nothing is still scored,
        # on FLOOR_REPORT) -- so the `if not vals` branch below could never fire, and every rollout
        # killed by `HarnessError: failed to prepare uv script ... Bus error` entered the metric as the
        # floor constant, median 0.5. Measured at 7.3% of traces in one training run. The reward must
        # keep scoring those rows; the MEASUREMENT must not.
        if trs[0].get("ok") is False:
            out.append((float(truth), None, None, "ERRORED:" + _why(trs[0]), False, d.get("idx")))
            continue
        # exactly score_cohortworld_phenocal.py:51 -- (values, level_name, malformed)
        vals, level, _malformed = full_report(trs[0], dis, hor)
        if not vals:
            out.append((float(truth), None, None, level, False, d.get("idx")))
            continue
        point = float(np.median(sorted(float(x) for x in vals)))
        base = float(R.base_rate(source, str(dis), int(hor)))
        out.append((float(truth), base, point, level, True, d.get("idx")))
    return out


def boot(fn, *arrs, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    k = len(arrs[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, k, k)
        try:
            v = fn(*[a[idx] for a in arrs])
        except Exception:
            continue
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--label", required=True)
    # DERIVED, not restated: this is CSGRPOAlgoConfig.rate_source's default, i.e. the table that
    # binarised the truth during training. A wrong value here silently scores against another
    # population's base rates; "default" is not even a key in the reference file and raised KeyError,
    # which is the good failure -- but the fix is to name the training default, not to guess again.
    ap.add_argument("--source", default="internal",
                    choices=["internal", "PhenoCal", "PhenoCal_C1", "PhenoCal_C2"])
    ap.add_argument("--only-idx", default=None,
                    help="file of item idx, whitespace separated; score ONLY those rows, for a matched"
                         " comparison against an arm stopped early")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    rs = rows(a.traces, a.source)
    if a.only_idx:
        keep = {int(x) for x in open(a.only_idx).read().split() if x.strip().lstrip("-").isdigit()}
        before = len(rs)
        rs = [r for r in rs if r[5] in keep]
        miss = keep - {r[5] for r in rs}
        print(f"  RESTRICTED to {len(keep)} idx from {a.only_idx}: kept {len(rs)} of {before} rows")
        if miss:
            print(f"  WARNING {len(miss)} requested idx are ABSENT from these traces -- this row is NOT")
            print( "          the matched comparison it claims to be; report the shortfall.")
    scored = [r for r in rs if r[4]]
    truth = np.array([r[0] for r in scored], float)
    base = np.array([r[1] for r in scored], float)
    point = np.array([r[2] for r in scored], float)
    lab = (truth > base).astype(int)

    print(f"{a.label}")
    print(f"  rollouts read       {len(rs)}   with a usable report {len(scored)} "
          f"({100*len(scored)/max(len(rs),1):.1f}%)")
    lv = {}
    for r in rs:
        lv[r[3]] = lv.get(r[3], 0) + 1
    print(f"  report_level        {dict(sorted(lv.items(), key=lambda x: (x[0] is None, x[0])))}")
    if len(scored) < 3 or len(set(lab)) < 2:
        print("  NOT SCORABLE: fewer than 3 usable reports, or one class only.")
        return
    print(f"  label balance       {int(lab.sum())} positive / {len(lab)-int(lab.sum())} negative "
          f"(label = truth > base_rate)")

    # `evaluate` takes (predicted, true, threshold) and derives the label (t > f), the score (p / f)
    # and its own bootstrap CI internally. The first version of this script recomputed all three
    # outside it -- a second copy of a fact, which is the failure the repo's own rule names. Deleted.
    rep = evaluate([(float(p), float(t), float(b)) for p, t, b in zip(point, truth, base)])
    slope = float(np.polyfit(truth, point, 1)[0])
    ci_sl = boot(lambda t, p: float(np.polyfit(t, p, 1)[0]) if len(set(t.tolist())) > 1 else None,
                 truth, point, n=a.boot)
    # DEGENERACY FLAG, same concept as score_cohortworld_phenocal's. An arm that never states and never
    # reads is scored on FLOOR_REPORT, so every prediction is the same constant. slope is then 0 by
    # construction and AUROC stays defined but reflects only the 1/base_rate ordering -- it measured
    # 0.4666 on a trained arm and 0.9236 on a synthetic constant arm, i.e. the value is arbitrary. Labelled
    # here so a degenerate cell cannot sit unlabelled beside a real one in the same table.
    # TWO EXTRA COLUMNS, for the paper's tables. Neither touches AUROC, slope or the
    # read rate above; they are additional views of the SAME (point, truth, base) triple.
    #   |r_hat-y|/b : absolute error in units of the base rate, so items spanning two orders of magnitude
    #                 of b are comparable. UNDEFINED where b == 0, so computed on the b > 0 rows ONLY with
    #                 that denominator printed -- over all rows it would divide by zero on 22.7% of this
    #                 cell, which is exactly the two-denominator trap this script already warns about.
    #   beats base  : share with |r_hat - y| < |b - y|, the prediction strictly closer to truth than
    #                 answering the base rate. No division, so defined on EVERY row.
    pos = base > 0
    if int(pos.sum()):
        eob = np.abs(point[pos] - truth[pos]) / base[pos]
        print(f"  |r_hat-y|/b         median {float(np.median(eob)):.4f}  mean {float(np.mean(eob)):.4f}"
              f"   ({int(pos.sum())} rows with b > 0 of {len(point)})")
    else:
        print("  |r_hat-y|/b         undefined: no row has b > 0")
    beats = np.abs(point - truth) < np.abs(base - truth)
    print(f"  beats base rate     {int(beats.sum())}/{len(beats)} = {100*float(beats.mean()):.1f}%"
          f"   (|r_hat-y| < |b-y|, all rows)")
    ndp = len(set(point.tolist()))
    if ndp <= 1:
        print(f"  DEGENERATE: all {len(point)} predictions are the same value ({point[0]:.4f}) -- this arm")
        print( "              produced no prediction. slope is 0 by construction; AUROC below reflects")
        print( "              only the 1/base_rate ordering and is not a discrimination measurement.")
    else:
        print(f"  distinct predictions {ndp}")
    # TWO DENOMINATORS, printed because they differ and the single "usable report" line above implied
    # they did not. `binary_metrics.evaluate` drops every row whose threshold is not > 0 (the score is
    # p/f), and base_rate's cascade returns 0.0 for a (disease, horizon) that no one in the train split
    # was ever diagnosed with: 211 of 929 rows in one measured run, 22.7%. So the AUROC is computed on
    # 718 rows and the slope on 929, and an internal AUROC quoted without its denominator carries the
    # smaller one.
    n_auroc = int(rep.get("n") or 0)
    n_drop = len(point) - n_auroc
    print(f"  AUROC rows           {n_auroc} of {len(point)}  "
          f"({n_drop} dropped: threshold b == 0, so p/f is undefined)")
    print(f"  slope rows           {len(point)}")
    au, auci = rep.get("auroc"), rep.get("auroc_ci")
    print(f"  AUROC               {au:.4f}" + (f"  95% CI [{auci[0]:.4f}, {auci[1]:.4f}]" if auci else ""))
    print(f"  calibration slope   {slope:.4f}  95% CI [{ci_sl[0]:.4f}, {ci_sl[1]:.4f}]")
    print(f"  acc {rep.get('acc'):.4f}  balanced {rep.get('balanced'):.4f}  f1 {rep.get('f1'):.4f}"
          if rep.get('acc') is not None else "")

    # THE FLOOR on the same rows: predict each item's own base rate. It carries the ranking of the base
    # rates and nothing patient-specific, so it is what a model must beat -- not 0.5.
    fl = evaluate([(float(b), float(t), float(b)) for t, b in zip(truth, base)])
    fl_sl = float(np.polyfit(truth, base, 1)[0])
    print(f"  -- floor (pred = base rate): AUROC {fl.get('auroc'):.4f}  slope {fl_sl:.4f}")


def selfcheck():
    """The filter must be able to fail. Both rows are identical except for `ok`."""
    import tempfile
    rec = lambda ok: json.dumps({
        "task": {"data": {"true_rate": 0.5, "disease": "A", "horizon": 5}},
        "traces": [{"ok": ok, "nodes": [],
                    "errors": [] if ok else [{"type": "HarnessError",
                                              "message": "harness setup: failed to prepare uv script"}]}]})
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write(rec(True) + "\n" + rec(False) + "\n")
        path = fh.name
    import cohortworld_env.reward as _R
    _R.base_rate = lambda *a, **k: 0.1          # the floor table has no ("A", 5)
    got = rows(path, "internal")
    os.unlink(path)
    kept = [r for r in got if r[4]]
    assert len(got) == 2, got
    assert len(kept) == 1, f"the ok=False row was NOT dropped: {got}"
    assert got[1][3] == "ERRORED:uv", got[1][3]
    print("selfcheck PASS: ok=True kept, ok=False dropped and labelled ERRORED:uv")
    print("  control -- without the filter both rows would score, the second at floor median 0.5")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        main()
