#!/usr/bin/env python3
"""AUROC and calibration slope for a SINGLE-TURN (leaderboard-protocol) PhenoCal arm.

Input is the record written by gen/singleturn_cohortcal.py: one JSON with `rows` of
{idx, true_rate, disease, horizon, pred, pred_board_parser, text}. The two-stage CohortWorld cells
are scored by gen/score_cohortworld_phenocal.py; this is the same metrics on a different INPUT
FORMAT, so nothing here restates a threshold, a label rule or a bootstrap:

  * the per-(disease, horizon) threshold comes from `score_cohortworld_phenocal.floor_for` -- the
    same function that cell uses, called rather than copied, with `--b-source` choosing train-split
    mean (default, the CohortCal table) or UKB marginal;
  * label / score / AUROC come from `binary_metrics.evaluate` (label = t > f, score = p / f);
  * the bootstrap is `score_internal_t3.boot`;
  * slope = polyfit(truth, pred, 1)[0], the board's orientation.

Both parsers are reported side by side. They are not interchangeable: `parse_pct` (the board's) takes
the FIRST number with no range check, which on a reasoning block returns a number from the reasoning,
and `parse_last_pct` takes the LAST percent after cutting the block and range-checks it. On the
24-item check they disagreed on 20/24, so the choice of parser IS the measurement.

    gen/score_singleturn.py --rec <single-turn.json> --label NAME [--version _C2]
    gen/score_singleturn.py --selfcheck
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cohortworld_env"))
from binary_metrics import evaluate                                  # noqa: E402
from score_cohortworld_phenocal import floor_for, B_SOURCES          # noqa: E402
from score_internal_t3 import boot                                   # noqa: E402


def arm(rows, floor, field):
    """(truth, pred, thr) triples for one parser's column, plus the rows it could not parse."""
    keep, dropped = [], 0
    for r in rows:
        p = r.get(field)
        k = (str(r.get("disease")), int(r["horizon"])) if r.get("horizon") is not None else None
        b = floor(k) if k is not None else None
        if p is None or b is None:
            dropped += 1
            continue
        keep.append((float(p), float(r["true_rate"]), float(b)))
    return keep, dropped


def report(tag, trip, dropped, boot_n):
    if len(trip) < 3:
        print(f"  {tag:18s} NOT SCORABLE ({len(trip)} rows, {dropped} dropped)")
        return
    pred = np.array([p for p, _t, _f in trip], float)
    truth = np.array([t for _p, t, _f in trip], float)
    thr = np.array([f for _p, _t, f in trip], float)
    rep = evaluate([(float(p), float(t), float(f)) for p, t, f in trip])
    slope = float(np.polyfit(truth, pred, 1)[0])
    ci = boot(lambda t, p: float(np.polyfit(t, p, 1)[0]) if len(set(t.tolist())) > 1 else None,
              truth, pred, n=boot_n)
    au, auci = rep.get("auroc"), rep.get("auroc_ci")
    print(f"  {tag:18s} n {len(trip):6d}  dropped {dropped:5d}  "
          f"AUROC {au:.4f}" + (f" [{auci[0]:.4f}, {auci[1]:.4f}]" if auci else "") +
          f"  slope {slope:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]  pred_mean {pred.mean():.4f}")
    return truth, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--version", default="_C2")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--b-source", choices=B_SOURCES, default=os.environ.get("CW_B_SOURCE", "train"),
                    help="threshold b: CohortCal train-split mean (paper Table 3) or UKB marginal (Table 4)")
    a = ap.parse_args()

    rec = json.load(open(a.rec))
    rows = rec["rows"]
    floor = floor_for(a.b_source, a.version)
    print(f"{a.label}")
    print(f"  model {rec.get('model')}  sys {os.path.basename(rec.get('sys',''))} "
          f"md5 {rec.get('sys_md5','')[:8]}  maxtok {rec.get('maxtok')}  n {rec.get('n')}")
    print(f"  cohort-think opened {rec.get('opened')} closed {rec.get('closed')}  "
          f"parsed {rec.get('parsed')}/{rec.get('n')}  distinct preds {rec.get('distinct_preds')}  "
          f"parsers disagree on {rec.get('board_parser_disagrees')}")

    got = None
    for tag, field in (("parse_last_pct", "pred"), ("board parse_pct", "pred_board_parser")):
        trip, dropped = arm(rows, floor, field)
        r = report(tag, trip, dropped, a.boot)
        if got is None and r is not None:
            got = r
    if got is not None:
        truth, thr = got
        fl = evaluate([(float(f), float(t), float(f)) for t, f in zip(truth, thr)])
        print(f"  {'-- floor (pred=thr)':18s} AUROC {fl.get('auroc'):.4f}  "
              f"slope {float(np.polyfit(truth, thr, 1)[0]):.4f}")


def selfcheck():
    """Both controls must FAIL the naive version: an unparsed row must not score, and a row whose
    (disease, horizon) has no threshold must not score."""
    floor = {("A", 5): 0.10}.get
    rows = [{"true_rate": 0.2, "disease": "A", "horizon": 5, "pred": 0.3},
            {"true_rate": 0.2, "disease": "A", "horizon": 5, "pred": None},     # unparsed
            {"true_rate": 0.2, "disease": "Z", "horizon": 5, "pred": 0.3},      # no threshold
            {"true_rate": 0.2, "disease": "A", "horizon": None, "pred": 0.3}]   # no horizon
    keep, dropped = arm(rows, floor, "pred")
    assert len(keep) == 1, keep
    assert dropped == 3, dropped
    print("selfcheck PASS: 1 scorable row kept, 3 dropped (unparsed / no threshold / no horizon)")
    print("  control -- without the guards all 4 would score, two of them against a missing threshold")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        main()
