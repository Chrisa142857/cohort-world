#!/usr/bin/env python3
"""Score an archived CohortWorld run on PhenoCal items with the PhenoCal board's own metrics.

    gen/score_cohortworld_phenocal.py --traces <traces.jsonl> --model qwen3.5-9b --out DIR

Emits the leaderboard's artifact schema (slope, corr, mae, ece, bss, binary{...})
so the figure scripts can load a CohortWorld cell beside a single-turn cell without a third code
path. Nothing is restated:
  * the report is read through `arm_audit.full_report` == `cohortworld_env.reward.rate_report`, the
    exact ladder the reward scored (never empty);
  * the point prediction is the report's median, the same statistic R uses (R = |b-y| - |median(F)-y|);
  * the per-(disease,horizon) floor is built exactly as score_claude_arm.py builds it;
  * slope = np.polyfit(truth, pred, 1)[0], the board's orientation; ACC/F1/AUROC via binary_metrics.evaluate.
"""
import argparse, json, os, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cohortworld_env"))
from arm_audit import full_report          # noqa: E402
from binary_metrics import evaluate         # noqa: E402
from parsers import ece                     # noqa: E402


def _finite(o):
    """Replace every non-finite float with None, in place, at any depth.

    `json.dump` writes NaN / Infinity as bare literals: Python reads them back, jq and JSON.parse do
    not, so a released artifact silently becomes unparseable for every non-Python consumer.
    """
    import math
    if isinstance(o, dict):
        for k, v in list(o.items()):
            if isinstance(v, float) and not math.isfinite(v):
                o[k] = None
            else:
                _finite(v)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            if isinstance(v, float) and not math.isfinite(v):
                o[i] = None
            else:
                _finite(v)
    return o


B_SOURCES = ("train", "ukb-marginal")


def floor_for(b_source, version):
    """(disease, horizon) -> b, the per-cell threshold every label and AUROC score is relative to.

    `train`: the CohortCal TRAIN-split mean per (disease, horizon), with the reward's own cascade
    (pair -> disease -> horizon -> grand mean), read from official/reference_rates_*.json by
    `cohortworld_env.reward.base_rate`. This is the b of the paper's CohortCal table.
    `ukb-marginal`: the fixed UKB marginal incidence of (disease, horizon), `gen/thresholds.py`; the b
    of the paper's CohortWorld tasks table.
    Never a mean over the full panel: that mean includes the test items' own outcomes.

    Extracted so the single-turn scorer CALLS it instead of restating it.
    """
    if b_source == "train":
        from cohortworld_env import reward as _R
        src = f"PhenoCal{version}"
        return lambda k: float(_R.base_rate(src, k[0], int(k[1])))
    if b_source == "ukb-marginal":
        from thresholds import base_rate as _marginal
        ver = version.lstrip("_") or "default"
        return lambda k: _marginal(k[0], int(k[1]), ver)
    raise ValueError(f"--b-source must be one of {B_SOURCES}, got {b_source!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--version", default="_C2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--b-source", choices=B_SOURCES, default=os.environ.get("CW_B_SOURCE", "train"),
                    help="threshold b: CohortCal train-split mean (paper Table 3) or UKB marginal (Table 4)")
    a = ap.parse_args()

    floor = floor_for(a.b_source, a.version)

    P, T, FL, LV, levels, n_rows, n_ok = [], [], [], [], {}, 0, 0
    for line in open(a.traces):
        if not line.strip():
            continue
        r = json.loads(line); n_rows += 1
        if not r.get("ok", True) or not r.get("traces"):
            continue
        d = r["task"]["data"]; tr = r["traces"][0]
        vals, level, _ = full_report(tr, d["disease"], d["horizon"])
        if not vals:
            continue
        k = (str(d["disease"]), int(d["horizon"]))
        b = floor(k)
        assert b is not None, f"no {a.b_source} threshold for {k}"
        P.append(st.median(vals)); T.append(float(d["true_rate"])); FL.append(b); LV.append(level)
        levels[level] = levels.get(level, 0) + 1; n_ok += 1

    # HOW MUCH OF THE REPORT WAS RESTATED RATHER THAN READ.
    # CORRECTION. An earlier version of this block claimed rung 3 `auto_observed`
    # "substitutes the last reading for an answer the model did not give". That is WRONG, and the
    # training stream settles it: at step 95 the policy's own rollouts end
    #   cw_look{} -> cw_move{move_id} -> cw_observe{} -> cw_commit{}
    # with cw_observe AND cw_commit carrying empty arguments, and the reward read exactly that rung.
    # `auto_observed` IS the trained answer channel: the reading the agent chose to stand on, with the
    # declared difference left at zero. It is not a substitution and not a measurement artifact.
    # What the `answered_*` keys measure is narrower and still worth having: the share of episodes that
    # RESTATED a number explicitly (rung 1 `stated_commit` or rung 2 `stated_text`) instead of letting
    # the reading stand. That is the only channel able to express a spread, so it bounds what any
    # dispersion-sensitive metric could ever see. It is NOT a count of "answers".
    STATED = ("stated_commit", "stated_text")
    ans = [i for i, lv in enumerate(LV) if lv in STATED]
    res_ans = {"n_answered": len(ans),
               "answer_rate": round(len(ans) / max(n_ok, 1), 4)}
    if len(ans) >= 20:
        aP = np.clip(np.array([P[i] for i in ans]), 0, 1)
        aT = np.array([T[i] for i in ans]); aFL = np.array([FL[i] for i in ans])
        if len(set((aT > aFL).tolist())) > 1:
            ra = evaluate([(float(x), float(t), float(f)) for x, t, f in zip(aP, aT, aFL)])
            res_ans["answered_auroc"] = round(float(ra["auroc"]), 4)
            res_ans["answered_auroc_ci"] = [round(float(x), 4) for x in (ra.get("auroc_ci") or (float("nan"),) * 2)]
            res_ans["answered_slope"] = round(float(np.polyfit(aT, aP, 1)[0]), 4)
            res_ans["answered_n_distinct_pred"] = int(len(set(aP.tolist())))

    a_, b_, fl_ = np.clip(np.array(P), 0, 1), np.array(T), np.array(FL)
    res = {"model": a.model, "version": a.version.lstrip("_"), "arm": "cohortworld", "traces": a.traces,
           "n_rows": n_rows, "n_parsed": n_ok, "fmt": round(n_ok / max(n_rows, 1), 4), "report_levels": levels,
           "pred_mean": round(float(a_.mean()), 5),
           "slope": round(float(np.polyfit(b_, a_, 1)[0]), 4),
           "corr": round(float(np.corrcoef(a_, b_)[0, 1]), 4),
           "mae": round(float(np.mean(np.abs(a_ - b_))), 5),
           "ece": round(float(ece(a_, b_)), 5),
           "bss": round(1 - float(np.sum((a_ - b_) ** 2)) / max(float(np.sum((fl_ - b_) ** 2)), 1e-12), 4),
           "binary": evaluate([(float(x), float(t), float(f)) for x, t, f in zip(a_, b_, fl_)])}
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"phenocal_{res['version']}_cohortworld_{a.model}_bss{res['bss']}_auroc{res['binary']['auroc']:.4f}_n{n_ok}.json")
    _finite(res)
    json.dump(res, open(path, "w"), indent=1, allow_nan=False)
    print(path)
    print(f"  n {n_ok}/{n_rows}  levels {levels}  slope {res['slope']}  bss {res['bss']}  auroc {res['binary']['auroc']:.4f} {res['binary']['auroc_ci']}")
    # THE FLOOR ON THE SAME ROWS -- predict each item's own threshold. Without it a slope has no
    # referent: 0.2959 reads as "poorly calibrated" only once you know the floor on those rows is
    # 0.587, and an AUROC reads as discrimination only against 0.5000. `bss` is already floor-relative;
    # these two are not, and they were the two being quoted.
    # DEGENERACY FLAG. An arm that never states and never reads is scored on FLOOR_REPORT, so every
    # prediction is the same constant and `corr` is NaN (json.dump would emit the literal `NaN`, which
    # Python reads back but jq / JSON.parse / jsonlite reject). Worse, AUROC stays *defined*: the score
    # is p/f, so a constant p ranks purely by 1/base_rate and can land anywhere -- measured 0.9236 on a
    # 40-item synthetic constant arm and 0.4666 on a trained arm. Such a cell is not a discrimination
    # measurement and must not be read as one, so it is labelled here rather than quietly reported.
    res.update(res_ans)
    res["n_distinct_pred"] = int(len(set(a_.tolist())))
    res["degenerate_constant_prediction"] = bool(res["n_distinct_pred"] <= 1)
    if res["degenerate_constant_prediction"]:
        print("  WARNING: every prediction is the same value -- this arm produced no prediction. "
              "slope is 0 by construction and auroc reflects only the 1/base_rate ordering.")
    fl = evaluate([(float(f), float(t), float(f)) for t, f in zip(b_, fl_)])
    res["floor_auroc"] = round(float(fl["auroc"]), 4)
    res["floor_slope"] = round(float(np.polyfit(b_, fl_, 1)[0]), 4)
    _finite(res)                     # no NaN/Infinity literals in a released artifact
    json.dump(res, open(path, "w"), indent=1, allow_nan=False)
    print(f"  -- floor (pred = that item's base rate): auroc {res['floor_auroc']:.4f}  slope {res['floor_slope']:.4f}")
    print(f"  -- RESTATED explicitly (stated_commit|stated_text): {res['n_answered']}/{n_ok} "
          f"= {100*res['answer_rate']:.1f}%" +
          (f"   auroc {res['answered_auroc']:.4f} {res['answered_auroc_ci']}  "
           f"slope {res['answered_slope']:.4f}  distinct {res['answered_n_distinct_pred']}"
           if "answered_auroc" in res else "   (too few, or one class only, to score)"))
    print( "     the unsuffixed auroc/slope above are the TRAINED answer channel (the reading the agent "
           "stood on); restating is the only channel that can carry a spread")


if __name__ == "__main__":
    main()
