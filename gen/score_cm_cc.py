#!/usr/bin/env python3
"""The matching step and the contrast step, read off the ANSWER, so every row can carry a score.

A CohortThink answer decomposes the same way the task does. The item's own cohort has a truth y, and
the group the episode is meant to widen to has a published rate r_ref, fixed by the item and not by
the policy. So y = r_ref + d, where r_ref is the reference LEVEL and d is the remaining difference.
Any answer r_hat, from an agent or from a model with no tools at all, splits the same way.

  MATCH     |r_hat - r_ref| < |b - r_ref|, the answer is nearer the reference group's rate than the
            population incidence b of that disease and horizon is. It asks whether the answer landed
            on the right group rather than on the disease's average, which is the matching step
            stated as a property of the number.
  CONTRAST  sign(r_hat - r_ref) == sign(y - r_ref), the remaining difference is applied in the right
            direction, with the tie counted only when both are zero.

Neither reads the trace, so a single-turn model with no episode is scored on the same two questions
as an agent. The earlier trace-level readings measured the walk and could not score a row without
one.

  gen/score_cm_cc.py --run DIR ... --singleturn FILE ... [--items PATH] [--experiments DIR]
"""
import argparse, json, glob, os, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cohortworld_env"))
from cohortworld_env import reward as R
from cohortworld_env.lattice import _shared

def ref_rate(version, target, disease, horizon):
    lat = _shared(version)
    if lat is None: return None
    return R._published_rate(lat, R.reference_cohort(version, target), disease, int(horizon))

def base(version, disease, horizon):
    try: return R.base_rate("internal", disease, int(horizon))
    except Exception: return None

def sign(x): return 0 if x == 0 else (1 if x > 0 else -1)

def score(rows):
    """rows: (r_hat, y, version, target, disease, horizon)

    Both numbers follow the paper's own definitions of Appendix A.7, moved from the position and the
    committed list onto the answer, so a model with no episode is scored on the same two questions.

    CM is the paper's EXACT match. There the answer is a position and the score is 1 only on the
    target cohort itself. Here the answer is a rate, and standing on the reference group means
    reporting the rate that group publishes, so exact means r_hat == r_ref. The nearer-than-b column
    beside it is the same comparison made tolerant, and it is reported because an exact rate match is
    something only a policy that reads can achieve.

    CC is the paper's direction accuracy, including its exclusion: an item whose truth sits exactly at
    the reference rate is the indistinguishable class, and the paper drops that class from the
    denominator because it is where a silent policy lands.
    """
    m_n = m_ex = mb_n = mb_ok = c_n = c_ok = 0
    for r_hat, y, ver, tgt, dis, hor in rows:
        if r_hat is None or y is None or not tgt: continue
        rr = ref_rate(ver, tgt, dis, hor)
        if rr is None: continue
        m_n += 1; m_ex += (r_hat == rr)
        b = base(ver, dis, hor)
        if b is not None:
            mb_n += 1; mb_ok += abs(r_hat - rr) < abs(b - rr)
        if sign(y - rr) != 0:                      # drop the indistinguishable class, as the paper does
            c_n += 1; c_ok += sign(r_hat - rr) == sign(y - rr)
    return m_n, m_ex, mb_n, mb_ok, c_n, c_ok

def from_run(d):
    global a
    F = glob.glob(os.path.join(a.experiments, d, "outputs", "*", "traces.jsonl"))
    if not F: return None
    rows = []
    for L in open(F[0]):
        rec = json.loads(L); it = (rec.get("task") or {}).get("data") or {}
        for t in (rec.get("traces") or [rec]):
            nodes = t.get("nodes") or []
            class Tr: pass
            tr = Tr(); tr.messages = [x["message"] for x in nodes if x.get("message")]
            if not tr.messages: continue
            vals, _l, _m = R.rate_report(tr, it.get("disease"), int(it["horizon"]))
            r_hat = statistics.median(vals) if vals else None
            rows.append((r_hat, it.get("true_rate"), it.get("version"), it.get("target_cohort"),
                         it.get("disease"), it.get("horizon")))
    return rows

def from_singleturn(f):
    import re
    rows = []
    items = {json.loads(l)["item_id"]: json.loads(l) for l in open(ITEMS_FILE)}
    for L in open(f):
        r = json.loads(L)
        m = re.findall(r'(\d+\.?\d*)\s*%?', r.get("text") or "")
        r_hat = float(m[-1]) / 100.0 if m else None
        if r_hat is not None and not (0.0 <= r_hat <= 1.0): r_hat = None
        it = items.get(r["item_id"], {})
        rows.append((r_hat, r.get("true_rate"), r.get("version"), it.get("target_cohort"),
                     r.get("disease"), r.get("horizon")))
    return rows

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run", nargs="*", default=[], help="run directories holding outputs/*/traces.jsonl (two-turn agent rows)")
ap.add_argument("--singleturn", nargs="*", default=[], help="jsonl files written by singleturn_t3.py (no-tool rows)")
ap.add_argument("--items", default=os.path.join(os.environ.get("CW_ITEMS", ""), "t3_rate_unseen.jsonl"),
                help="the CohortThink item file the single-turn rows answered; default $CW_ITEMS/t3_rate_unseen.jsonl")
ap.add_argument("--experiments", default=os.environ.get("CW_EXPERIMENTS", "experiments"),
                help="directory the --run names are relative to; default $CW_EXPERIMENTS or ./experiments")
a = ap.parse_args()
ITEMS_FILE = a.items
print(f"{'row':46}{'CM exact':>12}{'CM near-b':>12}{'CC dir':>12}{'n':>7}")
for d in a.run:
    rows = from_run(d)
    if not rows: print(f"{d[:48]:50} no traces"); continue
    mn, mex, mbn, mbo, cn, co = score(rows)
    print(f"{d[:44]:46}{100*mex/max(mn,1):>11.1f}%{100*mbo/max(mbn,1):>11.1f}%{100*co/max(cn,1):>11.1f}%{cn:>7}")
for f in a.singleturn:
    rows = from_singleturn(f)
    mn, mex, mbn, mbo, cn, co = score(rows)
    print(f"{f.split('/')[-1][:44]:46}{100*mex/max(mn,1):>11.1f}%{100*mbo/max(mbn,1):>11.1f}%{100*co/max(cn,1):>11.1f}%{cn:>7}")
