#!/usr/bin/env python3
"""CohortThink AUROC for Table 4, with b the UK Biobank marginal incidence of the item's disease and
horizon (`gen/thresholds.py`). The CohortCal table (Table 3) uses a different b, the train-split mean.

Label is r > b, score is r_hat / b, items with b == 0 are dropped, and the interval is a percentile
bootstrap over items with 2,000 resamples at seed 42. Reads either a two-turn run directory
(outputs/*/traces.jsonl, the answer read through `reward.rate_report`, the median of the committed
values) or a single-turn file written by `gen/singleturn_t3.py`.

Do not score this column with `reward.base_rate("internal")`: that is the train-split mean of the
CohortThink items, which is the environment's reward reference and not the b this column is defined with.

  gen/score_ct_auroc.py [--run DIR ...] [--singleturn FILE ...] [--experiments DIR] [--boot 2000] [--seed 42]
"""
import argparse, glob, json, os, re, statistics, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cohortworld_env"))
from cohortworld_env import reward as R   # noqa: E402
import thresholds as TH                    # noqa: E402


def auroc(y, s):
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(u)); np.add.at(sums, inv, r); r = (sums / cnt)[inv]
    P, N = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N) if P and N else float("nan")


def score(rows, boot, seed):
    ys, ss = [], []
    for r_hat, y, d, h, v in rows:
        if r_hat is None or y is None: continue
        try: b = TH.base_rate(d, int(h), version=v)
        except Exception: b = None
        if not b: continue
        ys.append(1.0 if y > b else 0.0); ss.append(r_hat / b)
    y, s = np.array(ys), np.array(ss); rng = np.random.default_rng(seed)
    bs = np.array([v for v in (auroc(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(boot))) if v == v])
    return len(y), auroc(y, s), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


class Tr: pass


def from_run(d, experiments):
    return from_traces(glob.glob(os.path.join(experiments, d, "outputs", "*", "traces.jsonl"))[0])


def from_traces(path):
    """One two-turn trace file, by path: a run directory's outputs/*/traces.jsonl or a released copy."""
    rows = []
    for L in open(path):
        rec = json.loads(L); it = rec["task"]["data"]
        for t in rec["traces"]:
            tr = Tr(); tr.messages = [x["message"] for x in (t.get("nodes") or []) if x.get("message")]
            if not tr.messages: continue
            v, _l, _m = R.rate_report(tr, it["disease"], int(it["horizon"]))
            rows.append((statistics.median(v) if v else None, it.get("true_rate"), it["disease"], it["horizon"], it["version"]))
    return rows


def from_singleturn(f):
    rows = []
    for L in open(f):
        r = json.loads(L); m = re.findall(r"(\d+\.?\d*)\s*%?", r.get("text") or "")
        p = float(m[-1]) / 100 if m else None
        if p is not None and not 0 <= p <= 1: p = None
        rows.append((p, r["true_rate"], r["disease"], r["horizon"], r["version"]))
    return rows


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run", nargs="*", default=[]); ap.add_argument("--singleturn", nargs="*", default=[])
ap.add_argument("--traces", nargs="*", default=[], help="two-turn trace files by path (the released paper_outputs copies)")
ap.add_argument("--experiments", default=os.environ.get("CW_EXPERIMENTS", "experiments"))
ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
print(f"{'row':50}{'n':>6}{'CT AUROC':>10}{'95% CI':>18}")
for d in a.run:
    n, au, lo, hi = score(from_run(d, a.experiments), a.boot, a.seed)
    print(f"{d[:48]:50}{n:>6}{au:>10.3f}   [{lo:.3f}, {hi:.3f}]")
for f in a.traces:
    n, au, lo, hi = score(from_traces(f), a.boot, a.seed)
    print(f"{os.path.basename(f)[:48]:50}{n:>6}{au:>10.3f}   [{lo:.3f}, {hi:.3f}]")
for f in a.singleturn:
    n, au, lo, hi = score(from_singleturn(f), a.boot, a.seed)
    print(f"{os.path.basename(f)[:48]:50}{n:>6}{au:>10.3f}   [{lo:.3f}, {hi:.3f}]")
