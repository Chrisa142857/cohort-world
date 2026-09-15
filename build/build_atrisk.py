#!/usr/bin/env python3
# release-gate: allow-file -- ran against the private UKB extract; shipped for provenance, paths shown as they were
"""Per-cell at-risk counts, BANDED, so `separated` uses the right denominator without leaking one.

`separated` in the edge table divides by the cohort's `n_band` lower bound. That is the wrong
quantity: the denominator for one outcome is the number still AT RISK of it,
`n_elig = total - prevalent` (`panel_common.py:223`), which is smaller -- often far smaller at the
coarse granularity, where prevalence depletes the at-risk set most. Dividing by the whole cohort
gives a narrower interval and therefore separates MORE often, so the flag was **anti-conservative
exactly where its docstring claimed to be "conservative twice over"**, and worst at g1.

Publishing the true at-risk count would fix the arithmetic and break the disclosure argument: one bit
per cell derived from an exact count is invertible in aggregate. So the count is BANDED on the same
rule the cohort size already uses, and the flag is computed from the band's LOWER edge. That is
simultaneously the right denominator, a conservative one, and a public one.

Reads member lists (private) and writes bands (public-safe). Members never leave this process.
"""
import argparse, collections, json, os, pickle, sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cohortworld_env"))
import panel_common as PC
import OK__build_cohortworld_env as B

# Bands cut so that the OPTIMAL SHRINKAGE WEIGHT is distinguishable between them, which a coarse
# four-band cut is not: 66.9% of served observations read `10-49` and 0.9% read anything else, so an agent
# asked how much to trust a rate was reading a constant. The weight is
# w = tau2 / (tau2 + p(1-p)/n) with tau2 = 0.003222 the between-cohort variance of the true rates and
# p = 0.0423 their mean, measured on the internal train split; across n in [10, 49] it runs from
# 0.44 to 0.80 -- the whole meaningful range, flattened into one symbol.
#
# The edges below step w by roughly 0.07 each:
#   n=10 -> 0.44   n=13 -> 0.50   n=17 -> 0.57   n=22 -> 0.64   n=30 -> 0.70   n=50 -> 0.80
#
# Disclosure: the floor stays at the k-anonymity MIN_N of 10 and no band is ever a single integer, so
# no exact count is recoverable; the narrowest band still hides the true value among three.
BANDS = [(10, 13, "10-12"), (13, 17, "13-16"), (17, 22, "17-21"), (22, 30, "22-29"),
         (30, 50, "30-49"), (50, 100, "50-99"), (100, 200, "100-199"),
         (200, 1000, "200-999"), (1000, 10**9, "1000+")]
_G = {}


def band_of(n):
    for lo, hi, lab in BANDS:
        if lo <= n < hi:
            return lab
    return None                                   # below the k-anonymity floor: not released at all


def _one(task):
    """At-risk band per (horizon, disease) for one cohort, at every granularity."""
    sid, members = task
    DX, BASE, OBS, FILT, H, END, MINN, GRANS, DIDX = (
        _G["DX"], _G["BASE"], _G["OBS"], _G["FILT"], _G["H"], _G["END"], _G["MINN"],
        _G["GRANS"], _G["DIDX"])
    out = {}
    for g in GRANS:
        prev_r, yr_r = [], []
        for pid, Y in members:
            pv = set()
            for c, t in DX.get(pid, {}).items():
                k = c[:g]
                if k and k[0] not in FILT and t - Y <= 0:
                    pv.add(k)
            prev_r.append(pv)
            yr_r.append(Y)
        idx = DIDX[g]
        for hi, Hy in enumerate(H):
            elig = [i for i in range(len(members)) if yr_r[i] + Hy <= END]
            if not elig:
                continue
            # One pass over eligible rows tallies prevalence for every code at once; the at-risk
            # count for a code is then the eligible total minus its prevalent count. Looping the
            # 1,374 codes of g3 per row instead would be ~50x slower for the same numbers.
            pc = collections.Counter()
            for i in elig:
                for k in prev_r[i]:
                    pc[k] += 1
            ne = len(elig)
            for d, di in idx.items():
                at = ne - pc.get(d, 0)
                if at >= MINN:
                    b = band_of(at)
                    if b:
                        out.setdefault(str(g), {}).setdefault(str(hi), {})[str(di)] = b
    return sid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inter", default="/ram/USERS/ziquanw/digitaltwinreward/ukb_qa_brain/"
                                       "output_panel/panel_intermediate.pkl")
    ap.add_argument("--filterout", default="ZRUVWXY")
    ap.add_argument("--grans", default="1,2,3")
    ap.add_argument("--workers", type=int, default=90)
    ap.add_argument("--versions", default="C1,C2,default")
    ap.add_argument("--panel", default="/work/USERS/ziquanw/cohortworld/official")
    ap.add_argument("--aug", default="/work/USERS/ziquanw/cohortworld_private/augment")
    ap.add_argument("--out", default="/work/USERS/ziquanw/cohortworld/official")
    a = ap.parse_args()

    print("[atrisk] loading panel ...", flush=True)
    D = pickle.load(open(a.inter, "rb"))["patients"]
    pids = [p for p, v in D.items() if v.get("gp_linked", True)]
    BASE = {p: float(D[p]["age_at_index"]) for p in pids}
    OBS = {p: float(D[p]["obs_years"]) for p in pids}
    DX = {p: {c: (int(d[:4]) + (int(d[5:7]) - 1) / 12.0) for c, d in D[p]["dx"].items()}
          for p in pids}
    del D
    GRANS = [int(x) for x in a.grans.split(",")]
    DIDX = {}
    for g in GRANS:
        f = os.path.join(a.panel, f"CohortWorld_C2_diseases_g{g}.json")
        DIDX[g] = {d: i for i, d in enumerate(json.load(open(f))["diseases"])}
    print(f"[atrisk] {len(pids):,} subjects; "
          + ", ".join(f"g{g}={len(DIDX[g])}" for g in GRANS), flush=True)

    _G.update(DX=DX, BASE=BASE, OBS=OBS, FILT=set(a.filterout), H=B.HORIZONS,
              END=B.STUDY_END_FRAC, MINN=PC.MIN_N, GRANS=GRANS, DIDX=DIDX)
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(a.workers)
    for v in a.versions.split(","):
        f = os.path.join(a.aug, f"CohortWorld_{v}_members.jsonl")
        tasks = []
        for line in open(f):
            d = json.loads(line)
            tasks.append((f"{v}:{d['state_id']}", [(m[0], float(m[1])) for m in d["members"]]))
        n = 0
        p = os.path.join(a.out, f"CohortWorld_{v}_atrisk_bands.jsonl")
        with open(p, "w") as fh:
            for sid, out in pool.imap_unordered(_one, tasks, chunksize=8):
                fh.write(json.dumps({"state_id": sid, "atrisk": out}) + "\n")
                n += 1
        cells = 0
        for line in open(p):
            for _g, hs in json.loads(line)["atrisk"].items():
                for _h, ds in hs.items():
                    cells += len(ds)
        print(f"[atrisk] {v}: {n:,} cohorts, {cells:,} banded cells -> {p}", flush=True)
    pool.close(); pool.join()
    print("[atrisk] done", flush=True)


main()
