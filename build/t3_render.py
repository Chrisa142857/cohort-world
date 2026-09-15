#!/usr/bin/env python3
# release-gate: allow-file -- ran against the private UKB extract; shipped for provenance, paths shown as they were
"""What `docs/item-rendering-spirit.md` implies for T3 (rate), measured rather than asserted.

Three questions, one script, because they share the candidate machinery:

  A. CALIBRATE THE MASK. The spirit says mask by CANDIDATE COUNT (target 2-8), not by axis count.
     For every C2 state, over every mask pattern of size 0, 1 and 2: how many cohorts stay
     consistent with what the item still states? Which patterns land in 2-8, how many states admit
     such a pattern at all, and what share of the lattice is usable that way.

  B. THE DECISIVE CLAIM. Internal T3's optimum is (w=1, s=0) -- "walk there and report what you
     read" -- while external transfer needs (w~0.25, s~2-3). The spirit predicts masking fixes the
     DIRECTION, because a masked item's answer spans several candidates and shrinkage becomes
     correct rather than a patch. Tested by running the external (w, s) sweep unchanged, with the
     agent's committed five values replaced by the five quantiles of the consistent candidate set.

     Its own positive control is mask=0: if the sweep cannot reproduce the KNOWN (w=1, s=0)
     pathology on unmasked items, nothing it says about masked ones is evidence.

  C. THE EXTERNAL SPLIT. Whether dials A and B are definable on PhenoCal cohorts at all, counted
     off the released panel and the axes sidecar.

The estimator, the grid and the orientation are copied from `gen/calibration_sweep.py` deliberately
-- the claim is a COMPARISON with the external optimum, so the two must be measured the same way:

    level   m' = w*median + (1-w)*base_rate
    spread  v' = m' * (1 + s*(v/median - 1))      clipped to [0, 1]

`score` there returns the mean CRPS (its `_crps` is 1-CRPS, so `1.0 - _crps` is the loss again) and
lower is better; the same holds here.

Nothing in this script reads a patient narrative, and nothing it writes is an item. It measures the
lattice and the released PhenoCal panel only.
"""
from __future__ import annotations
import argparse, collections, glob, itertools, json, os, random, statistics as st, sys
from array import array

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cohortworld_env import reward as R
from cohortworld_env.lattice import PANEL

AXES = ("age_band", "sex", "icd", "opcs", "bnf")
WS = [0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 1.0]     # gen/calibration_sweep.py, verbatim
SP = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
GRID = [(w, s) for w in WS for s in SP]
TARGET_LO, TARGET_HI = 2, 8                    # the spirit's candidate-count window
POP_ZERO_SHARE = 0.727                         # gen/build_items.py, published with the item set
PHENOCAL = os.environ.get(
    "CW_PHENOCAL", "/ram/USERS/ziquanw/digitaltwinreward/ukb_qa_brain/output_panel")


# --------------------------------------------------------------------------- the lattice

def load(version: str) -> list[dict]:
    """Every released cohort of one lattice, in one list. Splits are merged because the AGENT walks
    the merged lattice -- `Lattice.__init__` loads train+val+test -- so a candidate set that stopped
    at a split boundary would describe a lattice nobody navigates."""
    out = []
    gran = {"C1": 1, "C2": 2, "default": 3}[version]
    for p in sorted(glob.glob(f"{PANEL}/CohortWorld_{version}_*_states.jsonl")):
        for ln in open(p):
            d = json.loads(ln)
            ne = {int(k): ("all" if v == "all" else set(v or ()))
                  for k, v in (d.get("not_estimable") or {}).items()}
            rows = d[f"rates_g{gran}"]
            out.append({"sid": d["state_id"], "axes": d["axes"], "coarseness": d["coarseness"],
                        "n_band": d["n_band"], "ne": ne, "nh": len(rows), "nd": len(rows[0]),
                        "rates": array("f", [x for row in rows for x in row])})
    return out


def published(s: dict, hi: int, di: int) -> bool:
    v = s["ne"].get(hi)
    return not (v == "all" or (v and di in v))


def rate(s: dict, hi: int, di: int) -> float:
    return float(s["rates"][hi * s["nd"] + di])


def axval(s: dict, a: str):
    """One axis of a cohort as the ITEM would state it.

    Code axes reduce to the SET OF CODES and drop the onset-band index. That is what the item states
    after de-grouping: the onset band is governed by `coarseness`'s first component, which is a
    granularity of its own and not part of "which codes are on this axis". Keeping the index in the
    key makes 97.7% of C2 states unique at mask=0 and the whole exercise vacuous; dropping it
    reproduces the measured mask=0 row of docs/item-rendering-spirit.md (median 1, 0.8% with >=5).
    """
    v = s["axes"][a]
    return v if a in ("age_band", "sex") else tuple(sorted(c for c, _ in v))


def index(states, mask: tuple) -> dict:
    """retained-axes key -> list of state positions. One dict per mask pattern."""
    keep = [a for a in AXES if a not in mask]
    idx = collections.defaultdict(list)
    for i, s in enumerate(states):
        idx[tuple(axval(s, a) for a in keep)].append(i)
    return idx


def patterns(max_masked: int) -> list[tuple]:
    return [ms for k in range(max_masked + 1) for ms in itertools.combinations(AXES, k)]


# --------------------------------------------------------------------------- A. mask calibration

def section_a(states, out) -> dict:
    """Candidate counts for every state under every mask pattern of size <= 2."""
    pats = patterns(2)
    idxs = {ms: index(states, ms) for ms in pats}
    keys = {ms: [tuple(axval(s, a) for a in AXES if a not in ms) for s in states] for ms in pats}
    counts = {ms: [len(idxs[ms][k]) for k in keys[ms]] for ms in pats}

    p(out, "\n== A. MASK CALIBRATION "
           "=========================================================================")
    p(out, f"C2 lattice: {len(states)} released cohorts. Candidate = a cohort still consistent with "
           f"every axis the item\nstill states (code axes matched as SETS OF CODES; see axval).\n")
    p(out, f"{'mask pattern':34} {'median':>7} {'unique':>8} {'>=5':>7} {'in 2-8':>8} {'max':>7}")
    for k in (0, 1, 2):
        pooled = []
        for ms in [m for m in pats if len(m) == k]:
            c = counts[ms]
            pooled += c
            p(out, f"  {('none' if not ms else '+'.join(ms)):32} {st.median(c):>7.0f} "
                   f"{share(c, lambda x: x == 1):>7.1%} {share(c, lambda x: x >= 5):>6.1%} "
                   f"{share(c, lambda x: TARGET_LO <= x <= TARGET_HI):>7.1%} {max(c):>7}")
        p(out, f"  {('POOLED over ' + str(k) + '-axis masks'):32} {st.median(pooled):>7.0f} "
               f"{share(pooled, lambda x: x == 1):>7.1%} {share(pooled, lambda x: x >= 5):>6.1%} "
               f"{share(pooled, lambda x: TARGET_LO <= x <= TARGET_HI):>7.1%} {max(pooled):>7}")

    # Which states admit a mask that lands in the window at all, and by which pattern.
    admit, by_pat, sizes = 0, collections.Counter(), collections.Counter()
    for i in range(len(states)):
        ok = [ms for ms in pats if ms and TARGET_LO <= counts[ms][i] <= TARGET_HI]
        if ok:
            admit += 1
            sizes[min(len(ms) for ms in ok)] += 1
            for ms in ok:
                by_pat[ms] += 1
    frac = admit / len(states)
    p(out, f"\nStates admitting SOME mask of size 1-2 with candidates in [{TARGET_LO},{TARGET_HI}]: "
           f"{admit}/{len(states)} = {frac:.4f} {ci_str(boot_share([1]*admit + [0]*(len(states)-admit)))}")
    p(out, f"  reachable with a ONE-axis mask: {sizes[1]}/{len(states)} = {sizes[1]/len(states):.4f}"
           f"   needs TWO axes: {sizes[2]}/{len(states)} = {sizes[2]/len(states):.4f}")
    p(out, "  patterns that land in the window, by how many states they serve:")
    for ms, n in by_pat.most_common(10):
        p(out, f"    {'+'.join(ms):32} {n:>6} states ({n/len(states):.1%})")

    # The doc's one open interaction: does `coarseness`'s first component collide with principle 2?
    onset = collections.Counter(s["coarseness"].split("|")[0] for s in states)
    widths = collections.Counter()
    for s in states:
        b = s["axes"]["age_band"]
        lo, _, hi = b.partition("-")
        widths[(s["coarseness"].split("|")[0], int(hi) - int(lo) + 1 if hi.isdigit() else -1)] += 1
    p(out, f"\n  coarseness component 1 (the ONSET-band scheme, on '(first recorded ...)'): "
           + ", ".join(f"{k} {v/len(states):.1%}" for k, v in onset.most_common()))
    p(out, f"  cohort `axes.age_band` widths, by onset scheme: "
           + ", ".join(f"{k}:{w}y {v}" for (k, w), v in sorted(widths.items())))
    p(out, f"  distinct `axes.age_band` values: "
           f"{len({s['axes']['age_band'] for s in states})}, all of one width -- so the age band the "
           f"item de-groups and the onset scheme `coarseness` names are DIFFERENT fields")
    return {"counts": counts, "pats": pats, "admit_frac": frac}


# --------------------------------------------------------------------------- B. the (w, s) sweep

def draw_items(states, n, seed):
    """The archived T3 draw, minus the patient narrative: one cohort, one published (horizon,
    disease) cell, 50/50 zero / non-zero stratum (`gen/build_items.py` ZERO_FRACTION=0.5).

    A uniform draw over published cells is NOT an option and is not a judgement call: 96.6% of C2
    cells are exactly zero, so it would ask "is it zero" 29 times out of 30 and the optimum would be
    decided by the base-rate offset alone. The stratified draw is the item set's own construction,
    and POP_ZERO_SHARE re-weights any aggregate back to the population.
    """
    rng = random.Random(seed)
    order = list(range(len(states)))
    rng.shuffle(order)
    items = []
    for i in order:
        s = states[i]
        cells = [(hi, di) for hi in range(s["nh"]) for di in range(s["nd"])
                 if published(s, hi, di)]
        if not cells:
            continue
        nz = [c for c in cells if rate(s, *c) > 0]
        zc = [c for c in cells if rate(s, *c) == 0]
        want_zero = rng.random() < 0.5
        pick = (zc if want_zero else nz) or (nz if want_zero else zc)
        if not pick:
            continue
        hi, di = rng.choice(pick)
        items.append({"i": i, "hi": hi, "di": di, "truth": rate(s, hi, di)})
        if len(items) >= n:
            break
    return items


def draw_items_spread(states, cal_axis, idxs, keys, n, seed):
    """The same draw with ONE condition changed: the cell is chosen so the CANDIDATE SET carries a
    spread, instead of so the TARGET is non-zero.

    This is the control the stratified draw needs. `build_items.py` picks the cell on the target's
    own rate (50/50 zero / non-zero). Under a mask that is selection on the answer applied to ONE
    member of the candidate set: the target is forced non-zero while its consistent neighbours are
    drawn from a 96.6%-zero cell population, so the candidate median is biased toward zero by
    construction and no shrinkage weight can repair it. Selecting instead on `>=2 distinct published
    rates among the candidates` conditions on the thing the mask is supposed to create -- something
    to resolve -- and on nothing about the answer.
    """
    rng = random.Random(seed)
    order = [i for i in cal_axis]
    rng.shuffle(order)
    items = []
    for i in order:
        s = states[i]
        ms = cal_axis[i]
        cands = idxs[ms][keys[ms][i]]
        cells = []
        for hi in range(s["nh"]):
            for di in range(s["nd"]):
                if not published(s, hi, di):
                    continue
                rs = {rate(states[j], hi, di) for j in cands if published(states[j], hi, di)}
                if len(rs) >= 2:
                    cells.append((hi, di))
        if not cells:
            continue
        hi, di = rng.choice(cells)
        items.append({"i": i, "hi": hi, "di": di, "truth": rate(s, hi, di)})
        if len(items) >= n:
            break
    return items


def quantiles5(xs) -> list[float]:
    """The five values a report carries (`reward.MIN_REPORT`), read off the candidate set.

    The external sweep recalibrated the agent's own five committed numbers. The internal analogue of
    "what the agent committed" is what the consistent candidates say, summarised at the same width.
    """
    s = sorted(xs)
    n = len(s)
    return [s[min(n - 1, int(round(q * (n - 1))))] for q in (0.0, 0.25, 0.5, 0.75, 1.0)]


def recal(vals, base, w, s):
    """gen/calibration_sweep.py:score, one item's worth."""
    m = st.median(vals)
    mp = w * m + (1 - w) * base
    if not m:
        return [mp] * len(vals)
    return [max(0.0, min(1.0, mp * (1 + s * (x / m - 1)))) for x in vals]


def band_lo(b: str) -> int:
    b = str(b).replace("+", "").split("-")[0]
    return int(b) if b.isdigit() else 1


def wquantiles5(vals, wts) -> list[float]:
    """Size-weighted version of `quantiles5`, for the post-stratification check.

    A candidate set mixes a 1000+ cohort with a 10-49 one and the plain median counts them equally.
    Direct standardisation is the estimator this geometry calls for, so if the
    unweighted read carries no information, the weighted one is the first thing to rule out before
    the finding is called structural.
    """
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    tot = sum(wts) or 1.0
    out, acc, k = [], 0.0, 0
    cum = []
    for i in order:
        acc += wts[i]
        cum.append((acc - 0.5 * wts[i]) / tot)
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        k = min(range(len(order)), key=lambda j: abs(cum[j] - q))
        out.append(vals[order[k]])
    return out


def loss_matrix(states, items, cand_of, diseases, out, label, weighted=False):
    """L[item, grid cell] = CRPS, plus the read-nothing floor per item and the candidate count.

    Scored once; the bootstrap then only averages, which is what makes 1000 resamples over 42 cells
    cheap. The floor is `gen/calibration_sweep.py`'s own comparator -- the base rate spread +/-50%,
    a policy that never reads anything -- because an optimum that does not beat it says the item set
    has no discrimination to find, whichever direction it points.
    """
    L, ncand, kept, triv = [], [], [], []
    for it in items:
        s = states[it["i"]]
        hi, di = it["hi"], it["di"]
        cands = [j for j in cand_of(it) if published(states[j], hi, di)]
        vals = [rate(states[j], hi, di) for j in cands]
        if not vals:
            continue                      # nothing readable: the agent could not answer from the lattice
        b = R.base_rate("internal", diseases[di], HORIZONS[hi])
        q = wquantiles5(vals, [band_lo(states[j]["n_band"]) for j in cands]) if weighted \
            else quantiles5(vals)
        L.append([R.crps(recal(q, b, w, sp), it["truth"]) for w, sp in GRID])
        triv.append(R.crps([b * 0.5, b * 0.75, b, b * 1.25, b * 1.5], it["truth"]))
        ncand.append(len(cands))
        kept.append(it)
    p(out, f"  {label}: n={len(L)} items, candidates median {st.median(ncand):.0f} "
           f"(p10 {np.percentile(ncand,10):.0f}, p90 {np.percentile(ncand,90):.0f}), "
           f"{share(ncand, lambda x: x == 1):.1%} with a single candidate")
    return np.asarray(L), np.asarray(triv), np.asarray(ncand), kept


def sweep(L, triv, weights, n_boot, seed, out, label, grid=False):
    """Grid optimum with a bootstrap over ITEMS, refitting (w, s) inside every resample."""
    if len(L) == 0:
        p(out, f"  {label}: no scorable items")
        return None
    mean = lambda idx: np.average(L[idx], axis=0, weights=weights[idx])
    full = mean(np.arange(len(L)))
    floor = float(np.average(triv, weights=weights))
    j = int(np.argmin(full))
    rng = np.random.default_rng(seed)
    ws, ss, crps = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(L), len(L))
        m = mean(idx)
        k = int(np.argmin(m))
        ws.append(GRID[k][0]); ss.append(GRID[k][1]); crps.append(m[k])
    w, s = GRID[j]
    p(out, f"  {label}: OPTIMUM w={w} s={s}   CRPS {full[j]:.5f} "
           f"[{np.percentile(crps,2.5):.5f}, {np.percentile(crps,97.5):.5f}]")
    p(out, f"      bootstrap w* median {np.median(ws):.2f} "
           f"[{np.percentile(ws,2.5):.2f}, {np.percentile(ws,97.5):.2f}]  "
           f"P(w*<1) = {np.mean(np.asarray(ws) < 1.0):.3f}")
    p(out, f"      bootstrap s* median {np.median(ss):.2f} "
           f"[{np.percentile(ss,2.5):.2f}, {np.percentile(ss,97.5):.2f}]  "
           f"P(s*>0) = {np.mean(np.asarray(ss) > 0.0):.3f}")
    i10 = GRID.index((1.0, 0.0))
    # Does READING buy anything? Best cell that uses the reading at all, against the best cell that
    # discards it, against the read-nothing policy. w=0 still uses the item's disease and horizon.
    pos = min(full[k] for k, (w_, _) in enumerate(GRID) if w_ > 0)
    nul = min(full[k] for k, (w_, _) in enumerate(GRID) if w_ == 0)
    p(out, f"      read-and-report (w=1,s=0) {full[i10]:.5f}   best w>0 {pos:.5f}   "
           f"best w=0 {nul:.5f}   read-nothing +/-50% {floor:.5f}")
    p(out, f"      discrimination (best w=0 - best w>0) {nul-pos:+.5f}   "
           f"{'READING PAYS' if pos < nul else 'READING BUYS NOTHING'}")
    if grid:
        p(out, "      " + "  ".join(f"s={x:<5.1f}" for x in SP))
        for wi, w_ in enumerate(WS):
            p(out, f"      w={w_:<4.2f} " +
                   "  ".join(f"{full[wi*len(SP)+si]:.5f}" for si in range(len(SP))))
    return {"w": w, "s": s, "crps": float(full[j]), "p_w_lt1": float(np.mean(np.asarray(ws) < 1.0)),
            "p_s_gt0": float(np.mean(np.asarray(ss) > 0.0)), "crps_10": float(full[i10]),
            "reads_pay": bool(pos < nul), "floor": floor}


# --------------------------------------------------------------------------- C. external

def section_c(out):
    """Whether dial A and dial B are definable on PhenoCal cohorts, counted off the released panel."""
    p(out, "\n== C. THE EXTERNAL SPLIT (PhenoCal_C2 test) "
           "======================================================")
    path = f"{PHENOCAL}/PhenoCal_C2_test.jsonl"
    if not os.path.exists(path):
        p(out, f"  {path} unavailable")
        return {}
    nf, coarse, n = collections.Counter(), collections.Counter(), 0
    src = collections.Counter()
    for ln in open(path):
        d = json.loads(ln)
        m = d["metadata"]
        n += 1
        nf[len(m.get("profile_groups") or [])] += 1
        coarse[m.get("coarseness")] += 1
        src["hospital records only" in d["prompt"]] += 1
    zero = nf[0]
    p(out, f"  items {n};  cohorts described by 0 risk factors: {zero} = {zero/n:.4f} "
           f"-- nothing on the history axis to mask")
    p(out, f"  factors stated per item: median {median_of_counter(nf):.0f}, "
           f"1 factor {nf[1]/n:.1%}, >=3 {sum(v for k,v in nf.items() if k>=3)/n:.1%}")
    sup = sum(v for k, v in coarse.items() if k and k.endswith("-super"))
    p(out, f"  onset-band scheme (dial-B-like, but OBSERVED not set): "
           + ", ".join(f"{k} {v/n:.1%}" for k, v in coarse.most_common()))
    p(out, f"  already released at SUPERGROUP resolution: {sup}/{n} = {sup/n:.4f} "
           f"-- the coarse rung exists in the panel, the fine rung for those cohorts does not")
    p(out, f"  record source: hospital-only {src[True]/n:.1%}, GP-linked {src[False]/n:.1%}")
    sc = f"{PANEL}/PhenoCal_C2_axes_sidecar.jsonl"
    if os.path.exists(sc):
        t = o = b = 0
        for ln in open(sc):
            d = json.loads(ln); t += 1; o += bool(d["opcs"]); b += bool(d["bnf"])
        p(out, f"  axes sidecar: {t} cohort prefixes, opcs non-empty {o/t:.1%}, bnf non-empty "
               f"{b/t:.1%} -- the only axes `add`/`switch` can act on externally, and they are "
               f"RECOMPUTED, not part of the PhenoCal release")
    return {"zero_factor_share": zero / n, "super_share": sup / n}


# --------------------------------------------------------------------------- helpers

HORIZONS = [1, 2, 3, 5, 10]


def share(xs, f) -> float:
    return sum(1 for x in xs if f(x)) / len(xs) if xs else 0.0


def boot_share(bits, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.asarray(bits, dtype=float)
    m = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n_boot)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def ci_str(t) -> str:
    return f"[{t[0]:.4f}, {t[1]:.4f}]"


def median_of_counter(c) -> float:
    xs = sorted(k for k, v in c.items() for _ in range(v))
    return st.median(xs)


def p(out, line=""):
    print(line)
    out.write(line + "\n")


# --------------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="C2")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="-")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    states = load(a.version)
    diseases = [d["code"] for d in
                json.load(open(f"{PANEL}/CohortWorld_{a.version}_manifest.json"))["diseases"]]
    out = open(a.out, "w") if a.out != "-" else sys.stdout

    p(out, f"T3 UNDER THE ITEM-RENDERING SPIRIT -- {a.version}, seed {a.seed}, "
           f"n_boot {a.n_boot}, reward spec {R.REWARD_SPEC_VERSION}, env {R.ENV_VERSION}")
    A = section_a(states, out)
    counts, pats = A["counts"], A["pats"]
    idxs = {ms: index(states, ms) for ms in pats}
    keys = {ms: [tuple(axval(s, ax) for ax in AXES if ax not in ms) for s in states] for ms in pats}

    items = draw_items(states, a.n, a.seed)
    p(out, "\n== B. THE (w, s) OPTIMUM VERSUS MASK LEVEL "
           "=======================================================")
    p(out, f"Estimator, grid and orientation copied from gen/calibration_sweep.py. Lower CRPS is "
           f"better.\nExternal optimum for comparison: w ~ 0.25, s ~ 2-3.\n"
           f"Sampled items: {len(items)} (50/50 zero / non-zero stratum, as the item set is built)")

    rng = random.Random(a.seed + 1)
    one_axis = {it["i"]: rng.choice([ms for ms in pats if len(ms) == 1]) for it in items}
    cal_axis, no_admit = {}, 0
    for it in items:
        i = it["i"]
        ok = [ms for ms in pats if ms and TARGET_LO <= counts[ms][i] <= TARGET_HI]
        k = min((len(ms) for ms in ok), default=None)
        if k is None:
            no_admit += 1
        else:
            cal_axis[i] = rng.choice([ms for ms in ok if len(ms) == k])

    def cand(ms_of):
        def f(it):
            ms = ms_of(it["i"])
            return idxs[ms][keys[ms][it["i"]]]
        return f

    spread_items = draw_items_spread(states, cal_axis, idxs, keys, a.n, a.seed)
    arms = [("mask=0 (all five axes stated)", cand(lambda i: ()), items),
            ("mask=1 (one random axis)", cand(lambda i: one_axis[i]), items),
            (f"mask calibrated to {TARGET_LO}-{TARGET_HI} candidates",
             cand(lambda i: cal_axis[i]), [it for it in items if it["i"] in cal_axis]),
            (f"calibrated mask + cell selected on CANDIDATE SPREAD",
             cand(lambda i: cal_axis[i]), spread_items)]
    p(out, f"  items with no mask of size 1-2 landing in the window: {no_admit}/{len(items)}")
    p(out, f"  spread-selected arm: {len(spread_items)} items, "
           f"{share([it['truth'] for it in spread_items], lambda x: x == 0):.1%} with a zero truth "
           f"(not stratified -- this is what the condition yields)\n")

    res, res_nz = {}, {}
    for label, f, its in arms:
        L, triv, nc, kept = loss_matrix(states, its, f, diseases, out, label)
        if len(L) == 0:
            continue
        wts = np.ones(len(L))
        res[label] = sweep(L, triv, wts, a.n_boot, a.seed, out, label + " [equal weight]", grid=True)
        zs = np.asarray([1.0 if k["truth"] == 0 else 0.0 for k in kept])
        pw = np.where(zs > 0, POP_ZERO_SHARE / max(zs.mean(), 1e-9),
                      (1 - POP_ZERO_SHARE) / max(1 - zs.mean(), 1e-9))
        sweep(L, triv, pw, a.n_boot, a.seed, out, label + " [population-reweighted]")
        for nm, sel in (("zero-truth", zs > 0), ("non-zero-truth", zs == 0)):
            if sel.sum() >= 10:
                r = sweep(L[sel], triv[sel], np.ones(int(sel.sum())), a.n_boot, a.seed, out,
                          f"{label} [{nm}, n={int(sel.sum())}]", grid=(nm == "non-zero-truth"))
                if nm == "non-zero-truth":
                    res_nz[label] = r
        if "calibrated" in label:
            Lw, tw, _, kw = loss_matrix(states, its, f, diseases, out,
                                        label + " [size-weighted candidates]", weighted=True)
            zw = np.asarray([1.0 if k["truth"] == 0 else 0.0 for k in kw])
            sel = zw == 0
            if sel.sum() >= 10:
                sweep(Lw[sel], tw[sel], np.ones(int(sel.sum())), a.n_boot, a.seed, out,
                      f"{label} [size-weighted, non-zero-truth, n={int(sel.sum())}]")
        p(out, "")

    section_c(out)
    p(out, "\nVERDICT (mechanical, from the rows above). External target: w ~ 0.25, s ~ 2-3.")
    for tag, tab in (("all items", res), ("non-zero truth only", res_nz)):
        p(out, f"  -- {tag} --")
        for label, r in tab.items():
            if not r:
                continue
            moved = r["w"] < 1.0 and r["s"] > 0.0
            p(out, f"    {label:46} w={r['w']:<5} s={r['s']:<5} "
                   f"{'MOVES toward external' if moved else 'does NOT move':22} "
                   f"{'reading pays' if r['reads_pay'] else 'reading buys nothing'}")
    if out is not sys.stdout:
        out.close()
    return 0


def selfcheck():
    """The smallest thing that fails if any of the three moving parts breaks."""
    # 1. the estimator is calibration_sweep's, to the digit
    assert recal([0.1] * 5, 0.02, 1.0, 0.0) == [0.1] * 5
    assert abs(recal([0.1, 0.2, 0.3], 0.0, 0.0, 1.0)[0] - 0.0) < 1e-12
    v = recal([0.05, 0.1, 0.15], 0.02, 0.5, 2.0)
    m = 0.5 * 0.1 + 0.5 * 0.02
    assert abs(v[0] - max(0.0, m * (1 + 2 * (0.05 / 0.1 - 1)))) < 1e-12
    # 2. a single candidate is a point mass at the truth under (w=1, s=0) -- the pathology itself
    assert R.crps(recal(quantiles5([0.037]), 0.02, 1.0, 0.0), 0.037) == 0.0
    assert R.crps(recal(quantiles5([0.037]), 0.02, 0.25, 0.0), 0.037) > 0.0
    # 3. quantiles of a five-point set are that set
    assert quantiles5([0.1, 0.2, 0.3, 0.4, 0.5]) == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert quantiles5([0.4, 0.1]) == [0.1, 0.1, 0.1, 0.4, 0.4]
    # 4. masking can only add candidates, never remove them
    s = [{"axes": {"age_band": "60-64", "sex": "Male", "icd": [["E1", 1]], "opcs": [], "bnf": []}},
         {"axes": {"age_band": "60-64", "sex": "Male", "icd": [["I1", 1]], "opcs": [], "bnf": []}}]
    assert len(index(s, ())[tuple(axval(s[0], a) for a in AXES)]) == 1
    assert len(index(s, ("icd",))[tuple(axval(s[0], a) for a in AXES if a != "icd")]) == 2
    print("selfcheck OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
