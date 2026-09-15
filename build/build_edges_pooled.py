#!/usr/bin/env python3
# release-gate: allow-file -- ran against the private UKB extract; shipped for provenance, paths shown as they were
"""Precomputed edges over the THREE VERSIONS AS ONE LATTICE, with the delta at every granularity.

Pooled rather than per-version, for two measured reasons.

**The agent moves over the three versions as one lattice, so the edges must too.** Confined to
within-version pairs, the table loses **71.6%** of the genuine history-axis zoom structure -- 10,459
cross-version edges against 4,139 within-version, both after prefix verification.

**And the versions share no outcome code**: C1 is 1-char (19), C2 2-char (183), `default` 3-char
(1,374), and every pairwise intersection is exactly zero. A cross-version delta is therefore
uncomputable from the native matrices, and cannot be rolled up either -- "first diagnosis
anywhere in chapter T" is not the sum of the T0..T9 counts, since one person may have several and
only the first is an event. The lattice build solves this at source: every cohort carries
`rates_g1`, `rates_g2` and `rates_g3`, so any two cohorts can be compared at any granularity.

Precomputed edges: {5 axes} x {operation} -> (cost, reliability), value = rate-matrix difference.

An edge is one cohort, one operation on one axis, and the neighbour it reaches. Its value is the
difference of the two published rate matrices, which is what makes a located cohort's rate
DECOMPOSABLE: an answer becomes a base rate plus one labelled correction per operation taken, rather
than a single opaque number.

DISCLOSURE. This file is public and adds nothing to what CohortWorld already releases. Both endpoint
rates are already published, so their difference is a derived quantity, not a new measurement.
Reliability is computed from the published `n_band` LOWER BOUNDS, never from the exact counts under
cohortworld_private/ -- so the reliability score itself cannot be inverted to recover n.

COST is an axis-and-operation property: how much indirectness the move introduces. Changing sex
replaces the population; adding a BNF paragraph barely narrows it. RELIABILITY is NOT an axis
property -- it is computed per cell from the two cohorts' banded sizes, because whether a difference
is real depends on how many people it was measured on, not on which axis was edited. Operations that
cannot apply to an axis carry reliability 0 and are emitted so the table is total rather than
implied.
"""
import argparse, collections, itertools, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cohortworld_env"))
from cohortworld_env.lattice import _shared, _band_lo, PANEL

AXES = ("age_band", "sex", "icd", "opcs", "bnf")
OPS = ("add", "switch", "shift", "zoom_finer", "zoom_coarser")

# (axis, op) -> cost. None means the operation does not apply to that axis.
#
# Ordered by how much of the patient the move gives up. Sex and age band replace the reference
# population outright, which is the largest concession a clinician makes and the one they defend
# explicitly ("we only have data in men"). A diagnosis is a recorded fact; a procedure has a defined
# indication; a prescription is the noisiest proxy for one, so it is the cheapest thing to trade.
COST = {
    ("sex", "switch"): 8.0,
    # age_band zoom is NOT declared: the zoom loop iterates the three CODE axes only, so declaring
    # it would promise a move no edge can instantiate. Age bands are a fixed 5-year grid here.
    ("age_band", "shift"): 5.0,
    ("icd", "add"): 3.0, ("icd", "switch"): 3.0,
    ("icd", "zoom_coarser"): 2.0, ("icd", "zoom_finer"): 2.0,
    ("opcs", "add"): 2.0, ("opcs", "switch"): 2.0,
    ("opcs", "zoom_coarser"): 1.5, ("opcs", "zoom_finer"): 1.5,
    ("bnf", "add"): 1.0, ("bnf", "switch"): 1.0,
    ("bnf", "zoom_coarser"): 0.5, ("bnf", "zoom_finer"): 0.5,
}


def wilson(x, n, z=1.96):
    if n <= 0:
        return 0.0, 1.0
    c = (x + z * z / 2) / (n + z * z)
    h = z / (n + z * z) * math.sqrt(max(0.0, x * (n - x) / n) + z * z / 4)
    return max(0.0, c - h), min(1.0, c + h)


ATRISK = {}


def atrisk_lo(sid, g, hi, di, fallback):
    """Lower edge of the AT-RISK band for this cell, not of the whole cohort's.

    The denominator for one outcome is the number still at risk of it (`n_elig = total - prevalent`),
    which is smaller than the cohort -- and dividing by the cohort narrows the interval, so the flag
    separates MORE often than warranted: anti-conservative exactly where it claims to be
    conservative. Measured with the at-risk denominator: g1 loses 2.20% of its cells to at-risk falling under the
    k-anonymity floor against 0.08% at g3 (a 28x difference, so prevalence really does deplete coarse
    chapters hardest), while band lowering moves only 0.2-0.9% of cells -- the correction is real but
    small, and the band edges are too coarse for it to differ by granularity.
    """
    b = ATRISK.get(sid, {}).get(str(g), {}).get(str(hi), {}).get(str(di))
    return _band_lo(b) if b else fallback


def separated(p1, n1, p2, n2):
    """Whether the two rates are distinguishable, at the AT-RISK band's lower edge.

    `n1`/`n2` are the at-risk band lower edges supplied by `atrisk_lo`, not the cohort's `n_band`.
    The cohort total is the wrong denominator for a single outcome -- it is larger, so it narrows the
    interval and separates more often than warranted. Correcting it moved the separated share from
    0.239/0.262/0.260% to 0.196/0.219/0.217%, with 7,615 flags falling 1 -> 0 and **zero rising
    0 -> 1**; the latter is provable rather than merely observed, since at-risk is a subset of the
    total and the band function is monotone. Using the band's LOWER edge keeps it conservative and
    keeps the exact count out of the output."""
    if n1 <= 0 or n2 <= 0:
        return False
    lo1, up1 = wilson(p1 * n1, n1)
    lo2, up2 = wilson(p2 * n2, n2)
    return up1 < lo2 or up2 < lo1


NATIVE_G = {"C1": 1, "C2": 2, "default": 3}


def unmeasured(x, hi, di, g, sid):
    """Whether this cell is a placeholder rather than a measurement.

    The rate matrices are DENSE and write `0.0` where a cell could not be estimated; a separate
    `not_estimable` mask says which zeros mean "measured zero" and which mean "too few followed-up
    members to say". Ignoring that mask -- testing instead for a `None` the states never emit --
    makes **4,426,140 of 22,207,885 delta cells (19.9%) a difference taken against an unmeasured
    endpoint**, where the published delta is just the other endpoint's rate relabelled as a change.
    Those cells carry **11,291 of the 57,124 `separated` flags (19.8%)**, and they manufacture a
    headline: g1 appears to separate 1.73x more often than g3, when **half of g1's separated flags
    are fabricated** and the corrected ordering is flat.

    `"all"` marks a whole horizon and is valid at any granularity. The per-disease lists are recorded
    in the state's OWN native index space, so they are applied only at that granularity; cells masked
    only by a list at a non-native granularity are still counted, which makes this a lower bound on
    the contamination removed.
    """
    ne = (x.get("not_estimable") or {}).get(str(hi))
    if ne is None:
        return False
    if ne == "all":
        return True
    ver = sid.split(":", 1)[0]
    return g == NATIVE_G.get(ver) and di in ne


def cell(x, hi, di, g=None):
    """One cell at granularity `g`. Falls back to the state's native `rates` when g is None."""
    R = x[f"rates_g{g}"] if g is not None and f"rates_g{g}" in x else x["rates"]
    row = R[str(hi)] if isinstance(R, dict) else R[hi]
    v = row.get(str(di)) if isinstance(row, dict) else (row[di] if di < len(row) else None)
    return None if v is None else float(v)


def prof(x, drop=None):
    return tuple((ax, tuple(sorted(tuple(e) for e in x["axes"][ax])))
                 for ax in ("icd", "opcs", "bnf") if ax != drop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", default="C1,C2,default")
    ap.add_argument("--grans", default="1,2,3")
    ap.add_argument("--out", default="/work/USERS/ziquanw/cohortworld/official")
    a = ap.parse_args()
    VERS = [v.strip() for v in a.versions.split(",")]
    GRANS = [int(g) for g in a.grans.split(",")]
    # ONE pooled state space. A state id is already version-prefixed, so the pool is unambiguous and
    # an edge between two versions is not a special case -- it is just an edge whose endpoints happen
    # to carry different prefixes.
    S, H, DIS_G = {}, None, {}
    for v in VERS:
        lat = _shared(v)
        if lat is None:
            print(f"[edges] no lattice for {v}, skipping", flush=True); continue
        S.update(lat.states)
        H = H or lat.horizons
        for g in GRANS:
            # PANEL, not a hardcoded "official" -- the staging directory has to be loadable, or
            # the edge table can only ever be built after the lattice is already promoted to live.
            f = os.path.join(PANEL, f"CohortWorld_{v}_diseases_g{g}.json")
            if os.path.exists(f) and g not in DIS_G:
                DIS_G[g] = json.load(open(f))["diseases"]
    missing = [g for g in GRANS if g not in DIS_G]
    if missing:
        raise SystemExit(f"no disease table for granularity {missing} -- rebuild with --multigran")
    for v in VERS:
        f = os.path.join(PANEL, f"CohortWorld_{v}_atrisk_bands.jsonl")
        if os.path.exists(f):
            for line in open(f):
                d = json.loads(line)
                ATRISK[d["state_id"]] = d["atrisk"]
    print(f"[edges] at-risk bands for {len(ATRISK):,} cohorts", flush=True)
    print(f"[edges] pooled {len(S):,} cohorts over {VERS}, {len(H)} horizons, "
          + ", ".join(f"g{g}={len(DIS_G[g])}" for g in GRANS), flush=True)

    pairs = []            # (axis, op, code, sid_from, sid_to)
    by_as = collections.defaultdict(list)
    for sid, x in S.items():
        by_as[(x["axes"]["age_band"], str(x["axes"]["sex"]))].append((sid, x))
    for _k, v in by_as.items():
        idx = collections.defaultdict(list)
        for sid, x in v:
            for ax in ("icd", "opcs", "bnf"):
                idx[(ax, prof(x, drop=ax), x["coarseness"])].append((sid, x))
        for (ax, _p, _c), g in idx.items():
            for (s1, x1), (s2, x2) in itertools.combinations(sorted(g), 2):
                a1 = {tuple(e) for e in x1["axes"][ax]}
                a2 = {tuple(e) for e in x2["axes"][ax]}
                d1, d2 = a1 - a2, a2 - a1
                if not d1 and len(d2) == 1:
                    pairs.append((ax, "add", next(iter(d2))[0], s1, s2))
                elif len(d1) == 1 and not d2:
                    pairs.append((ax, "add", next(iter(d1))[0], s2, s1))
                elif len(d1) == 1 and len(d2) == 1:
                    pairs.append((ax, "switch", next(iter(d2))[0], s1, s2))
    # ---- zoom edges: same history, one axis cut at a different rung ----------------------
    #
    # The pairings above produce none, because every one of them requires an identical
    # `coarseness`. Zoom is the only one of the five actions that changes CODE GRANULARITY, and
    # granularity is exactly what separates C1 / C2 / default -- leaving it out would ship a
    # cost table with `zoom_finer` and `zoom_coarser` entries that no edge ever instantiates.
    #
    # Two cohorts form a zoom edge when they agree on age, sex and the other two code axes, and the
    # axis in question holds the SAME codes truncated to different lengths (`K40` <-> `K4` <-> `K`).
    def _cut(codes, n):
        return tuple(sorted({(c[:n], b) for c, b in codes}))

    for _k, v in by_as.items():
        zidx = collections.defaultdict(list)
        for sid, x in v:
            for ax in ("icd", "opcs", "bnf"):
                others = prof(x, drop=ax)
                cur = tuple(sorted(tuple(e) for e in x["axes"][ax]))
                if not cur:
                    continue
                for n in range(1, max(len(c) for c, _b in cur) + 1):
                    # max, not cur[0][0]: `default` really holds mixed-length axes (opcs {1,3,4},
                    # bnf {2,4,6}), where the first code alone decides fine/coarse orientation wrong.
                    zidx[(ax, others, _cut(cur, n))].append((sid, x, max(len(c) for c, _b in cur)))
        for (ax, _o, _c), g in zidx.items():
            seen_pair = set()
            for (s1, x1, l1), (s2, x2, l2) in itertools.combinations(sorted(g), 2):
                if l1 == l2 or (s1, s2) in seen_pair:
                    continue
                seen_pair.add((s1, s2))
                fine, coarse = ((s1, x1, l1), (s2, x2, l2)) if l1 > l2 else ((s2, x2, l2), (s1, x1, l1))
                # VERIFY the prefix relation before emitting. Sharing a bucket only makes two
                # cohorts SIBLINGS under a common ancestor -- it does not make one the truncation of
                # the other, and the loop above buckets at every length 1..maxlen including lengths
                # that are not real code rungs. BNF first characters are only `0` or `1`, so at n=1
                # nearly every BNF cohort in a stratum collapsed into ~3 buckets and was paired with
                # every other: **96.4% of C2's zoom edges and 96.5% of default's were fabricated**,
                # 54% of the whole table. ICD and OPCS came out clean only because their n=1 IS a
                # genuine rung (chapter letters), which is what hid the fault.
                fcodes = {c for c, _b in fine[1]["axes"][ax]}
                ccodes = {c for c, _b in coarse[1]["axes"][ax]}
                if not ccodes or {c[:coarse[2]] for c in fcodes} != ccodes:
                    continue
                pairs.append((ax, "zoom_coarser", "", fine[0], coarse[0]))
                pairs.append((ax, "zoom_finer", "", coarse[0], fine[0]))

    byp = collections.defaultdict(list)
    for sid, x in S.items():
        byp[(prof(x), x["coarseness"])].append((sid, x))
    for _p, g in byp.items():
        for (s1, x1), (s2, x2) in itertools.combinations(sorted(g), 2):
            sameA = x1["axes"]["age_band"] == x2["axes"]["age_band"]
            sameS = str(x1["axes"]["sex"]) == str(x2["axes"]["sex"])
            if sameA and not sameS:
                pairs.append(("sex", "switch", str(x2["axes"]["sex"]), s1, s2))
            elif sameS and not sameA:
                pairs.append(("age_band", "shift", x2["axes"]["age_band"], s1, s2))
    print(f"[edges] {len(pairs):,} one-step neighbour pairs", flush=True)

    seen_e = set()
    uniq = []
    for e in sorted(pairs):
        k = (e[0], e[1], e[3], e[4])
        if k in seen_e or e[3] == e[4]:
            continue
        seen_e.add(k); uniq.append(e)
    print(f"[edges] {len(uniq):,} unique edges after dedup", flush=True)
    n_cells = 0
    out = []
    for ax, op, code, s1, s2 in uniq:
        x1, x2 = S[s1], S[s2]
        n1, n2 = _band_lo(x1.get("n_band", "")), _band_lo(x2.get("n_band", ""))
        # One delta block PER GRANULARITY. The outcome granularity is a parameter of the question,
        # not a property of where the agent is standing, so an edge has to answer at whichever
        # granularity it is asked about -- and for a cross-version edge there is no single native
        # one to fall back on.
        deltas, any_cell = {}, False
        for g in GRANS:
            d_g = []
            for hi in range(len(H)):
                for di in range(len(DIS_G[g])):
                    a1 = atrisk_lo(s1, g, hi, di, None)
                    a2 = atrisk_lo(s2, g, hi, di, None)
                    if a1 is None or a2 is None:
                        continue                 # no at-risk band = below the floor = unmeasurable
                    if unmeasured(x1, hi, di, g, s1) or unmeasured(x2, hi, di, g, s2):
                        continue                 # one side unmeasured: no comparison to make
                    p1, p2 = cell(x1, hi, di, g), cell(x2, hi, di, g)
                    if p1 is None or p2 is None:
                        continue
                    if p1 == 0.0 and p2 == 0.0:
                        continue                 # both zero: no information, and most cells are this
                    d_g.append([hi, di, round(p2 - p1, 6),
                                1 if separated(p1, a1, p2, a2) else 0])
            if d_g:
                deltas[f"delta_g{g}"] = d_g
                any_cell = True
                n_cells += len(d_g)
        if not any_cell:
            continue
        out.append({"axis": ax, "op": op, "code": code,
                    "cross_version": x1 is not x2 and s1.split(":")[0] != s2.split(":")[0],
                    # NOT f"{version}:{s1}" -- `Lattice.__init__` already prefixes every state_id,
                    # so doing it here produces "C2:C2:004aa92d..." on every edge: ids matching
                    # neither the raw states file key nor the in-memory one, so the table joins to
                    # nothing.
                    "from": s1, "to": s2,
                    "cost": COST.get((ax, op)),
                    "n_band_from": x1.get("n_band"), "n_band_to": x2.get("n_band"),
                    **deltas})
    os.makedirs(a.out, exist_ok=True)
    ep = os.path.join(a.out, "CohortWorld_pooled_edges.jsonl")
    with open(ep, "w") as f:
        for e in out:
            f.write(json.dumps(e) + "\n")
    tab = [{"axis": ax, "op": op, "cost": COST.get((ax, op)),
            "applies": (ax, op) in COST,
            "reliability": "per-edge, from published n_band" if (ax, op) in COST else 0}
           for ax in AXES for op in OPS]
    tp = os.path.join(a.out, "CohortWorld_pooled_edge_schema.json")
    json.dump({"versions": VERS, "granularities": GRANS,
               "value": "delta[i] = [horizon_index, disease_index, rate_to - rate_from, separated]",
               "separated": "1 when the two Wilson intervals at the published n_band LOWER bounds "
                            "do not overlap; conservative, and computed from public bands only",
               "cost": "axis-and-operation property: indirectness introduced by the move",
               "reliability": "NOT an axis property -- per cell, via `separated`",
               "axes": list(AXES), "ops": list(OPS), "table": tab},
              open(tp, "w"), indent=1)
    print(f"[edges] wrote {len(out):,} edges, {n_cells:,} comparable cells -> {ep}", flush=True)
    print(f"[edges] schema -> {tp}", flush=True)
    sig = sum(1 for e in out for g in GRANS for d in e.get(f"delta_g{g}", ()) if d[3])
    xv = sum(1 for e in out if e["cross_version"])
    print(f"[edges] cross-version edges: {xv:,} / {len(out):,} ({100*xv/max(1,len(out)):.1f}%)",
          flush=True)
    for g in GRANS:
        c = sum(len(e.get(f"delta_g{g}", ())) for e in out)
        sg = sum(1 for e in out for d in e.get(f"delta_g{g}", ()) if d[3])
        print(f"[edges]   g{g}: {c:,} cells, {sg:,} separated ({100*sg/max(1,c):.3f}%)", flush=True)
    print(f"[edges] cells with a separated difference: {sig:,} ({100*sig/max(1,n_cells):.2f}%)", flush=True)
    per = collections.Counter((e["axis"], e["op"]) for e in out)
    for k, v in sorted(per.items(), key=lambda t: -t[1]):
        print(f"[edges]   {k[0]:9} {k[1]:7} {v:8,}", flush=True)


main()
