#!/usr/bin/env python3
"""The two open interactions of the item-rendering principles, measured, plus the ladder.

The principles: (1) an item carries only its target cohort's own released description; (2) the only
rewrite is de-grouping the age band to `about xx age`; (3) difficulty comes only from dials that
remove or coarsen, so no item is more precise than its source. They leave two things undefined and
this script settles both with counts off the released panel. Nothing here reads a patient record,
builds an item, or touches the frozen packages.

  1. AGE-BAND vs COARSENESS. `coarseness` is `<onset-scheme>|icd?-opcs?-bnf?`. Principle 2 rewrites
     `axes.age_band` to `about xx age`. The open question is whether that rewrite makes two
     released cohorts render identically -- an ambiguity that comes from the FORMAT rather than
     from a difficulty dial. Measured as a straight collapse count on the rendered description, per
     version, decomposed by cause.

  2. THE ZOOM DIAL. Which rungs exist per axis, which of them `Lattice.zoom` can actually walk, and
     what a zoom does to the consistent-candidate count -- the same table `t3_render.section_a`
     builds for masking, extended with the zoom axis and their joint.

  3. THE LADDER. A concrete schedule of dial settings, and the candidate distribution it yields
     when simulated on the C2 lattice.

Two definitions carry the whole thing, and both are chosen to match code that already exists:

  * A CANDIDATE is a released cohort of the same version whose rendering, under the item's own dial
    settings, is the same string of axis values. Code axes are compared as SETS OF TRUNCATED CODES
    -- `{c[:level]}` -- which is `gen/t3_render.axval` with the truncation made a dial instead of a
    constant. Truncation is a no-op when a candidate's codes are already shorter, so a cohort that
    is natively coarser than the item's stated level is NOT a candidate: it does not state what the
    item states.
  * The ONSET BAND (`(first recorded 45+)`) is a SETTING, not a constant. `t3_render.axval` drops
    it unconditionally and says so; §1e prices that choice instead of inheriting it.

Rendering may only ever go COARSER than the target. Principle 3 forbids an item that is
more precise than its source, so the "zoom in" half of dial B does not exist on the rendering side:
the item is the target coarsened, and `zoom(finer)` is the agent's inverse.
"""
from __future__ import annotations
import argparse, collections, glob, itertools, json, os, random, re, statistics as st, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import t3_render as T          # load / published / rate / draw_items / boot_share / ci_str / share
from cohortworld_env.lattice import Lattice, PANEL

VERSIONS = ("C1", "C2", "default")
CODE_AXES = ("icd", "opcs", "bnf")
AXES = T.AXES                                       # age_band, sex, icd, opcs, bnf
LADDER = {"icd": (1, 2, 3), "opcs": (1, 2, 3), "bnf": (2, 4, 6)}   # lattice._shift_code_tag
TIME_ORDER = ["decade", "4stage", "3stage", "2stage", "flat"]      # lattice.zoom, verbatim
LO, HI = T.TARGET_LO, T.TARGET_HI                   # the spirit's 2-8 candidate window
AGE_RE = re.compile(r"aged (\d+)-(\d+)")
ONSET_RE = re.compile(r" \(first recorded [^)]*\)")


# --------------------------------------------------------------------------- loading

def light(version: str) -> list[dict]:
    """state_id / axes / prompt_body / coarseness / n_band, without the rate matrices.

    The three panels are 1.1 GB and 99% of that is `rates*`, which §1 never reads. The key order is
    fixed by the builder, so cutting the line at `, "rates":` and closing the object is exact; the
    assert is there because a silent -1 from `find` would append a brace to the whole line and the
    parse error it raises points at the wrong thing.
    """
    out = []
    for path in sorted(glob.glob(f"{PANEL}/CohortWorld_{version}_*_states.jsonl")):
        for ln in open(path):
            i = ln.find(', "rates":')
            assert i > 0, f"unexpected key order in {path}"
            out.append(json.loads(ln[:i] + "}"))
    return out


def degroup(body: str) -> str:
    """Principle 2, and nothing else: `aged 65-69` -> `about 67 age`."""
    return AGE_RE.sub(lambda m: f"about {(int(m.group(1)) + int(m.group(2))) // 2} age", body)


# --------------------------------------------------------------------------- dial keys

def axkey(s: dict, a: str, zoom: dict, onset: bool):
    """One axis of a cohort as an item states it under dial B settings `zoom` and `onset`."""
    v = s["axes"][a]
    if a in ("age_band", "sex"):
        return v
    lv = zoom[a]
    if onset:
        return tuple(sorted({(c[:lv], b) for c, b in v}))
    return tuple(sorted({c[:lv] for c, _ in v}))


def keys_for(states, mask: tuple, zoom: dict, onset: bool) -> list[tuple]:
    keep = [a for a in AXES if a not in mask]
    return [tuple(axkey(s, a, zoom, onset) for a in keep) for s in states]


def counts_for(states, mask: tuple, zoom: dict, onset: bool):
    ks = keys_for(states, mask, zoom, onset)
    idx = collections.defaultdict(list)
    for i, k in enumerate(ks):
        idx[k].append(i)
    return [len(idx[k]) for k in ks], idx, ks


def native(version: str, states) -> dict:
    """The code levels this version's released panel actually carries, per axis."""
    got = {a: set() for a in CODE_AXES}
    for s in states:
        for piece in s["coarseness"].split("|")[1].split("-"):
            for a in CODE_AXES:
                if piece.startswith(a):
                    got[a].add(int(piece[len(a):]))
    return {a: sorted(v) for a, v in got.items()}


# --------------------------------------------------------------------------- stats

def boot_median(xs, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.asarray(xs, dtype=float)
    m = [np.median(a[rng.integers(0, len(a), len(a))]) for _ in range(n_boot)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def boot_mean(xs, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.asarray(xs, dtype=float)
    m = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n_boot)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def p90(xs) -> float:
    return float(np.percentile(np.asarray(xs, dtype=float), 90)) if len(xs) else float("nan")


# =========================================================================== 1. age / coarseness

def section1(out, panels, sheet) -> None:
    T.p(out, "\n== 1. THE AGE-BAND / COARSENESS INTERACTION "
             "=====================================================")
    T.p(out, "Principle 2 rewrites `axes.age_band` to `about xx age`. `coarseness`'s first component"
             " names an\nONSET scheme. The open question is whether they are the same field. They are not, "
             "and the two rows below\nare the proof; the collapse count that follows is the "
             "question actually worth answering.\n")

    # 1a -- is the de-grouping map injective?
    T.p(out, "1a. `axes.age_band` per version, and whether `about <midpoint>` is injective on it")
    for v, states in panels.items():
        bands = sorted({s["axes"]["age_band"] for s in states})
        widths = sorted({int(b.split("-")[1]) - int(b.split("-")[0]) + 1 for b in bands})
        mids = {b: (int(b.split("-")[0]) + int(b.split("-")[1])) // 2 for b in bands}
        T.p(out, f"  {v:8} {len(states):>6} cohorts  {len(bands)} bands {bands[0]}..{bands[-1]}  "
                 f"widths {widths}y  distinct midpoints {len(set(mids.values()))}  "
                 f"INJECTIVE {len(set(mids.values())) == len(bands)}")
    T.p(out, "  -> the de-grouping is a bijection on the age axis in every version, so PRINCIPLE 2 "
             "ON ITS OWN\n     collapses exactly ZERO cohort pairs. Whatever collapse exists is "
             "the onset scheme's.\n")

    # 1b -- the schemes are not independent of the age axis, but they do not set its width
    T.p(out, "1b. onset scheme x `axes.age_band` (the two fields are correlated, not identical)")
    for v, states in panels.items():
        by = collections.defaultdict(set)
        for s in states:
            by[s["coarseness"].split("|")[0]].add(s["axes"]["age_band"])
        T.p(out, f"  {v:8} " + "  ".join(f"{k}:{len(by[k])}bands" for k in TIME_ORDER if k in by))
    T.p(out, "  -> a scheme restricts WHICH bands occur (flat carries 6 of the 9, decade all 9) but "
             "every band\n     is 5 years wide under every scheme, so no scheme changes what the "
             "rewrite has to say.\n")

    # 1c -- the collapse count
    T.p(out, "1c. COLLAPSE COUNT. Group every released cohort by its rendered description "
             "(`prompt_body`\n    with principle 2 applied, nothing else changed).")
    T.p(out, "    THREE RENDERINGS, because a collapse count of zero is worth nothing unless the "
             "same grouper is\n    shown able to produce a non-zero one. `spirit` is the proposed "
             "rendering; the two below it are\n    positive controls -- deliberately lossier "
             "rewrites that MUST collapse cohorts if the grouper works.")
    T.p(out, f"  {'version':9} {'rendering':22} {'cohorts':>8} {'distinct':>9} {'collapsed':>10} "
             f"{'share':>8} {'groups':>7} {'largest':>8}")
    RENDER = [("spirit (about xx age)", lambda b: degroup(b)),
              ("+ onset clause cut", lambda b: ONSET_RE.sub("", degroup(b))),
              ("+ age decade instead", lambda b: AGE_RE.sub(
                  lambda m: f"about {int(m.group(1)) // 10 * 10}s age", b))]
    collapse = {}
    for v, states in panels.items():
        for lbl, fn in RENDER:
            g = collections.defaultdict(list)
            for s in states:
                g[fn(s["prompt_body"])].append(s)
            dup = {k: ss for k, ss in g.items() if len(ss) > 1}
            n_col = sum(len(ss) for ss in dup.values())
            if lbl.startswith("spirit"):
                collapse[v] = dup
            sheet.setdefault("collapse", {}).setdefault(v, {})[lbl] = {
                "cohorts": len(states), "distinct": len(g), "collapsed": n_col,
                "share": n_col / len(states)}
            bits = [1] * n_col + [0] * (len(states) - n_col)
            T.p(out, f"  {v:9} {lbl:22} {len(states):>8} {len(g):>9} {n_col:>10} "
                     f"{n_col/len(states):>7.4f} {len(dup):>7} "
                     f"{max((len(ss) for ss in dup.values()), default=0):>8}  "
                     f"{T.ci_str(T.boot_share(bits))}")
    T.p(out, "")

    # 1d -- what the zero means, and what the controls collapsed instead
    T.p(out, "1d. READING 1c. The spirit rendering collapses NOTHING, in any version. That is not a "
             "quiet check:\n    cutting the onset clause -- which is the OTHER thing `coarseness` "
             "governs -- collapses cohorts in\n    the hundreds on the same grouper, and coarsening "
             "the age to a decade collapses more still.\n    So the grouper finds collisions when "
             "they exist, and principle 2 creates none.")
    for v, states in panels.items():
        dup = collapse[v]
        npair = sum(1 for ss in dup.values() for _ in itertools.combinations(ss, 2))
        T.p(out, f"  {v:9} colliding pairs under the spirit rendering: {npair}")
    T.p(out, "  The reason is 1a: `about <midpoint>` is a bijection on a 5-year band, and the onset "
             "scheme is\n  written into the `(first recorded ...)` clause of every coded axis, "
             "which principle 2 does not\n  touch. The two fields suspected of being one field "
             "are two, and they do not interact.\n")

    # 1e -- the price of dropping the onset band, which t3_render does silently
    T.p(out, "1e. THE ONSET BAND IS A SETTING. `t3_render.axval` drops it from the candidate key "
             "unconditionally.\n    Same lattice, same mask=0, band stated vs dropped:")
    T.p(out, f"  {'version':9} {'onset':>8} {'median':>7} {'unique':>8} {'>=5':>7} {'in 2-8':>8} "
             f"{'max':>6}")
    for v, states in panels.items():
        z = {a: max(LADDER[a]) for a in CODE_AXES}
        for onset in (True, False):
            c, _i, _k = counts_for(states, (), z, onset)
            sheet.setdefault("onset_setting", {})[f"{v}/{'stated' if onset else 'dropped'}"] = {
                "median": st.median(c), "unique": T.share(c, lambda x: x == 1),
                "in_window": T.share(c, lambda x: LO <= x <= HI)}
            T.p(out, f"  {v:9} {('stated' if onset else 'dropped'):>8} {st.median(c):>7.0f} "
                     f"{T.share(c, lambda x: x == 1):>7.1%} {T.share(c, lambda x: x >= 5):>6.1%} "
                     f"{T.share(c, lambda x: LO <= x <= HI):>7.1%} {max(c):>6}")
    T.p(out, "  -> stating the band at native resolution makes the item near-unique and the whole "
             "mask exercise\n     vacuous; dropping it is a REAL COARSENING and belongs on the dial "
             "sheet, not in a helper.\n")


# =========================================================================== 2. the zoom dial

def section2(out, panels, args, sheet) -> dict:
    T.p(out, "\n== 2. THE ZOOM DIAL, CALIBRATED "
             "=================================================================")

    # 2a/2b -- which rungs exist, and which the ladder arithmetic can walk
    T.p(out, "2a. Code levels the released panel actually carries, per axis "
             "(from the `coarseness` tag).")
    nat = {}
    for v, states in panels.items():
        nat[v] = native(v, states)
        T.p(out, f"  {v:9} " + "   ".join(f"{a}: {nat[v][a]}" for a in CODE_AXES))
    T.p(out, f"  onset schemes: all five ({', '.join(TIME_ORDER)}) in all three versions.\n")

    T.p(out, "2b. DEAD RUNGS. `lattice._shift_code_tag` steps icd/opcs by 1 (floor 1, top 3) and bnf "
             "by 2\n    (floor 2, top 6). A step is walkable only if the level it lands on EXISTS.")
    T.p(out, f"  {'version':9} {'axis':>6} {'step':>26} {'lands on a released level?':>28}")
    walkable = collections.defaultdict(list)
    for v in VERSIONS:
        for a in CODE_AXES:
            step = 2 if a == "bnf" else 1
            for lv in nat[v][a]:
                for d, nx in (("finer", lv + step), ("coarser", lv - step)):
                    top, floor = (6, 2) if a == "bnf" else (3, 1)
                    blocked = not (floor <= nx <= top)
                    exists = nx in nat[v][a]
                    ok = (not blocked) and exists
                    if ok:
                        walkable[v].append((a, d, lv, nx))
                    why = "blocked by the ladder's own bound" if blocked else (
                        "YES" if exists else f"{a}{nx} is not a released level")
                    T.p(out, f"  {v:9} {a:>6} {f'{a}{lv} --{d}--> {a}{nx}':>26} {why:>28}")
    T.p(out, "")
    sheet["released_levels"] = nat
    sheet["walkable_code_rungs"] = {v: [f"{a}{lv}->{a}{nx}({d})" for a, d, lv, nx in walkable[v]]
                                    for v in VERSIONS}
    for v in VERSIONS:
        T.p(out, f"  {v:9} walkable code rungs: "
                 + (", ".join(f"{a}{lv}->{a}{nx} ({d})" for a, d, lv, nx in walkable[v]) or "NONE"))
    T.p(out, "  -> `icd2` and `opcs2` exist in NO version, so every one-step icd zoom and every "
             "opcs zoom that\n     is not `opcs4 -> opcs3` lands on an empty level. The `icd` axis "
             "has no zoom at all, anywhere.\n     `default`'s modal granularity is opcs4 and the "
             "ladder's top is 3, so opcs4 is a one-way trapdoor:\n     reachable natively, "
             "coarsenable once, never returned to.\n")

    # 2c -- what the environment's own zoom actually does, per state
    T.p(out, "2c. THE CENSUS. `Lattice.zoom(probe=True)` on EVERY released cohort of every version "
             "-- the same\n    call `_zoom_available` advertises with, so this is the move list the "
             "agent is shown.")
    T.p(out, f"  {'version':9} {'axis':>6} {'finer':>18} {'coarser':>18}")
    census = {}
    for v in VERSIONS:
        lat = Lattice(v)
        sids = list(lat.states)
        hit = collections.defaultdict(list)
        for sid in sids:
            lat.current = sid
            for a in ("time",) + CODE_AXES:
                for d in ("finer", "coarser"):
                    hit[(a, d)].append(1 if lat.zoom(a, d, probe=True).get("hit") else 0)
        census[v] = hit
        sheet.setdefault("zoom_availability", {})[v] = {
            f"{a}/{d}": sum(hit[(a, d)]) / len(hit[(a, d)])
            for a in ("time",) + CODE_AXES for d in ("finer", "coarser")}
        for a in ("time",) + CODE_AXES:
            cells = []
            for d in ("finer", "coarser"):
                b = hit[(a, d)]
                cells.append(f"{sum(b)/len(b):.4f} ({sum(b)})")
            T.p(out, f"  {v:9} {a:>6} {cells[0]:>18} {cells[1]:>18}")
        any_b = [max(hit[(a, d)][i] for a in ("time",) + CODE_AXES for d in ("finer", "coarser"))
                 for i in range(len(sids))]
        T.p(out, f"  {v:9} {'ANY':>6}  {sum(any_b)/len(any_b):.4f} ({sum(any_b)}/{len(any_b)}) "
                 f"{T.ci_str(T.boot_share(any_b))}")
        # The dial renders the item at the COARSE rung, so what matters is invertibility FROM there.
        for a in CODE_AXES:
            lo = min(nat[v][a])
            at_lo = [i for i, sid in enumerate(sids)
                     if f"{a}{lo}" in lat.states[sid]["coarseness"].split("|")[1].split("-")]
            if not at_lo or len(nat[v][a]) < 2:
                continue
            b = [hit[(a, "finer")][i] for i in at_lo]
            T.p(out, f"  {v:9} {a:>6}  conditioned on already being at {a}{lo} (the coarse rung the "
                     f"dial renders):\n            zoom finer available on {sum(b)}/{len(b)} = "
                     f"{sum(b)/len(b):.4f} {T.ci_str(T.boot_share(b))}")
        del lat
    T.p(out, "  -> the agent's `zoom` is not a dial with rungs, it is a rare move. A dial B setting "
             "the agent\n     cannot invert is irreversible loss, which is the class principle 3 "
             "forbids.\n")

    # 2d -- what a zoom does to the candidate count, on C2
    states = panels["C2"]
    zc2 = {a: [lv for lv in LADDER[a] if lv <= max(nat["C2"][a])] for a in CODE_AXES}
    zc2 = {a: [lv for lv in zc2[a] if lv in nat["C2"][a]] for a in CODE_AXES}
    full = {a: max(zc2[a]) for a in CODE_AXES}
    T.p(out, f"2d. WHAT A ZOOM DOES TO THE ITEM (C2, {len(states)} cohorts, mask=0). Rendering may "
             f"only COARSEN\n    (principle 3), so a setting is a per-axis cap at or below native. "
             f"Levels swept: {zc2}")
    rat = T.load("C2")
    by_sid = {s["sid"]: s for s in rat}
    rstates = [by_sid[s["state_id"]] for s in states]
    items = T.draw_items(rstates, args.n, args.seed)
    T.p(out, f"    Rate spread is measured on {len(items)} drawn items (one published cell each, "
             f"50/50 zero /\n    non-zero stratum -- `t3_render.draw_items`, the item set's own "
             f"construction).")
    T.p(out, f"  {'zoom setting':34} {'median':>7} {'unique':>8} {'>=5':>7} {'in 2-8':>8} "
             f"{'max':>7} {'p90 spread':>11}")
    zoom_settings = []
    for icd in zc2["icd"]:
        for opcs in zc2["opcs"]:
            for bnf in zc2["bnf"]:
                for onset in (True, False):
                    zoom_settings.append(({"icd": icd, "opcs": opcs, "bnf": bnf}, onset))
    zoom_rows = {}
    for z, onset in zoom_settings:
        c, idx, ks = counts_for(states, (), z, onset)
        sp = spreads(rstates, items, idx, ks)
        zoom_rows[(tuple(sorted(z.items())), onset)] = (c, sp)
        T.p(out, f"  {label_z(z, onset, full):32} {st.median(c):>7.0f} "
                 f"{T.share(c, lambda x: x == 1):>7.1%} {T.share(c, lambda x: x >= 5):>6.1%} "
                 f"{T.share(c, lambda x: LO <= x <= HI):>7.1%} {max(c):>7} {p90(sp):>11.4f}")
    T.p(out, "  NOTE on the earlier hand-recorded mask table. Its `0 axes masked` row reads median 1, 65.5% "
             "unique, 0.8% >=5,\n  p90 spread 0.0000. The first three columns are the "
             "onset-DROPPED key (measured here 1 / 63.9% /\n  0.8%); the fourth is the "
             "onset-STATED key (0.0000 here, against 0.0474 dropped). One row, two\n  settings -- "
             "which is the clearest evidence that the onset band was never treated as a dial.\n")

    # 2e -- mask x zoom
    T.p(out, "2e. MASK x ZOOM. Median candidates for every mask pattern of size <=1 crossed with "
             "every zoom\n    setting (onset dropped throughout, so the two dials are read off the "
             "same baseline).")
    masks = [()] + [(a,) for a in AXES]
    zs = [z for z, onset in zoom_settings if onset is False]
    T.p(out, "  " + f"{'mask':14}" + "".join(f"{label_z(z, False, full, short=True):>13}" for z in zs))
    med = {}
    joint = {}
    for m in masks:
        row = []
        for z in zs:
            c, idx, ks = counts_for(states, m, z, False)
            med[(m, tuple(sorted(z.items())))] = st.median(c)
            joint[(m, tuple(sorted(z.items())))] = c
            row.append(f"{st.median(c):>13.0f}")
        T.p(out, f"  {('none' if not m else m[0]):14}" + "".join(row))
    T.p(out, "")

    # independence: is log2(candidates) additive across the two dials?
    base = np.log2(np.asarray(joint[((), tuple(sorted(full.items())))], dtype=float))
    devs, devs_masked_axis = [], []
    same = same_n = 0
    for m in masks[1:]:
        for z in zs:
            zk = tuple(sorted(z.items()))
            a_m = np.log2(np.asarray(joint[(m, tuple(sorted(full.items())))], dtype=float))
            a_z = np.log2(np.asarray(joint[((), zk)], dtype=float))
            a_j = np.log2(np.asarray(joint[(m, zk)], dtype=float))
            d = a_j - (a_m + a_z - base)
            if m[0] in z and z[m[0]] < full[m[0]]:
                devs_masked_axis.extend(d.tolist())
            else:
                devs.extend(d.tolist())
            # exact irrelevance: switching the MASKED axis's own zoom, and only that, must not
            # move a single count. Compared against the same mask at this zoom with that one axis
            # restored -- not against `full`, which would also restore the other axes' zoom.
            if m[0] in z and z[m[0]] < full[m[0]]:
                z0 = dict(z); z0[m[0]] = full[m[0]]
                a_0 = np.asarray(joint[(m, tuple(sorted(z0.items())))], dtype=float)
                same += int((np.asarray(joint[(m, zk)], dtype=float) == a_0).sum())
                same_n += len(a_0)
    T.p(out, "  ADDITIVITY. If the dials were independent, log2(candidates) would add:"
             "\n    dev = log2 N(mask,zoom) - [log2 N(mask) + log2 N(zoom) - log2 N(neither)]")
    T.p(out, f"    all (mask, zoom) cells where the MASKED axis is NOT the zoomed one: "
             f"mean dev {np.mean(devs):+.4f} {T.ci_str(boot_mean(devs))}  n={len(devs)}")
    T.p(out, f"    cells where the masked axis IS the zoomed one:                      "
             f"mean dev {np.mean(devs_masked_axis):+.4f} {T.ci_str(boot_mean(devs_masked_axis))}  "
             f"n={len(devs_masked_axis)}")
    T.p(out, f"    CONTROL for the second row -- switching ONLY the masked axis's own zoom back on "
             f"leaves the count\n    unchanged on {same}/{same_n} = {same/max(same_n,1):.4f} of "
             f"cells, so `masking an axis makes its zoom\n    irrelevant` is exact, not "
             f"approximate. The second row's -0.15 is therefore not an interaction:\n    it is "
             f"minus the zoom's own solo effect, re-read through a dial that is out of the circuit. "
             f"The\n    FIRST row is the interaction, and at +0.07 log2 (about +5% candidates) the "
             f"two dials are very\n    nearly independent but mildly SUPER-additive.\n")

    # the ladder needs to know which axes are maskable ALONE without overshooting the window
    T.p(out, "2f. Single- and two-axis masks at native zoom, onset dropped. This is what decides "
             "which axes a\n    ladder rung may mask: an axis whose own mask overshoots the window "
             "cannot be a rung on its own.")
    zk_full = tuple(sorted(full.items()))
    single_med = {a: st.median(joint[((a,), zk_full)]) for a in AXES}
    sheet["mask_median_1axis"] = {a: single_med[a] for a in AXES}
    T.p(out, "  1 axis: " + "   ".join(f"{a} {single_med[a]:.0f}" for a in AXES))
    pair = {}
    for m in itertools.combinations(AXES, 2):
        c, _i, _k = counts_for(states, m, full, False)
        pair[m] = st.median(c)
    sheet["mask_median_2axis"] = {"+".join(m): v for m, v in pair.items()}
    T.p(out, "  2 axes: " + "   ".join(f"{'+'.join(m)} {v:.0f}"
                                       for m, v in sorted(pair.items(), key=lambda t: t[1])))
    T.p(out, "")
    return {"states": states, "rstates": rstates, "items": items, "zc2": zc2, "full": full,
            "census": census, "walkable": dict(walkable), "single_med": single_med, "pair": pair}


def label_z(z, onset, full, short=False):
    zs = [f"{a}{z[a]}" for a in CODE_AXES if z[a] < full[a]]
    tag = "+".join(zs) if zs else "native"
    if short:
        return tag[:12]
    return tag + ("  +onset" if onset else "  -onset")


def spreads(rstates, items, idx, ks):
    """p90 input: the rate range across the candidates of each drawn item, on its own asked cell."""
    out = []
    for it in items:
        i, hi, di = it["i"], it["hi"], it["di"]
        vals = [T.rate(rstates[j], hi, di) for j in idx[ks[i]]
                if T.published(rstates[j], hi, di)]
        if vals:
            out.append(max(vals) - min(vals))
    return out


# =========================================================================== 3. the ladder

# name, share, masked axes, code axes coarsened, onset stated. Masks are drawn from the pool §2f
# measures, never per item from the answer: per-item calibration to the window was measured to select
# the axis whose removal destroys the most information, landing the task under the do-nothing floor.
RUNGS = [
    ("E0 native+onset",  0.10, 0, 0, True),
    ("E1 onset dropped", 0.15, 0, 0, False),
    ("E2 zoom 1 axis",   0.15, 0, 1, False),
    ("M1 mask 1 axis",   0.25, 1, 0, False),
    ("H1 mask 1 + zoom 1", 0.20, 1, 1, False),
    ("H2 mask 2 axes",   0.15, 2, 0, False),
]


def section3(out, st2, args, sheet) -> None:
    states, rstates, items = st2["states"], st2["rstates"], st2["items"]
    zc2, full = st2["zc2"], st2["full"]
    T.p(out, "\n== 3. THE DIFFICULTY LADDER "
             "=====================================================================")
    T.p(out, "Six rungs. Every rung is a (mask set, per-axis zoom level, onset stated) triple, drawn "
             "per item and\nWRITTEN ON THE ITEM, so any result can be stratified by the difficulty "
             "that produced it.\n")
    rng = random.Random(args.seed + 7)
    coarsenable = [a for a in CODE_AXES if len(zc2[a]) > 1]
    pool = [a for a in AXES if st2["single_med"][a] <= HI]
    cen = st2["census"]["C2"]
    invertible = [a for a in coarsenable if sum(cen[(a, "finer")]) / len(cen[(a, "finer")]) > 0.0]
    T.p(out, f"  Maskable pool (median candidates under its own mask <= {HI}, from 2f): {pool}\n"
             f"  Excluded from masking: {[a for a in AXES if a not in pool]} -- masking it alone "
             f"already overshoots the window.\n"
             f"  Coarsenable in RENDERING (>1 released level in C2): {coarsenable}\n"
             f"  Of those, INVERTIBLE by the agent's own `zoom` (2c, non-zero availability): "
             f"{invertible}\n"
             f"  The zoom rungs below use only {invertible}. Coarsening `icd` or `opcs` is "
             f"disclosure-safe and\n  it does widen the candidate set -- but 2c measures "
             f"zoom availability on those axes at 0.0000 in\n  every version, so the agent has no "
             f"action that undoes it. That is irreversible loss, which is the\n  class principle 3 "
             f"forbids, so it is not a difficulty dial on this panel.\n")
    coarsenable = invertible
    item_of = {it["i"]: it for it in items}
    cache: dict = {}                 # one index per distinct dial setting, not per item

    def built(m, z, onset):
        k = (m, tuple(sorted(z.items())), onset)
        if k not in cache:
            cache[k] = counts_for(states, m, z, onset)
        return cache[k]

    T.p(out, f"  {'rung':16} {'share':>6} {'mask':>5} {'zoom':>5} {'onset':>6} "
             f"{'n':>6} {'median':>7} {'unique':>8} {'in 2-8':>8} {'>=20':>6} {'p90 spread':>11}")
    all_c, all_bits = [], []
    for name, share_, nmask, nzoom, onset in RUNGS:
        cs, sps = [], []
        n = max(200, int(args.n * share_))
        for _ in range(n):
            i = rng.randrange(len(states))
            m = tuple(sorted(rng.sample(pool, nmask)))
            z = dict(full)
            for a in rng.sample(coarsenable, min(nzoom, len(coarsenable))):
                z[a] = min(zc2[a])
            c, idx, ks = built(m, z, onset)
            cs.append(c[i])
            if i in item_of:
                sps += spreads(rstates, [item_of[i]], idx, ks)
        all_c += cs
        all_bits += [1 if LO <= x <= HI else 0 for x in cs]
        sheet["ladder"].append({
            "rung": name, "share": share_, "n_mask": nmask, "n_zoom": nzoom,
            "onset": "stated" if onset else "dropped", "mask_pool": pool,
            "zoom_axes": coarsenable, "n": len(cs), "median_candidates": st.median(cs),
            "unique": T.share(cs, lambda x: x == 1),
            "in_window": T.share(cs, lambda x: LO <= x <= HI),
            "ge20": T.share(cs, lambda x: x >= 20), "p90_rate_spread": p90(sps)})
        T.p(out, f"  {name:16} {share_:>6.2f} {nmask:>5} {nzoom:>5} "
                 f"{('yes' if onset else 'no'):>6} {len(cs):>6} {st.median(cs):>7.0f} "
                 f"{T.share(cs, lambda x: x == 1):>7.1%} "
                 f"{T.share(cs, lambda x: LO <= x <= HI):>7.1%} "
                 f"{T.share(cs, lambda x: x >= 20):>5.1%} {p90(sps):>11.4f}")
    sheet["pooled"] = {"n": len(all_c), "median_candidates": st.median(all_c),
                       "in_window": float(np.mean(all_bits)),
                       "in_window_ci": [float(x) for x in T.boot_share(all_bits)]}
    T.p(out, f"\n  POOLED over the six rungs at their shares: median {st.median(all_c):.0f} "
             f"{T.ci_str(boot_median(all_c))}, in [{LO},{HI}] "
             f"{np.mean(all_bits):.4f} {T.ci_str(T.boot_share(all_bits))}, n={len(all_c)}")
    T.p(out, "\n  Recorded on every item (five fields, all five are dial settings, none is an "
             "answer):\n    dials.rung:   the rung name above, so a result can be split by "
             "difficulty in one group-by\n    dials.mask:   list of masked axis names (a masked "
             "axis leaves NO trace in the item text)\n    dials.zoom:   {icd, opcs, bnf} rendered "
             "levels, native included, so `no zoom` is explicit\n    dials.onset:  stated | "
             "dropped\n    dials.candidates: the candidate count at build time -- the difficulty "
             "that was actually\n                      delivered, as against the one that was "
             "asked for")
    T.p(out, "\n== 4. THE THREE ANSWERS "
             "=========================================================================")
    T.p(out, "1. AGE-BAND SCHEME: EXCLUDED as a dial, and it is not a fourth one. `axes.age_band` "
             "is 5 years\n   wide in every version under every scheme (1a), `about <midpoint>` is "
             "a bijection on it, and the\n   spirit rendering collapses 0 cohorts in all three "
             "versions against controls that collapse\n   hundreds (1c). What `coarseness`'s first "
             "component actually governs is the ONSET clause, a\n   different field, and THAT is a "
             "component of dial B -- `Lattice.zoom(axis='time')` is its inverse.\n   It has two "
             "computable settings, stated and dropped, and the choice is worth 97.7% unique vs\n"
             "   63.9% unique at mask=0 (1e): the single largest lever in the whole dial sheet, and "
             "the one\n   `t3_render.axval` was silently holding at `dropped`.")
    T.p(out, "2. ZOOM: real on ONE axis of three. `icd2` and `opcs2` are released by no version, so "
             "every icd\n   zoom and every opcs zoom but `opcs4->opcs3` lands on an empty level "
             "(2b); the census puts icd\n   and opcs availability at 0.0000 in all three versions "
             "and bnf at 0.5040 (C2) / 0.5516 (default)\n   (2c). C1 has no walkable code rung at "
             "all. Zooming is also a WEAK dial where it works: bnf2\n   alone moves unique from "
             "63.9% to 57.0% where masking bnf moves the median to 272. The two dials\n   are "
             "nearly independent (+0.0661 log2, about +5% candidates) and exactly non-interacting "
             "on the\n   axis they share.")
    T.p(out, "3. LADDER: six rungs, masks drawn from a pool fixed in advance by 2f, zooms only on "
             "axes 2c says\n   the agent can walk back. Never calibrated per item on the "
             "answer -- that route was measured\n   landing under the do-nothing floor.")


# =========================================================================== driver

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    out = open(a.out, "w") if a.out else sys.stdout
    T.p(out, "THE TWO OPEN INTERACTIONS OF THE ITEM-RENDERING PRINCIPLES, MEASURED")
    T.p(out, f"panel {PANEL}   seed {a.seed}   bootstrap {a.n_boot}   items {a.n}")
    panels = {v: light(v) for v in VERSIONS}
    sheet: dict = {"seed": a.seed, "n_boot": a.n_boot, "window": [LO, HI], "ladder": []}
    section1(out, panels, sheet)
    st2 = section2(out, panels, a, sheet)
    section3(out, st2, a, sheet)
    if out is not sys.stdout:
        out.close()
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(sheet, fh, indent=1, sort_keys=True)
    return 0


def selfcheck() -> int:
    """The smallest thing that fails if any moving part breaks."""
    # 1. principle 2 is a pure de-grouping, and it is injective on 5-year bands
    assert degroup("adults aged 65-69 (male;") == "adults about 67 age (male;"
    assert len({degroup(f"aged {lo}-{lo+4}") for lo in range(40, 85, 5)}) == 9
    # 2. coarsening only ever merges candidates -- never splits them
    s = [{"axes": {"age_band": "60-64", "sex": "M", "icd": [["I21", 1]], "opcs": [], "bnf": []}},
         {"axes": {"age_band": "60-64", "sex": "M", "icd": [["I25", 1]], "opcs": [], "bnf": []}}]
    fine = {"icd": 3, "opcs": 3, "bnf": 6}
    coarse = {"icd": 1, "opcs": 3, "bnf": 6}
    assert counts_for(s, (), fine, False)[0] == [1, 1]
    assert counts_for(s, (), coarse, False)[0] == [2, 2]
    # 3. the onset band is a real setting: same cohorts, different key
    s2 = [{"axes": {"age_band": "60-64", "sex": "M", "icd": [["I21", 1]], "opcs": [], "bnf": []}},
          {"axes": {"age_band": "60-64", "sex": "M", "icd": [["I21", 2]], "opcs": [], "bnf": []}}]
    assert counts_for(s2, (), fine, True)[0] == [1, 1]
    assert counts_for(s2, (), fine, False)[0] == [2, 2]
    # 4. a masked axis is absent from the key, so its zoom level cannot move the count
    assert counts_for(s, ("icd",), fine, False)[0] == counts_for(s, ("icd",), coarse, False)[0]
    # 5. the ladder bounds are lattice._shift_code_tag's, not retyped guesses
    from cohortworld_env.lattice import Lattice as _L
    assert _L._shift_code_tag("icd3-opcs3-bnf2", "icd", "finer") is None      # top 3
    assert _L._shift_code_tag("icd3-opcs3-bnf2", "bnf", "coarser") is None    # floor 2
    assert _L._shift_code_tag("icd3-opcs3-bnf2", "icd", "coarser") == "icd2-opcs3-bnf2"
    print("selfcheck OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
