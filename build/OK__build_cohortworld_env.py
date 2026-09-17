#!/usr/bin/env python3
# release-gate: allow-file -- ran against the private UKB extract; shipped for provenance, paths shown as they were
"""CohortWorld — a verifier ENVIRONMENT built from UK Biobank.

STATE is the five axes and nothing else:
    (age band, sex, ICD history, OPCS history, BNF history)
each history axis being a set of (code prefix, age-band-at-first-occurrence) pairs. The index year is
used only during CONSTRUCTION -- to fix the age, to decide what counts as history, and to define the
outcome window -- and does NOT survive into the built environment. Consequently two index years that
yield the SAME five axes are the SAME state and are POOLED: a state carries one rate matrix computed
over all of its (person, index-year) observations.

Per state the environment stores an [H x N_disease] matrix of future incidence, over a FIXED disease
table, plus a banded person count. Actions are atomic edits to the OPCS and BNF axes only; applying
one yields a new five-axis tuple, and the transition is legal iff a released state carries that tuple.

WHY THIS SHAPE: THREE DEFECTS IT RULES OUT
  * state is keyed by its axes, so splitting states by an axis hash is profile-disjoint BY
    CONSTRUCTION. A person-level split leaves 56.6% of test descriptions verbatim in train -- the
    same failure mode as clinical/FIXED__grpo_train.py:96, "People-disjoint splits have 96% profile
    overlap".
  * the disease table is FIXED, so the question set does not depend on the answer. Emitting an item
    only for diseases someone in the cohort actually had would make "being asked" evidence of a
    non-zero rate.
  * history comes from cohortworld_hx_alldates.pkl, which keeps records past 2010. An extract that
    cuts every axis at 2010-01-01 describes 10 of 11 index years by records up to a decade stale
    while measuring the true later outcome window.

DISCLOSURE (profile B, user's call). Public files carry a BANDED person count and the raw computed
rate; exact counts live only under --private. rate is deliberately NOT coarsened, which leaves the
numerator recoverable for a minority of cells -- an accepted cost, recorded in the stats.
"""
import argparse, json, os, pickle, sys, collections, hashlib, bisect, time
import multiprocessing as mp

# ---- read-only state shared with pool workers via fork (copy-on-write); never mutated after setup --
_G = {}


def _keychunk(task):
    """Bucket key for a chunk of rows. Returned as a 128-bit digest rather than the key itself:
    the key holds a frozenset of string tuples, and pickling millions of those back to the parent
    costs more than computing them. The parent groups by digest and rebuilds the real key only for
    the buckets that actually get released."""
    idxs, gran, edges = task
    rows, HX, SEX = _G["rows"], _G["HX"], _G["SEX"]
    GI = {"I": gran[0], "O": gran[1], "B": gran[2]}
    bin_age, age_band = PC.bin_age, PC.age_band
    out = {}
    for ri in idxs:
        pid, t, age_Y, obs_Y, cut = rows[ri]
        g = {}
        for (age, s2, c2) in HX[pid][:cut]:
            n = GI[s2]
            # n < 0 is the body-system supergroup rung. Without this branch `c2[:-1]` silently
            # chopped the last character off the code -- neither a chapter nor a supergroup, and no
            # error, so the rung produced keys that matched nothing and coverage never improved.
            if n < 0:
                sc = supercode(s2, c2)
                if sc is None:
                    continue          # unnameable chapter: left out, never pooled as "other"
                k = (s2, sc)
            else:
                k = (s2, c2[:n] if n else "*")
            b = bin_age(age, edges) if edges else ""
            if k not in g or (isinstance(b, int) and b < g[k]):
                g[k] = b
        canon = f"{age_band(int(age_Y))}|{SEX[pid]}|" + ";".join(
            sorted(f"{s3}:{c3}:{b3}" for (s3, c3), b3 in g.items()))
        dg = hashlib.blake2b(canon.encode(), digest_size=16).digest()
        b = out.get(dg)
        if b is None:
            out[dg] = {pid: ri}
        else:
            prev = b.get(pid)
            if prev is None or t < rows[prev][1]:
                b[pid] = ri
    kept = {dg: list(b.values()) for dg, b in out.items()}
    # rows this worker collapsed as a person's later re-entry -- the parent adds them to its own
    # tally, otherwise the row-accounting assertion (rightly) fails
    return kept, len(idxs) - sum(len(v) for v in kept.values())


def _rowkey(ri, gran, edges):
    """The real bucket key for one row -- rebuilt in the parent for released buckets only."""
    rows, HX, SEX = _G["rows"], _G["HX"], _G["SEX"]
    pid, t, age_Y, obs_Y, cut = rows[ri]
    GI = {"I": gran[0], "O": gran[1], "B": gran[2]}
    g = {}
    for (age, s2, c2) in HX[pid][:cut]:
        n = GI[s2]
        if n < 0:
            sc = supercode(s2, c2)
            if sc is None:
                continue
            k = (s2, sc)
        else:
            k = (s2, c2[:n] if n else "*")
        b = PC.bin_age(age, edges) if edges else ""
        if k not in g or (isinstance(b, int) and b < g[k]):
            g[k] = b
    return (PC.age_band(int(age_Y)), SEX[pid], frozenset((s3, c3, b3) for (s3, c3), b3 in g.items()))


def _matrix(task):
    """Kaplan-Meier rate matrix for one state. Per-cell counts come back SPARSE: every disease with
    no events shares the same at-risk denominator, so only the diseases present in the state need
    their own entry -- dense counts for `default` would be 1,374 x 5 integers per state."""
    s_, members = task
    rows, DX = _G["rows"], _G["DX"]
    OUT_N, FILT, DIS, DIDX = _G["OUT_N"], _G["FILT"], _G["DIS"], _G["DIDX"]
    obs_r, prev_r, inc_r, yr_r = [], [], [], []
    for r in members:
        pid, Y, age_Y, obs_Y, cut = rows[r]
        pv, ic = set(), {}
        for c, t in DX[pid].items():
            k = c[:OUT_N]
            if not k or k[0] in FILT:
                continue
            dt = t - Y
            if dt <= 0:
                pv.add(k)
            elif k not in ic or dt < ic[k]:
                ic[k] = dt
        obs_r.append(obs_Y)
        yr_r.append(Y)
        prev_r.append(pv)
        inc_r.append({k: v for k, v in ic.items() if k not in pv})

    present = set()
    for pv in prev_r:
        present |= pv
    for ic in inc_r:
        present |= set(ic)
    nrow = len(members)

    # LANDMARK ELIGIBILITY + DEATH AS A COMPETING RISK.
    #
    # A row whose index sits less than H years before the end of records cannot be observed for H
    # years, so it is EXCLUDED from that horizon rather than entered as a censored observation.
    # Feeding it to Kaplan-Meier as censored was wrong in practice: index points run to 2020 while
    # records end 2021-03, so at H=10 only ~11% of rows had full administrative follow-up, KM
    # extrapolated from the few that did, and the tail blew up -- 23.7% of cells came out above their
    # naive event/at-risk ratio (one read KM 0.873 off 14 events in 36 people), and the curve hit
    # exactly 1.0 whenever the last at-risk row had an event, manufacturing 3,177 "homogeneous"
    # states that were an artefact, not a disclosure.
    #
    # Once that restriction is in place the ONLY thing that can end follow-up early is death, and
    # death is a COMPETING RISK, not censoring: a person who dies at year 3 without the disease did
    # not go on to acquire it. Treating death as censoring (which 1-KM does) assumes the dead would
    # have carried the survivors' risk, which inflates the estimate exactly where it matters most --
    # 10-year mortality runs 1.2% at ages 40-49 but 12.5% at 70-79, and states are age-banded.
    # With no censoring left, the Aalen-Johansen cumulative incidence reduces to the plain
    # proportion, so no survival curve is needed at all: deaths simply stay in the denominator as
    # non-events.
    nz = []                       # (horizon idx, disease idx, rate) for the non-zero cells only
    est = [[False] * len(DIS) for _ in HORIZONS]
    sparse = {}
    nonest = 0
    for hi, H in enumerate(HORIZONS):
        elig = [i for i in range(nrow) if yr_r[i] + H <= STUDY_END_FRAC]
        for di, d in enumerate(DIS):
            at = [i for i in elig if d not in prev_r[i]]
            if len(at) < PC.MIN_N:
                nonest += 1
                continue
            ev = 0
            for i in at:
                t = inc_r[i].get(d)
                if t is not None and t <= H and t <= obs_r[i]:
                    ev += 1
            if ev:
                nz.append((hi, di, round(ev / len(at), 6)))
            est[hi][di] = True
            if d in present:
                sparse.setdefault(d, []).append([hi, len(at), ev])

    ne = {}
    for hi in range(len(HORIZONS)):
        bad = [di for di in range(len(DIS)) if not est[hi][di]]
        if bad:
            ne[hi] = "all" if len(bad) == len(DIS) else bad
    return s_, nz, ne, {"present": sparse}, nonest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import panel_common as PC
from OK__build_cohortworld_v2 import VERSIONS, TIME_LEVELS, ICD_CH, sysname  # ICD_CH: disease-name table

PASSLOG = bool(os.environ.get("CW_PASSLOG"))
BASE_YEAR = 2010
STUDY_END_FRAC = 2021 + (31 + 28 + 31) / 365.25      # 2021-03-31 as a fractional year
HORIZONS = [1, 2, 3, 5, 10]
MIN_AGE = 18
N_BANDS = [(10, 49), (50, 199), (200, 999), (1000, None)]
# Relative price of generalising each history axis. OPCS/BNF are the actionable axes and are held at
# full resolution for longer; ICD is spent first. Set all three equal to weight the axes the same.
AXIS_COST = {"I": 1, "O": 2, "B": 2}
SYSFULL = {"I": "diagnoses", "O": "procedures", "B": "prescriptions"}


def _ekey(e):
    return (e[0], -1 if e[1] is None else e[1])


def n_band(n):
    for lo, hi in N_BANDS:
        if hi is None or n <= hi:
            return f"{lo}+" if hi is None else f"{lo}-{hi}"


OFFICIAL_RUNGS = {
    # Levels that exist in the classifications themselves, coarsest last. A 2-character ICD or OPCS
    # prefix is NOT one of them -- ICD blocks are ranges like A15-A19, which no fixed-width prefix
    # reproduces -- so generalising by character count invented levels that mean nothing to a
    # clinician. Source codes cap what is reachable: ICD arrives 3-character, OPCS 4, BNF 10.
    "I": [3, 1],        # category, chapter
    "O": [4, 3, 1],     # subcategory, category, chapter
    "B": [6, 4, 2],     # paragraph, section, chapter
}


def _axis_rungs(n, step):
    """One axis's generalisation rungs, finest to the CHAPTER, and no further.

    ICD/OPCS drop one character at a time to the chapter letter; BNF drops two at a time (paragraph 6
    -> section 4 -> chapter 2), a 1-char BNF prefix not being a level of the classification.

    There is deliberately NO presence-only rung. Such a rung ends in 0, meaning the code collapses to
    "*" and membership requires only that a record exist -- and "*" is not a code in any
    classification, it is the absence of one. Checked against members, a single such cohort of 17
    people carried 47 distinct OPCS codes between them, so the rung throws away exactly the history
    CohortWorld exists to carry, and leaves a category no reader can interpret. A profile too rare to
    reach k at its chapter is suppressed instead of being flattened into meaninglessness.
    """
    sysk = {1: None, 2: "B"}.get(step)
    raise RuntimeError("use axis_rungs(system, start) instead")


def axis_rungs(system: str, start: int) -> list[int]:
    """The official levels of one classification, from `start` down to its chapter."""
    return [r for r in OFFICIAL_RUNGS[system] if r <= start]


def code_ladder(gran0):
    """The three history axes generalise INDEPENDENTLY.

    Coarsening them in lockstep (a single ladder, as inherited from PhenoCal, which only ever had ONE
    history axis) means a person whose ICD history is rare has their OPCS and BNF
    history flattened too, even when both are common. Measured on the lockstep build: 80% of all
    person-slots ended up at the all-presence-only rung, in ~1,030 states averaging 600 people each,
    and 51% of slots had all three axes flattened at once. Generalising per axis keeps the axes that
    do not need help.

    Rungs are ordered by TOTAL information dropped across the three axes, so the greedy release still
    hands each person the finest profile that reaches k; ties are broken deterministically."""
    li = axis_rungs("I", gran0[0])
    lo = axis_rungs("O", gran0[1])
    lb = axis_rungs("B", gran0[2])
    ri = {v: k for k, v in enumerate(li)}
    ro = {v: k for k, v in enumerate(lo)}
    rb = {v: k for k, v in enumerate(lb)}
    combos = [(i, o, b) for i in li for o in lo for b in lb]
    # OPCS and BNF are the two axes an agent can ACT on, so their resolution is what makes the
    # environment controllable; ICD history is context the agent cannot change. Generalisation
    # therefore spends ICD precision first: one step down an actionable axis costs as much as
    # AXIS_COST["I"] / AXIS_COST["O"] steps down ICD. With the 1:2 weighting below, ICD is flattened
    # to presence-only before either actionable axis is coarsened at all.
    combos.sort(key=lambda c: (AXIS_COST["I"] * ri[c[0]] + AXIS_COST["O"] * ro[c[1]]
                               + AXIS_COST["B"] * rb[c[2]],
                               ro[c[1]] + rb[c[2]],          # then keep the actionable axes finest
                               ri[c[0]], -c[0], -c[1], -c[2]))
    return combos


def usable_years():
    """(index year -> horizons that still fit before the censoring date)."""
    out = {}
    for Y in range(BASE_YEAR, 2021):
        hs = [h for h in HORIZONS if Y + h <= STUDY_END_FRAC]
        if hs:
            out[Y] = hs
    return out


def render(ageband, sex, grid, edges):
    """Serialise a state to a PhenoCal-style prompt body.

    Two rules, each guarding against a silent corruption of the text:
      * generalisation collapses several codes onto one prefix (K21, K57 -> K); keeping one entry per
        ORIGINAL code makes a prompt read "digestive (first recorded 30-40); digestive (first
        recorded 50-60)". A generalised code has ONE first occurrence: the earliest. Entries are
        therefore merged on the code and the minimum band kept.
      * sex is stored as "Male"/"Female", so a test `sex == 'F'` is never true and would describe
        every cohort -- including every female one -- as male."""
    lab = PC.stage_label(edges) if edges else None
    by = collections.defaultdict(list)
    for (s, code, band) in sorted(grid):
        nm = sysname(s, code)
        # ICD chapters A and B share the label "infectious disease", and OPCS/BNF have the same
        # collisions, so a name-only entry made distinct states render identically. Carry the code.
        if code != "*" and nm != code:
            nm = f"{nm} [{code}]"
        show = lab and isinstance(band, int) and 0 <= band < len(lab)
        by[s].append(nm + (f" (first recorded {lab[band]})" if show else ""))
    lines = [f"Consider UK adults aged {ageband} "
             f"({'female' if str(sex).upper().startswith('F') else 'male'}; "
             f"linked primary-care and hospital records) with the following recorded history:"]
    for s in ("I", "O", "B"):
        lines.append(f"- {SYSFULL[s]}: " + ("; ".join(by[s]) if by[s] else "none recorded"))
    return "\n".join(lines)


def _state_id(m):
    """The published state_id: md5 of the AXES, identical to what phase 3 writes. Defined once so the
    augmentation joins to the frozen files on the same key instead of an internal counter."""
    axes = collections.defaultdict(list)
    for (s2, c2, b2) in sorted(m["grid"]):
        axes[s2].append([c2, b2 if b2 != "" else None])
    canon = json.dumps([m["ageband"], m["sex"], axes["I"], axes["O"], axes["B"], m["tag"]],
                       sort_keys=True)
    return hashlib.md5(canon.encode()).hexdigest()[:16]


def _augment(a, tag, rows, states, meta, HX, BASE, gran0, ladder):
    """Extra PRIVATE artifacts the task suite needs, added without touching the lattice.

    The build is deterministic (verified byte-identical across PYTHONHASHSEED values), so re-running
    reproduces exactly the frozen states; this only writes more about them.

      members    state_id -> [(person, index time)]. S2's split-half ground truth has to subsample a
                 large cell and re-apply the publication rule to the subsample, which needs the
                 members; the aggregate counts alone cannot do it.
      frontier   the frontier is undefined for 23.6% of states, because their time granularity is
                 `flat` and the axes then carry no bands to order codes by. Age at the latest OPCS/BNF
                 code is therefore computed per MEMBER on the continuous age scale, outside the
                 bucketing, and summarised: modal code, median age, and the share of members agreeing
                 on that code -- a frontier that only 30% of members share is not a state property and
                 the number says so.
      ancestors  greedy release puts each person at exactly ONE rung, so a fine cell's "coarse
                 ancestor" is not its superset -- it is the different people who failed at the fine
                 rung. The granularity free-lunch figure needs the true ancestor, i.e. ALL rows
                 sharing the coarser key. Computed here over every row, released or not.
    """
    os.makedirs(a.augment, exist_ok=True)
    import statistics as _st
    with open(os.path.join(a.augment, f"{tag}_members.jsonl"), "w") as f:
        for s_, mem in sorted(states.items()):
            f.write(json.dumps({"state_id": _state_id(meta[s_]),
                                "members": [[rows[r][0], round(rows[r][1], 4)] for r in mem]}) + "\n")
    with open(os.path.join(a.augment, f"{tag}_frontier.jsonl"), "w") as f:
        for s_, mem in sorted(states.items()):
            best = []
            for r in mem:
                pid, t, age_Y, obs_Y, cut = rows[r]
                last = None
                for (age, sy, c) in HX[pid][:cut]:
                    if sy in ("O", "B") and (last is None or age > last[0]):
                        last = (age, sy, c)
                if last:
                    best.append(last)
            if not best:
                f.write(json.dumps({"state_id": _state_id(meta[s_]), "frontier": None}) + "\n")
                continue
            mode = collections.Counter((sy, c) for _a, sy, c in best).most_common(1)[0]
            ages = [x[0] for x in best]
            f.write(json.dumps({
                "state_id": _state_id(meta[s_]), "frontier_axis": mode[0][0],
                "frontier_code": mode[0][1],
                "modal_share": round(mode[1] / len(best), 4),
                "age_at_frontier_median": round(_st.median(ages), 2),
                "age_at_frontier_iqr": [round(sorted(ages)[len(ages)//4], 2),
                                        round(sorted(ages)[3*len(ages)//4], 2)],
                "members_with_any": len(best), "members": len(mem)}) + "\n")
    # ancestors: one rung coarser, and the fully-collapsed rung, over ALL rows
    anc = {}
    for tagn, gran in (("one_step", ladder[1] if len(ladder) > 1 else ladder[0]),
                       ("collapsed", ladder[-1])):
        GI = {"I": gran[0], "O": gran[1], "B": gran[2]}
        buck = collections.defaultdict(list)
        for ri, (pid, t, age_Y, obs_Y, cut) in enumerate(rows):
            g = {}
            for (age, sy, c) in HX[pid][:cut]:
                n = GI[sy]
                k = (sy, c[:n] if n else "*")
                if k not in g:
                    g[k] = True
            buck[(PC.age_band(int(age_Y)), SEXG[pid], frozenset(g))].append(ri)
        for s_, mem in states.items():
            m = meta[s_]
            pid0, t0, age0, obs0, cut0 = rows[mem[0]]
            g = {}
            for (age, sy, c) in HX[pid0][:cut0]:
                n = GI[sy]
                g[(sy, c[:n] if n else "*")] = True
            key = (PC.age_band(int(age0)), SEXG[pid0], frozenset(g))
            anc.setdefault(s_, {})[tagn] = len(buck.get(key, ()))
    with open(os.path.join(a.augment, f"{tag}_ancestors.jsonl"), "w") as f:
        for s_ in sorted(states):
            f.write(json.dumps({"state_id": _state_id(meta[s_]), "own_members": len(states[s_]),
                                "ancestor_rows": anc.get(s_, {})}) + "\n")
    print(f"[env] augment -> {a.augment}: members / frontier / ancestors for {len(states):,} states",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=list(VERSIONS))
    ap.add_argument("--hx", default="output_panel/cohortworld_hx_alldates.pkl")
    ap.add_argument("--inter", default="output_panel/panel_intermediate.pkl")
    ap.add_argument("--filterout", default="ZRUVWXY")
    ap.add_argument("--out", default="output_panel")
    ap.add_argument("--private", default=os.environ.get("CW_PRIVATE")
                    or "CW_PRIVATE-is-not-set-point-it-at-the-private-store")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", default="50,10,40")
    ap.add_argument("--years", default="")
    ap.add_argument("--index-mode", default="event", choices=["event", "yearly"])
    ap.add_argument("--workers", type=int, default=0, help="0 = auto (cores-8, max 24)")
    ap.add_argument("--multigran", default="", help="also emit rate matrices at these outcome "
                    "granularities (comma-separated char counts, e.g. 1,2,3) so that an edge between "
                    "two versions has a comparable delta. The three versions share NO outcome code "
                    "-- C1 is 1-char, C2 2-char, default 3-char, pairwise intersection exactly zero "
                    "-- so a cross-version delta is otherwise uncomputable, and 71.6% of the genuine "
                    "history-axis zoom structure is cross-version. Granularity is one line in "
                    "`_matrix` (`k = c[:OUT_N]`), so the extra passes reuse the same members and the "
                    "same diagnosis times.")
    ap.add_argument("--augment", default="",
                    help="also emit the private artifacts the task suite needs (member lists, "
                         "frontier, coarse ancestors) into this dir, and assert the lattice is "
                         "identical to the frozen one")
    a = ap.parse_args()
    FILT = frozenset(a.filterout.upper())
    tr, va = (int(x) for x in a.ratios.split(",")[:2])
    *gran0, OUT_N = VERSIONS[a.version]
    gran0 = tuple(gran0)
    os.makedirs(a.private, exist_ok=True)

    D = pickle.load(open(a.inter, "rb"))["patients"]
    hx = pickle.load(open(a.hx, "rb"))
    assert hx.get("all_dates"), "need the all-dates history extract; the 2010-capped one is invalid here"
    # GP-LINKED ONLY: for an unlinked subject an absent BNF axis means UNKNOWN, not "no medication",
    # and the renderer cannot distinguish those two states.
    pids = sorted(p for p, v in D.items() if v.get("gp_linked", True))     # sorted => reproducible
    print(f"[env] GP-linked subjects: {len(pids):,}", flush=True)

    # Per person, history sorted by age so the pre-index history at ANY index year is a PREFIX.
    HX, AGES, BASE, OBS, SEX, DX = {}, {}, {}, {}, {}, {}
    for pid in pids:
        v = D[pid]
        h = sorted((age, s, c) for s, key in (("I", "icd"), ("O", "opcs"), ("B", "bnf"))
                   for c, age in hx[key].get(pid, {}).items())
        HX[pid] = h
        AGES[pid] = [x[0] for x in h]
        BASE[pid] = float(v["age_at_index"])
        OBS[pid] = float(v["obs_years"])
        SEX[pid] = v["sex"]
        DX[pid] = {c: (int(d[:4]) + (int(d[5:7]) - 1) / 12.0) for c, d in v["dx"].items()}
    del D, hx

    YEARS = usable_years()
    if a.years:
        keep = {int(x) for x in a.years.split(",")}
        YEARS = {y: h for y, h in YEARS.items() if y in keep}
    print(f"[env] index years {min(YEARS)}-{max(YEARS)} ({len(YEARS)})", flush=True)

    # ---- index points ----
    # A yearly grid samples each person on 1 January and therefore MISSES every five-axis combination
    # they occupy for less than a year -- and those combinations then never accumulate the 10 distinct
    # persons they need to be released. (Evidence it binds: 87% of released states sat in the 10-49
    # person band, i.e. right at the k threshold, while 84% were already at the FINEST code rung, so
    # granularity was not the limit.) Index points are therefore placed where a person's axes ACTUALLY
    # change -- the first occurrence of each new code prefix, and each age-band boundary -- plus the
    # window start. Every combination a person passes through is then sampled exactly once.
    #
    # The window cannot open before 2010: obs_years is follow-up measured from 2010-01-01, so every
    # person with obs > 0 is known to have survived to 2010. An earlier index would hand them
    # guaranteed survival time and bias early-horizon incidence down (immortal time bias).
    T_LO, T_HI = float(BASE_YEAR), STUDY_END_FRAC - min(HORIZONS)
    AGE_EDGES = range(0, 111, 5)                    # PC.age_band uses 5-year bands
    rows = []
    if a.index_mode == "yearly":
        for Y in sorted(YEARS):
            for pid in pids:
                age_Y, obs_Y = BASE[pid] + (Y - BASE_YEAR), OBS[pid] - (Y - BASE_YEAR)
                if obs_Y <= 0 or age_Y < MIN_AGE:
                    continue
                rows.append((pid, float(Y), age_Y, obs_Y, bisect.bisect_right(AGES[pid], age_Y)))
    else:
        gi0 = {"I": gran0[0], "O": gran0[1], "B": gran0[2]}
        for pid in pids:
            b0 = BASE[pid]
            ts = {T_LO}
            seen = set()
            for (age, s2, c2) in HX[pid]:            # first appearance of each finest-level prefix
                k = (s2, c2[:gi0[s2]])
                if k in seen:
                    continue
                seen.add(k)
                ts.add(BASE_YEAR + (age - b0))
            for e in AGE_EDGES:                      # crossing into a new age band
                if e > b0:
                    ts.add(BASE_YEAR + (e - b0))
            for t in ts:
                if not (T_LO <= t <= T_HI):
                    continue
                age_t, obs_t = b0 + (t - BASE_YEAR), OBS[pid] - (t - BASE_YEAR)
                if obs_t <= 0 or age_t < MIN_AGE:
                    continue
                rows.append((pid, t, age_t, obs_t, bisect.bisect_right(AGES[pid], age_t)))
    print(f"[env] index mode {a.index_mode}: {len(rows):,} (person, index-point) observations "
          f"= {len(rows)/len(pids):.1f} per person", flush=True)

    # ---- fixed disease table (needed by the matrix workers, so computed before the pool) ----
    DIS = sorted({c[:OUT_N] for pid in pids for c in DX[pid] if c and c[0] not in FILT})
    DIDX = {d: i for i, d in enumerate(DIS)}
    print(f"[env] fixed disease table: {len(DIS)} codes at {OUT_N}-char", flush=True)

    # Workers inherit these by fork (copy-on-write) instead of pickling them per task.
    globals()["SEXG"] = SEX
    _G.update(rows=rows, HX=HX, SEX=SEX, DX=DX, OUT_N=OUT_N, FILT=FILT, DIS=DIS, DIDX=DIDX)
    nw = a.workers or max(1, min(24, (os.cpu_count() or 8) - 8))
    pool = mp.get_context("fork").Pool(nw)
    print(f"[env] pool: {nw} workers", flush=True)

    # ---- adaptive local recoding over ALL rows pooled; k counted on DISTINCT PERSONS ----
    unreleased = list(range(len(rows)))
    states, meta, sid, collapsed = {}, {}, 0, 0
    for gran in code_ladder(gran0):
        for tag, edges in TIME_LEVELS:
            if not unreleased:
                break
            nin = len(unreleased)
            CH = max(2000, len(unreleased) // (nw * 4) + 1)
            tasks = [(unreleased[i:i + CH], gran, edges)
                     for i in range(0, len(unreleased), CH)]
            buckets = collections.defaultdict(list)
            # The workers group their own chunk, so the parent merges one list per DIGEST instead
            # of touching every row. With per-row merging the parent sat at 96% CPU while all 24
            # workers together managed only ~430% of a possible 2400% -- they were waiting on it.
            t0 = time.time()
            for part, dropped in pool.imap(_keychunk, tasks, chunksize=1):
                collapsed += dropped
                for dg, lst in part.items():
                    buckets[dg].extend(lst)
            t1 = time.time()
            still = []
            for dg in buckets:                 # digest order: deterministic, hash-seed free
                members = buckets[dg]
                # A person can sit in the same five-axis combination at several index points.
                # Counting each separately would let one person contribute many rows to the same
                # survival curve. Keep only their EARLIEST entry -- the landmark at which the state
                # becomes true -- so observations and distinct persons coincide.
                firstof = {}
                for r in members:
                    pid_r = rows[r][0]
                    if pid_r not in firstof or rows[r][1] < rows[firstof[pid_r]][1]:
                        firstof[pid_r] = r
                kept = sorted(firstof.values())
                collapsed += len(members) - len(kept)
                if len(kept) >= PC.MIN_N:
                    key = _rowkey(kept[0], gran, edges)   # real key, rebuilt only when released
                    states[sid] = kept
                    meta[sid] = dict(ageband=key[0], sex=key[1], grid=key[2], edges=edges,
                                     tag=f"{tag}|icd{gran[0]}-opcs{gran[1]}-bnf{gran[2]}")
                    sid += 1
                else:
                    still.extend(kept)
            unreleased = still
            if PASSLOG:
                print(f"[pass] {tag:>7s}|icd{gran[0]}-opcs{gran[1]}-bnf{gran[2]}  "
                      f"in {nin:>9,} rows  map {t1-t0:6.1f}s  reduce {time.time()-t1:6.1f}s  "
                      f"released {len(buckets):>7,} buckets -> {len(states):,} states", flush=True)
        if not unreleased:
            break
    placed = sum(len(m) for m in states.values())
    # Three disjoint fates for every index-point row: released into exactly one state, collapsed as a
    # re-entry of a person already counted in that state, or never reaching k. They must add up --
    # `placed = len(rows) - len(unreleased)` alone ignores the collapsed ones, which this assertion
    # catches.
    assert placed + collapsed + len(unreleased) == len(rows), \
        f"row accounting: {placed}+{collapsed}+{len(unreleased)} != {len(rows)}"
    ppl = len({rows[r][0] for m in states.values() for r in m})
    print(f"[env] STATES: {len(states):,} | rows released {placed:,} | collapsed re-entries "
          f"{collapsed:,} | never reached k {len(unreleased):,} | distinct persons covered "
          f"{ppl:,}/{len(pids):,} ({100*ppl/len(pids):.1f}%)", flush=True)

    if a.augment:
        # `tag` here is the TIME_LEVELS loop variable, not the output tag -- name it explicitly,
        # or every augmentation file comes out called "flat_*"
        _augment(a, f"CohortWorld_{a.version}", rows, states, meta, HX, BASE, gran0,
                 code_ladder(gran0))

    # ---- per state: [H x N_disease] matrix ----
    out_states, priv = [], []
    nonest = homog = 0
    results = pool.imap_unordered(_matrix, sorted(states.items()), chunksize=8)
    for s_, nz, notest, cells, ne in results:
        nonest += ne
        mat = [[0.0] * len(DIS) for _ in HORIZONS]
        for hi, di, v in nz:
            mat[hi][di] = v
        m = meta[s_]
        persons = {rows[r][0] for r in states[s_]}
        # Homogeneity: a cell at 1.0 says EVERY at-risk member acquired the disease, so anyone who
        # can place a person in this state learns that person's outcome -- an attribute disclosure
        # k-anonymity does not defend against. Filling 0 would fabricate the opposite extreme, so the
        # whole state is dropped.
        if any(v >= 1.0 for r in mat for v in r):
            homog += 1
            continue
        body = render(m["ageband"], m["sex"], m["grid"], m["edges"])
        axes = collections.defaultdict(list)
        for (s2, c2, b2) in sorted(m["grid"]):
            axes[s2].append([c2, b2 if b2 != "" else None])
        # id from the AXES, not the rendered text: the text is a view, the axes are the identity.
        sid_str = _state_id(m)
        out_states.append({
            "state_id": sid_str,
            "axes": {"age_band": m["ageband"], "sex": m["sex"],
                     "icd": axes["I"], "opcs": axes["O"], "bnf": axes["B"]},
            "prompt_body": body, "coarseness": m["tag"],
            "n_band": n_band(len(persons)),
            "rates": mat,
            # Rates are dense and carry 0 where a cell could not be estimated. This says which of
            # those zeros mean "measured zero" and which mean "too few followed-up members to say":
            # horizon index -> "all" or the list of disease indices that are NOT estimable. Horizons
            # absent here are fully estimable. The horizon and disease axes themselves live once in
            # the manifest, not in all 36k states.
            "not_estimable": notest})
        assert len(persons) == len(states[s_]), "one row per person per state is violated"
        # Individual outcomes, alongside the aggregate. A patient-level task cannot be scored
        # against "this subject's true rate" -- no such number exists, they either got the disease or
        # did not -- so the verifiable target is the outcome itself, scored with a proper scoring
        # rule over many subjects, exactly as risk models are evaluated. Private, like everything
        # subject-level here.
        outc = []
        for r in states[s_]:
            pid, Y, age_Y, obs_Y, cut = rows[r]
            got = {}
            for c, t in DX[pid].items():
                k = c[:OUT_N] if OUT_N < 3 else c
                if not k or k[0] in FILT:
                    continue
                dt = t - Y
                if dt > 0 and (k not in got or dt < got[k]):
                    got[k] = round(dt, 3)
            outc.append({"pid": pid, "index_year": round(Y, 4),
                         "obs_years": round(obs_Y, 3), "first_dx_years": got})
        priv.append({"state_id": sid_str, "n_persons": len(persons), **cells,
                     "outcomes": outc})
    # The --multigran block below opens a FRESH pool per granularity, so this one is closed here;
    # submitting to a closed pool raises `ValueError: Pool not running`.
    pool.close(); pool.join()
    del pool
    print(f"[env] states dropped for a homogeneous (rate==1.0) cell: {homog:,}", flush=True)
    tot_cells = len(out_states) * len(HORIZONS) * len(DIS)
    print(f"[env] non-estimable cells (at-risk < {PC.MIN_N}) filled 0: {nonest:,} / {tot_cells:,} "
          f"({100*nonest/max(1,tot_cells):.2f}%)", flush=True)

    byaxes = {}
    for st in out_states:
        byaxes[(st["axes"]["age_band"], st["axes"]["sex"],
                tuple(map(tuple, st["axes"]["icd"])), tuple(map(tuple, st["axes"]["opcs"])),
                tuple(map(tuple, st["axes"]["bnf"])), st["coarseness"])] = st["state_id"]
    assert len(byaxes) == len(out_states), "state axes are not unique"
    assert len({st["state_id"] for st in out_states}) == len(out_states), "state_id collision"
    assert len({st["prompt_body"] for st in out_states}) == len(out_states), \
        "two states render to the same prompt"

    # Enumerating vocab x states is quadratic and needless: every legal `add` edge is the REVERSE of
    # a legal `del` edge, because add(e) applied to the smaller axis reproduces the larger one. So
    # only deletions are searched -- O(total axis elements) -- and each hit is recorded both ways.
    adj = collections.defaultdict(list)
    for st in out_states:
        base = (st["axes"]["age_band"], st["axes"]["sex"], tuple(map(tuple, st["axes"]["icd"])))
        cur = {ax: tuple(map(tuple, st["axes"][ax])) for ax in ("opcs", "bnf")}
        for ax in ("opcs", "bnf"):
            for e in cur[ax]:
                shrunk = tuple(sorted(set(cur[ax]) - {e}, key=_ekey))
                key = (base[0], base[1], base[2],
                       shrunk if ax == "opcs" else cur["opcs"],
                       shrunk if ax == "bnf" else cur["bnf"], st["coarseness"])
                tgt = byaxes.get(key)
                if tgt and tgt != st["state_id"]:
                    adj[st["state_id"]].append(
                        {"axis": ax, "op": "del", "code": e[0], "band": e[1], "to": tgt})
                    adj[tgt].append(
                        {"axis": ax, "op": "add", "code": e[0], "band": e[1], "to": st["state_id"]})
    trans = [{"state_id": k,
              "actions": sorted(v, key=lambda a: (a["axis"], a["op"], a["code"],
                                                  -1 if a["band"] is None else a["band"], a["to"]))}
             for k, v in sorted(adj.items())]
    na = sum(len(t["actions"]) for t in trans)
    print(f"[env] transitions: {na:,} legal actions over {len(trans):,}/{len(out_states):,} states "
          f"({100*len(trans)/max(1,len(out_states)):.1f}% have >=1)", flush=True)

    # ---- extra outcome granularities -------------------------------------------------------
    if a.multigran:
        # The NATIVE granularity is included on purpose. Excluding it removed the only cheap check
        # on this whole block: `rates_g{native}` must come out equal to `rates`, which the main pass
        # computed independently. Without that equality the next bug is invisible.
        want = [int(x) for x in a.multigran.split(",")]
        keep = {st["state_id"]: st for st in out_states}
        # `states` is keyed by the internal counter; map it to the emitted state_id once.
        by_sid = {}
        for s_ in states:
            m = meta[s_]
            by_sid.setdefault(_state_id(m), s_)
        todo = [(by_sid[k], states[by_sid[k]]) for k in keep if k in by_sid]
        for g in want:
            DIS_g = sorted({c[:g] for pid in pids for c in DX[pid] if c and c[0] not in FILT})
            _G["OUT_N"], _G["DIS"], _G["DIDX"] = g, DIS_g, {d: i for i, d in enumerate(DIS_g)}
            # A FRESH pool per granularity. Workers inherit `_G` by fork, so mutating it in the
            # parent leaves the existing children on their snapshot -- they would go on computing at
            # the NATIVE granularity. For C2 and default that raises IndexError (a 3-char index into
            # a 19-wide row); for C1 it raises nothing at all and silently writes the native 1-char
            # rates into `rates_g2` / `rates_g3`, zero-padded, with a healthy-looking log line.
            gpool = mp.get_context("fork").Pool(nw)
            got = 0
            for s_, nz, _notest, _cells, _ne in gpool.imap_unordered(_matrix, sorted(todo),
                                                                     chunksize=8):
                mat = [[0.0] * len(DIS_g) for _ in HORIZONS]
                for hi, di, v in nz:
                    mat[hi][di] = v
                sid_str = _state_id(meta[s_])
                if sid_str in keep:
                    keep[sid_str][f"rates_g{g}"] = mat
                    got += 1
            gpool.close(); gpool.join()
            json.dump({"granularity": g, "diseases": DIS_g},
                      # NOT f"{tag}_..." -- `tag` is the TIME_LEVELS loop variable, so every file
                      # would come out named `flat_diseases_g1.json` (the trap the comment at the
                      # `_augment` call warns about).
                      open(os.path.join(a.out,
                                        f"CohortWorld_{a.version}_diseases_g{g}.json"), "w"))
            same = "n/a"
            if g == OUT_N:
                # Self-check: the native pass here must reproduce the main pass exactly.
                bad = sum(1 for st in out_states if st.get(f"rates_g{g}") != st["rates"])
                same = f"native check: {len(out_states)-bad}/{len(out_states)} match"
                assert bad == 0, f"multigran native pass disagrees with the main pass on {bad} states"
            print(f"[env] granularity {g}-char: {len(DIS_g):,} codes, matrices on {got:,} states"
                  f"  {same}", flush=True)
        _G["OUT_N"], _G["DIS"], _G["DIDX"] = OUT_N, DIS, DIDX

    # ---- profile-disjoint split: the state IS the profile, so hash it ----
    def bucket(sid_str):
        h = int(hashlib.md5(f"{a.seed}:{sid_str}".encode()).hexdigest(), 16) % 100
        return "train" if h < tr else ("val" if h < tr + va else "test")
    tag = f"CohortWorld_{a.version}"
    SP = collections.defaultdict(list)
    for st in sorted(out_states, key=lambda x: x["state_id"]):
        SP[bucket(st["state_id"])].append(st)
    for sp in ("train", "val", "test"):
        with open(os.path.join(a.out, f"{tag}_{sp}_states.jsonl"), "w") as f:
            for st in SP[sp]:
                f.write(json.dumps(st) + "\n")
    # The horizon and disease axes are identical for every state, so they live here once instead of
    # being repeated in each of tens of thousands of state records (~283 MB of duplication for
    # `default`). Names come along so a state can be serialised into a question without a second
    # lookup table.
    json.dump({"version": a.version, "horizons": HORIZONS,
               "diseases": [{"code": d,
                             "name": PC.icd3_name(d) if len(d) == 3
                             else ICD_CH.get(d[0], "other") + f" [{d}]"} for d in DIS],
               "rates_shape": [len(HORIZONS), len(DIS)],
               "rates_meaning": "cumulative incidence of a FIRST diagnosis within the horizon, "
                                "among state members not already diagnosed; death is a competing "
                                "risk (dead members stay in the denominator as non-events)"},
              open(os.path.join(a.out, f"{tag}_manifest.json"), "w"), indent=1)

    with open(os.path.join(a.out, f"{tag}_transitions.jsonl"), "w") as f:
        for t in trans:
            f.write(json.dumps(t) + "\n")
    with open(os.path.join(a.private, f"{tag}_private_counts.jsonl"), "w") as f:
        for p in sorted(priv, key=lambda x: x["state_id"]):
            f.write(json.dumps(p) + "\n")

    stats = {"version": a.version, "gp_linked_only": True, "history": "all-dates (not 2010-capped)",
             "hx_granularity": dict(zip(("icd", "opcs", "bnf"), gran0)), "outcome_icd_chars": OUT_N,
             "index_years_used_in_construction": sorted(YEARS), "index_year_in_state": False,
             "horizons": HORIZONS, "diseases": len(DIS), "subjects": len(pids),
             "index_mode": a.index_mode, "index_points": len(rows),
             "rows_released": placed, "rows_collapsed_reentry": collapsed,
             "rows_unreleased": len(unreleased), "persons_covered": ppl,
             "states": len(out_states), "states_by_split": {s: len(SP[s]) for s in SP},
             "split": "profile-disjoint (hash of the state's own rendered axes)",
             "seed": a.seed, "ratios": a.ratios,
             "k_min_persons": PC.MIN_N, "min_events_rule": "dropped (disclosure profile B)",
             "published_n": "banded persons", "rate": "raw, not coarsened",
             "non_estimable_cells_filled_zero": nonest,
             "states_dropped_homogeneous": homog,
             "legal_actions": na, "states_with_actions": len(trans)}
    json.dump(stats, open(os.path.join(a.out, f"{tag}_env_stats.json"), "w"), indent=1)
    print(f"\n[env] {tag}: states {len(out_states):,} "
          f"(train/val/test {len(SP['train']):,}/{len(SP['val']):,}/{len(SP['test']):,}) | "
          f"diseases {len(DIS)} | actions {na:,}")


if __name__ == "__main__":
    main()
