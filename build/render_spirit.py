#!/usr/bin/env python3
# release-gate: allow-file -- ran against the private UKB extract; shipped for provenance, paths shown as they were
"""The PUBLISHABLE item generator for T1, T2 and T3.

The earlier item builder rendered one REAL member of the target cohort into every T1 and T3 item
(`render_patient` reads the private history tables). 82.8% of the 3,112 items it produced are unique
on (codes, age, sex) inside the item set alone -- a lower bound on uniqueness across UKB's ~500,000
people -- so those items cannot be published and neither can any score computed on them.
This file renders items under three principles (the rendering "spirit"): (1) an item carries only
its target cohort's own released description, nothing read from a private table; (2) the only
rewrite is de-grouping the age band to `about xx age`; (3) difficulty comes only from dials that
remove or coarsen, so no item is more precise than its source.

PRINCIPLE 1 IS ENFORCED STRUCTURALLY, NOT BY AUDIT. The user's instruction was explicit: the
publishable code path must not be able to reach the private history tables, and no post-hoc
disclosure review is wanted, because a construction that is sound always passes one and passing
tells you nothing. Two mechanisms, in this order:

  (a) A `sys.addaudithook` installed as the FIRST statement of `main()`, before any project import,
      that RAISES on any `open` of a private data path (`cohortworld_private/augment/*`,
      `*_private_counts.jsonl`, the `/ram` source pickles, and any `*_subject.jsonl` this build's
      private sibling writes). It is not "reads them but does not emit": the process cannot open
      them. `--selfcheck` fires the hook on purpose, so the guard is shown able to find something
      before its silence is read as evidence.
  (b) An import-graph assertion after the build: no module loaded into `sys.modules` names a
      private table, with ONE declared exception -- `cohortworld_env.lattice`, whose single
      reference is the lazily-built `Lattice._priv()`; it is never called here, and (a) would raise
      if it were.

The private `subject-unseen` split is a SEPARATE entry point (a private builder, not released), which
imports this module for the schema and does open those tables. It is the spirit's sole exception and
it writes only under the private root.

Run:
  .venv/bin/python build/render_spirit.py --selfcheck
  .venv/bin/python build/render_spirit.py --out <rundir>/artifacts/<name>.txt --emit <itemdir>
"""
from __future__ import annotations
import argparse, collections, glob, hashlib, json, os, random, re, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PANEL = os.environ.get("CW_PANEL", f"{REPO}/official")   # as cohortworld_env.lattice resolves it

# --------------------------------------------------------------------------- (a) the audit hook

PRIVATE_PATTERNS = (
    "cohortworld_private",               # per-subject member lists, ancestors, frontiers
    "_private_counts.jsonl",             # per-state exact event/at-risk counts
    "cohortworld_hx_alldates",           # the source history pickles, matched by file name rather
    "panel_intermediate",                #   than by a path that would also name a machine
    "_subject.jsonl",                    # whatever the private sibling entry point writes
)
"""The private history tables, by name. This is a DENYLIST and that is a deliberate ceiling: an
allowlist would have to enumerate the interpreter's own file access (every `.py`, every `.pyc`,
`site-packages`, `/proc`), and a guard nobody can read is a guard nobody maintains.

Matching on the store and file names rather than on absolute roots is what lets this file ship: a
root is a property of the machine that held the extract, and naming it here would publish that. It
also widens the guard, since the tables are caught wherever they are mounted.
# ponytail: denylist over the named private sources; widen it if a sixth is ever created."""


class PrivateAccess(RuntimeError):
    """A publishable build tried to open a private history table."""


def install_private_guard():
    """Make the private tables unreachable from this process. Cannot be uninstalled -- that is why
    it goes in first, before any project import."""
    def hook(event, args):
        if event in ("open", "os.open") and args and isinstance(args[0], (str, bytes)):
            p = args[0].decode() if isinstance(args[0], bytes) else args[0]
            for pat in PRIVATE_PATTERNS:
                if pat in p:
                    raise PrivateAccess(
                        f"the publishable item path tried to open {p!r}, which matches the private "
                        f"pattern {pat!r}. Principle 1: no private table is readable here.")
    sys.addaudithook(hook)


DECLARED_LAZY_EXCEPTION = "cohortworld_env.lattice"
"""The one module allowed to NAME a private table: `Lattice._priv()` builds a per-state count cache
on first call, for build/score paths that are not this one. This build never calls it, and the hook
above would raise if it did."""


def assert_import_graph_clean():
    """(b): nothing loaded reaches a private table, bar the one declared lazy exception."""
    bad = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f or not f.endswith(".py") or name == DECLARED_LAZY_EXCEPTION:
            continue
        if not (f.startswith(REPO) or f.startswith(HERE)):
            continue                                   # stdlib / site-packages
        if os.path.abspath(f) == os.path.abspath(__file__):
            continue                                   # this file names them to forbid them
        try:
            src = open(f).read()
        except OSError:
            continue
        for pat in ("cohortworld_private/augment", "hx_alldates", "panel_intermediate",
                    "private_counts"):
            if pat in src:
                bad.append((name, pat))
                break
    return bad


# --------------------------------------------------------------------------- constants

VERSIONS = ("C1", "C2", "default")
GRAN = {"C1": 1, "C2": 2, "default": 3}
AXES = ("age_band", "sex", "icd", "opcs", "bnf")
CODE_AXES = ("icd", "opcs", "bnf")
MASK_POOL = ("icd", "opcs")          # bnf is the failure boundary; age/sex leak via entry
HORIZ = [1, 2, 3, 5, 10]
MIN_K = 10                           # gen/panel_common.MIN_N, restated as the item-side floor
SEED = 42
RATIO = (90, 5)                      # T1/T3 train/seen/unseen 90/5/5
T2_RATIO = (50, 25)                  # T2 KEEPS 50/25/25 -- its supply cannot survive a 5% band.
"""Two ratios, and the reason there are two.

`split_of` hashes the state id, and T1/T3 emit EXACTLY ONE item per released cohort (measured:
18,378 items over 18,378 distinct cohorts, 1.000 per cohort), so for those two tasks the item
ratio IS the cohort ratio and `RATIO` realises itself to within hash noise. T2 does not: its 465
items come from 169 from-cohorts, 2.75 per cohort and up to 42, and it is direction-balanced
WITHIN each (size, horizon, granularity) cell PER SPLIT -- so shrinking a split's band of cohorts
costs items superlinearly, because a cell with no counter-example emits nothing at all.

Measured on the frozen pool (`gen/t2_compose.balanced`, non-age strata, bal key size+horizon+
granularity), T2's whole direction-balanced separable supply by ratio:

    ratio        train  seen  unseen   unseen by stratum
    50/25/25       198    40      72   {bnf add: 2, sex switch: 70}
    80/10/10       286    28      38   {bnf add: 2, sex switch: 36}
    85/7/8         292    64       2   {bnf add: 2}
    90/5/5         328    36       2   {bnf add: 2}

At 90/5/5 the FULL T2 build (separable + `indistinguishable` controls) is train 492 / seen 54 /
unseen **3** -- one item per class on a three-class task, 20x below the n>=60 bar. `sex switch` is
97% of T2's unseen supply at any ratio and its from-cohorts simply do not hash into a 5% band.

The supply cannot be grown to rescue 90/5/5. Two measurements settle that:
  * `t2_compose` §0/§1: the `1,052/58/68` column a rebuild was sized off is
    `bal any-gran`, which counts the same edges' cells at FOREIGN granularities. Version <-> outcome
    granularity is a bijection and `DeltaTaskset.load` tests exact membership, so the server refuses
    them. The clause-4 repair measured 316/32/38 -- exactly what was already there. It buys emission
    efficiency (every emitted item is servable), not supply.
  * `t2_compose` §2: the one stratum with large supply, `age_band shift` (2,198 pooled balanced cells), was
    DROPPED on a scored ladder -- reading beats a free age rule there by +0.0056 [+0.0000, +0.0167]
    -- and re-adding it requires de-grouping the age branch of `internal._t2_clause`, which is a
    FROZEN module, so it changes the question every earlier score answered.

THE TRADE, stated: the docstring of `split_of` below promised a cohort lands in the same split on
T1, T2 and T3. Two ratios break that. What it costs, measured on the frozen T2 items with T1/T3 at
90/5/5: 52/108 T2-unseen and 60/60 T2-seen FROM-cohorts fall in T1/T3's train band. It is not the
clean property it reads as, because it is never fully held: even at one shared 50/25/25,
41/108 (38.0%) of T2-unseen items have their **to** endpoint in T1/T3-train -- `split_of` only ever
constrained the from endpoint. And the exposure is not exploitable: a T2 prompt STATES NO COHORT
(`DeltaTaskset.load` renders the operation clause, disease and horizon and
nothing else), so a memorised (description -> rate) association from T1/T3 training has no trigger
in a T2 prompt; the agent must `cw_look` first, and then it has read.
"""
ZERO_FRACTION = 0.5                  # build_items.py's stratified zero/non-zero cell draw, kept
POP_ZERO_SHARE = 0.727

# The benchmark's deliberate no-match share, and the population no-match share measured by census.
NOMATCH_SHARE_BENCH = 0.38
POP_NOMATCH_SHARE = {"C1": 0.8825, "C2": 0.9066, "default": 0.9124}

# `zoom` = coarsen bnf to level 2; `onset` = drop the `(first recorded ...)` clause.
RUNGS = {
    "E0": {"share": 0.10, "n_mask": 0, "zoom": False, "onset": True},
    "E1": {"share": 0.15, "n_mask": 0, "zoom": False, "onset": False},
    "E2": {"share": 0.15, "n_mask": 0, "zoom": True,  "onset": False},
    "M1": {"share": 0.25, "n_mask": 1, "zoom": False, "onset": False},
    "H1": {"share": 0.20, "n_mask": 1, "zoom": True,  "onset": False},
    "H2": {"share": 0.15, "n_mask": 2, "zoom": False, "onset": False},
}
CAND_CEILING = 20                    # redraw once above this, then drop

AGE_RE = re.compile(r"aged (\d+)-(\d+)")
ONSET_RE = re.compile(r" \(first recorded [^)]*\)")
PAIR_RE = re.compile(r"([^;:]+?) \[([A-Za-z0-9]+)\]")
LINE_OF = {"icd": "- diagnoses: ", "opcs": "- procedures: ", "bnf": "- prescriptions: "}


# --------------------------------------------------------------------------- panel

def load_states(version: str) -> dict:
    """{state_id -> state}, from the released panel only."""
    out = {}
    for path in sorted(glob.glob(f"{PANEL}/CohortWorld_{version}_*_states.jsonl")):
        for ln in open(path):
            d = json.loads(ln)
            out[d["state_id"]] = d
    return out


def code_levels(s: dict) -> dict:
    """The per-axis code level this cohort is natively released at, off its `coarseness` tag."""
    out = {}
    for piece in s["coarseness"].split("|")[1].split("-"):
        for a in CODE_AXES:
            if piece.startswith(a):
                out[a] = int(piece[len(a):])
    return out


def band_lo(b: str) -> int:
    return int(str(b).split("-")[0].rstrip("+") or 0)


def namemap(states: dict) -> dict:
    """{(axis, code) -> label}, harvested from the RELEASE's own prompt bodies.

    The one thing bnf coarsening needs that string surgery cannot supply: the label of the shorter
    code. It is not composed -- it is copied from another released cohort that states that code.
    Verified over all three panels: 0 codes with two labels, 0 coarse codes missing a label.
    """
    out = {}
    for s in states.values():
        for line in s["prompt_body"].split("\n")[1:]:
            head, _, rest = line.partition(": ")
            ax = {"- diagnoses": "icd", "- procedures": "opcs",
                  "- prescriptions": "bnf"}.get(head)
            if ax is None:
                continue
            for nm, code in PAIR_RE.findall(rest):
                k = (ax, code)
                nm = nm.strip()
                assert out.get(k, nm) == nm, f"{k} labelled both {out[k]!r} and {nm!r}"
                out[k] = nm
    return out


# --------------------------------------------------------------------------- the two dials

def degroup(body: str) -> str:
    """Principle 2, the ONLY transformation: `aged 65-69` -> `about 67 age`. One regex.

    Bijective on all three panels (measured: 0 of 20,180 cohorts collapse), and vaguer than the
    band, which is the direction principle 2 requires.
    """
    return AGE_RE.sub(lambda m: f"about {(int(m.group(1)) + int(m.group(2))) // 2} age", body)


def render(s: dict, mask=(), bnf_level=None, onset=True, names=None) -> str:
    """The item's description: string surgery on the release's own `prompt_body`, never a re-render.

    Dial A (`mask`) deletes a whole axis line, leaving NO trace -- so a masked axis is
    indistinguishable from one the agent has not read yet, which is what makes `add`/`switch` its
    inverse. An EMPTY axis still says `none recorded`, because that is the release's own wording.
    Dial B coarsens `bnf` to `bnf_level` and/or drops the onset clause. Both only ever remove.
    """
    lines = s["prompt_body"].split("\n")
    head, rest = lines[0], lines[1:]
    out = [head]
    lv = code_levels(s)
    for ln in rest:
        ax = {"- diagnoses": "icd", "- procedures": "opcs",
              "- prescriptions": "bnf"}.get(ln.partition(": ")[0])
        assert ax is not None, f"unexpected axis line {ln!r}"
        if ax in mask:
            continue                                   # dial A: the line is gone, silently
        if ax == "bnf" and bnf_level is not None and bnf_level < lv["bnf"]:
            ln = _coarsen(s, bnf_level, names)         # dial B, code half
        out.append(ln)
    body = "\n".join(out)
    if not onset:
        body = ONSET_RE.sub("", body)                  # dial B, onset half
    return degroup(body)


def _coarsen(s: dict, level: int, names: dict) -> str:
    """The `- prescriptions:` line restated at a coarser BNF level, in the release's own words.

    Codes are truncated (a prefix the panel itself releases); several may merge onto one, and a
    merged entry keeps the EARLIEST onset band, which is `OK__build_cohortworld_env.render`'s own
    rule for a generalised code. Labels come from `namemap`, i.e. from other released cohorts.

    The entry's code is taken to be the LAST bracketed code in the part that is actually one of this
    cohort's axis codes, and its trailing text is the onset clause MATCHED, not the remainder of the
    string. Slicing the remainder is wrong on a real and common shape: the panel renders some BNF
    entries as `skin [1304] [1304]` (its `sysname` already carries the code, and `render` appends it
    again), so the remainder carries a second copy of the 4-character code into a `bnf2` item --
    measured, 901 of 4,563 unseen T3 items then state a code FINER than their own declared zoom
    level, which is principle 3 exactly. `--selfcheck` checks every bracketed code in the coarsened
    line, not just the first regex match.
    """
    axis = dict(s["axes"]["bnf"])
    best = {}
    line = s["prompt_body"].split("\n")[3]
    for part in line.partition(": ")[2].split("; "):
        codes = [c for _n, c in PAIR_RE.findall(part) if c in axis]
        if not codes:
            continue
        c = codes[-1]
        m = ONSET_RE.search(part)
        band, k = axis[c], c[:level]
        if k not in best or (band is not None and (best[k][0] is None or band < best[k][0])):
            best[k] = (band, m.group(0) if m else "")
    parts = [f"{names[('bnf', k)]} [{k}]{best[k][1]}" for k in sorted(best)]
    return LINE_OF["bnf"] + ("; ".join(parts) if parts else "none recorded")


def dkey(s: dict, mask=(), zoom=None, onset=True):
    """What a cohort STATES, under one item's dial settings. The candidate relation.

    Truncation is a no-op on codes that are already shorter, so a cohort natively COARSER than the
    item's stated level gets a different key and is NOT a candidate -- it does not state what the
    item states. `gen/spirit_dials.axkey` with the same semantics; `--selfcheck` asserts agreement.
    """
    k = []
    for a in AXES:
        if a in mask:
            continue
        if a in ("age_band", "sex"):
            k.append(s["axes"][a])
        else:
            lv = (zoom or {}).get(a)
            v = s["axes"][a]
            k.append(tuple(sorted({((c[:lv] if lv else c), b) if onset else (c[:lv] if lv else c)
                                   for c, b in v})))
    return tuple(k)


class Candidates:
    """Counts of released cohorts consistent with a description, cached per dial setting."""

    def __init__(self, states: dict):
        self.states = list(states.values())
        self._idx = {}

    def count(self, s, mask, zoom, onset) -> int:
        key = (tuple(sorted(mask)), tuple(sorted((zoom or {}).items())), onset)
        ix = self._idx.get(key)
        if ix is None:
            ix = self._idx[key] = collections.Counter(
                dkey(t, mask, zoom, onset) for t in self.states)
        return ix[dkey(s, mask, zoom, onset)]


# --------------------------------------------------------------------------- splits

def split_of(sid: str, ratio=RATIO) -> str:
    """`gen/t2_compose.split_of` verbatim, so a cohort lands in the SAME split on T1, T2 and T3.

    Profile-disjoint by construction: `state_id` is a hash of the axes, so the disjointness unit is
    the profile. Subject disjointness is a separate property and is measured, not assumed -- see
    the private subject builder (not released), which is where a subject exists at all.
    """
    h = int(hashlib.md5(f"cwsplit2:{sid}".encode()).hexdigest(), 16) % 100
    a, b = ratio
    return "train" if h < a else ("seen" if h < a + b else "unseen")


# --------------------------------------------------------------------------- the one schema

SCHEMA_KEYS = ("item_id", "task", "split", "version", "description", "patient", "ask", "operation",
               "dials", "source", "answer", "labels")
FORBIDDEN_KEYS = ("subject", "eid", "member", "members", "index_year", "at_risk", "n", "pid")
LEGACY_KEYS = ("age_band", "sex", "disease", "horizon", "answer_rate", "target_cohort", "n_band",
               "granularity", "zero_stratum", "pop_zero_share", "hops_from_entry", "axis", "op",
               "code", "cross_version", "answer_direction", "answer_delta", "_audit", "cost")


def mkitem(*, item_id, task, split, version, description, ask, operation, dials, source, answer,
           labels, legacy) -> dict:
    """ONE item shape, every task, every split.

    Every canonical key is present on every item; a key that does not apply to a task is `None`,
    never absent, so a reader that does `d["operation"]` gets `None` rather than a KeyError that
    differs by task, and a diff between two splits shows values and never shapes.

    `patient` is a byte-identical alias of `description` and the flat `legacy` block is spliced into
    the object ROOT, because the frozen tasksets read flat top-level keys
    (`internal.py:512-636`) and editing them is what this build may not do. Both views come from
    one value in one place; `--selfcheck` asserts they agree on every item.
    """
    d = {"item_id": item_id, "task": task, "split": split, "version": version,
         "description": description, "patient": description,
         "ask": ask, "operation": operation, "dials": dials, "source": source,
         "answer": answer, "labels": labels}
    assert tuple(d) == SCHEMA_KEYS, tuple(d)
    full = {**d, **{k: legacy.get(k) for k in LEGACY_KEYS}}
    for k in FORBIDDEN_KEYS:
        assert k not in full, f"{k!r} is a forbidden key on a published item"
    return full


def check_k(n_band: str, where: str):
    """Requirement 4: fail loudly rather than emit an item whose label is below k=10.

    A cohort below `panel_common.MIN_N` is not in the panel at all, so this can only fire if the
    panel itself is wrong -- which is the point: it is a floor on the ITEM, checked where the item
    is written, not an inherited assumption.
    """
    lo = band_lo(n_band)
    if lo < MIN_K:
        raise ValueError(f"{where}: n_band {n_band!r} has k={lo} < {MIN_K}")


# --------------------------------------------------------------------------- T1 / T3

def hops_index(version: str, states: dict, entry_of) -> dict:
    """{state_id -> hops from its own entry cohort}, by ONE BFS per entry rather than per item.

    A fresh 3-hop BFS per item drops whatever it does not reach inside the cap (measured: 14%
    unreachable, a build defect). One full BFS per entry cohort is cheaper AND complete, so
    `hops_from_entry` is a measurement rather than a filter.
    """
    adj = collections.defaultdict(set)
    for ln in open(f"{PANEL}/CohortWorld_pooled_edges.jsonl"):
        e = json.loads(ln)
        adj[e["from"]].add(e["to"])
        adj[e["to"]].add(e["from"])
    entry = {sid: entry_of(version, s["axes"]["age_band"], s["axes"]["sex"])
             for sid, s in states.items()}
    reach = {}
    for home in set(entry.values()) - {None}:
        dist, front, d = {home: 0}, [home], 0
        while front:
            d += 1
            nxt = []
            for u in front:
                for v in adj.get(u, ()):
                    if v not in dist:
                        dist[v] = d
                        nxt.append(v)
            front = nxt
        reach[home] = dist
    # Each target is measured from ITS OWN entry -- the cohort `toolset.setup_task` will actually
    # place that item's agent on -- not from the nearest of all entries.
    return {f"{version}:{sid}": reach.get(e, {}).get(f"{version}:{sid}", -1)
            for sid, e in entry.items()}


def _cells(s: dict, g: int, rng) -> tuple | None:
    """One (horizon, disease) cell, drawn from the ESTIMABLE cells, stratified zero / non-zero.

    `not_estimable` is the panel's own k>=10 certificate for the cell: `OK__build_cohortworld_env`
    writes a disease index into it exactly when its at-risk denominator is below `MIN_N`. So
    honouring it IS requirement 4 at cell level, and `--selfcheck` checks it against the released
    at-risk band store with a positive and a negative control.
    """
    ne = s.get("not_estimable") or {}
    mat = s.get(f"rates_g{g}") or s["rates"]
    cand = []
    for hi in range(len(mat)):
        m = ne.get(str(hi), ne.get(hi))
        if m == "all":
            continue
        bad = set(m or ())
        cand += [(hi, di, mat[hi][di]) for di in range(len(mat[hi])) if di not in bad]
    if not cand:
        return None
    nz = [c for c in cand if c[2] > 0]
    zc = [c for c in cand if c[2] == 0]
    want_zero = rng.random() < ZERO_FRACTION
    pick = (zc if want_zero else nz) or (nz if want_zero else zc)
    return rng.choice(pick) if pick else None


def draw_rung(version: str, s: dict, rng) -> str:
    """A rung from the schedule, restricted to the rungs this cohort can actually be rendered at.

    C1 admits no code zoom at all (`icd2`/`opcs2` are released by no version, and C1
    carries one level per axis), and a cohort already at `bnf2` has no coarser bnf rung. The version
    balance is not defaulted silently; the choice taken is --
    C1 is mask-only, the rung is written on every item, and any headline can be re-cut by rung.
    """
    zoomable = code_levels(s)["bnf"] > 2
    ok = [k for k, v in RUNGS.items() if zoomable or not v["zoom"]]
    w = [RUNGS[k]["share"] for k in ok]
    return rng.choices(ok, weights=w)[0]


def build_t1_t3(version, states, cand, names, hops, dis, rng, out, log):
    """One T1 and one T3 item per eligible released cohort. Neither opens a member file."""
    g = GRAN[version]
    drops = collections.Counter()
    for sid in sorted(states):
        s = states[sid]
        full = f"{version}:{sid}"
        n_codes = sum(len(s["axes"][a]) for a in CODE_AXES)
        if n_codes == 0:
            drops["entry cohort (no codes): nothing to locate"] += 1
            continue
        check_k(s["n_band"], full)
        lv = code_levels(s)
        sp = split_of(full)
        h = hops.get(full, -1)

        # ---- T1: no dials at all. Acceptance condition: candidates == 1.
        native = dict(lv)
        c1 = cand.count(s, (), native, True)
        if h < 0:
            drops["(label only) target not reachable from its own entry cohort"] += 1
        if c1 != 1:
            drops["T1 description names >1 released cohort"] += 1
        else:
            desc = render(s, (), None, True, names)
            out["t1_locate"][sp].append(mkitem(
                item_id=f"T1:{version}:{sid}:E0:na", task="T1", split=sp, version=version,
                description=desc, ask=None, operation=None,
                dials={"rung": "E0", "mask": [], "zoom": native, "onset": "stated",
                       "candidates": 1},
                source={"kind": "lattice", "cohort": full, "start_cohort": None,
                        "n_band": s["n_band"], "release": f"official/CohortWorld_{version}_*_states.jsonl"},
                answer={"kind": "position", "rate": None, "cohort": full, "direction": None,
                        "covered": True},
                labels={"zero_stratum": None, "pop_zero_share": POP_ZERO_SHARE,
                        "hops_from_entry": h, "coverage_c": None, "gp_linked": None,
                        "nomatch": False, "separable": None, "regime": "match",
                        "pop_nomatch_share": POP_NOMATCH_SHARE[version], "k": band_lo(s["n_band"])},
                legacy={"age_band": s["axes"]["age_band"], "sex": s["axes"]["sex"],
                        "target_cohort": full, "n_band": s["n_band"], "granularity": g,
                        "pop_zero_share": POP_ZERO_SHARE, "hops_from_entry": h}))

        # ---- T3: the dials, in full. This is the only task that carries them.
        cell = _cells(s, g, rng)
        if cell is None:
            drops["T3 no estimable (horizon, disease) cell"] += 1
            continue
        hi, di, rate = cell
        disease = dis[g][di]
        rung = draw_rung(version, s, rng)
        R = RUNGS[rung]
        zoom = dict(lv)
        if R["zoom"]:
            zoom["bnf"] = 2
        got = None
        for attempt in range(2):                      # redraw ONCE, then drop
            mask = tuple(sorted(rng.sample(MASK_POOL, R["n_mask"]))) if R["n_mask"] else ()
            n = cand.count(s, mask, zoom, R["onset"])
            assert n >= 1, "a cohort is always consistent with its own degraded rendering"
            if n < CAND_CEILING:
                got = (mask, n)
                break
        if got is None:
            drops[f"T3 candidates >= {CAND_CEILING} twice"] += 1
            continue
        mask, ncand = got
        desc = render(s, mask, zoom["bnf"], R["onset"], names)
        out["t3_rate"][sp].append(mkitem(
            item_id=f"T3:{version}:{sid}:{rung}:{HORIZ[hi]}", task="T3", split=sp, version=version,
            description=desc, ask={"disease": disease, "horizon": HORIZ[hi], "granularity": g},
            operation=None,
            dials={"rung": rung, "mask": list(mask), "zoom": zoom,
                   "onset": "stated" if R["onset"] else "dropped", "candidates": ncand},
            source={"kind": "lattice", "cohort": full, "start_cohort": None,
                    "n_band": s["n_band"], "release": f"official/CohortWorld_{version}_*_states.jsonl"},
            answer={"kind": "rate", "rate": round(float(rate), 6), "cohort": None,
                    "direction": None, "covered": True},
            labels={"zero_stratum": bool(rate == 0), "pop_zero_share": POP_ZERO_SHARE,
                    "hops_from_entry": h, "coverage_c": None,
                    "gp_linked": None, "nomatch": False, "separable": None, "regime": "match",
                    "pop_nomatch_share": POP_NOMATCH_SHARE[version], "k": band_lo(s["n_band"])},
            legacy={"age_band": s["axes"]["age_band"], "sex": s["axes"]["sex"], "disease": disease,
                    "horizon": HORIZ[hi], "answer_rate": round(float(rate), 6),
                    "target_cohort": full, "n_band": s["n_band"], "granularity": g,
                    "zero_stratum": bool(rate == 0), "pop_zero_share": POP_ZERO_SHARE,
                    "hops_from_entry": h}))
    for k, v in sorted(drops.items()):
        log(f"    {version:>7s}  dropped {v:>6,}  {k}")
    return drops


# --------------------------------------------------------------------------- T2

def build_t2(states_all, names, out, log):
    """T2, re-wrapped into the one schema. The COMPOSITION is t2_compose's and is not re-derived.

    `build/t2_compose.py` settles the T2 item set (age stratum dropped, ratio 50/25/25,
    direction-balanced within (size, horizon, granularity), one `indistinguishable` control per
    separable pair drawn from the same (axis, op)). Re-deriving it here would be a second
    implementation of a scored decision, so this calls it and only changes the shape of the record.

    Dials do not apply to T2: `setup_task` places the agent ON the from cohort, so
    `cw_look` restores anything the item redacts before the agent has acted. The `description` is
    therefore the from cohort's own released body, de-grouped, at native levels, onset stated.
    """
    sys.path[:0] = [HERE, REPO]
    import t2_compose as T2C
    pool, _n_rows, _n_ok = T2C.build_pool(clause4=True)
    strata = [k for k in sorted(pool) if k != T2C.AGE]
    built = T2C.build(pool, strata, ratio=T2_RATIO, bal_key=T2C.BAL_KEY, ind_match=T2C.IND_MATCH,
                      seed=SEED)
    for sp, rows in built.items():
        for r in rows:
            a = r["_audit"]
            v = a["from"].split(":", 1)[0]
            s = states_all[v][a["from"].split(":", 1)[1]]
            check_k(s["n_band"], a["from"])
            check_k(a["n_band_to"], a["to"])
            assert split_of(a["from"], T2_RATIO) == sp, "t2 split disagrees with split_of"
            lv = code_levels(s)
            out["t2_delta"][sp].append(mkitem(
                item_id=f"T2:{v}:{a['from'].split(':', 1)[1]}:{r['axis']}-{r['op']}:{r['horizon']}",
                task="T2", split=sp, version=v,
                description=render(s, (), None, True, names),
                ask={"disease": r["disease"], "horizon": r["horizon"],
                     "granularity": r["granularity"]},
                operation={"axis": r["axis"], "op": r["op"], "code": r["code"], "cost": r["cost"]},
                dials={"rung": "E0", "mask": [], "zoom": lv, "onset": "stated", "candidates": None},
                source={"kind": "lattice", "cohort": None, "start_cohort": a["from"],
                        "n_band": s["n_band"],
                        "release": f"official/CohortWorld_{v}_*_states.jsonl"},
                answer={"kind": "direction", "rate": None, "cohort": None,
                        "direction": r["answer_direction"], "covered": True},
                labels={"zero_stratum": None, "pop_zero_share": POP_ZERO_SHARE,
                        "hops_from_entry": 0, "coverage_c": None, "gp_linked": None,
                        "nomatch": False, "separable": a["separable"], "regime": "match",
                        "pop_nomatch_share": POP_NOMATCH_SHARE[v], "k": band_lo(s["n_band"])},
                legacy={"axis": r["axis"], "op": r["op"], "code": r["code"],
                        "granularity": r["granularity"], "disease": r["disease"],
                        "horizon": r["horizon"], "cost": r["cost"], "cross_version": False,
                        "answer_direction": r["answer_direction"],
                        "answer_delta": r["answer_delta"], "_audit": a,
                        "age_band": s["axes"]["age_band"], "sex": s["axes"]["sex"],
                        "n_band": s["n_band"], "pop_zero_share": POP_ZERO_SHARE}))
    log(f"    T2 composition taken from build/t2_compose.py, strata {len(strata)}, "
        f"ratio {T2_RATIO[0]}/{T2_RATIO[1]}/{100 - sum(T2_RATIO)} -- T2's OWN ratio, see T2_RATIO")


# --------------------------------------------------------------------------- selfcheck

def selfcheck():
    """Every non-trivial claim this file makes, with a control that shows the check can fail."""
    # 1. the private guard fires -- shown able to find something before its silence means anything
    import subprocess
    r = subprocess.run([sys.executable, "-c",
                        f"import sys; sys.path.insert(0, {HERE!r});\n"
                        "import render_spirit as RS; RS.install_private_guard()\n"
                        "open('cohortworld_private/augment/CohortWorld_C2_members.jsonl')"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "PrivateAccess" in r.stderr, r.stderr[-400:]
    # ...and it does NOT fire on the released panel, or the build could not run at all
    r2 = subprocess.run([sys.executable, "-c",
                         f"import sys; sys.path.insert(0, {HERE!r});\n"
                         "import render_spirit as RS; RS.install_private_guard()\n"
                         f"open({PANEL!r} + '/CohortWorld_C2_manifest.json').read()"],
                        capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr[-400:]
    print("  guard: raises on the private member file, passes the released panel")

    S = load_states("C2")
    N = namemap(S)
    ss = [S[k] for k in sorted(S)[:400]]
    dbl = [x for x in S.values() if "] [" in x["prompt_body"].split("\n")[3]
           and code_levels(x)["bnf"] > 2]
    assert dbl, "the panel no longer carries the doubled-code shape; drop this control"
    ss += dbl[:200]

    # 2. de-grouping is one substitution and it only ever loses precision
    b = "Consider UK adults aged 60-64 (male; x) with the following recorded history:"
    assert degroup(b).startswith("Consider UK adults about 62 age (male;"), degroup(b)
    assert degroup("aged 60-64 aged 70-74") == "about 62 age about 72 age"
    # positive control: the same grouper DOES collapse when the age is rendered by decade
    coarse = collections.Counter(re.sub(r"aged (\d)(\d)-\d+", r"aged \g<1>0s", s["prompt_body"])
                                 for s in S.values())
    fine = collections.Counter(degroup(s["prompt_body"]) for s in S.values())
    assert max(fine.values()) == 1, "de-grouping collapsed two released cohorts"
    assert max(coarse.values()) > 1, "the collapse detector cannot detect a collapse"
    print(f"  degroup: 0/{len(S):,} C2 cohorts collapse; decade control collapses "
          f"{sum(v - 1 for v in coarse.values() if v > 1):,}")

    # 3. every dial only ever REMOVES characters from the released body
    for s in ss:
        base = degroup(s["prompt_body"])
        for mask in ((), ("icd",), ("icd", "opcs")):
            for onset in (True, False):
                d = render(s, mask, None, onset, N)
                assert len(d) <= len(base), (mask, onset)
                assert d.split("\n")[0] == base.split("\n")[0]
                assert len(d.split("\n")) == 4 - len(mask)
        if code_levels(s)["bnf"] > 2:
            z = render(s, (), 2, True, N)
            assert len(z) <= len(base) and "prescriptions" in z
            # every code stated after coarsening is a PREFIX of a code the release stated
            src = {c for c, _ in s["axes"]["bnf"]}
            # EVERY bracket on the line, not just what PAIR_RE pairs up: the release's own
            # `skin [1304] [1304]` shape hides a second code from the pair regex, and that is the
            # bug this assertion exists to catch.
            for code in re.findall(r"\[([A-Za-z0-9]+)\]", z.split("\n")[-1]):
                assert len(code) == 2, f"{s['state_id']}: coarsened line states {code!r}"
                assert any(c.startswith(code) for c in src), code
            bnfline = z.split("\n")[-1]
            assert "] [" not in bnfline, f"{s['state_id']}: duplicated code survived coarsening"
    print(f"  dials: on {len(ss)} C2 cohorts every setting is non-increasing in length, "
          "and every coarsened bnf code is a released prefix")

    # 4. the candidate relation agrees with gen/spirit_dials.axkey, the function the dial measurements use
    sys.path.insert(0, HERE)
    import spirit_dials as SD
    for s in ss[:50]:
        lv = code_levels(s)
        for onset in (True, False):
            mine = dkey(s, (), lv, onset)
            theirs = tuple(SD.axkey(s, a, lv, onset) for a in AXES)
            assert mine == theirs, (mine, theirs)
    print("  candidates: dkey == spirit_dials.axkey on 50 cohorts x 2 onset settings")

    # 5. a cohort is always a candidate for its own degraded rendering (so `candidates >= 1`,
    #    which is why NO published item can be no-match -- see the report's regime section)
    C = Candidates(S)
    for s in ss[:120]:
        lv = code_levels(s)
        for mask in ((), ("icd",), ("icd", "opcs")):
            assert C.count(s, mask, lv, False) >= 1
    print("  no-match: candidates >= 1 for every (cohort, mask) tried -- 0 is unreachable by dial")

    # 6. k: the panel's own certificate, with a positive AND a negative control through _atrisk
    sys.path.insert(0, REPO)
    from cohortworld_env.lattice import _atrisk
    ok = bad = 0
    for s in ss[:60]:
        ne = s.get("not_estimable") or {}
        for hi in range(5):
            m = ne.get(str(hi), ne.get(hi))
            if m == "all":
                continue
            allbad = set(m or ())
            good = [d for d in range(len(s["rates"][hi])) if d not in allbad]
            if good:
                b1 = _atrisk("C2", "C2:" + s["state_id"], 2, hi, good[0])
                ok += bool(isinstance(b1, str) and b1[:1].isdigit())
            if allbad:
                b2 = _atrisk("C2", "C2:" + s["state_id"], 2, hi, sorted(allbad)[0])
                bad += bool(isinstance(b2, str) and b2[:1].isdigit())
    assert ok > 0, "no estimable cell carried a published at-risk band"
    print(f"  k: estimable cells with a published at-risk band {ok}; NOT-estimable cells with one "
          f"{bad} (the negative control -- it must stay far below {ok})")
    try:
        check_k("1-9", "fixture")
        raise AssertionError("check_k accepted k=1")
    except ValueError:
        pass
    check_k("10-49", "fixture")
    print("  k: check_k rejects '1-9' and accepts '10-49'")

    # 7. the schema round-trips and the legacy view agrees with the canonical one
    it = mkitem(item_id="T3:x", task="T3", split="train", version="C2", description="d",
                ask={"disease": "E1", "horizon": 5, "granularity": 2}, operation=None,
                dials={"rung": "E0", "mask": [], "zoom": {}, "onset": "stated", "candidates": 1},
                source={"kind": "lattice", "cohort": "C2:a", "start_cohort": None,
                        "n_band": "10-49", "release": "official/x"},
                answer={"kind": "rate", "rate": 0.5, "cohort": None, "direction": None,
                        "covered": True},
                labels={"nomatch": False}, legacy={"answer_rate": 0.5, "disease": "E1"})
    assert it["patient"] == it["description"] == "d"
    assert it["answer_rate"] == it["answer"]["rate"] and it["disease"] == it["ask"]["disease"]
    print("  schema: canonical and legacy views agree; forbidden keys rejected")
    print("selfcheck OK")


def verify(itemdir: str, log):
    """Re-render every written item from its named release under its own recorded
    dials and assert byte equality with what was written, plus the schema and the forbidden keys.

    This is the machine form of "publishable by construction". It is not a disclosure audit -- it
    never looks for a leak. It checks that the description IS a function of the release and the
    dials, which is what makes a leak impossible rather than absent.
    """
    S = {v: load_states(v) for v in VERSIONS}
    N = namemap({k: x for v in VERSIONS for k, x in S[v].items()})
    n = collections.Counter()
    for path in sorted(glob.glob(f"{itemdir}/*.jsonl")):
        base = os.path.basename(path)
        if base.endswith("_unseen_subject.jsonl"):
            # not opened, not counted: the guard installed at the top of `main` would raise, which
            # is the point -- the verifier of the PUBLIC set cannot read the private one either.
            n["private, not opened by this process (the guard forbids it)"] += 0
            continue
        for ln in open(path):
            d = json.loads(ln)
            assert all(k in d for k in SCHEMA_KEYS), base
            assert set(d) == set(SCHEMA_KEYS) | set(LEGACY_KEYS) | {"patient"}, \
                (base, sorted(set(d) ^ (set(SCHEMA_KEYS) | set(LEGACY_KEYS) | {"patient"})))
            for k in FORBIDDEN_KEYS:
                assert k not in d, (base, k)
            assert d["patient"] == d["description"]
            sid = (d["source"]["cohort"] or d["source"]["start_cohort"]).split(":", 1)[1]
            s = S[d["version"]][sid]
            dl = d["dials"]
            got = render(s, tuple(dl["mask"]), dl["zoom"].get("bnf"), dl["onset"] == "stated", N)
            assert got == d["description"], (d["item_id"], got, d["description"])
            assert d["source"]["release"].startswith("official/")
            check_k(d["source"]["n_band"], d["item_id"])
            if dl["zoom"].get("bnf") is not None and d["task"] == "T3":
                for code in re.findall(r"\[([A-Za-z0-9]+)\]", got.split("\n")[-1]):
                    assert len(code) == dl["zoom"]["bnf"], (d["item_id"], code)
            n[base] += 1
    for k, v in sorted(n.items()):
        log(f"  re-rendered byte-identical: {v:>7,}  {k}")
    return dict(n)


# --------------------------------------------------------------------------- driver

def main():
    install_private_guard()                    # (a), FIRST, before any project import
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="artifact path (.txt; a .json is written beside it)")
    ap.add_argument("--emit", default=None, help="item directory to write; nothing is written without it")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--versions", default=",".join(VERSIONS))
    ap.add_argument("--verify", default=None, help="re-render and check an emitted item directory")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    if a.verify:
        print("=== VERIFY (re-render, byte equality) ===")
        verify(a.verify, print)
        return print("verify OK")

    L = []

    def log(line=""):
        print(line, flush=True)
        L.append(line)

    rng = random.Random(a.seed)
    versions = a.versions.split(",")
    log(f"THE PUBLISHABLE ITEM SET  seed {a.seed}  "
        f"ratio T1/T3 {RATIO[0]}/{RATIO[1]}/{100 - sum(RATIO)}  "
        f"T2 {T2_RATIO[0]}/{T2_RATIO[1]}/{100 - sum(T2_RATIO)}")
    log(f"panel {PANEL}")
    log("principle 1 (no private table readable) is enforced by an audit hook (raises on any "
        "private open) plus an import-graph assertion; the private split is a separate entry point.")
    log("")

    dis = {g: json.load(open(f"{PANEL}/CohortWorld_C2_diseases_g{g}.json"))["diseases"]
           for g in (1, 2, 3)}
    sys.path[:0] = [HERE, REPO]
    from cohortworld_env.reward import entry_cohort

    out = {t: collections.defaultdict(list) for t in ("t1_locate", "t2_delta", "t3_rate")}
    states_all, drops = {}, {}
    log("=== T1 / T3 ===")
    for v in versions:
        S = states_all[v] = load_states(v)
        N = namemap(S)
        C = Candidates(S)
        H = hops_index(v, S, entry_cohort)
        nr = sum(1 for x in H.values() if x >= 0)
        log(f"    {v:>7s}  {len(S):,} released cohorts, {len(N):,} code labels, "
            f"{nr:,} ({nr/len(S):.4f}) reachable from their OWN entry cohort")
        drops[v] = build_t1_t3(v, S, C, N, H, dis, rng, out, log)

    log("")
    log("=== T2 ===")
    build_t2(states_all, namemap({k: v for S in states_all.values() for k, v in S.items()}),
             out, log)

    # ---- (b) the import graph, checked after everything the build imports has been imported
    bad = assert_import_graph_clean()
    log("")
    log("=== PRINCIPLE 1, STRUCTURALLY ===")
    log(f"  audit hook active on {len(PRIVATE_PATTERNS)} private patterns; the build completed, so "
        "it opened none of them")
    log(f"  import graph: modules naming a private table, excluding the declared lazy exception "
        f"{DECLARED_LAZY_EXCEPTION}: {bad if bad else 'none'}")
    assert not bad, bad

    # ---- report
    log("")
    log("=== COUNTS ===")
    J = {"seed": a.seed, "ratio": RATIO, "t2_ratio": T2_RATIO, "counts": {}, "dials": {},
         "regime": {}, "splits": {}}
    for tag in ("t1_locate", "t2_delta", "t3_rate"):
        c = {sp: len(out[tag][sp]) for sp in sorted(out[tag])}
        J["counts"][tag] = c
        log(f"  {tag:<10} " + "  ".join(f"{k}={v:,}" for k, v in c.items())
            + f"   total {sum(c.values()):,}")

    log("")
    log("=== DIALS DELIVERED (T3; measured, never assumed) ===")
    t3 = [r for v_ in out["t3_rate"].values() for r in v_]
    if t3:
        by = collections.Counter(r["dials"]["rung"] for r in t3)
        cands = [r["dials"]["candidates"] for r in t3]
        inwin = sum(1 for c in cands if 2 <= c <= 8) / len(cands)
        log("  rung      share   median cand   in [2,8]")
        for k in RUNGS:
            rows = [r for r in t3 if r["dials"]["rung"] == k]
            if not rows:
                log(f"  {k:<8}  {0:.4f}   (none built)")
                continue
            cc = [r["dials"]["candidates"] for r in rows]
            log(f"  {k:<8}  {by[k]/len(t3):.4f}   {st.median(cc):>10.1f}   "
                f"{sum(1 for c in cc if 2 <= c <= 8)/len(cc):.4f}   (schedule {RUNGS[k]['share']})")
        log(f"  ALL       1.0000   {st.median(cands):>10.1f}   {inwin:.4f}")
        J["dials"] = {"rung_share": {k: by[k] / len(t3) for k in RUNGS},
                      "median_candidates": st.median(cands), "in_window_2_8": inwin,
                      "candidates_max": max(cands)}

    log("")
    log("=== REGIME MIX vs THE TARGET p ===")
    for tag in ("t1_locate", "t2_delta", "t3_rate"):
        rows = [r for v_ in out[tag].values() for r in v_]
        nm = sum(1 for r in rows if r["labels"]["nomatch"])
        log(f"  {tag:<10} no-match {nm}/{len(rows):,} = {nm/max(1,len(rows)):.4f}   "
            f"target p = {NOMATCH_SHARE_BENCH}")
        J["regime"][tag] = {"nomatch": nm, "n": len(rows),
                            "share": nm / max(1, len(rows)), "target": NOMATCH_SHARE_BENCH}
    log("  Every published item is regime `match`, and that is STRUCTURAL, not a shortfall: a")
    log("  cohort is always consistent with its own masked / coarsened / onset-dropped rendering,")
    log("  so `candidates >= 1` for every dial setting (asserted in --selfcheck and again in the")
    log("  build loop). No dial can manufacture a no-match item out of a released cohort. p = 0.38")
    log("  is therefore reachable only where a REAL person is the subject -- `subject-unseen`,")
    log("  built by the private subject builder, which is private by the same construction.")

    log("")
    log("=== SPLITS ===")
    for tag in ("t1_locate", "t2_delta", "t3_rate"):
        prof = {sp: {r["source"]["cohort"] or r["source"]["start_cohort"] for r in out[tag][sp]}
                for sp in out[tag]}
        sps = sorted(prof)
        cross = {f"{x}|{y}": len(prof[x] & prof[y])
                 for i, x in enumerate(sps) for y in sps[i + 1:]}
        log(f"  {tag:<10} cross-split PROFILE overlap {cross}")
        assert set(cross.values()) <= {0}, cross
        J["splits"][tag] = {"profile_overlap": cross}
    log("  Subject disjointness is not a property a published item HAS -- no published item names,")
    log("  renders or is drawn from a person. It is measured where subjects exist, on")
    log("  `subject-unseen`; see that build's report.")

    # ---- write
    if a.emit:
        os.makedirs(a.emit, exist_ok=True)
        J["emitted"] = {}
        for tag in ("t1_locate", "t2_delta", "t3_rate"):
            for sp, rows in sorted(out[tag].items()):
                rng.shuffle(rows)
                path = f"{a.emit}/{tag}_{sp}.jsonl"
                with open(path, "w") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                J["emitted"][f"{tag}_{sp}"] = {
                    "n": len(rows), "bytes": os.path.getsize(path),
                    "md5": hashlib.md5(open(path, "rb").read()).hexdigest()}
        log("")
        log(f"=== EMITTED to {a.emit} ===")
        for k, v in sorted(J["emitted"].items()):
            log(f"  {k + '.jsonl':<34} {v['n']:>7,} items  {v['bytes']:>10,} bytes  md5 {v['md5']}")

    text = "\n".join(L)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        open(a.out, "w").write(text + "\n")
        json.dump(J, open(a.out.replace(".txt", ".json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
