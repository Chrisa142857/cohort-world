"""Binds the frozen CohortWorld lattice to the tools and the task generators.

Everything the agent can see comes from `output_panel/` (banded counts, rates); exact `n` and `x`
live in a private store outside this repository and are used only to BUILD tasks and score them, never
served through a tool.
"""
from __future__ import annotations
import json, os, re
from typing import Any, Iterator

# The BUILT lattice, not the first draft. `data/` holds the 13:50 build whose adaptive ladder still
# collapsed codes to "*" -- 84% of its states carry a "*" axis, which says only "some record exists"
# and throws away the history CohortWorld is supposed to hold. `official/` is the rebuild that
# removed that rung, and it is what the item builders and the sidecar were generated against.
# Pointing the environment at `data/` meant the agent navigated a 41,315-state "*" lattice while
# every item referenced the 7,113-state real one; entry landed on cohorts described as
# "diagnoses: any (unspecified)" for questions that said "no recorded conditions".
#
# Derived from `__file__`, never written out by name. A hardcoded path to the author's checkout
# is absolute -- which is all T42 used to check -- and absolute to the WRONG REPOSITORY in any clone:
# `reward.py` resolves `official/` off `__file__` while this line named the source tree, so a fresh
# clone read its reference table from itself and its lattice from somewhere else, and twelve tests
# asserted "the lattice is the rebuild" against a repository they were not in. Set CW_PANEL to point
# elsewhere on purpose; do not spell a repo path into the package.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANEL = os.environ.get("CW_PANEL", f"{_REPO}/official")
PRIV = os.environ.get("CW_PRIV", f"{PANEL}/priv")
# Chapter names for the two editable code systems, inlined so the environment package is
# self-contained: importing them from the build scripts under clinical/ worked only when
# that directory happened to be on sys.path, and the rollout crashed with
# ModuleNotFoundError once the package ran on its own.
OPCS_CH = {'A': 'nervous system', 'B': 'endocrine/breast', 'C': 'eye', 'D': 'ear', 'E': 'respiratory tract', 'F': 'mouth', 'G': 'upper digestive', 'H': 'lower digestive', 'J': 'liver/pancreas', 'K': 'heart', 'L': 'arteries/veins', 'M': 'urinary', 'N': 'male genital', 'P': 'lower female genital', 'Q': 'upper female genital', 'R': 'pregnancy-related', 'S': 'skin', 'T': 'soft tissue', 'U': 'diagnostic imaging', 'V': 'bones/joints of skull-spine', 'W': 'other bones/joints'}

BNF_CH = {'01': 'gastro-intestinal', '02': 'cardiovascular', '03': 'respiratory', '04': 'central nervous system', '05': 'infections', '06': 'endocrine', '07': 'obstetrics/gynaecology', '08': 'malignant disease/immunosuppression', '09': 'nutrition/blood', '10': 'musculoskeletal', '11': 'eye', '12': 'ENT', '13': 'skin', '14': 'immunological/vaccines', '15': 'anaesthesia'}

COARSER = {"default": "C2", "C2": "C1", "C1": None}
FINER = {v: k for k, v in COARSER.items() if v}
# The 80/10/10 state split does NOT live here. `Lattice.regime_of` / `chapters_for` / `RATIOS` sat in
# this module with no caller anywhere in the repo -- the live split is `gen/build_tasks.py`, which
# runs once when the item files are built and never at rollout or scoring time. Dead code inside the
# frozen surface is a mutation no test can catch (two of them survived the mutation run for exactly
# this reason), so they are deleted rather than annotated a second time.


def _axkey(axes: dict) -> tuple:
    return (axes["age_band"], axes["sex"],
            tuple(map(tuple, axes["icd"])), tuple(map(tuple, axes["opcs"])),
            tuple(map(tuple, axes["bnf"])))


_SHARED: dict[str, "Lattice"] = {}


def _prefix_match(got, base) -> bool:
    """Two renderings of the same history at different code resolutions."""
    if len(got) != len(base):
        return False
    for (c1, b1), (c2, b2) in zip(sorted(got), sorted(base)):
        if b1 != b2 or not (c1.startswith(c2) or c2.startswith(c1)):
            return False
    return True


def _band_lo(b: str) -> int:
    """Lower edge of a published n_band ("1000+", "50-99"), for ranking cohorts by size."""
    b = str(b).replace("+", "").split("-")[0]
    return int(b) if b.isdigit() else 0


_ATRISK: dict = {}
# The at-risk store is keyed by OUTCOME GRANULARITY, and each version's native one is fixed:
# C1 has 19 outcome codes, C2 183, default 1,374.
_GRAN = {"C1": 1, "C2": 2, "default": 3}


_DROP_COST: dict = {}


def _drop_cost(axis: str) -> float:
    """What `<axis>.drop` costs: whatever `add` costs on the same axis.

    READ OFF THE RELEASED EDGE TABLE, not written out here. The three numbers are 1.0 / 2.0 / 3.0
    for bnf / opcs / icd today, and a second copy of them in this file is exactly the drift this
    repository keeps paying for -- the cost table is the source of truth for what an axis is worth,
    and a `drop` priced from a restated literal would keep the old price through a re-release.

    It RAISES rather than falling back to a default: an axis with no released `add` edge anywhere
    means the edge table and this module disagree about what the axes are, and a quietly-priced move
    would hide that.
    """
    if not _DROP_COST:
        from . import legal_moves as LM
        for ms in LM.load().values():
            for m in ms:
                if m["op"] == "add":
                    _DROP_COST.setdefault(m["axis"], float(m["cost"]))
    if axis not in _DROP_COST:
        raise KeyError(f"cannot price {axis!r}.drop: the released edge table has no {axis!r} add "
                       f"edge to price it from (it prices {sorted(_DROP_COST)})")
    return _DROP_COST[axis]


_ID_LINE = re.compile(rb'"state_id"\s*:\s*"([^"]+)"')


def _atrisk_index(version: str):
    """(path, {state_id: byte offset}, cache) for one version's at-risk store.

    These files are 544-693 MB and the old code parsed one entirely into a dict on the first
    `observe` in each process. With `-c 16` that is 16 tool-server processes at ~6.4 GB RSS each on
    a node that also carries another user's ~300 GB job: free RAM fell to 23 GB, swap to zero, and
    three rollouts died of `httpx.ReadTimeout` racing the MCP startup timeout.

    A state's cell is ~94 KB and a rollout touches a few dozen states, so hold offsets (7,113 ints)
    and seek. The offsets are written next to the data as a small `.idx` sidecar, because scanning
    693 MB per process would trade the RAM problem for an NFS one; the write is tmp + rename so
    concurrent processes cannot read a half-written index.
    """
    f = f"{PANEL}/CohortWorld_{version}_atrisk_bands.jsonl"
    if not os.path.exists(f):
        return (None, {}, {})
    idx = f + ".idx"
    try:
        if os.path.getmtime(idx) >= os.path.getmtime(f):
            with open(idx) as fh:
                return (f, json.load(fh), {})
    except (OSError, ValueError):
        pass
    offsets = {}
    with open(f, "rb") as fh:
        while True:
            off = fh.tell()
            line = fh.readline()
            if not line:
                break
            m = _ID_LINE.search(line, 0, 4096)
            if m:
                offsets[m.group(1).decode()] = off
    tmp = f"{idx}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(offsets, fh)
        os.replace(tmp, idx)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return (f, offsets, {})


def _atrisk(version: str, sid: str, gran: int, hi: int, di: int):
    """The at-risk band for one (state, granularity, horizon, disease) cell.

    `CohortWorld_<V>_atrisk_bands.jsonl` was built for exactly this and **nothing in the package read
    it** -- `observe` returned the cohort's `n_band`, which is a different number: 3,575 of 3,615
    released C2 states carry the same band, so an agent asked to judge reliability was reading a
    constant, while the true `n_persons` has a median of 12 and a p90 of 21 and the optimal shrinkage
    weight moves sharply across that range.

    Three things about the file's shape, each of which returned None until it was handled:
      * it is nested `atrisk[granularity][horizon_index][disease_index]`, THREE levels, not two;
      * the store is SPARSE -- `build_atrisk.py` writes a cell only when its at-risk count clears the
        k-anonymity floor, so a missing entry means "fewer than 10 still at risk", not "unknown";
      * a missing horizon means the state had no landmark-eligible rows at that horizon at all.
    The last two are reported as what they are, because "fewer than 10 people are still at risk" is
    precisely the signal an agent needs in order to distrust a rate.

    PANEL is absolute and is reused rather than composed: a relative default here would raise inside
    the tool server's subprocess runtime, whose cwd is not the repo, and that has cost two runs.
    """
    if version not in _ATRISK:
        _ATRISK[version] = _atrisk_index(version)
    path, offsets, cache = _ATRISK[version]
    if path is None:
        return None
    cell = cache.get(sid)
    if cell is None:
        off = offsets.get(sid)
        if off is None:
            return None
        with open(path) as fh:
            fh.seek(off)
            cell = json.loads(fh.readline())["atrisk"]
        if len(cache) > 512:            # a rollout visits far fewer states than this
            cache.clear()
        cache[sid] = cell
    g = cell.get(str(gran))
    if g is None:
        return None
    h = g.get(str(hi))
    if h is None:
        return "no follow-up window at this horizon"
    return h.get(str(di), "<10 still at risk")


_CUT = ', "rates_g1":'
"""Where a state line stops being useful to a tool server. See `Lattice.__init__`."""


def _shared(version: str, *, granular: bool = False):
    """One Lattice per version per process: zoom crosses versions constantly and rebuilding the
    index each time would dominate the rollout. Returns None for a version not present on disk, so
    a partially built lattice degrades to fewer resolutions instead of crashing a rollout."""
    key = (version, bool(granular))
    if key not in _SHARED:
        try:
            _SHARED[key] = Lattice(version, granular=granular)
        except FileNotFoundError:
            _SHARED[key] = None
    return _SHARED[key]


class Lattice:

    masked: str = ""
    """One cohort id whose RATES this episode may not read. Empty means nothing is withheld.

    A CLASS attribute, so every Lattice built before this existed -- and every task that names no
    target -- behaves exactly as it did. Only `CohortWorldToolset.setup_task` sets it, and only on a
    per-episode SHALLOW COPY of the `_shared()` object (see that method for why).

    WHY IT EXISTS. Every internal T3 item IS a lattice cohort: the item carries `target_cohort` and
    its `answer_rate` is that cohort's published rate, and all 18,593 items' targets are still in the
    served lattice. So the answer was readable with `cw_observe` from the right state: 500/500
    probed train items returned their own answer exactly (`gen/leak_audit.py --no-mask`), and in
    one run's 2,597 training rollouts 33 (1.27%) took a published reading while standing on the target
    -- worth a measured +0.43 mean reward. Standing there is FINE and is now the design:
    the entry is meant to be the item's own small, specific cohort, from which the agent retreats to
    a larger reference cohort it can trust and corrects back by the difference it reasoned about.
    Reading the answer off the target is what is not fine, so the mask withholds the RATE and nothing
    else -- the state stays advertised by `cw_look` and reachable by `cw_move`.
    """

    GRANULAR_FIELDS = ("rates_g1", "rates_g2", "rates_g3")
    """The per-granularity rate matrices: 88.4% of a state's bytes, and NOT read by a tool server.

    Measured on the panel: a state is 45,031 B, of which rates_g3 is 34,548 (76.7%), rates_g2 4,712
    (10.5%) and rates_g1 549 (1.2%). Parsed into Python the C2 lattice is 2.40 GB with them and 0.29 GB
    without -- 8.3x, because a JSON float becomes a 24-byte object in a list of pointers.

    WHO ACTUALLY READS THEM, checked across the whole package rather than assumed: `estimable_cells`
    (internal.py:962) and nothing else at runtime; every other mention is in `gen/`, the offline
    builders. `estimable_cells` is called only from `internal.py:997` on the taskset's `draw_cells`
    path, which runs ONCE PER ENV-SERVER WORKER. `toolset.py` does not import it at all -- and the
    toolset is what runs once per rollout, 128 of them concurrently, holding 548.7 GiB between them
    while vLLM sat at 0-2 requests with an empty queue.

    So the fields are loaded only when the caller says it needs them. The default is not to.
    `gen/nomatch_share.py:76` already does this by hand for the same reason: "the line is cut before
    rates_g2 because g2 and g3 are 99% of the bytes and nothing here reads them".

    THE FAILURE THIS MUST NOT CAUSE, and why it needs a test rather than care: `estimable_cells` reads
    `state.get(f"rates_g{granularity}") or state.get("rates")`. That `or` SWALLOWS the mistake -- a
    taskset that lost the granular matrices would silently draw its cells from the wrong matrix and
    raise nothing. tests/test_lattice_granular.py asserts the taskset path still gets them.
    """

    def __init__(self, version: str = "C1", *, granular: bool = False):
        self.version = version
        self.granular = bool(granular)
        self.states = {}
        # Read whichever layout is on disk: the builder writes one file per split, and an earlier
        # combined `_states.jsonl` was only ever a convenience produced by hand. Requiring the
        # combined file made every rollout die in the toolset constructor with FileNotFoundError.
        paths = [f"{PANEL}/CohortWorld_{version}_{sp}_states.jsonl"
                 for sp in ("train", "val", "test")]
        paths = [q for q in paths if os.path.exists(q)] or \
                [f"{PANEL}/CohortWorld_{version}_states.jsonl"]
        # CUT BEFORE PARSING, not after. Dropping the granular matrices from the dict already saved the
        # RAM (2.40 GB -> 0.29 GB), but json.loads still had to parse all 45 KB of every line to build
        # objects that were then thrown away: the lattice took 3.94 s of a 6.0 s tool-server cold start.
        # The three fields are CONTIGUOUS AND LAST in the builder's output -- the order is
        # state_id, axes, prompt_body, coarseness, n_band, rates, not_estimable, rates_g1, g2, g3 --
        # and rates_g1 begins at byte 5,088 of 44,560, so cutting there removes 88.6% of the text.
        # `gen/nomatch_share.py:83` already does exactly this by hand, for exactly this reason.
        #
        # The cut is CONDITIONAL on finding the marker: a file whose fields were ordered differently
        # falls back to parsing the whole line rather than producing invalid JSON or, worse, a state
        # missing `rates`. `_CUT` therefore cannot silently corrupt a panel it does not recognise.
        for l in (line for q in paths for line in open(q)):
            if not self.granular:
                i = l.find(_CUT)
                if i > 0:
                    l = l[:i] + "}"
            d = json.loads(l)
            # Version-prefix the id. The three lattices share 10-15k raw state_ids, and only 23.6%
            # of the C1/C2 collisions hold the SAME number of people: an identical key is filled by
            # different members in each version, because greedy release reaches it down a different
            # ladder. Merging on the bare id would silently fuse two different cohorts.
            d["state_id"] = f"{version}:{d['state_id']}"
            if not self.granular:
                for _k in self.GRANULAR_FIELDS:
                    d.pop(_k, None)
            self.states[d["state_id"]] = d
        mf = json.load(open(f"{PANEL}/CohortWorld_{version}_manifest.json"))
        self.horizons = mf["horizons"]
        self.diseases = [d["code"] for d in mf["diseases"]]
        self.dname = {d["code"]: d["name"] for d in mf["diseases"]}
        self.didx = {d: i for i, d in enumerate(self.diseases)}
        self.vocab = json.load(open(f"{PANEL}/CohortWorld_{version}_vocab.json"))
        self.byaxes = {(_axkey(d["axes"]), d["coarseness"]): sid
                       for sid, d in self.states.items()}
        # Arbitrary until a task places the agent. Starting a rollout here would drop the agent in
        # an unrelated cohort -- a question about 40-44 year old men answered from a 65-69 year old
        # women's cohort -- and every navigation then misses, because nothing near that cohort
        # resembles the question. Tasks must call `enter()`.
        self.current = next(iter(self.states))

    def enter(self, age_band: str, sex: str, prefer_coarsest: bool = True) -> dict[str, Any]:
        """Place the agent in a cohort of the right age and sex, to start from.

        The right entry is the most generic cohort matching the question's demographics: large,
        stable, and a place the specialising moves can work outward from. Landing somewhere unrelated
        makes the action space useless, which is what happened when the starting position was
        whichever state the dictionary happened to yield first.
        """
        sid = self.entry_for(age_band, sex, prefer_coarsest)
        if not sid:
            return {"hit": False, "reason": f"no cohort for {age_band} {sex}"}
        self.current = sid
        return {"hit": True, **self.state_of(self.current)}

    def goto(self, sid: str) -> bool:
        """Place the agent on a NAMED cohort, for a task whose question is about a specific one.

        T2 asks about one edge, so the agent has to start at that edge's tail -- "stand on a group
        without that feature" is the first line of its instructions. There is no tool for this and
        there must not be: `cw_move` only accepts advertised ids, which is what stops an agent from
        typing its way to an answer. This is the setup path, not an action.
        """
        if sid in self.states:
            self.current = sid
            return True
        return False

    def entry_for(self, age_band: str, sex: str, prefer_coarsest: bool = True) -> str:
        """Which cohort `enter` would choose, WITHOUT moving the agent there.

        Split out because T1's reward is `S(final) - S(entry)` and scoring runs in a different
        process from the tool server: the reward has to ask where the agent started without a lattice
        mutation as a side effect. The selection lives here once, so the position the reward assumes
        and the position the agent actually got cannot drift apart.
        """
        cands = [sid for sid, d in self.states.items()
                 if d["axes"]["age_band"] == age_band
                 and str(d["axes"]["sex"]).upper().startswith(str(sex).upper()[:1])]
        if not cands:
            return ""
        def _load(sid):
            """Fewest codes first, then largest cohort.

            Generality is the NUMBER of codes a cohort insists on, not the rung its codes are cut
            at. Ranking by rung was right while the ladder still had a "*" presence-only rung, when
            a coarse rung meant "any record"; with "*" gone, the most generic cohort for 40-44 men
            is the single zero-code one -- no history required, n_band 1000+ -- and it sits at the
            FINEST rung (decade|icd3-opcs3-bnf4). Ranking by rung put it last and started the agent
            in a 10-49 person cohort with a genitourinary history, answering a question that said
            "no recorded conditions".
            """
            d = self.states[sid]
            n = len(d["axes"]["icd"]) + len(d["axes"]["opcs"]) + len(d["axes"]["bnf"])
            return (n, -_band_lo(d.get("n_band", "")))

        cands.sort(key=_load, reverse=not prefer_coarsest)
        return cands[0]

    # ---- private truth (build/score only) ----------------------------------
    def _priv(self):
        if not hasattr(self, "_p"):
            self._p = {}
            for l in open(f"{PRIV}/CohortWorld_{self.version}_private_counts.jsonl"):  # release-gate: allow (build/score-only; not released)
                d = json.loads(l)
                for code, recs in d["present"].items():
                    for hi, n, x in recs:
                        self._p[(f"{self.version}:{d['state_id']}", hi, self.didx[code])] = (n, x)
        return self._p

    # ---- tools -------------------------------------------------------------
    def _action_summary(self) -> dict[str, Any]:
        """The parameters this cohort's actions accept -- nothing else.

        What used to be here was three things, only one of which was an action space:
        `opcs_areas` / `bnf_areas`, drawn from the OPCS and BNF **code systems** and therefore
        byte-identical at every cohort; `switch_class`, one static English sentence; and `legal_add`,
        which was real but shipped only on a successful landing and covered `add` alone. The agent
        picked from the code-system menu because that was the only thing that looked like a list, and
        **1,114 of arm A's 2,757 tool returns (40.4%) came back `{"hit": false, "bound": "n<10"}`** --
        64.7% of every miss, 88% of them issued before it had ever seen a legal list, with 30% of
        rollouts never landing at all.

        The legal set now comes from the edge table, where the edge already records
        `from, axis, op, code, to, cost, n_band_to`. Serving it from there makes "advertised" and
        "reachable" one set by construction instead of an agreement between two code paths -- and it
        covers every operation, not just `add`, which matters because `switch` is the commonest move
        (mean 2.31 out-edges per C2 state against `add`'s 0.96) and `age_band shift` (11.9% of all
        edges), `sex switch`, `icd add` and `icd switch` had no tool at all despite being priced in
        the cost table. Verified on the whole lattice: 97,401/97,401 advertised moves resolve to a
        released cohort.
        """
        from . import legal_moves as LM
        return LM.summary(self.current, extra=self._served_drops(self.current))

    def _subset_index(self):
        """(age_band, sex) -> [(the cohort's codes as a set, its id)], built once per Lattice.

        Only the CODES are kept, not the states, so this is a few hundred kilobytes next to the rate
        matrices it indexes. Keyed by (age_band, sex) because a generalising move drops a recorded
        criterion and keeps the person: a destination in another age band or another sex is a
        different patient, not a broader reference class.
        """
        if not hasattr(self, "_sub"):
            idx: dict = {}
            for sid, d in self.states.items():
                ax = d["axes"]
                codes = frozenset((k, tuple(e)) for k in ("icd", "opcs", "bnf")
                                  for e in (ax.get(k) or []))
                idx.setdefault((ax.get("age_band"), str(ax.get("sex"))), []).append((codes, sid))
            self._sub = idx
        return self._sub

    def _served_drops(self, sid: str) -> list[dict]:
        """The `drop` moves THIS EPISODE advertises: none at all unless the item named a target.

        THE JUDGE IS FROZEN, and this is where. `masked` is set only for an item that carries a
        `target_cohort` -- which is exactly the internal T3 items whose entry IS their own cohort and
        whose rate is withheld. Those are the episodes the generalising move exists for: the agent
        cannot read where it stands, so it must be able to widen. Every other task -- the external
        PhenoCal judge, T1, T2 -- keeps the action space without `drop` byte for byte, so the archived
        zero-shot screens stay valid comparators and no baseline has to be re-run.

        Keyed on the MASK rather than on a separate flag on purpose: "this episode starts on its own
        unreadable cohort" and "this episode may generalise" are the same fact, and a second switch
        could be set without the first. `_drop_moves` stays ungated because `reward` computes the
        reference cohort from it, off the shared lattice, where no episode's mask applies.

        It was true even before the gate that no judge ENTRY advertises a drop -- all 18 C2
        demographic entries are zero-code cohorts -- but that is an accident of the entry rule, not a
        guarantee: one `cw_move` from there lands on a coded cohort, and the judge would have seen
        the new op. `test_the_judge_never_sees_the_generalising_op` checks exactly that cohort.
        """
        return self._drop_moves(sid) if self.masked else []

    def _drop_moves(self, sid: str) -> list[dict]:
        """`<axis>.drop.<code>`: give up one recorded criterion and land on the broadest cohort that
        still contains this patient. The only way this environment has to GENERALISE.

        Ungated: this is the RULE. `_served_drops` decides which episodes are shown it.

        The five op groups the edge table serves -- `bnf.add`, `opcs.add`, `icd.add`,
        `age_band.shift`, `sex.switch` -- all specialise or move the person. There was no move along
        a history axis that made the reference class BIGGER, which is the move a clinician actually
        makes when a subgroup is too small to trust, and it is the move the T3 procedure now depends
        on: the agent starts on the item's own cohort, whose rate is withheld, and has to retreat to
        one it can read.

        NOT an exact-match "remove one code" edge, which is a dead end. Measured against THIS
        implementation on 500 train items (`gen/drop_feasibility.py`): exact removal leaves 137/500
        (27.4%) with no released neighbour at all and 431/500 (86.2%) unable to reach a 200+ cohort,
        because the intermediate rungs were never released. Generalisation leaves 0/500 stranded and
        58.6% reaching 200+/1000+ in the one move.
        The rule here is GENERALISATION: drop the criterion, then land on the MOST SPECIFIC released
        cohort whose axes are a subset of what remains. Every such cohort still contains this
        patient, and the demographic-only cohort (no codes at all) is always a subset, so a legal
        destination always exists once the state has any code to drop.

        Computed rather than served from the edge table because "most specific subset" is a fact
        about the whole (age band, sex) stratum, not a relation between two cohorts -- there is no
        edge to record. `Lattice` is where it belongs: it is the object that holds the states.
        `legal_moves` takes the result as `extra` so that advertised, counted and applied stay one
        set.

        PRICE: `_drop_cost`, which reads `add`'s price on the same axis off the released edge table.
        The cost table prices the AXIS -- how much a code on it narrows the cohort -- and not the
        direction, so undoing an `icd add` costs what the add cost. Deliberately NOT cheaper: cost
        orders the `cw_look` listing, and making generalisation the cheapest thing on the menu would
        advertise it as the preferred move at every cohort rather than letting the agent judge.
        Nothing in the reward reads `cost`; the budget charges one per call whatever the move.
        """
        d = self.states[sid]
        ax = d["axes"]
        codes = frozenset((k, tuple(e)) for k in ("icd", "opcs", "bnf") for e in (ax.get(k) or []))
        if not codes:
            return []
        pool = self._subset_index().get((ax.get("age_band"), str(ax.get("sex"))), ())
        out, seen = [], set()
        for axis, ent in sorted(codes):
            rem = codes - {(axis, ent)}
            # Ties on specificity are broken by id, so the listing is the same on every process.
            best = max(((len(cc), s) for cc, s in pool if cc <= rem), default=None)
            if best is None:                       # only possible if the stratum has no subset state
                continue
            to = best[1]
            mv = {"from": sid, "axis": axis, "op": "drop", "code": ent[0], "to": to,
                  "cost": _drop_cost(axis), "n_band_to": self.states[to]["n_band"],
                  "onset_rung": ent[1]}
            from . import legal_moves as LM
            if (mid := LM._mid(mv)) in seen:       # one code twice on an axis at two onset rungs
                continue
            seen.add(mid)
            out.append(mv)
        return out

    def move(self, move_id: str) -> dict[str, Any]:
        """Take one advertised move, by the id the state listed. A lookup, never a search.

        `edit()` searched the bands for a code, because `add(axis, code)` had no room to say WHICH
        destination it meant -- and **16.7% of (axis, op, code) combinations lead to more than one
        cohort** (up to 24). That search disagreed with the edge table on 68.6% of the moves both
        could express: 24,876 spurious `n<10` and 19,885 landings somewhere else entirely.
        """
        from . import legal_moves as LM
        drops = self._served_drops(self.current)
        m = LM.apply(self.current, move_id, extra=drops)
        if m is None:
            return {"hit": False, "reason": "not a legal move from here",
                    "moved": False, **LM.summary(self.current, extra=drops)}
        self.current = m["to"]
        d = self.states[m["to"]]
        # Where you are and how many options each group holds -- NOT the whole menu. Re-sending the
        # full listing after every move cost ~680 tokens a step and put 7 of 30 rollouts into
        # `context_length`. `cw_look` prints the menu when the agent wants it.
        out = {"hit": True, "took": move_id, "cost": m["cost"], "state_id": m["to"],
               "axes": d["axes"], "n_band": d["n_band"], "coarseness": d["coarseness"],
               "description": d["prompt_body"],
               **LM.counts(m["to"], extra=self._served_drops(m["to"]))}
        out["diseases_here"] = len(self.diseases)
        return out

    def _legal_edits(self, with_ids: bool = False) -> dict[str, list]:
        """The codes that can actually be added from here, found by index rather than by probing.

        The candidate lists come from the OPCS-4 and BNF code systems, so they advertise **1,605**
        options at every state while only about 30 lead to a released cohort -- 0.1% to 2.0% by
        entry. The agent was left to find that out by spending budget on it, which is the same tax
        `zoom_available` removed for zoom, and it is why 89.4% of navigation moves missed.

        Computed by scanning the states in this (age band, sex) stratum for ones whose axes differ
        from the current cohort by exactly one added code, so it costs a stratum scan rather than
        1,605 lookups. Advertising a legal move set is not advertising the answer: which cohorts
        EXIST is the shape of the world, while what their rates ARE stays hidden.
        """
        d = self.states[self.current]
        key = (d["axes"]["age_band"], str(d["axes"]["sex"]))
        if not hasattr(self, "_stratum"):
            idx: dict = {}
            for sid, st in self.states.items():
                idx.setdefault((st["axes"]["age_band"], str(st["axes"]["sex"])), []).append((sid, st))
            self._stratum = idx
        cur = {ax: {tuple(e) for e in d["axes"][ax]} for ax in ("icd", "opcs", "bnf")}
        out: dict[str, list] = {"opcs": [], "bnf": []}
        for sid, st in self._stratum.get(key, ()):
            if sid == self.current:
                continue
            other = {ax: {tuple(e) for e in st["axes"][ax]} for ax in ("icd", "opcs", "bnf")}
            if other["icd"] != cur["icd"]:
                continue
            diff_ax = [ax for ax in ("opcs", "bnf") if other[ax] != cur[ax]]
            if len(diff_ax) != 1:
                continue
            ax = diff_ax[0]
            added = other[ax] - cur[ax]
            if len(added) == 1 and not (cur[ax] - other[ax]):
                out[ax].append((next(iter(added))[0], sid))
        if with_ids:
            return {ax: sorted(set(v)) for ax, v in out.items()}
        return {ax: sorted({c for c, _s in v}) for ax, v in out.items()}

    def _zoom_available(self) -> dict[str, list]:
        """Which zoom moves actually lead somewhere, per axis.

        The action space listed the codes for `add`/`switch` but said nothing about which zoom rungs
        exist, so the agent spent its budget discovering the shape of the ladder by hitting the walls:
        over 120 rollouts, 89.4% of navigation moves missed, and about 360 of those were
        "no finer resolution for icd" / "no coarser resolution for bnf" / "no finer time scale than
        decade" -- questions the environment could have answered for free.

        This is legal-move information, not answer information. ALFWorld tells the agent what is in
        the room; withholding the shape of the ladder does not make the task harder in an interesting
        way, it just taxes the budget. The rate matrix stays hidden, which is where the difficulty is.

        It asks `zoom` itself in probe mode rather than reimplementing the rung arithmetic. A second
        copy of that logic drifted immediately on the first try: the probe passed `axis=None` where
        the real move passes the axis and a derived parent code, and advertised `bnf: []` for a
        cohort where `zoom("bnf", "coarser")` in fact succeeded. Advertising a legal move as illegal
        is worse than saying nothing.
        """
        home = self.current
        avail: dict[str, list] = {}
        try:
            for ax in ("time", "icd", "opcs", "bnf"):
                got: list = []
                for d in ("finer", "coarser"):
                    if self.zoom(ax, d, probe=True).get("hit"):
                        got.append(d)
                avail[ax] = got
        finally:
            self.current = home
        return avail

    def action_space(self) -> dict[str, Any]:
        """The codes that may be put on each editable axis, with a readable label.

        `sysname` only names the top level of each system (1-char OPCS, 2-char BNF), so at C2's
        2-/4-char granularity every candidate came back labelled with its own code. Here the label
        is the chapter it belongs to plus the code itself, which is what a reader needs to pick.
        """
        out = {}
        for ax, sysk in (("opcs", "O"), ("bnf", "B")):
            items = []
            for c in self.vocab["candidates"][ax]:
                code = c["code"]
                chap = OPCS_CH.get(code[0], "other") if sysk == "O" else BNF_CH.get(code[:2], "other")
                items.append({"code": code, "area": chap})
            out[ax] = items
        return {"editable_axes": ["opcs", "bnf"],
                "switch_class": self.vocab["switch_class"], "candidates": out}

    def state_of(self, sid: str) -> dict[str, Any]:
        """The full observable state: the five axes, the banded size, and the legal candidate codes.

        The candidates belong here rather than behind a tool: they are the action space's degrees of
        freedom at this point, not a fact to be discovered. Deriving them from the code systems --
        never from the lattice -- is what keeps them from leaking the support set: on `default` they
        offer 1,260 OPCS codes of which only 425 appear in any released state.
        """
        d = self.states[sid]
        return {"state_id": sid, "axes": d["axes"], "n_band": d["n_band"],
                "coarseness": d["coarseness"], "description": d["prompt_body"],
                "action_space": self._action_summary()}

    def _land(self, sid: str | None) -> dict[str, Any]:
        """Every move returns the full state, action space included: the degrees of freedom change
        with the cohort, so they are part of what an agent observes on arrival, not a separate
        lookup it has to pay for."""
        if sid is None:
            return {"hit": False, "bound": "n<10"}
        self.current = sid
        out = {"hit": True, **self.state_of(sid)}
        out["diseases_here"] = len(self.diseases)
        return out

    def select(self, axes: dict) -> dict[str, Any]:
        want = _axkey(axes)
        for (k, _c), sid in self.byaxes.items():
            if k == want:
                return self._land(sid)
        return self._land(None)

    def edit(self, sid: str, axis: str, code: str, op: str) -> dict[str, Any]:
        d = self.states[sid]
        cur = [tuple(x) for x in d["axes"][axis]]
        band = max((b for _c, b in cur if isinstance(b, int)), default=None)
        if op == "add":
            new = sorted(set(cur) | {(code, band)}, key=lambda e: (e[0], -1 if e[1] is None else e[1]))
        else:                                   # switch: drop one same-class member, add this code
            n = self.vocab["granularity"]["opcs_chars" if axis == "opcs" else "bnf_chars"]
            pre = code[:max(1, n - 1)] if axis == "opcs" else code[:max(2, n - 2)]
            new = sorted({e for e in cur if not e[0].startswith(pre)} | {(code, band)},
                         key=lambda e: (e[0], -1 if e[1] is None else e[1]))
        axes = {**d["axes"], axis: [list(e) for e in new]}
        r = self.select(axes)
        if r.get("hit"):
            return r
        # Retry across the bands this code actually takes in the lattice.
        #
        # The band above is copied from the codes already on the axis, so on an EMPTY axis there is
        # none and it falls back to `None` -- while every released cohort stores a real age band,
        # `("1103", 0)`. The exact lookup could then never match, and the empty axis is exactly where
        # every rollout starts. Measured before this: of the 1,605 candidate `add` moves advertised
        # at the entry cohort, **four of the six entries had zero legal moves** and the other two had
        # three, although the entry's own (age band, sex) stratum holds 21 released one-code cohorts
        # and the lattice holds 835. The neighbours were there; a `None` was refusing them.
        #
        # The band belongs to the cohort being moved to, not to the edit being made, so trying the
        # bands the lattice actually uses for this code is the correct lookup rather than a fallback.
        for b in self._bands_for(axis, code):
            if b == band:
                continue
            cand = sorted({e for e in new if e[0] != code} | {(code, b)},
                          key=lambda e: (e[0], -1 if e[1] is None else e[1]))
            r = self.select({**d["axes"], axis: [list(e) for e in cand]})
            if r.get("hit"):
                return r
        # Last resort: the same stratum index `_legal_edits` advertises from. Going through a
        # different code path than the advertisement is how a move gets advertised as legal and then
        # refused -- the exact drift that had to be fixed in `zoom_available`. One index, one answer.
        if op == "add":
            sid2 = self._neighbour_by_add(axis, code)
            if sid2:
                self.current = sid2
                return {"hit": True, **self.state_of(sid2)}
        return r

    def _neighbour_by_add(self, axis: str, code: str) -> str | None:
        """The released cohort that differs from here by exactly this one added code, if any."""
        for ax, sid in self._legal_edits(with_ids=True).get(axis, ()):
            if ax == code:
                return sid
        return None

    def _bands_for(self, axis: str, code: str) -> list:
        """Every band this code appears with on this axis, commonest first."""
        if not hasattr(self, "_bandidx"):
            idx: dict = {}
            for st in self.states.values():
                for ax in ("icd", "opcs", "bnf"):
                    for c, b in st["axes"][ax]:
                        idx.setdefault((ax, c), {}).setdefault(b, 0)
                        idx[(ax, c)][b] += 1
            self._bandidx = idx
        got = self._bandidx.get((axis, code), {})
        return [b for b, _n in sorted(got.items(), key=lambda t: -t[1])]

    def zoom(self, axis: str, direction: str, code: str = "",
             probe: bool = False) -> dict[str, Any]:
        """Change the resolution of ONE axis, or of the time scale, and move to the cohort there.

        The lattice is not one resolution but many: code granularity is per axis (C2 alone carries 33
        combinations such as `icd2-opcs2-bnf4` and `icd2-opcs2-bnf2`), while the time scale is one
        global setting shared by all three axes (decade / 4stage / 3stage / 2stage / flat). So `axis`
        is one of icd / opcs / bnf / time, and the move is one step along that dimension.

        An earlier version zoomed whole VERSIONS and looked for identical axes on the other side.
        That could not work for matching a patient: the same person's cohort carries 1-character
        codes in C1 and 2-character codes in C2, so an exact axis match never existed and the move
        failed for reasons that looked arbitrary. Finer resolution may only be available in another
        version -- C1's ICD ladder stops at one character -- so this crosses versions when it has to
        and says when it did.

        Going finer needs to know WHICH finer code, since one coarse code covers many; pass it in
        `code`, or leave it out to be told which finer codes lead to a released cohort.
        """
        if axis not in ("icd", "opcs", "bnf", "time"):
            return {"hit": False, "reason": "axis must be icd, opcs, bnf or time"}
        if direction not in ("coarser", "finer"):
            return {"hit": False, "reason": "direction must be coarser or finer"}
        here = self.states[self.current]
        tag_time, tag_code = here["coarseness"].split("|")
        if axis == "time":
            order = ["decade", "4stage", "3stage", "2stage", "flat"]
            i = order.index(tag_time) if tag_time in order else 0
            j = i + (1 if direction == "coarser" else -1)
            if not 0 <= j < len(order):
                return {"hit": False, "reason": f"no {direction} time scale than {tag_time}"}
            want = f"{order[j]}|{tag_code}"
            return self._nearest(here, want, axis=None, code="", probe=probe)
        want_code = self._shift_code_tag(tag_code, axis, direction)
        if want_code is None:
            return {"hit": False, "reason": f"no {direction} resolution for {axis}"}
        # Coarsening is determined -- M3 has exactly one parent, M -- so the code is derived rather
        # than asked for. Only refining is ambiguous, since one coarse code covers many finer ones.
        if direction == "coarser":
            step = 2 if axis == "bnf" else 1
            code = ""
            cur = [c for c, _b in here["axes"][axis]]
            if cur:
                code = cur[0][:max(1, len(cur[0]) - step)]
        return self._nearest(here, f"{tag_time}|{want_code}", axis, code, probe=probe)

    @staticmethod
    def _shift_code_tag(tag_code: str, axis: str, direction: str) -> str | None:
        vals = {}
        for piece in tag_code.split("-"):
            for name in ("icd", "opcs", "bnf"):
                if piece.startswith(name):
                    vals[name] = int(piece[len(name):])
        step = 2 if axis == "bnf" else 1
        floor = 2 if axis == "bnf" else 1
        cur = vals.get(axis)
        if cur is None:
            return None
        nxt = cur + step if direction == "finer" else cur - step
        top = {"icd": 3, "opcs": 3, "bnf": 6}[axis]
        if nxt < floor or nxt > top:
            return None
        vals[axis] = nxt
        return f"icd{vals['icd']}-opcs{vals['opcs']}-bnf{vals['bnf']}"

    def _nearest(self, here: dict, want_coarseness: str, axis: str | None,
                 code: str, probe: bool = False) -> dict[str, Any]:
        """The cohort at `want_coarseness` whose axes match here, refining `axis` to `code` if given.

        Searches THIS VERSION ONLY. Crossing versions to find a rung was the original design -- a
        resolution one step finer often exists only in a version whose ladder starts there -- but
        the three versions do not share an outcome granularity (C1 chapter, C2 two-character,
        default category), so arriving in another version silently replaces the disease table the
        question was posed in. Observed: a `zoom {axis: opcs, direction: finer}` -- a PROCEDURE-axis
        move -- flipped the lattice C2 -> C1 and collapsed the disease axis from 183 codes to 19, so
        the outcome the task asked about ("G4") no longer existed and every later observe failed.
        A move that makes the question unanswerable is worse than a move that is unavailable.
        """
        a = here["axes"]
        base = {k: [tuple(x) for x in a[k]] for k in ("icd", "opcs", "bnf")}
        want = dict(base)
        if axis and code:
            band = next((b for c, b in base[axis]), None)
            want[axis] = [(code, band)]
        cands = []
        for ver in (self.version,):
            lat = self
            for sid, d in lat.states.items():
                if d["coarseness"] != want_coarseness or \
                        d["axes"]["age_band"] != a["age_band"] or d["axes"]["sex"] != a["sex"]:
                    continue
                ok = True
                for k in ("icd", "opcs", "bnf"):
                    got = [tuple(x) for x in d["axes"][k]]
                    if k == axis:
                        if code and [c for c, _b in got] != [code]:
                            ok = False
                    elif got != base[k]:
                        # a code one level apart is still the same history, seen at another
                        # resolution: accept it as a prefix match
                        if not _prefix_match(got, base[k]):
                            ok = False
                    if not ok:
                        break
                if ok:
                    cands.append((ver, sid, d))
        if not cands:
            if axis and not code:
                opts = self._finer_options(here, want_coarseness, axis)
                if opts:
                    return {"hit": False, "reason": f"say which {axis} code to zoom into",
                            "options": opts[:20]}
            return {"hit": False, "reason": f"no cohort at {want_coarseness} matching this history"}
        _ver, sid, _ = cands[0]
        if probe:
            # Probe only: do NOT move, and do NOT build the full state. `state_of` calls
            # `_action_summary`, which calls `_zoom_available`, which calls this -- so a probe that
            # returned a full state would recurse without end, and one that moved would teleport the
            # agent while merely asking what is legal.
            return {"hit": True}
        # No lattice swap here. `self.__dict__.update(other.__dict__)` used to live on this line and
        # is what turned a procedure-axis zoom into a silent version change; the search is now
        # confined to self.version, so the disease table the question was posed in always survives.
        self.current = sid
        out = {"hit": True, **self.state_of(sid)}
        out["note"] = f"resolution is now {want_coarseness}"
        return out

    def _finer_options(self, here: dict, want_coarseness: str, axis: str) -> list[str]:
        a = here["axes"]
        cur = {c for c, _b in a[axis]}
        out = set()
        for ver in (self.version,):
            lat = self
            for d in lat.states.values():
                if d["coarseness"] != want_coarseness or \
                        d["axes"]["age_band"] != a["age_band"] or d["axes"]["sex"] != a["sex"]:
                    continue
                for c, _b in d["axes"][axis]:
                    if any(c.startswith(p) for p in cur) and c not in cur:
                        out.add(c)
        return sorted(out)

    def _name_hits(self, q: str) -> list[str]:
        """Every code whose English name overlaps this query. One source for the match, because the
        refusal below must show the SAME codes the resolver considered -- showing a different set is
        how 23.3% of one run's `cw_observe` calls became unrecoverable (see `_resolve_disease`)."""
        low = (q or "").strip().lower()
        if len(low) < 3:
            return []
        return [c for c, name in self.dname.items() if low in name.lower() or name.lower() in low]

    def _resolve_disease(self, q: str, prefer: str | None = None) -> str | None:
        """Accept the ICD-10 code or the name the prompt shows.

        Prompts name diseases in English, so a model asked about "Osteoporosis with pathological
        fracture; and 9 related (M80-M89)" naturally passes that back; demanding the bare code made
        every `observe` fail. Lost once already when the zoom rewrite replaced a span of this file --
        `observe` kept calling it and every rollout hit AttributeError inside the tool.
        """
        q = (q or "").strip()
        if q in self.didx:
            return q
        # A bracketed code wins over the label around it. PhenoCal writes the outcome as
        # "injury / poisoning [T8]" using the chapter's English name, while CohortWorld C2 calls the
        # same code "Complications following infusion, transfusion and therapeutic injection
        # (T80-T88)". The names never match, so name matching alone rejected the very disease the
        # question was about; the code inside the brackets is the part both sides agree on.
        m = re.search(r"\[([A-Za-z]\d{0,2})\]", q)
        if m:
            for cand in (m.group(1), m.group(1).upper()):
                if cand in self.didx:
                    return cand
        if q.upper() in self.didx:
            return q.upper()
        low = q.lower()
        for code, name in self.dname.items():
            if low == name.lower():
                return code
        # A parenthesised code range is the other spelling both sides agree on: C2 names a disease
        # "Osteoporosis with pathological fracture; and 9 related (M80-M89)" and a model asked about
        # it echoes the whole label, so match on the leading code of that range.
        m = re.search(r"\(([A-Za-z]\d{1,2})\s*-", q)
        if m and m.group(1).upper() in self.didx:
            return m.group(1).upper()
        # From here down, a match must be UNIQUE. The old last resort was "the first disease whose
        # English name contains this string, or is contained in it", and it almost always hit --
        # against the real C2 table 'H' -> F0 (mental/behavioural), 'M' -> D0 (blood/neoplasm),
        # 'N' -> A0 (infectious), 'K' -> K0 (digestive). `observe` then returned `status: published`
        # with a real rate for the wrong outcome, which is worse than returning nothing: refusing
        # costs the agent one call and tells it which codes exist here, while a confident wrong rate
        # silently corrupts the measurement the whole task is scored on.
        #
        # Uniqueness rather than removal, because removal loses spellings that are RIGHT -- but the
        # worked example this comment used to give was itself wrong, so here is the measured one.
        # A query is resolved PER VERSION, and the same string lands differently in each table:
        #   `N6`          C2  the exact code, one prefix hit -> resolves (this is the rescue)
        #                 default  five prefix hits N60-N64 -> None, correctly refused
        #                 C1  no hit at all -> None
        #   `congenital`  C1  one name hit -> `Q`;  C2/default  no hit -> None
        # So uniqueness rescues `N6` in C2 and `congenital` in C1 and refuses `N6` in default, which
        # is exactly the intended behaviour and is NOT what the old comment claimed (it named `N6` in
        # default as a rescue). Replaying all 6,718 archived `cw_observe` calls, 20 change and every
        # one changes True -> None: 17 are "read the wrong disease" calls this now refuses, against
        # 5 lucky guesses lost. An unexplained -- or mis-explained -- guard is what gets deleted next,
        # which is why the numbers are here rather than the intent.
        hits = [c for c in self.didx if c.upper().startswith(q.upper())]
        if len(hits) == 1:
            return hits[0]
        # A word, not a code. Two characters or fewer IS a code in every version's table, so a short
        # query never reaches the name index -- that restriction is what keeps 'H' and 'K' out, and
        # no uniqueness test would, since a single letter can be unique in a small name table by
        # accident.
        hits = self._name_hits(q)
        if len(hits) == 1:
            return hits[0]
        # AMBIGUOUS BY CONSTRUCTION, and `prefer` is the one code that is not a guess.
        #
        # The uniqueness rule above is right and stays: returning a rate for a disease the agent did
        # not mean corrupts the measurement silently. But against the real name tables it refuses the
        # query the PROMPT ITSELF teaches. C2 names every code in a chapter after the chapter --
        # `circulatory [I0]` ... `circulatory [I9]` -- so the word the question hands the model
        # ("circulatory") matches 10 codes and can never be unique. Measured on one training run, 2,561 traces:
        # 27 of 124 `cw_observe` calls (23.3%) died here, every one of them passing a chapter word
        # from its own prompt; `read a rate elsewhere` sat at 1.68% and did not trend up over 5 steps.
        #
        # `prefer` is the code `setup_task` recorded as THIS episode's asked disease. When the word
        # the agent passed matches that code's name, the asked code is the only thing the agent can
        # mean -- and `observe(disease=None)` already reads exactly that cell by default, so this adds
        # no reach the action space did not have. It resolves a SPELLING, it does not pick a disease:
        # when `prefer` is not among the hits the query is still refused.
        if prefer and prefer in hits:
            return prefer
        return None

    def observe(self, disease: str, horizon: int, prefer: str | None = None) -> dict[str, Any]:
        d = self.states[self.current]
        code = self._resolve_disease(disease, prefer)
        if code is None or horizon not in self.horizons:
            # Say WHICH table is in force. After a zoom the disease axis changes (C1 has 19 codes,
            # C2 183, default 1374), so a code that was valid a move ago is not any more, and the
            # error alone left the model guessing spellings of a code that could not exist here.
            # SHOW WHAT WAS NEARLY MATCHED. The menu used to be `self.diseases[:40]`, which is the
            # table's alphabetical head, so a model asking for 'circulatory' was shown A0-A7
            # (infectious) -- it could not recover, and 27 calls in one measured run did not. The near-matches are
            # already computed; handing those back turns a dead end into one more turn.
            near = self._name_hits(disease)
            menu = near or list(self.diseases)
            return {"error": f"no disease matching {disease!r} in the {self.version} disease table",
                    "diseases_here": [{"code": c, "name": self.dname[c]} for c in menu][:40],
                    "total_diseases_here": len(self.diseases),
                    "hint": (f"{len(near)} codes here carry that name; pass one of their codes"
                             if near else
                             "zoom changes the disease table; these are the codes valid right now")}
        disease = code
        if self.current == self.masked:
            # THE ANSWER LIVES HERE. This item's `answer_rate` is this cohort's published rate, so
            # returning it would be handing over the label, not measuring an estimate. The status is
            # withheld too, not just the number: `NO_FOLLOWUP_WINDOW` / `TOO_FEW_AT_RISK` / `rate ==
            # 0` are themselves most of the answer for the zero stratum. Nothing else about the state
            # is hidden -- the agent can stand here, and is meant to.
            return {"status": "RATE_WITHHELD_HERE", "disease": disease, "horizon": horizon,
                    "hint": "This group's rates are not published. Move to a group you can read and "
                            "reason from there about how this one differs."}
        hi, di = self.horizons.index(horizon), self.didx[disease]
        ne = {int(k): v for k, v in d.get("not_estimable", {}).items()}
        v = ne.get(hi)
        if v == "all":
            return {"status": "NO_FOLLOWUP_WINDOW", "horizon": horizon}
        if v and di in v:
            return {"status": "TOO_FEW_AT_RISK", "disease": disease, "horizon": horizon}
        at = _atrisk(self.version, self.current, _GRAN.get(self.version, 2), hi, di)
        return {"status": "published", "disease": disease, "disease_name": self.dname[disease],
                "horizon": horizon, "rate": d["rates"][hi][di],
                # The number still AT RISK OF THIS OUTCOME, not the cohort's size. They are different
                # numbers and only the first one says how much the rate can be trusted: the cohort
                # band was `10-49` on 3,575 of 3,615 released C2 states, so an agent asked to judge
                # reliability was reading a constant. True n_persons has a median of 12 and a p90 of
                # 21, and the optimal shrinkage weight moves sharply across that range.
                "at_risk": at, "cohort_n_band": d["n_band"]}
