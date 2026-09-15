"""The legal action parameters for every cohort, indexed straight off the edge table.

An interactive environment must hand the agent the parameters its actions accept, the way WebShop
shows the links that exist on the page. CohortWorld did not: the state carried `opcs_areas` /
`bnf_areas`, which are drawn from the OPCS/BNF CODE SYSTEMS and are therefore **identical at every
cohort**, and `switch_class`, a static English sentence. Only `legal_add` was real, and it shipped
only on a successful landing. Measured on rollouts against that state: **1,114 of 2,757 tool returns (40.4%) came back
`{"hit": false, "bound": "n<10"}`**, 88% of them issued before the agent had ever been shown a legal
list, and 30% of rollouts never landed successfully at all.

Nothing new has to be computed. An edge in `CohortWorld_pooled_edges.jsonl` already records
`from`, `axis`, `op`, `code`, `to`, `cost` and `n_band_to` -- the edge IS the legal move, and its
`separated` flags are a per-cell reliability annotation that has no bearing on whether the move
exists. Grouping the edges by `from` yields the parameter set directly, for every operation rather
than for `add` alone -- which matters because `switch` is the commonest move (mean 2.31 out-edges
per C2 state against `add`'s 0.96) and was the one with no parameter list at all.
"""
import json, os, sys, collections

PANEL = "official"
EDGES = f"{PANEL}/CohortWorld_pooled_edges.jsonl"
OUT = f"{PANEL}/CohortWorld_legal_moves.jsonl"


def _band_of(state, axis, code):
    for c in (state.get("axes") or {}).get(axis) or []:
        if isinstance(c, (list, tuple)) and str(c[0]) == str(code) and isinstance(c[1], int):
            return c[1]
    return None


def _axis_codes(state, axis):
    v = (state.get("axes") or {}).get(axis) or []
    if isinstance(v, str):
        return [v]
    return [c[0] if isinstance(c, (list, tuple)) else c for c in v]


def build():
    """Index the edges by source, and work out what each `switch` REPLACES.

    A switch takes two parameters -- what leaves the axis and what arrives -- and the edge records
    only what arrives. So the same `(axis, op, code)` appears more than once with different
    destinations, differing in which existing code it displaced: from `bnf [02,03,05,10,13]`,
    `switch 04` reaches both `[03,04,05,10,13]` (04 replaced 02) and `[02,04,05,10,13]` (04 replaced
    03). To the agent those two entries are byte-identical and unchoosable. **12,756 of the 87,368
    served moves, 14.6%, are redundant-looking for this reason**, and hunting through them is what
    the degenerate repetition loops were doing: truncated turns run 33,368 characters at a sentence
    uniqueness of 0.19, the top sentence repeated a median 35 times, against 2,964 characters and
    uniqueness 1.00 for a clean turn.

    Diffing the source and destination axes recovers the missing parameter, so every move is
    distinguishable and reads as the clinical operation it is.
    """
    import sys
    sys.path.insert(0, "cohortworld_env")
    from cohortworld_env.lattice import _shared
    lats = {v: _shared(v) for v in ("C1", "C2", "default")}
    states = {}
    for L in lats.values():
        states.update(L.states)

    idx = collections.defaultdict(list)
    n = 0
    for line in open(EDGES):
        e = json.loads(line)
        n += 1
        rec = {"axis": e["axis"], "op": e["op"], "code": e["code"],
               "to": e["to"], "cost": e["cost"], "n_band_to": e["n_band_to"]}
        if e["from"] in states and e["to"] in states:
            if e["op"] == "switch":
                before = set(_axis_codes(states[e["from"]], e["axis"]))
                after = set(_axis_codes(states[e["to"]], e["axis"]))
                gone = sorted(before - after)
                if gone:
                    rec["replaces"] = gone[0] if len(gone) == 1 else ",".join(gone)
            # The onset rung the code lands on. `replaces` alone still left 4.15% of moves
            # indistinguishable: from `bnf [["14", 2]]`, `switch 14->05` reaches both `[["05", 2]]`
            # and `[["05", 1]]`, the same code arriving with a different first-recorded stage.
            # Lower is earlier (band 1 renders "40-55", band 3 "58+"), and the stage labels differ
            # by version, so the rung is carried as an ordinal and explained once in the summary.
            b = _band_of(states[e["to"]], e["axis"], e["code"])
            if b is not None:
                rec["onset_rung"] = b
        idx[e["from"]].append(rec)
    with open(OUT, "w") as f:
        for sid, moves in sorted(idx.items()):
            moves.sort(key=lambda m: (m["cost"], m["axis"], m["op"], str(m["code"])))
            f.write(json.dumps({"state_id": sid, "moves": moves}) + "\n")
    print(f"[legal] {n:,} edges -> {len(idx):,} states with at least one legal move")
    return idx


def verify(idx, sample_n=300, seed=42):
    """The invariant: every advertised move must land. If one does not, the index is wrong."""
    import random
    sys.path.insert(0, "cohortworld_env")
    from cohortworld_env.lattice import _shared
    lat = {v: _shared(v) for v in ("C1", "C2", "default")}
    rng = random.Random(seed)
    sids = rng.sample(sorted(idx), min(sample_n, len(idx)))
    tried = hit = 0
    bad = collections.Counter()
    for sid in sids:
        v = sid.split(":", 1)[0]
        L = lat.get(v)
        if L is None:
            continue
        for m in idx[sid][:6]:
            L.current = sid
            op, ax, code = m["op"], m["axis"], m["code"]
            if op in ("add", "switch"):
                r = L.edit(sid, ax, code, op=op)
            elif op == "zoom":
                r = L.zoom(ax, code)
            else:
                continue                       # age_band shift has no tool; see the report
            tried += 1
            if r.get("hit"):
                hit += 1
            else:
                bad[(op, ax, r.get("bound") or r.get("reason", "miss"))] += 1
    print(f"[legal] invariant: {hit}/{tried} advertised moves landed = "
          f"{hit/tried*100:.1f}%" if tried else "[legal] nothing testable")
    for k, c in bad.most_common(8):
        print(f"          FAILED {c:>5}  op={k[0]} axis={k[1]} -> {k[2]}")
    return hit, tried


if __name__ == "__main__":
    i = build()
    verify(i)
