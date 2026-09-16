"""The legal action set, served from the edge table, with the destination the edge already names.

Three defects, measured on arm A of one run (120 rollouts, 2,757 tool returns), share one cause -- the
state never told the agent what its actions accept:

  * **1,114 returns (40.4%) were `{"hit": false, "bound": "n<10"}`**, 64.7% of every navigation miss.
    88% of them were issued before the agent had ever been shown a legal list, because `legal_add`
    ships only on a successful landing (275 times in 120 rollouts). 30% of rollouts never landed at
    all, and the first successful landing took a median of 8 tool calls.
  * What the state DID carry was `opcs_areas` / `bnf_areas`, drawn from the OPCS/BNF **code
    systems** and therefore byte-identical at every cohort, plus `switch_class`, one static English
    sentence. Those are the menu the agent picked from, and most of it leads nowhere.
  * **18.9% of the lattice's legal moves had no tool at all**: `age_band shift` (11,568 edges),
    `sex switch` (2,884), `icd switch` (2,539), `icd add` (1,395) -- every one of them priced in the
    cost table, so they were designed in and left unimplemented. Only `opcs` and `bnf` were really
    editable, which is why relaxing age, the most direct way to buy generality, was impossible.

Nothing here is newly computed. An edge already records `from, axis, op, code, to, cost, n_band_to`
-- the edge IS the legal move and it names its own destination, so serving moves from this index
makes "advertised" and "reachable" the same set by construction rather than by agreement between two
code paths. `separated` is a per-cell reliability annotation and has no bearing on whether a move
exists, which is why the index is built before any reliability filter.
"""
from __future__ import annotations
import json, collections, os

# ABSOLUTE by default, matching lattice.py. A relative default is not a style choice here: this
# module is imported inside the `cw` tool server's subprocess runtime, whose cwd is not the repo, so
# a relative path makes the import raise and the server never serves -- every rollout then dies with
# `tool server 'cw' not serving in runtime` before a single LLM call. That failure had already
# happened once today, with CW_SYS, and was reintroduced here.
# ... and derived from `__file__` rather than spelled out, matching lattice.py. A path hardcoded to
# the author's checkout is absolute and still wrong in a clone: it points at the SOURCE
# repository, so the clone's reward read one tree's reference table and this module read another's.
PANEL = os.environ.get(
    "CW_PANEL",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "official"))
INDEX = f"{PANEL}/CohortWorld_legal_moves.jsonl"

_CACHE: dict = {}


def load() -> dict[str, list[dict]]:
    if not _CACHE:
        for line in open(INDEX):
            d = json.loads(line)
            _CACHE[d["state_id"]] = d["moves"]
    return _CACHE


def moves(sid: str, same_version_only: bool = True, extra=()) -> list[dict]:
    """Every legal (axis, op, code) from this cohort, cheapest first. Empty means a dead end.

    `extra` is for moves that are NOT in the edge table because they cannot be: `drop`
    generalises by dropping one recorded criterion and landing on the most specific released cohort
    whose axes are a subset of what remains, and "most specific subset" is a fact about the whole
    stratum, not an edge between two cohorts. `Lattice._drop_moves` computes it -- it is the object
    that holds the states -- and passes the result in here, so every caller still sees ONE legal
    move set and `summary`/`counts`/`apply` keep agreeing with each other by construction.

    It DEFAULTS EMPTY, and that is load-bearing: `reward.delta_reference()` calls `moves()` to build
    a pinned constant out of the entry cohort's edges, and a default that silently widened that set
    would change a hashed reward constant while looking like an environment change.

    Cross-version edges are withheld from the SERVED action set by default. 10,033 of the 97,401
    edges (10.3%) cross between C1, C2 and default, and they are genuine -- the pooled edge table
    needs them to compare versions. But a served rollout must stay in its item's version: the three
    lattices share no outcome code at all, so a move into another version silently changes what the
    question is about. That is the same defect as the earlier `zoom` bug, where crossing versions
    collapsed the disease axis from 183 codes to 19.
    """
    ms = list(load().get(sid, [])) + list(extra)
    if not same_version_only:
        return ms
    v = sid.split(":", 1)[0] + ":"
    return [m for m in ms if m["to"].startswith(v)]


def _render(m: dict) -> dict:
    """One move as the agent chooses it: an id, and enough to tell it from its siblings.

    `code` alone was not enough. A `switch` takes two parameters -- what leaves the axis and what
    arrives -- and the edge records only what arrives, so from `bnf [02,03,05,10,13]`, `switch 04`
    appeared TWICE, identically, once replacing 02 and once replacing 03. Adding the onset rung the
    code lands on settles the rest. Indistinguishable moves fell 14.6% -> 4.15% -> **0.82%**.

    This mattered because unchoosable options are what the degenerate loops were chewing on:
    a truncated turn ran 33,368 characters at a sentence uniqueness of 0.19, its top sentence
    repeated a median 35 times, against 2,964 characters and uniqueness 1.00 for a clean turn.
    """
    out = {"id": _mid(m), "code": m["code"], "size_after": m["n_band_to"]}
    if m.get("replaces"):
        out["replaces"] = m["replaces"]
    if m.get("onset_rung") is not None:
        out["onset_rung"] = m["onset_rung"]
    return out


def _mid(m: dict) -> str:
    """A stable id for one move. The agent picks an id; it never composes an action.

    Composing was the defect. `cw_add(axis, code)` cannot say WHICH destination it means: **16.7% of
    (axis, op, code) combinations lead to more than one cohort** (up to 24), because the band on the
    axis is part of the move and the signature had no room for it. So `edit()` searched the bands and
    returned `n<10` when the search failed -- which is why it disagrees with the edge table on 68.6%
    of the moves both can express (24,876 spurious `n<10`, 19,885 landing somewhere else). Picking a
    listed id is a lookup, and lookup cannot drift from what was advertised.
    """
    # The destination suffix is carried in FULL. Truncating it to 6 hex collided on 46 of the
    # 97,401 moves (0.047%) -- two different destinations sharing one id, which would silently send
    # the agent somewhere it did not choose. A rare collision in an identifier is not a small bug.
    # The destination is carried WHOLE, version prefix included. Stripping the prefix collided 46
    # of the 97,401 ids: a profile can exist in more than one lattice version under the same 16-hex
    # hash, so `C2:e5e7ba17...` and `default:e5e7ba17...` became one id and the agent could be sent
    # into a different version than the one it picked. 10,033 edges are cross-version, so the
    # version is part of the move, not decoration. The raw edge table has no duplicate on
    # (from, axis, op, code, to) at all -- every collision here was manufactured by the id function.
    return f"{m['axis']}.{m['op']}.{str(m['code'])}->{m['to']}"


def counts(sid: str, extra=()) -> dict:
    """Group counts only -- what a move returns, so the menu is not re-sent on every step.

    Returning the full listing after every move grew the transcript by roughly 680 tokens a step
    over 11-13 steps. With `max_tokens` at 6144 against a 32,768 window that pushed **7 of 30
    rollouts into `context_length`**, trading one silent no-answer failure for another: truncation
    had just fallen 20.0% -> 6.7%, and overflow ate the gain.

    A move now reports where you are and how many options each group holds; `cw_look` prints them.
    That is the structure WebShop has -- acting shows you the resulting page, not the whole site map.
    """
    by = collections.Counter()
    cost = {}
    for m in moves(sid, extra=extra):
        k = f"{m['axis']}.{m['op']}"
        by[k] += 1
        cost.setdefault(k, m["cost"])
    total = sum(by.values())
    return {"legal_move_counts": {k: {"n": v, "cost": cost[k]} for k, v in sorted(by.items())},
            "legal_move_count": total,
            "note": ("cw_look lists these moves with their ids." if total else
                     "This cohort has no legal move: you can only observe or commit from here.")}


def summary(sid: str, per_axis: int = None, extra=()) -> dict:
    """What the agent is shown: the parameters its actions accept, here, now.

    Grouped by (axis, op) and truncated per group with an explicit count (`n`), so a long list costs
    context without ever misrepresenting itself as complete -- the earlier `legal_add` was cut to 40
    silently.

    The default is 6 and NOT 0. It was 0 -- "no truncation" -- for two days while this docstring
    described the six-per-group behaviour, because neither caller (`Lattice._action_summary`,
    `Lattice.move`'s miss path) passed the argument. Measured on one run: **0 of 1,017 served move
    groups were truncated**, `cw_look` replied a median 4,198 characters, p90 8,544, max 9,687, and
    `context_length` stops were back at 5 and 9 in the next two runs. A guard that is written, documented
    and never switched on is the failure mode this repository keeps paying for; the default carries
    it now, so a caller has to opt OUT rather than remember to opt in.

    The default is `reward.PER_AXIS` and NOT a literal 6, which is what it was through the freeze.
    A default argument is not a contract: it was in no `config.yaml` and no `CONSTANTS()` call,
    so 6 could become 3 -- moving the achievable policy set --
    while every score stayed formally comparable. The value is unchanged, so nothing already scored
    is invalidated by naming it; `test_the_move_listing_is_truncated_at_the_pinned_width` pins it.

    `per_axis` is 6, not 12. Entry cohorts carry a median 32 legal moves and up to 50, so the full
    listing made `cw_look` a 2,479-character reply at exactly the point every rollout calls it
    first. Prompt tokens then reached a p99 of 25,924 against a 32,768 window, and **1.4% of turns
    hit the 4,096-token generation cap mid-sentence -- which ends the rollout with no tool call and
    no answer, scored a flat 1.0**. That single failure mode produced 19 of arm A's 21 worst-case
    items, 11 of B's 13 and 31 of C's 36, so the arms were separating on how verbose their prompts
    made the model rather than on how well it reasoned. Six per group cuts the entry reply 35%.
    """
    if per_axis is None:
        from .reward import PER_AXIS
        per_axis = PER_AXIS
    by = collections.defaultdict(list)
    for m in moves(sid, extra=extra):
        by[(m["axis"], m["op"])].append(m)
    out, total = {}, 0
    for (ax, op), ms in sorted(by.items(), key=lambda kv: (ms_cost(kv[1]), kv[0])):
        total += len(ms)
        out[f"{ax}.{op}"] = {
            "cost": ms[0]["cost"],
            "n": len(ms),
            "moves": [_render(m) for m in (ms if per_axis <= 0 else ms[:per_axis])]
                     + ([{"id": "...", "note": f"{len(ms) - per_axis} more not shown"}]
                        if 0 < per_axis < len(ms) else []),
        }
    note = ("These are every move this cohort accepts. A code not listed here has no cohort behind "
            "it. `replaces` is the code this switch removes from the axis; `onset_rung` is how "
            "early the arriving code was first recorded, lower being earlier."
            if total else
            "This cohort has no legal move: you can only observe or commit from here.")
    return {"legal_moves": out, "legal_move_count": total, "note": note}


def ms_cost(ms: list[dict]) -> float:
    return ms[0]["cost"] if ms else 0.0


def apply(sid: str, move_id: str, extra=()) -> dict | None:
    """The destination the edge already names, or None if that id is not legal here."""
    for m in moves(sid, extra=extra):
        if _mid(m) == move_id:
            return m
    return None
