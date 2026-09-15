"""The disease word the PROMPT hands the agent must be readable.

A training run measured the hole this closes: 2,561 traces, 124 `cw_observe` calls, 27 of them (23.3%) dead on
`no disease matching '<chapter word>'`. Every one passed a word from its own question. C2 names every
code in a chapter after the chapter -- `circulatory [I0]` ... `circulatory [I9]` -- so the unique-hit
rule in `_resolve_disease` could never resolve it, and the refusal listed the table's alphabetical head
(A0-A7, infectious) rather than the ten codes it had just matched and discarded. `read a rate
elsewhere` sat at 1.68% across 5 steps with no trend.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cohortworld_env"))

from cohortworld_env.lattice import _shared  # noqa: E402

ASKED = "I2"  # a circulatory code that exists in C2


@pytest.fixture(scope="module")
def lat():
    got = _shared("C2")
    if got is None:
        pytest.skip("C2 panel not on disk")
    return got


@pytest.mark.parametrize("word,prefer", [
    ("circulatory", "I2"), ("eye/ear", "H4"), ("respiratory", "J4"),
    ("injury", "S0"), ("mental/behavioural", "F0"),
])
def test_a_chapter_word_resolves_to_the_asked_code(lat, word, prefer):
    """The five shapes that run actually failed on."""
    assert lat._resolve_disease(word, prefer) == prefer


def test_the_word_alone_is_still_ambiguous(lat):
    """Without the episode's asked code there is nothing to disambiguate WITH, and guessing is the
    failure this guard exists to prevent. Refusing is correct here."""
    assert lat._resolve_disease("circulatory") is None
    assert len(lat._name_hits("circulatory")) > 1


def test_prefer_outside_the_hits_is_refused(lat):
    """`prefer` resolves a SPELLING; it must not let the agent read a disease it did not name."""
    assert lat._resolve_disease("circulatory", "J4") is None
    assert lat._resolve_disease("respiratory", "I2") is None


def test_the_old_rescues_and_refusals_are_untouched(lat):
    """The measured behaviour `_resolve_disease`'s own comment records, re-asserted."""
    assert lat._resolve_disease("N6") == "N6"      # unique prefix hit -> the rescue
    assert lat._resolve_disease("H") is None       # one letter -> never the name index
    assert lat._resolve_disease("K") is None
    assert lat._resolve_disease("injury / poisoning [T8]") == "T8"   # bracketed code wins


def test_the_refusal_shows_what_it_nearly_matched(lat):
    """A menu the agent cannot act on is what made the 27 failures unrecoverable."""
    out = lat.observe("circulatory", 3)
    assert "error" in out
    codes = [d["code"] for d in out["diseases_here"]]
    assert codes, "a refusal with an empty menu teaches nothing"
    assert all(c.startswith("I") for c in codes), codes
    assert "carry that name" in out["hint"]


def test_a_true_non_match_still_gets_the_table(lat):
    """When nothing was nearly matched, the old menu is still the right one."""
    out = lat.observe("not a disease at all", 3)
    assert "error" in out
    assert out["diseases_here"], "must still list the codes valid here"
    assert "zoom changes the disease table" in out["hint"]


def test_the_toolset_hands_the_asked_code_to_the_resolver():
    """The fix is worthless if `observe` does not pass it. Asserted on the source, because
    constructing a toolset needs a live MCP server."""
    src = (Path(__file__).resolve().parents[1]
           / "cohortworld_env/cohortworld_env/toolset.py").read_text()
    assert "prefer=self.asked_disease" in src


def test_prefer_resolves_through_observe_the_path_a_rollout_takes(lat):
    """`observe(word, h, prefer=code)` must return exactly what `observe(code, h)` returns, on a cell
    where the first name hit would return something ELSE.

    The gap this closes, named by a pre-launch review: every other test here calls
    `_resolve_disease` directly, and `observe` only for its refusal menu. A rollout never calls
    `_resolve_disease` -- it calls `cw_observe` -> `Lattice.observe`, which threads `prefer` through. A fix
    correct at the inner function could be unwired at the outer one.

    Three things this test had to get right, each found by it failing or skipping first:
      * the identity is against the EXPLICIT-CODE call, not against "a rate came back". An earlier version
        asserted `out["disease"] == prefer` and failed -- because `NO_FOLLOWUP_WINDOW` returns only
        `{status, horizon}` and never echoes the disease.
      * it must stand on a cell where the FIRST hit differs from `prefer`, or the identity is satisfied by
        a resolver that ignores `prefer` and takes hits[0]. An earlier version skipped itself for exactly
        that reason, which is honest but proves nothing, so the cell is now SEARCHED for.
      * the state is searched, never hardcoded: state ids change on a rebuild (that is what MAJOR means
        here), and a hardcoded id would turn into a false red.
    """
    import copy

    # COPY. `_shared` hands back one Lattice per version, reused by every test in this session, and
    # `observe` reads `self.current`. Walking the shared object would leak this test's position into the
    # others -- the same shared-mutable-state bug fixed in `toolset.setup_task`.
    lat = copy.copy(lat)

    hits = lat._name_hits("circulatory")
    assert len(hits) > 1, "the premise is that this word is ambiguous; it is not, so the test proves nothing"
    first, prefer = hits[0], hits[-1]

    cell = None
    for sid in list(lat.states)[:400]:
        lat.current = sid
        for h in lat.horizons:
            a, b = lat.observe(first, h), lat.observe(prefer, h)
            if a != b and "error" not in a and "error" not in b:
                cell = (sid, h, a, b)
                break
        if cell:
            break
    assert cell, ("found no cell where two codes of the same chapter give different replies; without one "
                  "this test cannot tell `prefer` from first-hit")
    sid, h, by_first, by_code = cell
    lat.current = sid

    by_word = lat.observe("circulatory", h, prefer=prefer)
    assert "error" not in by_word, f"the prompt's own word was refused even with prefer={prefer!r}: {by_word}"
    assert by_word == by_code, (
        f"at {sid} h={h}: observe(word, prefer={prefer!r}) gave {by_word} but observe({prefer!r}) gave "
        f"{by_code}; `prefer` did not reach the resolver through observe")
    assert by_word != by_first, (
        f"the reply is identical to the FIRST hit {first!r}, so this cell cannot distinguish `prefer` from "
        f"a first-hit resolver and the assertion above is vacuous here")


def test_observe_without_prefer_still_refuses_the_ambiguous_word(lat):
    """The control for the test above: the same call WITHOUT `prefer` must still be refused.

    Without this, `test_prefer_resolves_through_observe...` would pass just as well against a resolver
    that had dropped the uniqueness rule altogether and started guessing.
    """
    out = lat.observe("circulatory", 3)
    assert "error" in out, (
        f"the ambiguous word resolved with NO prefer to guide it: {out}. The uniqueness guard is gone and "
        f"the agent can now read a disease it did not name")
