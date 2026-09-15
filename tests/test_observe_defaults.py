"""`cw_observe()` with no arguments reads the cell the question asks about.

The measurement that forced it: the two-stage question names the disease in WORDS, because the
demonstrations never show codes, while `observe` takes a code. On the form-A smoke, three of the four
cw_observe calls that followed a successful move failed with `no disease matching '<name>' in the
<version> disease table`, and the single call that got the code right produced the only usable
prediction in ~300 rollouts.

Expressed in the SIGNATURE rather than in prompt prose, which is this environment's own rule: verifiers
builds the tool's JSON Schema from the annotations, so optional parameters are what tells the model the
arguments may be omitted.
"""
import asyncio
import sys
# import path set by tests/conftest.py (repo-relative)
from cohortworld_env.toolset import CohortWorldToolset, CohortWorldToolsetConfig


class _Data:
    version, age_band, sex, budget = "C2", "45-49", "Female", 26
    disease, horizon = "D0", 5


class _Task:
    data = _Data()


def _toolset():
    ts = CohortWorldToolset(CohortWorldToolsetConfig(version="C2"))
    asyncio.run(ts.setup_task(_Task()))
    return ts


def test_the_asked_cell_is_remembered():
    ts = _toolset()
    assert (ts.asked_disease, ts.asked_horizon) == ("D0", 5)


def test_observe_with_no_arguments_reads_the_asked_cell():
    ts = _toolset()
    bare = asyncio.run(ts.observe())
    explicit = asyncio.run(ts.observe(disease="D0", horizon=5))
    for k in ("status", "rate", "disease", "horizon"):
        assert bare.get(k) == explicit.get(k), (
            f"the bare call must read the asked cell; differed on {k}: {bare.get(k)} vs {explicit.get(k)}")


def test_an_explicit_cell_still_wins():
    """The control. If the default silently overrode the arguments, the test above would still pass --
    what proves the default is a DEFAULT is that a different cell gives a different answer."""
    ts = _toolset()
    a = asyncio.run(ts.observe())
    b = asyncio.run(ts.observe(disease="I9", horizon=1))
    assert a.get("disease") != b.get("disease"), (
        f"an explicit disease was ignored: {a.get('disease')} vs {b.get('disease')}")


def test_a_toolset_without_a_task_does_not_raise():
    ts = CohortWorldToolset(CohortWorldToolsetConfig(version="C2"))
    out = asyncio.run(ts.observe())
    # a toolset with no asked cell answers a bare observe with where the agent stands and a
    # note saying there is no cell to read, instead of an "asked cell" error.
    assert "note" in out and "not a rate" in out["note"] and "state_id" in out
