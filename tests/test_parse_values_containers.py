"""`parse_values` must read the number the agent committed, whatever container it arrives in.

THE FAILURE THIS IS FOR, measured. `cw_commit`'s advertised schema is
`anyOf[array of number, string, null]` -- no bare number -- but prime-rl sets
VLLM_ENFORCE_STRICT_TOOL_CALLING=0 (inference/server.py:27), so the schema is advertised and NOT
enforced and `cw_commit(rates=0.05)` arrives as a float. `_NUM_RE.findall(0.05)` then raised
`TypeError: expected string or bytes-like object, got 'float'`, verifiers wrapped it as
`TaskError: task RateTask scoring`, and THE WHOLE TRACE WAS DISCARDED.

It was latent for as long as the policy did not act: a run with 93.9% of rollouts making no tool call
never hit it. The first run whose credit reaches the action turn lost 293 traces in one step --
Error 31.9% against the earlier ~1%. Nothing archived depended on the old behaviour: the old behaviour was an
exception, and an exception produces no score.
"""
import sys
# import path set by tests/conftest.py (repo-relative)
from cohortworld_env.reward import parse_values


def test_the_exact_input_that_discarded_293_traces():
    assert parse_values(0.05) == [0.05], "a bare float still raises; that is 31.9% of a step's traces"


def test_an_int_is_a_number_too():
    assert parse_values(3) == [3.0]


def test_a_bool_is_not_a_rate():
    """True must not become 1.0, and must not reach the regex that cannot take it either."""
    assert parse_values(True) == []
    assert parse_values(False) == []


def test_the_containers_that_already_worked_still_do():
    """The control: the fix must not change the paths archived scores were produced on."""
    assert parse_values(None) == []
    assert parse_values([0.01, 0.02, 0.03]) == [0.01, 0.02, 0.03]
    assert parse_values("the rate is 0.04 and 0.06") == [0.04, 0.06]
    assert parse_values('{"values": [0.1, 0.2]}') == [0.1, 0.2]


def test_nothing_raises_on_any_container_a_tool_call_could_carry():
    """The point is not any single case: it is that SCORING NEVER THROWS on what an unenforced schema
    lets through. A raise here costs the whole trace, which is why this is asserted as a class."""
    for v in (0.05, 3, True, False, None, [], [0.1], ["x"], "0.2", "", {}, {"values": [0.3]},
              [0.1, "x", 0.2], 0, -1.0, 1e9):
        parse_values(v)          # must not raise
