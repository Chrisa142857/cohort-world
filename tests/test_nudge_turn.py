"""Form A's asked turn: it must fire when a reply only talks, and keep firing until the episode is done.

Two bugs this file exists for, both of which a test asserting "the env has a run method" would pass.

1. `if reward.tool_results(trace)` tested a GENERATOR object for truthiness, which is always true, so the
   ask never fired and the first run with it reproduced the run without it exactly.
2. The loop then gated on `acted()`, which is true after ANY tool call. The ask names `cw_look`, so an
   episode that OBEYED spent its one ask on `cw_look`, received the full legal-move listing, and was
   immediately recorded as having acted -- ending with the listing unused. Measured in training: `cw_move`
   is accepted 120/155 (77.4%) when a listing is in context and 0/122 when it is not. The single
   configuration in which the action is reliably available was the one the episode ended in.
"""
import asyncio
import sys
# import path set by tests/conftest.py (repo-relative)
from cohortworld_env.internal import (acted, NUDGE, NUDGE_AFTER_LOOK, ASKS,
                                      NUDGES_PER_EPISODE, CohortWorldEnv)
from cohortworld_env import reward as R


class _T:
    def __init__(self, messages):
        self.messages = messages


def _talk_only():
    return _T([{"role": "user", "content": "q"},
               {"role": "assistant", "content": "<cohort-think>reasoning</cohort-think>"}])


def _read_a_rate():
    return _T([{"role": "user", "content": "q"},
               {"role": "assistant", "content": "", "tool_calls": [{"name": "cw_observe"}]},
               {"role": "tool", "name": "cw_observe",
                "content": '{"status": "published", "rate": 0.01, "state_id": "C2:abc"}'}])


def test_a_talking_episode_has_not_acted():
    assert acted(_talk_only()) is False, (
        "a reply with no tool call must count as not having acted, or the asked turn never fires")


def test_an_episode_that_read_has_acted():
    assert acted(_read_a_rate()) is True


def test_the_predicate_is_not_vacuous():
    """The control: the two inputs must DISAGREE. A predicate returning the same answer for both -- which
    is exactly what the generator-truthiness bug did -- passes the two tests above only if they happen to
    expect that answer, so the disagreement is the real assertion."""
    assert acted(_talk_only()) != acted(_read_a_rate())


def test_the_ask_names_only_a_no_argument_call():
    """It asks for cw_look, which takes no parameters, and shows the markup. Asking for `cw_move` with an
    id was measured at 0 calls in 23 nudged episodes: move ids are only ever listed by cw_look, so that
    version asked for an argument the policy could not know."""
    assert "cw_look" in NUDGE and "<tool_call>" in NUDGE
    assert "cw_move with the id" not in NUDGE


# --------------------------------------------------------------------------- the loop itself

class _Segment:
    def __init__(self, terminated): self.terminated = terminated


class _Interaction:
    """Records every turn the env asks for. `script` gives each turn's `terminated`."""
    def __init__(self, script, trace):
        self.script, self.trace, self.asks = list(script), trace, []
    async def turn(self, prompt=None):
        self.asks.append(prompt)
        return _Segment(self.script[len(self.asks) - 1] if len(self.asks) <= len(self.script) else False)


class _Agent:
    def __init__(self, interaction): self._i = interaction
    def interaction(self, task):
        agent = self
        class _CM:
            async def __aenter__(self): return agent._i
            async def __aexit__(self, *a): return False
        return _CM()


class _Agents:
    def __init__(self, agent): self.agent = agent


class _Cfg:
    reward = sorted(R.T3_THINK_PROMPT_VARIANTS)[0]


class _Task:
    config = _Cfg()


def _run_loop(script, trace):
    inter = _Interaction(script, trace)
    # `__new__` WITHOUT `__init__`: constructing a real Env needs a seed taskset ("CohortWorldEnv needs
    # a seed taskset -- every rollout starts from one of its tasks"), which would put a 16,770-task load
    # between this test and the four lines it is about.
    #
    # It used to pass `self=None`, on the true-at-the-time claim that `run()` touches no attribute of
    # self. That stopped being true when `_peers` was added for the patient matrix, and all three loop
    # tests broke with `AttributeError: 'NoneType' object has no attribute '_peers'` -- caught by the
    # full suite, not by me. A bare instance costs nothing and cannot go stale the same way.
    env = CohortWorldEnv.__new__(CohortWorldEnv)
    CohortWorldEnv._pool = []          # these tasks carry no patients_per_population, so M == 1 and
                                       # `_peers` returns [] before it ever looks at the pool
    asyncio.run(CohortWorldEnv.run(env, _Task(), _Agents(_Agent(inter))))
    return inter.asks


def test_an_episode_that_only_looked_is_asked_again():
    """THE REGRESSION TEST. Turn 1 talks; the ask fires; turn 2 obeys it with a cw_look and does not
    terminate. The env must ask a SECOND time, because the listing cw_look just returned is the only
    context in which cw_move is ever accepted.

    This is the assertion that fails on the previous code: with `acted()` in the loop condition, the
    cw_look made `acted()` true and the loop returned after one ask."""
    asks = _run_loop([False, False, False], _read_a_rate())
    assert asks[0] is None, "the first turn is the bare opening turn, not an ask"
    real = [a for a in asks if a is not None]
    assert len(real) == 2, (
        f"an episode that looked but did not commit must be asked again; got {len(real)} ask(s). "
        "Under the old `segment.terminated or acted(trace)` condition this is 0.")
    assert real[0] == NUDGE and real[1] == NUDGE_AFTER_LOOK, (
        f"the two asks must be DIFFERENT questions: got {real[0][:40]!r} then {real[1][:40]!r}. "
        "NUDGE's operative sentence asks for cw_look; repeating it asks an episode that has already "
        "looked to look again, which is the state the second ask exists to escape.")
    assert "cw_look" in real[0] and "cw_move" in real[1], (
        "ask 1 must name the call that BUILDS the listing and ask 2 the call that SPENDS it")


def test_a_terminated_episode_is_not_asked():
    """The control for the test above: if it never stopped asking, the test above would pass for the
    wrong reason. A committed (terminated) episode must receive NO ask at all."""
    asks = _run_loop([True, True, True], _read_a_rate())
    assert asks == [None], f"a terminated opening turn must end the loop; got {asks!r}"


def test_the_ask_budget_is_bounded():
    """And it stops: an episode that never terminates is asked exactly NUDGES_PER_EPISODE times, not
    forever. The navigation budget is 26, so 2 asks is affordable; an unbounded loop would not be."""
    asks = _run_loop([False] * 10, _talk_only())
    real = [a for a in asks if a is not None]
    assert len(real) == NUDGES_PER_EPISODE == 2 == len(ASKS), (
        f"asked {len(real)} times; NUDGES_PER_EPISODE={NUDGES_PER_EPISODE}, len(ASKS)={len(ASKS)}. "
        "An unbounded loop would burn the 26-call navigation budget on asks.")
