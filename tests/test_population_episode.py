"""One episode answers M patients, so one episode IS one population.

The design this pins: a dataset row stays ONE patient. `CohortWorldEnv.run` draws M-1 further patients
from the taskset at ROLLOUT TIME and answers all M concurrently in the same episode. A group of N such
episodes is then a true M x N matrix, entirely present in `Algorithm.score_group` -- without a dataset
change (item_id keeps its meaning), without a dispatcher change, and without a curriculum change
(`curriculum/base.py:57-58` raises on a group with multiple task keys, and the group still has one).

It is verifiers' own pattern: `environments/proposer_solver/proposer_solver/taskset.py:144-149` fans one
seat out to `self.config.n` concurrent runs of a task minted inside `run()`.
"""
import asyncio
import sys
# import path set by tests/conftest.py (repo-relative)
from cohortworld_env.internal import CohortWorldEnv, NUDGES_PER_EPISODE
from cohortworld_env import reward as R


class _Seg:
    terminated = True          # every answer ends on its opening turn: we are counting traces, not turns


class _Interaction:
    def __init__(self, task, log): self.task, self.log, self.trace = task, log, None
    async def turn(self, prompt=None):
        self.log.append(self.task)
        return _Seg()
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False


class _Agent:
    def __init__(self, log): self.log, self.ran = log, []
    def interaction(self, task): return _Interaction(task, self.log)
    async def run(self, task): self.ran.append(task)


class _Agents:
    def __init__(self, agent): self.agent = agent


class _Cfg:
    def __init__(self, m): self.reward, self.patients_per_population = sorted(R.T3_THINK_PROMPT_VARIANTS)[0], m


class _Data:
    def __init__(self, i): self.item_id = f"item-{i}"


class _Task:
    def __init__(self, i, m): self.key, self.data, self.config = f"k{i}", _Data(i), _Cfg(m)


def _run(m, pool_size=64):
    env = CohortWorldEnv.__new__(CohortWorldEnv)          # no seed taskset needed: we inject the pool
    CohortWorldEnv._pool = [_Task(i, m) for i in range(pool_size)]
    log = []
    agent = _Agent(log)
    asyncio.run(CohortWorldEnv.run(env, _Task(999, m), _Agents(agent)))
    return log, agent


def test_M_equals_1_answers_exactly_one_patient():
    """The historical shape, which every archived run used. It must not move."""
    log, _ = _run(1)
    assert len({t.key for t in log}) == 1, f"M=1 answered {len({t.key for t in log})} patients"
    assert log[0].key == "k999", "M=1 must answer the DISPATCHED patient, not a drawn one"


def test_M_greater_than_1_answers_M_distinct_patients_in_one_episode():
    """THE POINT. One episode, M distinct patients, the anchor among them."""
    for m in (2, 8, 16):
        log, _ = _run(m)
        keys = {t.key for t in log}
        assert len(keys) == m, f"M={m}: episode answered {len(keys)} distinct patients, expected {m}"
        assert "k999" in keys, f"M={m}: the dispatched patient is not in its own population"


def test_the_population_is_reproducible_from_the_anchor():
    """A population drawn from an unseeded RNG would make the metric unreproducible on replay -- the one
    property a scored artifact cannot lose. Two runs of the same anchor must draw the same peers."""
    a, _ = _run(8)
    b, _ = _run(8)
    assert {t.key for t in a} == {t.key for t in b}, "the same anchor drew a different population"


def test_it_is_not_vacuous():
    """The control. If `_peers` silently returned nothing, every test above still passes at M=1 and the
    M>1 test would fail -- but if it returned the SAME task M times, the distinct-count test would catch
    it and this one states why: the peers must differ from the anchor."""
    log, _ = _run(8)
    assert len([t for t in log if t.key == "k999"]) == 1, (
        "the anchor was answered more than once; the peers are copies, not other patients")


def test_a_non_think_variant_bypasses_the_fan_out():
    """Every other reward variant keeps the loop its archived scores were produced with."""
    env = CohortWorldEnv.__new__(CohortWorldEnv)
    CohortWorldEnv._pool = [_Task(i, 8) for i in range(8)]
    task = _Task(1, 8); task.config.reward = "not-a-think-variant"
    agent = _Agent([])
    asyncio.run(CohortWorldEnv.run(env, task, _Agents(agent)))
    assert agent.ran == [task] and agent.log == [], (
        "a non-think variant must go straight to agents.agent.run, with no fan-out and no asks")


def test_the_setup_override_matches_the_base_signature():
    """The check that was missing, and the reason a first launch of the fan-out died.

    `Env.setup` is `(self, agents: Agents)` and `env.py:271` calls `await self.setup(agents)`. An override
    taking only `self` raises TypeError on EVERY episode: 40 episodes, 0 traces, 0 requests to vLLM, and
    the run died on "10 consecutive zero-output batch equivalents" four minutes in. The pre-launch check
    asserted only that the override was a coroutine function -- true of both the right and the wrong
    signature -- so it could not fail. Comparing arity against the base is what discriminates.
    """
    import inspect
    import verifiers.v1 as vf
    base = inspect.signature(vf.Env.setup)
    ours = inspect.signature(CohortWorldEnv.setup)
    assert list(ours.parameters) == list(base.parameters), (
        f"setup override takes {list(ours.parameters)} but the base takes {list(base.parameters)}; "
        f"env.py:271 calls `await self.setup(agents)`, so a mismatch kills every episode")
    assert inspect.iscoroutinefunction(CohortWorldEnv.setup), "setup must stay a coroutine"
