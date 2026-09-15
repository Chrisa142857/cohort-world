"""T2 (delta) on CohortWorld's own cohorts, served as `cohortworld-delta`.

A new package rather than a reuse of `cohortworld-lower`: `t3_lower` is the task that was DELETED,
not T2, and a shim whose name says one task while it serves another is how the previous layout got
into trouble in the first place.
"""
from cohortworld_env.internal import CohortWorldEnv, DeltaTaskset

# The package exports its Env subclass as well as its Taskset: `utils/loaders.py:168-181` resolves a run's
# Env from the taskset's own package and falls back to `SingleAgentEnv` when there is none. Without this
# line T1/T2 ran on `SingleAgentEnv`, whose `run` is exactly `await agents.agent.run(task)`, so the
# two-turn loop in `CohortWorldEnv.run` was unreachable for them however the taskset was configured -- the
# reason a policy trained under that protocol was measured at zero tool calls on both tasks. With
# `think = false`, the default, `CohortWorldEnv.run` takes the same single line, so every archived T1/T2
# score is produced by the same behaviour as before.

__all__ = ["DeltaTaskset", "CohortWorldEnv"]
