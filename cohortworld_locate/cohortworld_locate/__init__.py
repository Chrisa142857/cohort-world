"""T1 (locate) on CohortWorld's own cohorts, served as `cohortworld-locate`.

Repointed from `cohortworld_env.taskset.LocateTaskset`, which was the earlier locate-by-rate-matrix
task reading `tasks/` -- files from before the lattice rebuild, where 88% of the cohort keys no longer
resolve. That class is deleted; this is the live locate-by-navigation task.
"""
from cohortworld_env.internal import CohortWorldEnv, LocateTaskset

# The package exports its Env subclass as well as its Taskset: `utils/loaders.py:168-181` resolves a run's
# Env from the taskset's own package and falls back to `SingleAgentEnv` when there is none. Without this
# line T1/T2 ran on `SingleAgentEnv`, whose `run` is exactly `await agents.agent.run(task)`, so the
# two-turn loop in `CohortWorldEnv.run` was unreachable for them however the taskset was configured -- the
# reason a policy trained under that protocol was measured at zero tool calls on both tasks. With
# `think = false`, the default, `CohortWorldEnv.run` takes the same single line, so every archived T1/T2
# score is produced by the same behaviour as before.

__all__ = ["LocateTaskset", "CohortWorldEnv"]
