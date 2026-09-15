"""T3 (rate) on CohortWorld's own cohorts -- the internal half of the design.

A taskset id maps to a same-named importable package that exports its Taskset subclass in `__all__`;
that is why this is a package rather than a module inside `cohortworld_env`, which already answers to
`cohortworld-env`. The implementation lives in `cohortworld_env.internal` so it sits beside the
lattice and toolset it depends on, and so its scoring is literally the same object the external
PhenoCal task uses rather than a second copy that can drift.

**One task per package, not three selected by a field.** `verifiers/v1/utils/loaders.py` refuses a
package exporting more than one Taskset subclass, and that rule exists for a reason this project paid
for: the first generic parameter of `vf.Taskset` fixes the TASK CLASS at runtime, so a taskset that
yielded three classes had two of them silently reconstructed as the third on the served path and
scored by the wrong rubric. T1 is `cohortworld-locate`, T2 is `cohortworld-delta`.
"""
from cohortworld_env.internal import CohortWorldEnv, RateTaskset

# The package exports its Env subclass as well as its Taskset: `utils/loaders.py:168-181` resolves a run's
# Env from the taskset's own package and falls back to `SingleAgentEnv` when there is none. The addition
# is one asked turn when a reply reasons but takes no action -- form A, chosen after form B produced zero
# tool calls in ~190 rollouts. See `CohortWorldEnv`.
__all__ = ["RateTaskset", "CohortWorldEnv"]
