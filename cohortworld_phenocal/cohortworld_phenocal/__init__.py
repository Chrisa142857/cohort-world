"""PhenoCal questions posed inside the CohortWorld environment.

This is the PUBLIC task layer. Patient-level items carry individual histories and cannot be
released; PhenoCal's 238k items are already k-anonymous and published, so anyone can train and
evaluate on CohortWorld with these even without UK Biobank access.

Scores are directly comparable to the PhenoCal leaderboard: same questions, same ground truth,
evaluated on PhenoCal's own official profile-disjoint test split. What changes is how the model may
answer -- instead of a single-turn guess it can navigate the lattice, read neighbouring cohorts and
reason from them.
"""
from cohortworld_env.internal import CohortWorldEnv

from .taskset import PhenoCalTaskset

__all__ = ["PhenoCalTaskset", "CohortWorldEnv"]
"""`CohortWorldEnv` is re-exported so verifiers can FIND it.

`verifiers/v1/utils/loaders.environment_class` resolves the Env from the TASKSET's package via
`_plugin_class`, which reads `__all__`. `cohortworld_env/__init__.py` deliberately imports nothing (the
tool server runs `python -m cohortworld_env.toolset`, and an eager import there would load that module
twice with unshared state), so it has no `__all__`; `_plugin_class` raises AttributeError and
`environment_class` SWALLOWS it -- `except (ModuleNotFoundError, TypeError, AttributeError): return
SingleAgentEnv`. Every eval was therefore running the default single-turn Env, silently, while training
(which loads its Env itself) got the two-stage nudge loop: 301 of 304 sampled training traces from one run
carry a NUDGE user turn.

The cost of that silence was two full external cells. On PhenoCal items the policy writes its
<cohort-think> block and calls no tool, so SingleAgentEnv ends the episode; 2 x 31,784 rollouts came back
100% `floor` with both arms at 0.3422 -- the floor constant -- and 0 `cw_commit` in 300 sampled traces
while 271 had already written a number in prose. Internal evals were unaffected only because the policy
calls a tool in turn 1 there, so the default loop continues on its own.

Exported HERE rather than from `cohortworld_env` because this package already imports at module level and
is not bound by that file's no-import rule. `_plugin_class` requires exactly one Env subclass in
`__all__`; `PhenoCalTaskset` is a Taskset, so this stays unambiguous."""
