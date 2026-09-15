"""CohortWorld core: the lattice, the toolset, the pinned reward contract, and the three tasksets.

This file imports NOTHING. verifiers starts the tool server with `python -m
cohortworld_env.toolset`, which imports this package first; an eager `from .toolset import ...` here
would leave that module in `sys.modules` before runpy executes it as `__main__`, loading it twice so
the two copies would not share state. Import the submodules directly:

    from cohortworld_env.lattice import Lattice
    from cohortworld_env.internal import RateTaskset, LocateTaskset, DeltaTaskset
    from cohortworld_env.reward import ALPHA, CONSTANTS, crps
    from cohortworld_env.toolset import CohortWorldToolset
"""
