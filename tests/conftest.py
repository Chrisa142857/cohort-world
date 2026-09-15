"""Repo-relative import paths for the test suite, so a clone runs its own packages and gen/ scripts.

The tests used to insert `<author's checkout>/cohortworld_env` into sys.path, which in any other clone
imported nothing or, worse, another checkout. Everything here is derived from this file's location.
Data-dependent tests (the item freeze and the environment contract) skip with a message when
CW_ITEMS or CW_OFFICIAL is unset; see the guard at the top of each of those files.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("cohortworld_env", "cohortworld_phenocal", "cohortworld_internal", "cohortworld_locate",
            "cohortworld_delta", "gen"):
    p = os.path.join(REPO, sub)
    if p not in sys.path:
        sys.path.insert(0, p)
