"""The hashed environment modules must not name a machine.

Every path that could differ between two installations is an argument (CW_ITEMS, CW_PHENOCAL,
CW_OFFICIAL, ...). A literal lab path in a hashed module would ship in the public release and would
make an unset variable fail on a path that names our cluster. This test reads the same normalised
source `semantic_hash()` reads, so a literal hidden in code (not in a comment or docstring) is caught.
"""
import os, re
from cohortworld_env import reward as R


def test_no_lab_paths_in_hashed_source():
    src = R._normalise(R._semantic_source())
    for pat in (r"/work/USERS/", r"/ram/USERS/", r"/export_home/", r"cohortworld_private"):  # release-gate: allow (the patterns under test)
        assert not re.search(pat, src), f"{pat} appears in a hashed module"


def test_version_is_registered():
    assert R.VERSIONED_SEMANTICS.get(R.ENV_VERSION) == R.semantic_hash()


def test_unset_items_fails_loudly_not_on_a_lab_path():
    """Checked in a fresh interpreter: reloading `internal` in-process swaps its task classes and
    leaves the placeholder behind for every test that follows."""
    import os, subprocess, sys
    env = {k: v for k, v in os.environ.items() if k != "CW_ITEMS"}
    out = subprocess.run([sys.executable, "-c", "from cohortworld_env import internal; print(internal.ITEMS)"],
                         env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert "CW_ITEMS" in out and "/" not in out
