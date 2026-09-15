"""The concurrency budget check, and the launch it would have refused.

A training run was launched with max_inflight 128 and patients_per_population 16. The TRACE arithmetic had been
checked and was right; the CONCURRENCY arithmetic had not been made at all, because `batch_size` counts
traces while `max_inflight` counts episodes and M multiplies only the second. 381 live tool servers,
MemAvailable 19 GiB of 2015, on a shared node.
"""
import sys
# import path set by tests/conftest.py (repo-relative)
from preflight_concurrency import check, concurrent_answers, available_gib


def _cfg(inflight, m):
    return {"orchestrator": {"concurrency": {"max_inflight": inflight},
                             "train": {"source": [{"env": {"taskset": {"task": {
                                 "patients_per_population": m}}}}]}}}


def test_it_refuses_the_launch_that_actually_happened():
    """THE REGRESSION. This is the configuration that was launched and had to be killed."""
    ok, msg = check(_cfg(128, 16), avail_gib=1852, per_server_gib=2.63, reserve_gib=400)
    assert not ok, f"the check passed the configuration that nearly OOMed the node: {msg}"
    assert "2048 concurrent" in msg, msg


def test_it_admits_the_corrected_one():
    """The control. A check that refuses everything is not a check -- it must pass the fix."""
    ok, msg = check(_cfg(8, 16), avail_gib=1852, per_server_gib=2.63, reserve_gib=400)
    assert ok, f"the check refuses the corrected configuration too: {msg}"


def test_M_is_what_multiplies_it():
    """The two numbers were each individually fine. Only their product is the hazard, so the check has
    to be on the product -- with M=1 the SAME max_inflight is admitted."""
    assert check(_cfg(128, 1), avail_gib=1852, per_server_gib=2.63, reserve_gib=400)[0], (
        "max_inflight 128 fits at M=1; the check must not refuse it")
    assert not check(_cfg(128, 16), avail_gib=1852, per_server_gib=2.63, reserve_gib=400)[0]


def test_M_defaults_to_one_for_every_config_that_does_not_set_it():
    """An env that does not fan out answers one patient per episode. A config with no
    patients_per_population must be read as M=1, not as missing-and-therefore-unchecked."""
    cfg = {"orchestrator": {"concurrency": {"max_inflight": 128}, "train": {"source": [{}]}}}
    inflight, m, answers = concurrent_answers(cfg)
    assert (inflight, m, answers) == (128, 1, 128)


def test_available_memory_is_read_not_restated():
    """The value a run is admitted against must come from the machine, or the check certifies a node
    that no longer exists."""
    assert 1.0 < available_gib() < 100000.0
