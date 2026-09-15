"""The granular rate matrices are opt-in, and the ONE path that needs them still gets them.

Why this file exists. rates_g1/g2/g3 are 88.4% of a state's bytes and 8.3x of the parsed lattice
(2.40 GB -> 0.29 GB for C2). A tool server never reads them, and there were 128 tool servers holding
548.7 GiB while vLLM sat at 0-2 requests with an empty queue. So they are loaded only on request.

The hazard is not the saving, it is HOW THE MISTAKE WOULD LOOK. `estimable_cells` reads
`state.get(f"rates_g{granularity}") or state.get("rates")`. If the granular matrices are missing the
`or` silently substitutes the ungranular one: the cells drawn are wrong, no exception is raised, and
nothing in a log says so. The assertion below is that the taskset path still receives them.
"""
import os
import sys
# import path set by tests/conftest.py (repo-relative)
from cohortworld_env.lattice import Lattice, _shared


def _any_state(lat):
    return next(iter(lat.states.values()))


def test_the_default_drops_them():
    lat = Lattice("C2")
    st = _any_state(lat)
    assert "rates" in st, "the ungranular matrix is what a tool server observes with; it must stay"
    for f in Lattice.GRANULAR_FIELDS:
        assert f not in st, f"{f} was loaded by default; the tool server pays 8.3x for a field it never reads"


def test_granular_true_keeps_them():
    lat = Lattice("C2", granular=True)
    st = _any_state(lat)
    for f in Lattice.GRANULAR_FIELDS:
        assert f in st, f"{f} is missing with granular=True; draw_cells would silently use `rates`"


def test_shared_caches_the_two_shapes_separately():
    """The control that matters most: one cache keyed only by version would hand a tool server the
    granular lattice (no saving) or hand draw_cells the stripped one (silently wrong cells), depending
    on which asked first. The key must include the shape."""
    a = _shared("C2")
    b = _shared("C2", granular=True)
    assert a is not b, "_shared returned the same object for both shapes; the cache key ignores `granular`"
    assert all(f not in _any_state(a) for f in Lattice.GRANULAR_FIELDS)
    assert all(f in _any_state(b) for f in Lattice.GRANULAR_FIELDS)


def test_the_draw_path_asks_for_them():
    """Read the call site rather than trusting the comment: the one runtime reader of rates_g must be
    the one that passes granular=True."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cohortworld_env", "cohortworld_env", "internal.py")).read()
    i = src.index("def _drawn_cells(")
    body = src[i:i + 2000]
    assert "_shared(d.get(\"version\"), granular=True)" in body, (
        "_drawn_cells no longer asks for the granular lattice; estimable_cells' `or` fallback would "
        "then draw every cell from the ungranular matrix without raising")


def test_the_cut_yields_exactly_what_the_full_parse_would():
    """Truncating the JSON text before `rates_g1` must be indistinguishable from parsing it all and
    popping the three fields. If the builder ever reorders its output, this is what notices: the cut
    would otherwise silently drop `rates` or `not_estimable` and every observe would return None.
    """
    import json, sys
    # import path set by tests/conftest.py (repo-relative)
    from cohortworld_env.lattice import PANEL, _CUT, Lattice
    line = open(f"{PANEL}/CohortWorld_C2_train_states.jsonl").readline()
    full = json.loads(line)
    for f in Lattice.GRANULAR_FIELDS:
        full.pop(f, None)
    i = line.find(_CUT)
    assert i > 0, f"the cut marker {_CUT!r} is not in a real state line; the panel's field order moved"
    cut = json.loads(line[:i] + "}")
    assert cut == full, "the truncated parse differs from the full parse minus the granular fields"
    for need in ("state_id", "axes", "coarseness", "n_band", "rates", "not_estimable"):
        assert need in cut, f"the cut dropped {need!r}, which the tool server reads"


def test_a_line_without_the_marker_is_parsed_whole():
    """The fallback. A panel whose fields are ordered differently must be parsed in full, not cut into
    invalid JSON -- `find` returns -1 and the guard is `> 0`."""
    import json, sys
    # import path set by tests/conftest.py (repo-relative)
    from cohortworld_env.lattice import _CUT
    line = json.dumps({"state_id": "x", "rates": [[0.1]], "axes": {}, "coarseness": 1, "n_band": "a"})
    assert line.find(_CUT) == -1, "this fixture is supposed to lack the marker"
    assert json.loads(line)["rates"] == [[0.1]], "a marker-less line must survive untouched"


def test_the_lattice_still_observes_the_same_rate():
    """End to end: the number `cw_observe` publishes must not move. Same state, same cell, cut vs full."""
    import sys
    # import path set by tests/conftest.py (repo-relative)
    from cohortworld_env.lattice import Lattice
    lean, full = Lattice("C2"), Lattice("C2", granular=True)
    sid = next(iter(lean.states))
    a, b = lean.states[sid]["rates"], full.states[sid]["rates"]
    assert a == b, "the cut changed the rate matrix the agent reads"
