"""The one base rate every benchmark is scored against: the UKB marginal incidence.

A single lookup, shared by PhenoCal and CohortWorld, so a number from one can be put beside a number
from the other. It replaces the per-key mean over items that both used before, which on the internal
set was estimated from a median of ONE item per key.

`version` selects which lattice's row set the marginal was computed over. The three row sets differ
because the index points are defined by that version's history granularity, so an internal item must
be matched to its own version. PhenoCal items belong to no lattice; they use C2, the variant they are
named for.

The released table publishes rates only (2 significant digits); a cell in which fewer than ten
distinct participants had an event is null, and `base_rate` returns None for it, as for a code
that is not in the table. See the table's `meta.release_suppression`.
"""
import json, os

# Same data root as the environment: $CW_OFFICIAL if set, else official/ beside this checkout.
MATRIX = os.environ.get("CW_MARGINAL", os.path.join(
    os.environ.get("CW_OFFICIAL", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "official")),
    "marginal_incidence_landmark_gplinked_3ver.json"))
_M = None


def _load():
    global _M
    if _M is None:
        _M = json.load(open(MATRIX))
    return _M


def base_rate(code, horizon, version="C2"):
    """Marginal incidence of `code` within `horizon` years, or None if the code is not in the panel."""
    M = _load()
    V = M["versions"].get(version)
    if not V:
        return None
    cell = V["rates"].get(str(len(str(code))), {}).get(str(code), {}).get(str(int(horizon)))
    return cell["rate"] if cell else None


def meta():
    return _load()["meta"]
