#!/usr/bin/env bash
# release-gate: allow-file -- ran against the private UKB extract; shipped for provenance, paths shown as they were
# One version, one node. Writes to a STAGING dir so the live official/ is untouched until all
# three succeed -- a half-replaced lattice would leave state ids from two builds side by side.
set -u
V="$1"; W="$2"
# Roots are arguments, never literals. ${X:?} stops with the name of what is missing, which is both
# safer than a wrong default and the only form that does not publish the machine it ran on.
CW=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)          # the repository root, derived
SRC=${CW_PANEL_SRC:?set CW_PANEL_SRC to the directory holding the panel pickles}
PRIV=${CW_PRIVATE:?set CW_PRIVATE to the private store}/rebuild_new
OUT=${CW_OUT:-$CW/rebuild/official_new}
PY=${CW_PYTHON:-$CW/.venv/bin/python}
mkdir -p "$OUT" "$PRIV/augment"
exec "$PY" "$CW/build/OK__build_cohortworld_env.py" \
  --version "$V" --workers "$W" \
  --hx "$SRC/cohortworld_hx_alldates.pkl" \
  --inter "$SRC/panel_intermediate.pkl" \
  --out "$OUT" --private "$PRIV" --augment "$PRIV/augment" \
  --multigran 1,2,3
