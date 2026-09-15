#!/usr/bin/env bash
# release-gate: allow-file -- ran against the private UKB extract; shipped for provenance, paths shown as they were
# One version, one node. Writes to a STAGING dir so the live official/ is untouched until all
# three succeed -- a half-replaced lattice would leave state ids from two builds side by side.
set -u
V="$1"; W="$2"
CW=/work/USERS/ziquanw/cohortworld
OUT=$CW/rebuild/official_new
PRIV=/work/USERS/ziquanw/cohortworld_private/rebuild_new
mkdir -p "$OUT" "$PRIV/augment"
exec $CW/.venv/bin/python $CW/build/OK__build_cohortworld_env.py \
  --version "$V" --workers "$W" \
  --hx /ram/USERS/ziquanw/digitaltwinreward/ukb_qa_brain/output_panel/cohortworld_hx_alldates.pkl \
  --inter /ram/USERS/ziquanw/digitaltwinreward/ukb_qa_brain/output_panel/panel_intermediate.pkl \
  --out "$OUT" --private "$PRIV" --augment "$PRIV/augment" \
  --multigran 1,2,3
