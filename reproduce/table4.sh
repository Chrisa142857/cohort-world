#!/usr/bin/env bash
# Table 4, the CohortWorld tasks, for ONE model served at an OpenAI-compatible endpoint.
#
#   CW_ITEMS=<item dir> reproduce/table4.sh <base_url> <served-name> <label> [agent|single]
#
# agent (default): the two-turn CohortThink protocol on the 943 unseen items (verifiers eval on
#                  configs/eval_cohortthink_unseen.toml). Scored three ways from the same traces:
#                  CM and CC by gen/score_cm_cc.py, CT AUROC by gen/score_ct_auroc.py (UKB marginal b), and calls
#                  per item by gen/count_calls.py.
# single:          the no-tool row, the same 943 questions answered in one turn with the estimator
#                  prompt (gen/singleturn_t3.py), scored by gen/score_cm_cc.py and gen/score_ct_auroc.py.
set -euo pipefail
CW=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
URL=${1:?base_url}; NAME=${2:?served-name}; LABEL=${3:?label}; MODE=${4:-agent}
: "${CW_ITEMS:?set CW_ITEMS to the item directory}"
export CW_ITEMS
OUT=${CW_OUT:-$CW/outputs}/table4_${LABEL}; mkdir -p "$OUT"
case "$MODE" in
  agent)
    sed -e "s#^base_url = .*#base_url = \"$URL\"#" -e "s#^model = .*#model = \"$NAME\"#" \
        "$CW/configs/eval_cohortthink_unseen.toml" > "$OUT/eval_config_resolved.toml"
    ( cd "$OUT" && eval @ "$OUT/eval_config_resolved.toml" -m "$NAME" -c "${CW_CONC:-64}" -o "$OUT/outputs" --no-push )
    T=$(find "$OUT/outputs" -name traces.jsonl | head -1)
    python "$CW/gen/score_cm_cc.py" --experiments "$(dirname "$OUT")" --run "$(basename "$OUT")"
    # CT AUROC: the released marginal table suppresses cells with fewer than ten participants, so 58
    # of the 943 items drop and the value lands within about 0.03 of the paper (see reproduce/README.md).
    python "$CW/gen/score_ct_auroc.py" --experiments "$(dirname "$OUT")" --run "$(basename "$OUT")"
    python "$CW/gen/count_calls.py" --traces "$T" ;;
  single)
    H=${URL#http://}; H=${H%%/*}
    python "$CW/gen/singleturn_t3.py" --host "${H%%:*}" --port "${H##*:}" --model "$NAME" --out "$OUT/singleturn.jsonl" --conc "${CW_CONC:-64}"
    python "$CW/gen/score_cm_cc.py" --singleturn "$OUT/singleturn.jsonl"
    python "$CW/gen/score_ct_auroc.py" --singleturn "$OUT/singleturn.jsonl" ;;   # same caveat as above
  *) echo "mode must be agent or single"; exit 1 ;;
esac
