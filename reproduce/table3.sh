#!/usr/bin/env bash
# Table 3, the CohortCal leaderboard, for ONE model served at an OpenAI-compatible endpoint.
#
#   CW_PHENOCAL=<dir with PhenoCal_C2_*.jsonl> reproduce/table3.sh <base_url> <served-name> <label> [single|agent]
#
# single (default): the one-turn estimator protocol every leaderboard row uses; writes and scores a
#                   record. AUROC / beats b / slope are printed by gen/score_singleturn.py.
# agent:            the "w/ CW" row, the same items answered inside CohortWorld by the untrained model
#                   under the one-turn acting prompt (arm B); runs the verifiers eval on
#                   configs/eval_cohortcal_agent.toml and scores the traces.
set -euo pipefail
CW=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
URL=${1:?base_url}; NAME=${2:?served-name}; LABEL=${3:?label}; MODE=${4:-single}
: "${CW_PHENOCAL:?set CW_PHENOCAL to the CohortCal directory}"
OUT=${CW_OUT:-$CW/outputs}/table3_${LABEL}; mkdir -p "$OUT"
case "$MODE" in
  single)
    python "$CW/gen/singleturn_cohortcal.py" --base-url "$URL" --model "$NAME" --out "$OUT/singleturn.json" \
      --items "$CW_PHENOCAL/PhenoCal_C2_test.jsonl" --conc "${CW_CONC:-64}" ${CW_NO_THINKING:+--no-thinking}
    python "$CW/gen/score_singleturn.py" --rec "$OUT/singleturn.json" --label "$LABEL" ;;
  agent)
    sed -e "s#^base_url = .*#base_url = \"$URL\"#" -e "s#^model = .*#model = \"$NAME\"#" \
        "$CW/configs/eval_cohortcal_agent.toml" > "$OUT/eval_config_resolved.toml"
    export CW_SYS="$CW/prompts/sys_clinician_contrast_v2.txt"   # arm B, the one-turn acting prompt
    ( cd "$OUT" && eval @ "$OUT/eval_config_resolved.toml" -m "$NAME" -c "${CW_CONC:-64}" -o "$OUT/outputs" --no-push )
    python "$CW/gen/score_cohortworld_phenocal.py" --traces "$(find "$OUT/outputs" -name traces.jsonl | head -1)" \
      --model "$LABEL" --out "$OUT/scored" ;;
  *) echo "mode must be single or agent"; exit 1 ;;
esac
