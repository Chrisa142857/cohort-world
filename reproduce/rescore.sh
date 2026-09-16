#!/usr/bin/env bash
# Re-score the paper's own model outputs, no model needed. Prints Table 3 and Table 4 from the
# records under paper_outputs/ of the dataset. CT AUROC lands within about 0.03 of the paper for
# the reason given in reproduce/README.md (the released marginal table suppresses small cells);
# every other number reproduces exactly.
#
#   CW_OFFICIAL=data/official CW_PHENOCAL=data/cohortcal CW_ITEMS=data/items \
#   reproduce/rescore.sh data/paper_outputs
set -euo pipefail
CW=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
P=${1:?path to paper_outputs}
: "${CW_OFFICIAL:?}" "${CW_PHENOCAL:?}" "${CW_ITEMS:?}"
export CW_OFFICIAL CW_PHENOCAL CW_ITEMS CW_PANEL="${CW_PANEL:-$CW_OFFICIAL}" CW_SIDECAR="${CW_SIDECAR:-$CW_OFFICIAL}"

echo "== Table 3, CohortCal (b = CohortCal train-split mean)"
for r in qwen3.5-2b qwen3.5-4b qwen3.5-9b qwen3.5-27b gemma3-4b gemma3-27b medgemma4b medgemma27b llama8b mistral24b claude-sonnet-5 gemini-3.5-flash-lite; do
  python "$CW/gen/score_singleturn.py" --rec "$P/table3/$r.singleturn.json" --label "$r" --b-source train
done
python "$CW/gen/score_cohortworld_phenocal.py" --traces "$P/table3/qwen3.5-9b_cohortworld-agent.traces.jsonl" \
  --model "qwen3.5-9b-with-CohortWorld" --b-source train --out "${CW_OUT:-$CW/outputs}/rescore_table3_agent"

echo; echo "== Table 4, CohortWorld tasks (CM, CC, calls exact; CT within ~0.03, see README)"
for m in qwen3.5-9b qwen3.6-27b; do
  python "$CW/gen/score_cm_cc.py"    --singleturn "$P/table4/${m}_no-tools.singleturn.jsonl"
  python "$CW/gen/score_ct_auroc.py" --singleturn "$P/table4/${m}_no-tools.singleturn.jsonl"
  for r in agent csgrpo csgrpo_psi; do
    f="$P/table4/${m}_$r.traces.jsonl"
    python "$CW/gen/score_cm_cc.py"    --traces "$f"
    python "$CW/gen/score_ct_auroc.py" --traces "$f"
    python "$CW/gen/count_calls.py"    --traces "$f"
  done
done
