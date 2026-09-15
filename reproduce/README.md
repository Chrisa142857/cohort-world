# Reproducing the paper's tables

Both entry points take an OpenAI-compatible endpoint. Serve the model however you like (we used vLLM
with `--enable-auto-tool-choice` and the model family's tool-call parser), then point the script at it.
Every path is an argument or an environment variable; nothing in the scripts names a machine.

```bash
export CW_PANEL=/path/to/cohorts             # the released cohort collection (official/): lattice, moves, disease tables
export CW_OFFICIAL=$CW_PANEL                 # reference rates and the marginal table, same directory
export CW_SIDECAR=$CW_PANEL                  # CohortCal axes sidecars, same directory
export CW_ITEMS=/path/to/items               # t3_rate_{train,seen,unseen}.jsonl and the T1/T2 files
export CW_PHENOCAL=/path/to/cohortcal        # PhenoCal_C2_{train,val,test}.jsonl
export CW_VLLM_KEY=dummy                     # the client requires the variable to exist
```

## Table 3, CohortCal

One row per model. `single` is the leaderboard protocol every row uses: the estimator system prompt
(`prompts/sys_estimator.txt`), one user turn, temperature 0, a bare percentage parsed from the reply.

```bash
reproduce/table3.sh http://host:8000/v1 qwen3.5-9b Qwen3.5-9B single
```

prints AUROC on `r > b`, the share of answers that beat `b`, and the calibration slope, each with a
2,000-resample bootstrap interval at seed 42, where `b` is the mean recorded rate of the item's
disease and horizon over the CohortCal training split (`--b-source train`, the default). The `agent` mode runs the same items inside
CohortWorld under the two-turn protocol (`configs/eval_cohortcal_agent.toml`) and scores the traces.

## Table 4, the CohortWorld tasks

`agent` runs the 943 unseen CohortThink items under the two-turn protocol and scores the traces three
ways from one file: CM and CC by `gen/score_cm_cc.py`, CT AUROC by `gen/score_ct_auroc.py`, and calls
per item by `gen/count_calls.py`. `single` runs the same 943 questions in one turn with no tools.

```bash
reproduce/table4.sh http://host:8000/v1 qwen3.5-9b Qwen3.5-9B agent
reproduce/table4.sh http://host:8000/v1 qwen3.5-9b Qwen3.5-9B single
```

CM is the share of items whose answer equals the published rate of the item's reference group, the
cohort one step up the coding tree that publishes a rate. CC is the three-way sign agreement between
the difference the answer adds on top of that rate and the recorded one, over the items where the
recorded difference is not zero. CT AUROC uses a different `b` from Table 3, the UK Biobank marginal
incidence of the item's disease and horizon (`gen/thresholds.py`, `--b-source ukb-marginal`).

**CT AUROC reproduces to within about 0.03 of the paper.** Its `b` is the UK Biobank marginal
incidence table. The released copy suppresses every cell in which fewer than ten distinct
participants had an event, a stricter rule than UK Biobank's published minimum of five, and
publishes rates only, rounded to two significant digits; the paper was scored on the unsuppressed
table. Items whose (disease, horizon) falls in a suppressed cell drop out, 930 scored in the paper
against 872 here, and the eight CT values move by up to 0.025 (for example 0.638 becomes 0.620 for
Qwen3.5-9B with CS-GRPO and psi, 0.642 becomes 0.625 for Qwen3.6-27B with CS-GRPO, and 0.621
becomes 0.596 for the untrained Qwen3.6-27B agent). The ordering of the rows is unchanged.
CM, CC and calls per item use no `b` and reproduce exactly. Do not substitute
`reward.base_rate("internal")`: it is the train-split mean of the CohortThink items, the environment's
reward reference, not the UK Biobank marginal this column is defined with.

## What a run leaves behind

Each call writes `outputs/<table>_<label>/` with the resolved eval config, the traces, and the scored
numbers, so any value can be traced to the config and the traces that produced it. Sampling is
temperature 0.6 for the agents and 0 for the single-turn rows; agent rows therefore vary between
runs by about the width of their intervals.
