# CS-GRPO — the training code for CohortWorld

The environment in this repository is trained with **CS-GRPO**, an algorithm that samples an $M \times N$
matrix of rollouts (M patients per population, N populations per group) and pays each rollout its own
contribution to the population's AUROC and calibration slope, rather than a per-rollout scalar. This
directory is the whole of it.

## Why it lives here and not in the trainer

CS-GRPO runs as a plug-in to [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl). The trainer is
upstream code we do not own, so what is vendored here is our algorithm plus the *smallest diff* that
registers it — 304 added lines, of which 3 are the registration itself.

    cs_grpo.py                      the algorithm. Drop into
                                    prime-rl/src/prime_rl/orchestrator/algo/cs_grpo.py
    prime_rl_ef9dea1_cs_grpo.patch  the upstream diff: the CSGRPOAlgoConfig fields, and the 3-line
                                    entry in ALGORITHM_CLASSES. Apply at prime-rl ef9dea1.
    configs/                        the run configuration, with every non-default value commented with
                                    the measurement that set it.

## Reproducing the Qwen3.5-9B run

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl && cd prime-rl
git checkout ef9dea1
cp <this>/cs_grpo.py src/prime_rl/orchestrator/algo/cs_grpo.py
git apply <this>/prime_rl_ef9dea1_cs_grpo.patch
uv run sft @ <this>/configs/cw_sft_qwen3.5-9b_cohort-think.toml   # the cold start
uv run rl  @ <this>/configs/cw_rl_csgrpo_qwen3.5-9b.toml          # then CS-GRPO
```

Training reads the released data through the same five variables the evaluation uses (`CW_PANEL`,
`CW_OFFICIAL`, `CW_SIDECAR`, `CW_ITEMS`, `CW_PHENOCAL`; see the env block in `reproduce/README.md`):
the reward ranks answers against `$CW_ITEMS/t3_rate_train.jsonl` and the reference table in
`$CW_OFFICIAL`, both at run time.

The stages, in order, for each model (each stage starts from the checkpoint the previous one wrote):

| stage | Qwen3.5-9B | Qwen3.6-27B |
|---|---|---|
| supervised cold start | `cw_sft_qwen3.5-9b_cohort-think.toml` | `cw_sft_qwen3.6-27b_cohort-think.toml` |
| CS-GRPO | `cw_rl_csgrpo_qwen3.5-9b.toml` | `cw_rl_csgrpo_qwen3.6-27b.toml` |
| CS-GRPO with psi = 0.3 | `cw_rl_csgrpo_qwen3.5-9b_psi0.3_from-csgrpo-step100.toml` | `cw_rl_csgrpo_qwen3.6-27b_psi0.3_from-csgrpo-step40.toml` |

The cold start reads the demonstrations from `data/sft/cohort_think`. They are not in the public
dataset, because their answers state rate differences, and they are available on request (see
`data/README.md`). The reward's reference table is built under `$CW_OFFICIAL` from the train items'
answers, which the public dataset hides, so training needs the labels, also available on request. To
reproduce the paper's training rewards exactly, set `CW_OFFICIAL` to the directory that holds the table
those runs read (`official/paper_training`, on request as well).

`cs_grpo.py` is self-checking: `python -m prime_rl.orchestrator.algo.cs_grpo` runs `demo()`, which
asserts the credit assignment, the M×N matrix path, the success buffer, the row/group baseline, and the
single-label refusal. Each check was written by first constructing the bug it pins and confirming the
check goes red; the docstrings record which mutation each one catches.

## The pieces, and what each was measured to fix

| piece | what it does | the measurement that put it there |
|---|---|---|
| `_assign_matrix` | scores every population (column) and centres each cell against the same patient's cells in the other populations (row) | a group is N rollouts of ONE task upstream, so a population had to be faked from a rolling context; an early run scored 0 populations in ~600 groups |
| `success_buffer` | past usable cells top up a population that is short of `min_population` | one run read a rate in 1.68% of rollouts, so M·p = 0.27 against a floor of 4, and 12 of 12 groups scored 0 cells |
| group-baseline fallback | a row with one scored cell centres against the group's scored cells | one cell is its own mean, so it received advantage 0; 22 of 160 rows were in that state |
| single-label refusal | `contribution_metric` returns None when no opposite-label member exists | it was silently returning the slope term at double weight — a second scale pooled into the same baseline |
| `credit_turn = "action"` | the ladder lands on the turn carrying the tool call | under `first`, all 800 trainable branches had their nonzero advantages on turn 1, including all 37 that called a tool |

The ladder (`call/move/read`) is written as a **group-centred indicator**, so it pays nothing once every
rollout does the thing. In the Qwen3.5-9B run it extinguished itself: ladder/metric fell from 290x to 1.6x as the read
rate went 1.95% → 99%, and the reward kept rising after it went silent.
