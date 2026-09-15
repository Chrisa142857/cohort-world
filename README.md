# CohortWorld

**Aligning real-world data-driven diagnostics and language models.**

CohortWorld is an interactive environment for language agents built from UK Biobank. Every state is a real, $k$-anonymous cohort keyed on a recorded history of diagnoses (ICD-10), procedures (OPCS-4) and prescriptions (BNF), and every cohort publishes the measured incidence of each disease over each follow-up horizon. An agent estimates one patient's risk by walking to a recorded group, reading its rate, changing one feature at a time and reading again, so its answer is a published number reached by a comparison it has stated rather than a prevalence recalled from pretraining.

Paper: *CohortWorld: Aligning Real-World Data-Driven Diagnostics and Language Models* (under review, ICLR 2027). Code, cohorts and benchmark are being organised for release in this repository; the layout below is the target and will settle over the next weeks.

## What is released

| component | what it is | size |
|---|---|---|
| **CohortWorld environment** | the cohort lattice, four tools, tasks and rewards, served through [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) | 20,180 cohorts at three coding depths |
| **CohortCal** | a calibration benchmark: written histories with a disease and horizon, answered with a percentage, scored against the recorded incidence | 31,784 unseen items over 9,704 cohorts disjoint from CohortWorld |
| **CohortThink** | the diagnostic simulation inside the environment, with CohortMatch and CohortContrast read off its episodes | 16,770 / 880 / 943 items (train / seen / unseen) |
| **CS-GRPO** | the training algorithm, a population reward for a population metric, as a plug-in to [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) | `train/` |
| **Checkpoints** | Qwen3.5-9B and Qwen3.6-27B after the cold start and after CS-GRPO | to follow |

All released cohorts hold at least $k = 10$ participants, sizes are published as bands, and no individual-level record leaves UK Biobank. A pseudo-subject is a history and an index time drawn from a cohort, not a person's record.

## The environment in one screen

An episode starts on a cohort with its rate withheld. Four tools:

```
cw_look     {}                              where you stand and every move this cohort accepts, each with an id
cw_move     {move_id}                       take one listed move: add / drop / switch a code, shift the age band, zoom the coding depth
cw_observe  {disease?, horizon?}            read one published (disease, horizon) rate where you stand
cw_commit   {rates?}                        end the episode; a rate question takes a list of values, a cohort question takes nothing
```

A budget of 26 calls. A group specific enough to match one patient is usually too small to publish a rate, so the skill the environment asks for is how much specificity to give up for a reading worth believing.

Three tasks, all served as questions in words with no cohort identity, size or rate in them:

- **CohortThink** (CT) estimate the probability of a first diagnosis of $D$ within $h$ years for this history. Scored by AUROC on $r > b$ with $b$ the disease's own incidence, plus calibration slope and the share of answers closer to the truth than $b$.
- **CohortMatch** (CM) did the answer land on the level of the reference group, the cohort one step up the coding tree that publishes a rate. Exact match of $\hat r$ against that group's rate.
- **CohortContrast** (CC) did the difference the answer adds on top of that level point the right way. Three-way sign agreement, items with no difference excluded.

CM and CC are properties of the number, so a model with no tools is scored on the same two questions as an agent.

## Quick start

> The commands below describe the intended release layout. Until the reorganisation lands, paths may differ.

```bash
# environment packages
pip install -e cohortworld_env -e cohortworld_internal -e cohortworld_locate -e cohortworld_delta

# serve a model (any OpenAI-compatible endpoint works; we use vLLM)
vllm serve Qwen/Qwen3.5-9B --served-model-name qwen3.5-9b --port 8000 --enable-auto-tool-choice --tool-call-parser qwen3_xml

# evaluate on the CohortThink unseen split
eval @ configs/eval_cohortthink_unseen.toml -m qwen3.5-9b -c 64 -o outputs/

# score
python gen/score_internal_t3.py --traces outputs/<run>/traces.jsonl --label qwen3.5-9b
```

Every path, split, threshold and seed is an argument; nothing is a literal in a script body. A run directory records the resolved config, the git SHA and working-tree state, input hashes and the frozen environment version, so any number in the paper traces back to the directory that produced it.

## Training

`train/` holds CS-GRPO as a plug-in to prime-rl: the algorithm (`cs_grpo.py`), a small upstream patch, and the exact run configurations with every non-default value annotated by the measurement that set it. The recipe is a supervised cold start on 1,845 code-generated `<cohort-think>` demonstrations (the comparison group in each is written by code from the cohort collection, so it is correct by construction), then CS-GRPO, which samples an $M \times N$ matrix of pseudo-subjects and responses and pays each response its own contribution to the population's AUROC and calibration slope. See `train/README.md`.

## Repository layout

TODO, the code and data are being organised for release.

## Ethics and scope

This work uses UK Biobank under an approved application. A number produced here is the incidence of a group a person matches and is not a personal prognosis; it carries UK Biobank's selection and needs recalibration before use on another population. Nothing in this repository is a clinical decision aid.

## Citation

```bibtex
@inproceedings{cohortworld2027,
  title     = {CohortWorld: Aligning Real-World Data-Driven Diagnostics and Language Models},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2027}
}
```

## License

Code: TODO. Cohorts and benchmark: released under the terms permitted by the UK Biobank data use agreement; see `official/LICENSE` (to follow).
