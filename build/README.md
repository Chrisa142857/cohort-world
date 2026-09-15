# build/ — how the released cohorts and items were made

These scripts produced the data under `official/` and the task item set. They are here so a reader can
see exactly how every released file was computed. **They cannot be run from this repository.** They
read the UK Biobank extract (primary-care and hospital records) and a private store of exact cohort
counts, which only a UK Biobank application approved for this purpose can access. The release data
is their output, after the disclosure rules below; nothing here is needed to use the environment, the
benchmark or the training code.

Paths inside the scripts are left as they were when they ran, so what each one read is visible. Each
file that names a private location carries a `release-gate: allow-file` marker for the release check.

| script | writes |
|---|---|
| `rebuild_run.sh` | driver: one version (C1, C2, default) of the lattice build per call |
| `OK__build_cohortworld_env.py` (+ `OK__build_cohortworld_v2.py`, `panel_common.py`) | `CohortWorld_{version}_{train,val,test}_states.jsonl`, `_manifest.json`, `_transitions.jsonl`, `_env_stats.json`; exact counts to the private store |
| `OK__cohortworld_vocab.py` | `CohortWorld_{version}_vocab.json`: the legal codes for add / switch, from the code systems |
| `build_atrisk.py` | `CohortWorld_{version}_atrisk_bands.jsonl`: per-cell at-risk sizes, banded |
| `build_edges_pooled.py` | `CohortWorld_pooled_edges.jsonl`: edges over the three versions as one lattice (build-only, not released) |
| `build_legal_moves.py` | `CohortWorld_legal_moves.jsonl`: the moves each cohort accepts |
| `render_spirit.py` (+ `spirit_dials.py`, `t2_compose.py`, `t2_spirit.py`, `t2_ceiling.py`, `t3_render.py`) | the T1 / T2 / T3 item files (train / seen / unseen) |
| `make_cohort_think_demos.py` (+ `code_descriptions.py`, `demo_quality.py`) | the supervised cold-start demonstrations used by `train/` |

Disclosure rules applied to everything released: every cohort holds at least k = 10 participants,
sizes are published as bands, and no individual-level record or exact private count leaves UK Biobank.
Item files at the individual level (k = 1) are not released.
