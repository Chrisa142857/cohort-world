# data/

The data are not in this repository. They are on Hugging Face:
**https://huggingface.co/datasets/Ziquan/cohort-world**. The dataset card there describes every
file and the disclosure rules applied.

```bash
hf download Ziquan/cohort-world --repo-type dataset --local-dir data
export CW_PANEL=$PWD/data/official CW_OFFICIAL=$PWD/data/official CW_SIDECAR=$PWD/data/official
export CW_PHENOCAL=$PWD/data/cohortcal CW_ITEMS=$PWD/data/items
```

The incidence labels are hidden: only the 20 cohorts listed in `official/labelled_cohorts.json` keep
their rate tables, and the first 20 rows of the CohortCal train split keep their answers. To evaluate a
model or enter the leaderboard, send its predictions as the dataset card describes. The labels, the
agent traces and the supervised cold-start demonstrations are available on request.

`paper_outputs/` holds the single-turn model outputs behind Tables 3 and 4, one file per published
row, with an `index.json` per table, without the true rates. `reproduce/rescore.sh data/paper_outputs`
prints both tables from them once the labels and the agent traces are in place. The trained
checkpoints are separate model repositories, linked from the dataset card.

`MANIFEST.tsv` lists every released file with its size in bytes and its SHA-256. Check a download
against it with:

```bash
cd data && tail -n +2 MANIFEST.tsv | awk -F'\t' '{print $3"  "$1}' | sha256sum -c
```

`.gitignore` excludes every data format (`*.jsonl` and the rest), so a download into `data/` is never
committed by accident.
