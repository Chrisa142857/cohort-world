# data/

The data are not in this repository. They are on Hugging Face:
**https://huggingface.co/datasets/Chrisa142857/cohort-world**. The dataset card there describes every
file and the disclosure rules applied.

```bash
huggingface-cli download Chrisa142857/cohort-world --repo-type dataset --local-dir data
export CW_PANEL=$PWD/data/official CW_OFFICIAL=$PWD/data/official CW_SIDECAR=$PWD/data/official
export CW_PHENOCAL=$PWD/data/cohortcal CW_ITEMS=$PWD/data/items
```

`MANIFEST.tsv` lists every released file with its size in bytes and its SHA-256. Check a download
against it with:

```bash
cd data && tail -n +2 MANIFEST.tsv | awk -F'\t' '{print $3"  "$1}' | sha256sum -c
```

`.gitignore` excludes every data format (`*.jsonl` and the rest), so a download into `data/` is never
committed by accident.
