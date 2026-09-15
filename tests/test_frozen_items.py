"""The item set is frozen WITH the code, and these four tests are what makes that sentence true.

ENV_VERSION pins the item ROOT: `internal.ITEMS` is inside the hashed semantic surface, so
repointing it moves `semantic_hash()` and `test_the_version_names_the_code_it_ships_with` goes red
until the table records the new version. That pins the NAME. It does not pin the CONTENT -- a rebuild
written back over the same directory changes every answer in the benchmark and moves no hash the
environment checks, which is unrecorded constant drift one level down: the version would be unchanged and
nothing on disk would say the numbers are not comparable.

So the content is pinned here, by md5 and by line count, together with the two properties the release
depends on and that a rebuild is most likely to break quietly: k-anonymity on every published label,
and cohort-disjoint splits.

Each of the four was verified to FAIL by mutating the invariant it names -- a wrong hash, a k of 9, a
cohort moved across two splits, a private cohort planted in train -- because a freeze test that
cannot go red is a comment.
"""
import collections
import hashlib
import json

import os

import pytest

if not os.environ.get("CW_ITEMS"):
    pytest.skip("CW_ITEMS is unset: the item-freeze tests need the released item set (see data/README.md)",
                allow_module_level=True)

from cohortworld_env.internal import ITEMS  # noqa: E402

PUBLISHED = ("t1_locate", "t2_delta", "t3_rate")
SPLITS = ("train", "seen", "unseen")
PRIVATE = ("t1_locate_unseen_subject", "t3_rate_unseen_subject")

HASHES = {
    # file                              md5                                 lines
    "t1_locate_train.jsonl":           ("c1ad6c742de551f09058cf85b4ad7170", 16589),
    "t1_locate_seen.jsonl":            ("7e32b9bc13e450c91c94644aae19b1f9",   870),
    "t1_locate_unseen.jsonl":          ("0d6e6bc1c0f8c7e53d08f7a5375708d2",   919),
    "t1_locate_unseen_subject.jsonl":  ("e85e9f0ac52580636cd1177b527f1359",  4249),
    "t2_delta_train.jsonl":            ("59283cbfd430552ab780204fdeaa787a",   297),
    "t2_delta_seen.jsonl":             ("1388a754cb871c7e66712a0112e31c64",    60),
    "t2_delta_unseen.jsonl":           ("44e29cdd87dc1623d86a15c3c7025309",   108),
    "t3_rate_train.jsonl":             ("0b9d02f9e0adb8f2cc50ae826055416d", 16770),
    "t3_rate_seen.jsonl":              ("24f6e15682abbd997b5d9b9ca12cf7da",   880),
    "t3_rate_unseen.jsonl":            ("ee4b68a9b18bd476563c8a5770d6562c",   943),
    "t3_rate_unseen_subject.jsonl":    ("c82181e4687073587a2844f02c7d79cf",  4249),
}
"""The eleven files, by content. The same eleven md5s fold into one `items_md5` line in every run's
config.yaml, so a run directory names the item set as precisely as this does."""

_CACHE: dict = {}


def rows(name):
    """One item file, parsed once per session -- 79 MB across the eleven files."""
    if name not in _CACHE:
        _CACHE[name] = [json.loads(l) for l in open(f"{ITEMS}/{name}.jsonl")]
    return _CACHE[name]


def cohort(r):
    """The disjointness unit: the released cohort the item was rendered from. T2 renders an EDGE,
    so its rows carry `start_cohort` instead. Same unit either way."""
    return r["source"]["cohort"] or r["source"]["start_cohort"]


def test_the_environment_reads_the_frozen_item_set_and_its_content_is_the_frozen_content():
    """Name and content together. Either half alone is bypassable: `CW_ITEMS` repoints the name
    without touching a file, and a rebuild rewrites the files without touching the name."""
    # Pinned by content, not by path: the path is wherever a clone put the released files. The two
    # individual-level (k = 1) subject files are not released, so their entries are skipped when absent.
    for name, (want_md5, want_n) in HASHES.items():
        if name.startswith(PRIVATE) and not os.path.exists(f"{ITEMS}/{name}"):
            continue
        blob = open(f"{ITEMS}/{name}", "rb").read()
        got = hashlib.md5(blob).hexdigest()
        assert got == want_md5, f"{name}: md5 {got}, frozen at {want_md5} -- the item set was rebuilt"
        assert blob.count(b"\n") == want_n, f"{name}: {blob.count(chr(10).encode())} lines, frozen at {want_n}"


@pytest.mark.parametrize("task", PUBLISHED)
def test_every_published_label_is_k_anonymous(task):
    """k >= 10 on every label of every PUBLISHED item -- the release rule the whole rendering spirit
    exists to keep. The private `unseen_subject` files
    are excluded by construction, not by exception: they carry k=1 because each row IS one person,
    which is exactly why they are never published.

    A missing `k` fails as loudly as a small one. The check reduces over `min()`, and a `None` that
    slipped through as "no cell counted" would make the whole file vacuous."""
    for split in SPLITS:
        ks = [r["labels"].get("k") for r in rows(f"{task}_{split}")]
        assert ks and all(isinstance(k, int) for k in ks), f"{task}/{split}: a label carries no k"
        assert min(ks) >= 10, f"{task}/{split}: min k = {min(ks)}, below the release floor of 10"


def test_the_published_splits_are_cohort_disjoint():
    """Profile disjointness, the only holdout the published splits claim.
    Measured 0 on every pair of every task; asserted as 0, not as "small", because the split
    function hashes the cohort's own state_id, so any overlap at all is a broken split function
    rather than a tolerable rate."""
    for task in PUBLISHED:
        sets = {s: {cohort(r) for r in rows(f"{task}_{s}")} for s in SPLITS}
        for a, b in (("train", "seen"), ("train", "unseen"), ("seen", "unseen")):
            assert not sets[a] & sets[b], \
                f"{task}: {len(sets[a] & sets[b])} cohorts in both {a} and {b}"


@pytest.mark.skipif(not os.path.exists(f"{ITEMS}/t3_rate_unseen_subject.jsonl"),
                    reason="the individual-level subject split is not released")
def test_the_private_subject_split_touches_no_training_cohort():
    """The private split is the performance-reporting holdout, so its observations must not be
    describable by a cohort the agent could have been trained on. Its cohorts land in `unseen`
    (1610 of them) and in NEITHER train nor seen -- that is the invariant, and the unseen overlap
    is the design, not a leak.

    Also pinned: one row per (subject, index_year) per file, and T1's observations are exactly
    T3's, so the two private files score the same 3,990 people."""
    train_seen = {cohort(r) for t in PUBLISHED for s in ("train", "seen") for r in rows(f"{t}_{s}")}
    obs = {}
    for name in PRIVATE:
        rs = rows(name)
        coh = {r["source"]["cohort"] for r in rs}
        assert not coh & train_seen, f"{name}: {len(coh & train_seen)} cohorts are train/seen cohorts"
        assert all(r["source"]["private"]["private"] for r in rs), f"{name}: a row is not marked private"
        obs[name] = collections.Counter(r["item_id"].split(":")[2] for r in rs)
        assert max(obs[name].values()) == 1, f"{name}: an observation appears twice"
    assert set(obs[PRIVATE[0]]) == set(obs[PRIVATE[1]]), "T1 and T3 score different people"
