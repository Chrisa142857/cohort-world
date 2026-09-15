"""The two percentage parsers the CohortCal leaderboard was scored with, vendored from the
`clinical.f2_eval` module of the private analysis repository so the release scores without it.

`parse_pct` is the board's parser: the FIRST number anywhere in the reply, after special tokens are
stripped, with no range check. `parse_last_pct` is for models that reason before answering: the
LAST percentage after the reasoning block, range-checked to a probability. On a reply that carries a
reasoning block the two disagree, so the choice of parser is part of the measurement and both are
recorded side by side.
"""
import re

import numpy as np


def parse_pct(t):
    t = re.sub(r"<[^>]*>", " ", t or "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*%?", t)
    return float(m.group(1)) / 100.0 if m else None


def _prob_or_none(v):
    return v if 0.0 <= v <= 1.0 else None


def parse_last_pct(t):
    t = t or ""
    if "</think>" in t:
        t = t.split("</think>")[-1]
    if "assistantfinal" in t:
        t = t.split("assistantfinal")[-1]
    t = re.sub(r"<[^>]*>", " ", t)
    ms = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*%", t)
    if ms:
        return _prob_or_none(float(ms[-1]) / 100.0)
    ms = re.findall(r"\b([0-9]+(?:\.[0-9]+)?)\b", t)
    return _prob_or_none(float(ms[-1]) / 100.0) if ms else None


def ece(p, o, bins=10):
    """Expected calibration error, equal-width bins, top bin closed on the right so a prediction of
    exactly 1.0 is counted. Vendored with the parsers so gen/score_cohortworld_phenocal.py keeps its
    artifact schema without the private analysis repository."""
    p, o = np.array(p), np.array(o); e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (p >= lo) & ((p <= hi) if b == bins - 1 else (p < hi))
        if m.sum():
            e += m.mean() * abs(p[m].mean() - o[m].mean())
    return e
