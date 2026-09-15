"""The selectable T3 reward form: default byte-identical, variants as specified.

Every number below is read off `rate_reward` on a stub trace, never recomputed beside it (see
`test_contract.stub_commit`). The default-equivalence claim on real traces was checked by an offline
replay; here it is the formula that is pinned.
"""
import json

import pytest

import cohortworld_env.reward as R
from cohortworld_env.internal import ITEMS, RateConfig, RateData, RateTask, RateTaskConfig
from tests.test_contract import stub_commit


def _fixture_item():
    """The first C2 train item whose ENTRY cohort publishes the asked cell AND whose truth is off the
    base rate, so the side term and the entry reference are both exercised."""
    with open(f"{ITEMS}/t3_rate_train.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["version"] != "C2":
                continue
            who = dict(version=d["version"], age_band=d["age_band"], sex=d["sex"])
            dis, hor = d["disease"], int(d["horizon"])
            if R.entry_rate(disease=dis, horizon=hor, **who) is None:
                continue
            if abs(float(d["answer_rate"]) - R.base_rate("internal", dis, hor)) < 1e-6:
                continue
            return who, dis, hor, float(d["answer_rate"])
    pytest.skip("no C2 train item with a published entry cell")


def test_default_is_the_contract_plus_one_metric():
    who, dis, hor, y = _fixture_item()
    t = stub_commit([0.01, 0.02, 0.03, 0.04, 0.05])
    plain = R.rate_reward(t, truth=y, source="internal", disease=dis, horizon=hor)
    named = R.rate_reward(t, truth=y, source="internal", disease=dis, horizon=hor,
                          variant="base", **who)
    assert plain == named
    assert set(plain) == {"reward", "crps_loss", "R", "point", "base", "report_level",
                          "report_malformed", "m", "side"}
    assert plain["reward"] == R.ALPHA * plain["R"] + (1.0 - plain["crps_loss"])
    assert plain["side"] == (1.0 if (plain["point"] - plain["base"]) * (y - plain["base"]) > 0 else 0.0)


def test_entry_variants_measure_against_the_entry_rate_and_add_the_side_term():
    who, dis, hor, y = _fixture_item()
    ref = R.entry_rate(disease=dis, horizon=hor, **who)
    t = stub_commit([0.01, 0.02, 0.03, 0.04, 0.05])
    base = R.rate_reward(t, truth=y, source="internal", disease=dis, horizon=hor)
    e = R.rate_reward(t, truth=y, source="internal", disease=dis, horizon=hor,
                      variant="entry", **who)
    s = R.rate_reward(t, truth=y, source="internal", disease=dis, horizon=hor,
                      variant="entry_step", **who)
    assert e["ref"] == ref and e["R_ref"] == abs(ref - y) - abs(e["point"] - y)
    assert e["reward"] == pytest.approx(R.ALPHA * e["R_ref"] + (1.0 - base["crps_loss"]))
    assert s["reward"] == pytest.approx(e["reward"] + R.T3_STEP_BETA * base["side"])
    # the contract's own components are untouched by the variant
    for k in ("crps_loss", "R", "point", "base", "report_level", "m", "side"):
        assert e[k] == base[k] == s[k]


def test_entry_variant_falls_back_to_b_where_the_entry_cell_is_silent():
    who, dis, hor, y = _fixture_item()
    silent = dict(who, disease="ZZZ9", horizon=hor)   # no such disease in any lattice
    t = stub_commit([0.01, 0.02, 0.03, 0.04, 0.05])
    base = R.rate_reward(t, truth=y, source="internal", disease="ZZZ9", horizon=hor)
    e = R.rate_reward(t, truth=y, source="internal", disease="ZZZ9", horizon=hor,
                      variant="entry", **who)
    assert R.entry_rate(**silent) is None
    assert e["ref"] == base["base"] and e["R_ref"] == base["R"] and e["reward"] == base["reward"]


def test_unknown_variant_is_refused_at_config_load_and_at_scoring():
    with pytest.raises(ValueError):
        RateTaskConfig(reward="entry_stpe")
    with pytest.raises(KeyError):
        R.rate_reward(stub_commit([0.1] * 5), truth=0.1, source="internal", disease="D0", horizon=5,
                      variant="entry_stpe", version="C2", age_band="45-49", sex="Female")


def test_config_reaches_the_task_and_the_score():
    who, dis, hor, y = _fixture_item()
    c = RateConfig.model_validate({"id": "cohortworld-internal", "task": {"reward": "entry_step"}})
    assert c.task.reward == "entry_step" and c.model_dump()["task"]["reward"] == "entry_step"
    data = RateData(idx=0, prompt="p", true_rate=y, zero_stratum=(y == 0.0), pop_zero_share=0.727,
                    disease=dis, horizon=hor, **who)
    assert RateTask(data, None).config.reward == R.DEFAULT_T3_VARIANT
    task = RateTask(data, c.task)
    got = task._score(stub_commit([0.01, 0.02, 0.03, 0.04, 0.05]))
    want = R.rate_reward(stub_commit([0.01, 0.02, 0.03, 0.04, 0.05]), truth=y, source="internal",
                         disease=dis, horizon=hor, variant="entry_step", **who)
    assert got == want and "R_ref" in got


def test_entry_reference_is_a_per_item_constant_to_grpo():
    """The `entry` reference cancels in `reward - group mean`: two rollouts of one item differ by
    exactly the same amount under `entry` as under `base`. Found the hard way."""
    who, dis, hor, y = _fixture_item()
    a, b_ = stub_commit([0.01, 0.02, 0.03, 0.04, 0.05]), stub_commit([0.03, 0.05, 0.08, 0.10, 0.20])
    def r(t, variant):
        return R.rate_reward(t, truth=y, source="internal", disease=dis, horizon=hor, variant=variant, **who)["reward"]
    assert r(a, "entry") - r(b_, "entry") == pytest.approx(r(a, "base") - r(b_, "base"), abs=1e-12)


def test_point_and_tanh_variants_read_only_the_median_and_add_the_documented_terms():
    import math
    who, dis, hor, y = _fixture_item()
    t = stub_commit([0.01, 0.02, 0.03, 0.04, 0.05])
    base = R.rate_reward(t, truth=y, source="internal", disease=dis, horizon=hor)
    p = R.rate_reward(t, truth=y, source="internal", disease=dis, horizon=hor, variant="point", **who)
    assert p["reward"] == pytest.approx(R.ALPHA * base["R"] + (1.0 - abs(base["point"] - y)))
    h = R.rate_reward(t, truth=y, source="internal", disease=dis, horizon=hor, variant="tanh", **who)
    z = (base["point"] - base["base"]) * (y - base["base"]) / R.T3_TANH_SCALE ** 2
    assert h["cov"] == pytest.approx(R.T3_TANH_GAMMA * math.tanh(z)) and abs(h["cov"]) <= R.T3_TANH_GAMMA
    assert h["reward"] == pytest.approx(base["reward"] + h["cov"])
    # a wider report with the same median: tanh term unchanged (median-only), CRPS term moves as in base
    w = stub_commit([0.0, 0.03, 0.03, 0.03, 1.0])
    hw = R.rate_reward(w, truth=y, source="internal", disease=dis, horizon=hor, variant="tanh", **who)
    bw = R.rate_reward(w, truth=y, source="internal", disease=dis, horizon=hor)
    assert hw["cov"] == pytest.approx(h["cov"]) and hw["reward"] - bw["reward"] == pytest.approx(h["cov"])
    # the point form does not pay for widening at a fixed median
    pw = R.rate_reward(w, truth=y, source="internal", disease=dis, horizon=hor, variant="point", **who)
    assert pw["reward"] == pytest.approx(p["reward"])
    for k in ("crps_loss", "R", "point", "base", "report_level", "m", "side"):
        assert p[k] == base[k] == h[k]


def test_point_tanh_is_exactly_the_sum_of_its_two_parts():
    who, dis, hor, y = _fixture_item()
    t = stub_commit([0.01, 0.02, 0.03, 0.04, 0.05])
    kw = dict(truth=y, source="internal", disease=dis, horizon=hor)
    base = R.rate_reward(t, **kw)
    p = R.rate_reward(t, **kw, variant="point", **who)
    h = R.rate_reward(t, **kw, variant="tanh", **who)
    pt = R.rate_reward(t, **kw, variant="point_tanh", **who)
    assert pt["reward"] == pytest.approx(p["reward"] + h["cov"])
    assert pt["cov"] == h["cov"]
    # still median-only: widening the report at a fixed median changes nothing
    w = stub_commit([0.0, 0.03, 0.03, 0.03, 1.0])
    assert R.rate_reward(w, **kw, variant="point_tanh", **who)["reward"] == pytest.approx(pt["reward"])
    for k in ("crps_loss", "R", "point", "base", "report_level", "m", "side"):
        assert pt[k] == base[k]
