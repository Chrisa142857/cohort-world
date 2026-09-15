"""The `cohort_think` T3 variant: the tag fixes a TASK, not a structure.

Two things are paid and both must satisfy the property `_t3_entry` failed: they must VARY across the
rollouts of ONE item, or GRPO's `reward - group mean` removes them. Nothing may be earnable by
reproducing a template. Every number is read off `rate_reward`, never recomputed beside it.
"""
import json, math
from functools import lru_cache

import pytest

import cohortworld_env.reward as R
from cohortworld_env.internal import ITEMS, RateConfig, RateTaskConfig, RateTaskset
from tests.test_contract import Stub, tool_msg


def _fixture_item():
    with open(f"{ITEMS}/t3_rate_train.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["version"] != "C2":
                continue
            who = dict(version=d["version"], age_band=d["age_band"], sex=d["sex"])
            dis, hor = d["disease"], int(d["horizon"])
            e = R.entry_rate(disease=dis, horizon=hor, **who)
            if e and e > 0 and float(d["answer_rate"]) > 0:
                return who, dis, hor, float(d["answer_rate"]), e
    pytest.skip("no usable C2 train item")


def _obs(state_id, disease, horizon, rate):
    return tool_msg("cw_observe", {"state_id": state_id, "status": "published",
                                   "disease": disease, "horizon": horizon, "rate": rate})


def _asst(text, calls=()):
    return {"role": "assistant", "content": text,
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": n, "arguments": "{}"}} for i, n in enumerate(calls)]}


def _move_result(to, code="04"):
    """The PRODUCTION shape of a landed move: `lat.move()` returns {"hit": True, "took":
    "<axis>.<op>.<code>-><state>", "cost", "state_id", ...}. `think_moved` reads `hit` + `took`; a
    fixture without them would test a shape the tool never emits."""
    return tool_msg("cw_move", {"hit": True, "took": f"bnf.add.{code}->{to}", "cost": 1.0, "state_id": to})


def _commit(v):
    return "ANSWER: [" + ", ".join(f"{v:.10f}" for _ in range(5)) + "]"


def _score(msgs, reply, who, dis, hor, y, variant="cohort_think"):
    t = Stub(messages=list(msgs), last_reply=reply)
    return R.rate_reward(t, truth=y, source="internal", disease=dis, horizon=hor,
                         variant=variant, **who)


# --------------------------------------------------------------------------- edge_prediction

def _episode(dis, hor, e, e2, think_text, after_text=""):
    """look, observe at A, THINK + move, observe at B, then commit. The think text sits in the
    message that carries the move call, i.e. it precedes the move by construction."""
    return [
        _asst("", ["cw_look"]), tool_msg("cw_look", {"state_id": "A"}),
        _asst("", ["cw_observe"]), _obs("A", dis, hor, e),
        _asst(think_text, ["cw_move"]), _move_result("B"),
        _asst(after_text, ["cw_observe"]), _obs("B", dis, hor, e2),
    ]


def test_declaration_before_the_move_is_captured_and_scored_against_the_measured_delta():
    who, dis, hor, y, e = _fixture_item()
    e2 = e * 1.5
    true_delta = e2 - e
    msgs = _episode(dis, hor, e, e2, f"<cohort-think>this history should raise it, I expect {true_delta:+.6f}</cohort-think>")
    ep = R.edge_prediction(Stub(messages=msgs), dis, hor)
    assert ep["declared"] == pytest.approx(true_delta, abs=1e-6)   # the text carries 6 decimals
    assert ep["measured"] == pytest.approx(true_delta)
    assert ep["src"] == "A" and ep["dst"] == "B" and ep["declared_before_move"] == 1.0
    s = _score(msgs, _commit(e2), who, dis, hor, y)
    # improvement over declaring zero: 1 - exp(-|true|/scale), which is close to 1 for this edge
    # the score is the IMPROVEMENT over declaring zero, so a perfect declaration is worth
    # 1 - exp(-|true delta| / scale): bounded by how far the edge actually moved, not a flat 1.0
    assert s["think_edge"] == pytest.approx(1.0 - math.exp(-abs(true_delta) / R.T3_THINK_EDGE_SCALE), abs=2e-2)
    assert s["think_edge"] > 0.0
    assert s["think_moved"] == 1.0 and s["think_declared"] == 1.0


def test_a_number_written_after_the_reading_is_a_copy_and_is_not_paid():
    """Order is the whole point: the number must precede the move."""
    who, dis, hor, y, e = _fixture_item()
    e2 = e * 1.5
    msgs = _episode(dis, hor, e, e2, "<cohort-think>let me look first</cohort-think>",
                    after_text=f"so the change was {e2 - e:+.6f}")
    ep = R.edge_prediction(Stub(messages=msgs), dis, hor)
    assert ep["declared"] is None and ep["declared_before_move"] == 0.0
    assert _score(msgs, _commit(e2), who, dis, hor, y)["think_edge"] == 0.0


def test_only_a_SIGNED_number_counts_as_a_declaration():
    """The unsigned fallback manufactured declarations -- `1.0` from "1 year", `0.0` from "0 diagnoses"
    -- 78% of the "declarations" in an audited smoke run. An unsigned reading in passing is not one."""
    who, dis, hor, y, e = _fixture_item()
    e2 = e * 1.5
    msgs = _episode(dis, hor, e, e2, f"<cohort-think>the rate here is {e:.6f}; adding this drug should move it by +0.003</cohort-think>")
    assert R.edge_prediction(Stub(messages=msgs), dis, hor)["declared"] == pytest.approx(0.003)
    msgs = _episode(dis, hor, e, e2, "<cohort-think>over 1 year with 0 diagnoses the rate is 0.004</cohort-think>")
    assert R.edge_prediction(Stub(messages=msgs), dis, hor)["declared"] is None
    msgs = _episode(dis, hor, e, e2, "<cohort-think>I expect \u22120.004</cohort-think>")     # unicode minus
    assert R.edge_prediction(Stub(messages=msgs), dis, hor)["declared"] == pytest.approx(-0.004)


def test_wrong_prediction_decays_and_no_move_scores_zero():
    who, dis, hor, y, e = _fixture_item()
    e2 = e * 1.5
    far = _episode(dis, hor, e, e2, "<cohort-think>I expect +0.5</cohort-think>")
    assert _score(far, _commit(e2), who, dis, hor, y)["think_edge"] < 0.0, "a wild guess scores BELOW zero-declaration"
    stay = [_asst("", ["cw_look"]), tool_msg("cw_look", {"state_id": "A"}),
            _asst("<cohort-think>I expect -0.002</cohort-think>", ["cw_observe"]), _obs("A", dis, hor, e)]
    s = _score(stay, _commit(e), who, dis, hor, y)
    assert s["think_edge"] == 0.0 and s["think_moved"] == 0.0


def test_the_constant_zero_template_earns_exactly_nothing():
    """With `exp(-|d-m|/s)` alone, the string `+0` scored 0.63-0.74 on real entry
    edges (median |delta| 0.0026, 23.4% exactly zero) -- more than a sign-correct informed prediction.
    The term now pays the IMPROVEMENT over declaring zero, so `+0` is worth exactly 0 whatever the
    edge measured, and a correct non-zero declaration is worth more than zero."""
    who, dis, hor, y, e = _fixture_item()
    for e2 in (e, e * 1.5, e * 0.2):
        msgs = _episode(dis, hor, e, e2, "<cohort-think>I expect +0</cohort-think>")
        assert _score(msgs, _commit(e2), who, dis, hor, y)["think_edge"] == pytest.approx(0.0, abs=1e-12), e2
    e2 = e * 1.5
    good = _episode(dis, hor, e, e2, f"<cohort-think>I expect {e2 - e:+.6f}</cohort-think>")
    assert _score(good, _commit(e2), who, dis, hor, y)["think_edge"] > 0.0


def test_no_destination_reading_means_no_edge_payment_and_no_same_cohort_reread():
    """Mutations M8 (pay with measured=None) and M3d (drop `here != before[0]`) survived the previous
    suite; this is their test. A move that lands but is never read at the asked cell pays 0, and a
    re-read at the SAME cohort is not a destination reading."""
    who, dis, hor, y, e = _fixture_item()
    landed = [
        _asst("", ["cw_look"]), tool_msg("cw_look", {"state_id": "A"}),
        _asst("", ["cw_observe"]), _obs("A", dis, hor, e),
        _asst("<cohort-think>I expect +0.003</cohort-think>", ["cw_move"]),
        tool_msg("cw_move", {"hit": True, "took": "bnf.switch.04->B", "state_id": "B"}),
    ]
    s = _score(landed, _commit(e), who, dis, hor, y)
    assert s["think_moved"] == 1.0, "a landed move counts as moved even without readings"
    assert s["think_edge"] == 0.0
    reread = landed + [_asst("", ["cw_observe"]), _obs("A", dis, hor, e)]
    assert R.edge_prediction(Stub(messages=reread), dis, hor)["measured"] is None
    assert _score(reread, _commit(e), who, dis, hor, y)["think_edge"] == 0.0


def test_non_finite_and_out_of_range_declarations_are_ignored():
    who, dis, hor, y, e = _fixture_item()
    e2 = e * 1.5
    for bad in ("+1e999", "-1e999", "+5.0", "nan"):
        msgs = _episode(dis, hor, e, e2, f"<cohort-think>I expect {bad}</cohort-think>")
        ep = R.edge_prediction(Stub(messages=msgs), dis, hor)
        assert ep["declared"] is None, bad


# --------------------------------------------------------------------------- rank_contribution

def test_rank_contribution_is_centred_and_moves_with_the_prediction():
    who, dis, hor, y, e = _fixture_item()
    ref = R.rank_reference()
    assert len(ref) == R.T3_THINK_REF_N
    lo, hi = R.rank_contribution(0.0, y), R.rank_contribution(1.0, y)
    mid = R.rank_contribution(y, y)
    assert -0.5 <= lo <= 0.5 and -0.5 <= hi <= 0.5
    assert mid > lo and mid > hi, "the truth itself must rank better than either extreme"
    assert len({round(R.rank_contribution(p, y), 6) for p in (0.0, y / 4, y, 4 * y, 1.0)}) >= 3


def test_rank_reference_is_pinned_by_content():
    h1 = R.rank_reference_hash()
    R.rank_reference.cache_clear()
    assert R.rank_reference_hash() == h1
    assert R.CONSTANTS()["t3_think_ref_sha16"] == h1


# --------------------------------------------------------------------------- the GRPO properties

def test_the_paid_terms_vary_across_the_rollouts_of_ONE_item():
    who, dis, hor, y, e = _fixture_item()
    e2 = e * 1.5
    rollouts = [
        (_episode(dis, hor, e, e2, f"<cohort-think>I expect {e2 - e:+.6f}</cohort-think>"), _commit(y)),
        (_episode(dis, hor, e, e2, f"<cohort-think>I expect {e2 - e + 0.01:+.6f}</cohort-think>"), _commit(y)),   # off by one e-fold
        (_episode(dis, hor, e, e2, "<cohort-think>no number here</cohort-think>"), _commit(0.9)),
        ([_asst("", ["cw_look"]), tool_msg("cw_look", {"state_id": "A"})], _commit(e)),
    ]
    edges = [_score(m, r, who, dis, hor, y)["think_edge"] for m, r in rollouts]
    ranks = [_score(m, r, who, dis, hor, y)["think_rank"] for m, r in rollouts]
    vals = {round(x, 6) for x in edges}
    assert len(vals) >= 3 and 0.0 in vals and max(edges) > 0.0, edges
    assert min(edges) < 0.0, "a wild declaration must score BELOW declaring zero"
    assert len({round(x, 6) for x in ranks}) >= 2


def test_nothing_is_earned_by_reproducing_the_old_template():
    """The second form paid for `entry=<reading> k=<multiplier>`. That is a template, and a policy
    that learns a template has not learned the task. It must be worth exactly nothing now."""
    who, dis, hor, y, e = _fixture_item()
    msgs = [_asst("", ["cw_look"]), tool_msg("cw_look", {"state_id": "A"}),
            _asst("", ["cw_observe"]), _obs("A", dis, hor, e)]
    with_tpl = _score(msgs, f"<cohort-think>entry={e:.10f} k=2.0</cohort-think>\n" + _commit(2 * e), who, dis, hor, y)
    without = _score(msgs, _commit(2 * e), who, dis, hor, y)
    assert with_tpl["reward"] == pytest.approx(without["reward"])


def test_default_and_point_tanh_untouched():
    who, dis, hor, y, e = _fixture_item()
    msgs = [_asst("", ["cw_look"]), tool_msg("cw_look", {"state_id": "A"})]
    b = _score(msgs, _commit(e), who, dis, hor, y, variant="base")
    assert not any(k.startswith("think_") for k in b)
    pt = _score(msgs, _commit(e), who, dis, hor, y, variant="point_tanh")
    ct = _score(msgs, _commit(e), who, dis, hor, y)
    assert ct["reward"] == pytest.approx(pt["reward"] + R.T3_THINK_W_RANK * ct["think_rank"]), \
        "with no move and no declaration, only the ranking term is added"


# --------------------------------------------------------------------------- the prompt

def test_the_prompt_fixes_a_task_and_no_structure_and_is_conditional():
    plain = next(iter(RateTaskset(RateConfig(split="train", num_tasks=1)).load()))
    think = next(iter(RateTaskset(RateConfig(split="train", num_tasks=1,
                                             task=RateTaskConfig(reward="cohort_think"))).load()))
    assert "<cohort-think>" not in plain.data.prompt
    assert "entry=" not in think.data.prompt and "k=" not in think.data.prompt, \
        "the tag must not prescribe a structure"
    assert "cw_look" not in think.data.prompt and "cw_move" not in think.data.prompt, \
        ("the tool affordance is the function schema prime-rl renders, not prose in the "
         "prompt. Naming the tools here is what made the RL question differ from the one the cold "
         "start was trained on.")
    # THE invariant of a two-stage design: the question RL poses is the question the demonstrations
    # were written against, built by the one function both stages call. Checked by construction here.
    from cohortworld_env.internal import cohort_think_question, strip_disease_code, _disease_name
    d = think.data
    expected = cohort_think_question(
        think.data.prompt.split("\n\nYour task is to:")[0],
        strip_disease_code(_disease_name(d.version, d.disease)), d.horizon, d.sex)
    assert think.data.prompt == expected, \
        ("the think prompt must BE cohort_think_question's output. The cold start learns "
         "p(answer | question); posed a different one, 555 of 556 rollouts "
         "answered in one turn, never reading a rate.")
    assert "Reason about which larger group of patients" in think.data.prompt


# =========================================================================== cohort_think_cal
# The variant pays `ALPHA * R2 + (1 - Q2) + W_RANK * rank` -- a CAPPED, SCALED squared point loss
# and the unchanged ranking term; no covariance term, and the edge term is reported but unpaid
# (see the T3_CAL_* docstring in reward.py for the measured reasons). Every number below is read
# off `rate_reward`, never recomputed beside it.

OFFICIAL_REF_SHA16 = "02cdb918f2d7917a"     # rank reference of the official item set (items_md5 12d8d1c574520e819ea43b1e05a4e85b)
OFFICIAL_DELTA_REF_SHA16 = "24948d2aa109e9e9"   # delta reference (494 item corrections) on the same item set
OFFICIAL_DELTA_REF_N = 494                  # of T3_CAL_DELTA_REF_N = 512 drawn; the rest publish no reference
# The sample is `answer_rate - rate(reference_cohort)`, one value per item, not entry-edge deltas:
# the entry IS the item's own cohort and its rate is withheld, so an entry-edge distribution would
# describe a journey nobody takes.
CAL = "cohort_think_cal"


def _look_only():
    return [_asst("", ["cw_look"]), tool_msg("cw_look", {"state_id": "A"})]


def test_cal_paid_terms_vary_across_the_rollouts_of_ONE_item():
    who, dis, hor, y, e = _fixture_item()
    e2 = e * 1.5
    rollouts = [
        (_episode(dis, hor, e, e2, f"<cohort-think>I expect {e2 - e:+.6f}</cohort-think>"), _commit(y)),
        (_episode(dis, hor, e, e2, f"<cohort-think>I expect {-(e2 - e):+.6f}</cohort-think>"), _commit(y)),  # wrong sign
        (_episode(dis, hor, e, e2, "<cohort-think>no number here</cohort-think>"), _commit(0.9)),
        (_look_only(), _commit(e)),
    ]
    s = [_score(m, r, who, dis, hor, y, variant=CAL) for m, r in rollouts]
    edges = [x["think_edge"] for x in s]
    assert len({round(v, 6) for v in edges}) >= 3 and 0.0 in edges and min(edges) < 0.0 < max(edges), edges
    assert len({round(x["think_rank"], 6) for x in s}) >= 2
    assert len({round(x["reward"], 9) for x in s}) >= 3, "the paid part must vary within the group"


def test_cal_leaves_base_and_every_existing_variant_byte_identical():
    """Values pinned on the official item set, compared with
    `==`, not approx. Skipped on any other item set, where the fixture item differs."""
    if R.rank_reference_hash() != OFFICIAL_REF_SHA16:
        pytest.skip("pins are for the official item set")
    who, dis, hor, y, e = _fixture_item()
    e2 = e * 1.5
    rollouts = [
        (_episode(dis, hor, e, e2, f"<cohort-think>I expect {e2 - e:+.6f}</cohort-think>"), _commit(y)),
        (_episode(dis, hor, e, e2, f"<cohort-think>I expect {-(e2 - e):+.6f}</cohort-think>"), _commit(y)),
        (_episode(dis, hor, e, e2, "<cohort-think>no number here</cohort-think>"), _commit(0.9)),
        (_look_only(), _commit(e)),
        (_episode(dis, hor, e, e2, "<cohort-think>I expect +0</cohort-think>"), _commit(e2)),
    ]
    b = 0.017189128014842302
    common = [dict(crps_loss=0.0, R=0.0828108719851577, point=0.1, base=b, report_level=2.0, report_malformed=0.0, m=5.0, side=1.0),
              dict(crps_loss=0.0, R=0.0828108719851577, point=0.1, base=b, report_level=2.0, report_malformed=0.0, m=5.0, side=1.0),
              dict(crps_loss=0.8, R=-0.7171891280148424, point=0.9, base=b, report_level=2.0, report_malformed=0.0, m=5.0, side=1.0),
              dict(crps_loss=0.09960300000000001, R=-0.01679212801484231, point=0.000397, base=b, report_level=2.0, report_malformed=0.0, m=5.0, side=0.0),
              dict(crps_loss=0.0994045, R=-0.016593628014842307, point=0.0005955, base=b, report_level=2.0, report_malformed=0.0, m=5.0, side=0.0)]
    covs = [0.49468537753955866, 0.49468537753955866, 0.5, -0.24286656713171856, -0.24046371857262452]
    pins = {
        "base": [dict(c, reward=r) for c, r in zip(common, [1.1035135899814472, 1.1035135899814472, -0.696486410018553, 0.8794068399814471, 0.8798534649814471])],
        "point": [dict(c, reward=r) for c, r in zip(common, [1.1035135899814472, 1.1035135899814472, -0.696486410018553, 0.8794068399814471, 0.8798534649814471])],
        "tanh": [dict(c, reward=r, cov=cv) for c, r, cv in zip(common, [1.598198967521006, 1.598198967521006, -0.19648641001855305, 0.6365402728497285, 0.6393897464088226], covs)],
        "point_tanh": [dict(c, reward=r, cov=cv) for c, r, cv in zip(common, [1.598198967521006, 1.598198967521006, -0.19648641001855305, 0.6365402728497285, 0.6393897464088226], covs)],
        "cohort_think": [
            dict(common[0], reward=1.626864581006632, cov=covs[0], think_rank=0.5, think_edge=0.07331226971252314, think_declared=1.0, think_moved=1.0, think_relevant=0.0, think_moved_code="04", think_edge_true=0.0001985),
            dict(common[1], reward=1.6197856325913462, cov=covs[1], think_rank=0.5, think_edge=-0.06826669859319134, think_declared=1.0, think_moved=1.0, think_relevant=0.0, think_moved_code="04", think_edge_true=0.0001985),
            dict(common[2], reward=-0.174105999995774, cov=covs[2], think_rank=0.4476082004555809, think_edge=0.0, think_declared=0.0, think_moved=1.0, think_relevant=0.0, think_moved_code="04", think_edge_true=0.0001985),
            dict(common[3], reward=0.6476450564488173, cov=covs[3], think_rank=0.2220956719817767, think_edge=0.0, think_declared=0.0, think_moved=0.0, think_relevant=0.0, think_moved_code="", think_edge_true=0.0),
            dict(common[4], reward=0.6504945300079115, cov=covs[4], think_rank=0.2220956719817767, think_edge=0.0, think_declared=1.0, think_moved=1.0, think_relevant=0.0, think_moved_code="04", think_edge_true=0.0001985)],
    }
    for variant, want in pins.items():
        got = [_score(m, r, who, dis, hor, y, variant=variant) for m, r in rollouts]
        assert got == want, variant
    assert not any(k.startswith(("think_", "cal_")) for k in pins["base"][0])


def test_cal_returns_cohort_think_keys_plus_its_own():
    """internal.RateTask's think_* metrics and gen/cohort_think_gate.py read these keys off the
    score dict; a variant that dropped them would fail the process gate for an unrelated reason."""
    who, dis, hor, y, e = _fixture_item()
    msgs = _episode(dis, hor, e, e * 1.5, "<cohort-think>I expect +0.001</cohort-think>")
    ct = _score(msgs, _commit(y), who, dis, hor, y)
    cal = _score(msgs, _commit(y), who, dis, hor, y, variant=CAL)
    assert set(cal) >= set(ct) and {"cal_q2", "cal_r2", "cal_edge", "cal_delta", "cal_declared"} <= set(cal)
    for k in set(ct) - {"reward"}:
        assert cal[k] == ct[k], k                  # diagnostics come from cohort_think's code, unchanged


def test_cal_both_edge_terms_are_reported_unpaid_and_only_cal_delta_moves_the_reward():
    """Two diagnostics, one payment. The fixture's first edge measures 0.5 e < s0 -- template-sized --
    so `cal_edge` (the gated edge term, which requires |d| >= s0) is 0 for EVERY declaration on it,
    exact ones included, while `think_edge` (cohort_think's edge term) is positive for the exact one
    and negative for the wild one. The reward follows NEITHER: it follows `cal_delta`, which pays the
    exact small declaration the size gate refuses."""
    who, dis, hor, y, e = _fixture_item()
    e2 = e * 1.5
    assert abs(e2 - e) < R.T3_THINK_EDGE_SCALE
    informed = _score(_episode(dis, hor, e, e2, f"<cohort-think>I expect {e2 - e:+.6f}</cohort-think>"), _commit(y), who, dis, hor, y, variant=CAL)
    zero = _score(_episode(dis, hor, e, e2, "<cohort-think>I expect +0</cohort-think>"), _commit(y), who, dis, hor, y, variant=CAL)
    wild = _score(_episode(dis, hor, e, e2, "<cohort-think>I expect +0.5</cohort-think>"), _commit(y), who, dis, hor, y, variant=CAL)
    copied = _score(_episode(dis, hor, e, e2, "<cohort-think>later</cohort-think>", after_text=f"it moved {e2 - e:+.6f}"),
                    _commit(y), who, dis, hor, y, variant=CAL)
    assert informed["think_edge"] > 0.0 > wild["think_edge"] and zero["think_edge"] == 0.0
    assert copied["think_declared"] == 0.0 and copied["think_edge"] == 0.0
    assert informed["cal_edge"] == zero["cal_edge"] == wild["cal_edge"] == copied["cal_edge"] == 0.0
    assert informed["cal_delta"] > 0.0 == zero["cal_delta"] == wild["cal_delta"] == copied["cal_delta"]
    assert informed["reward"] > zero["reward"] == wild["reward"] == copied["reward"], "cal_edge and think_edge are UNPAID; cal_delta is what moves the reward"
    assert informed["reward"] == pytest.approx(zero["reward"] + R.T3_CAL_W_DELTA * informed["cal_delta"], abs=1e-12)
    # and for a zero-delta move every declaration is worth exactly the same as silence
    same = _score(_episode(dis, hor, e, e, "<cohort-think>I expect -0.002</cohort-think>"), _commit(y), who, dis, hor, y, variant=CAL)
    assert same["think_edge"] <= 0.0 and same["cal_edge"] == same["cal_delta"] == 0.0 and same["reward"] == zero["reward"]


def test_cal_outcome_term_is_capped_scaled_l2_and_mean_consistent_in_form():
    who, dis, hor, y, e = _fixture_item()
    b = R.base_rate("internal", dis, hor)
    s2 = R.cal_l2_scale()
    for p in (0.0, b, y, 0.3, 0.9):
        s = _score(_look_only(), _commit(p), who, dis, hor, y, variant=CAL)
        q2 = min((s["point"] - y) ** 2 / s2, R.T3_CAL_Q2_CAP)     # the PARSED point: _commit writes 10 decimals
        assert s["cal_q2"] == pytest.approx(q2, abs=1e-12)
        assert s["cal_r2"] == pytest.approx(min((b - y) ** 2 / s2, R.T3_CAL_Q2_CAP) - q2, abs=1e-12)
        assert s["cal_edge"] == s["cal_delta"] == 0.0     # nothing declared
        assert s["reward"] == pytest.approx(R.ALPHA * s["cal_r2"] + 1.0 - q2 + R.T3_THINK_W_RANK * s["think_rank"] + R.T3_CAL_W_DELTA * s["cal_delta"], abs=1e-12)
    at_truth = _score(_look_only(), _commit(y), who, dis, hor, y, variant=CAL)
    assert at_truth["cal_q2"] == 0.0 and at_truth["cal_r2"] >= 0.0
    # the cap: beyond |p - y| = sqrt(CAP * S2) the term is flat, so one wild commit cannot dominate a group
    far = sorted(_score(_look_only(), _commit(p), who, dis, hor, y, variant=CAL)["cal_q2"] for p in (0.7, 0.85, 1.0))
    assert far[0] == far[-1] == R.T3_CAL_Q2_CAP and (1.0 - y) ** 2 / s2 > R.T3_CAL_Q2_CAP


def test_cal_outcome_term_does_not_reward_dispersion_beyond_the_truth():
    """b explicit (the reward's b is the pinned train cascade, not the fixture's entry rate): truth =
    b + 0.02, p1 = truth, p2 = truth + 0.05. point_tanh pays the overshoot (its tanh gain outruns
    the L1 cost whenever |y - b| > 0.012); cohort_think_cal does not."""
    who, dis, hor, _y, e = _fixture_item()
    b = R.base_rate("internal", dis, hor)
    y = b + 0.02
    p1, p2 = y, y + 0.05
    pt = [_score(_look_only(), _commit(p), who, dis, hor, y, variant="point_tanh")["reward"] for p in (p1, p2)]
    ct = [_score(_look_only(), _commit(p), who, dis, hor, y)["reward"] for p in (p1, p2)]
    cal = [_score(_look_only(), _commit(p), who, dis, hor, y, variant=CAL)["reward"] for p in (p1, p2)]
    assert pt[1] > pt[0] and ct[1] > ct[0], (pt, ct)
    assert cal[1] < cal[0], cal


def test_cal_selection_proxy_on_a_fixed_synthetic_group():
    """The acceptance in miniature: one item, a GRPO group of rollouts committing {truth, truth + 0.05
    (dispersed beyond the truth), b (uninformative), 0 (the entry-reading attractor)}. The old term's
    argmax is the dispersed point; the new term's argmax is the truth. Read as `reward - group mean`
    so it is the advantage GRPO would see."""
    who, dis, hor, _y, e = _fixture_item()
    b = R.base_rate("internal", dis, hor)
    y = b + 0.02
    group = {"truth": y, "dispersed": y + 0.05, "base": b, "zero": 0.0}
    def adv(variant):
        r = {k: _score(_look_only(), _commit(p), who, dis, hor, y, variant=variant)["reward"] for k, p in group.items()}
        m = sum(r.values()) / len(r)
        return {k: v - m for k, v in r.items()}
    old, new = adv("cohort_think"), adv(CAL)
    assert max(old, key=old.get) == "dispersed" and max(new, key=new.get) == "truth", (old, new)
    assert new["zero"] < 0.0 and new["base"] < new["truth"]


def test_cal_ranks_the_sample_mean_above_the_median_and_the_l1_form_does_not():
    """Through rate_reward, not arithmetic: a right-skewed truth sample, a forecaster that knows
    only the sample. Grid argmax of the MEAN reward over the sample: within 0.005 of the sample
    mean under cohort_think_cal; inside the L1-minimiser interval (between the 3rd and 4th order
    statistics, [0, 0.02], which holds the median) under `point`; and point_tanh -- the outcome
    term of cohort_think -- lands neither at the mean nor at the median (0.0375 here: the tanh term
    moves the optimum off both, which is the non-properness the variant removes)."""
    who, dis, hor, _y, e = _fixture_item()
    truths = [0.0, 0.0, 0.0, 0.02, 0.05, 0.2]
    mean_t, median_t = sum(truths) / len(truths), R.median(truths)
    grid = [i / 400 for i in range(0, 121)]            # 0 .. 0.30 step 0.0025
    def argmax(variant):
        best = max(grid, key=lambda p: sum(_score(_look_only(), _commit(p), who, dis, hor, t, variant=variant)["reward"] for t in truths))
        return best
    assert abs(argmax(CAL) - mean_t) < 0.005, argmax(CAL)
    assert 0.0 <= argmax("point") <= 0.02 and mean_t - argmax("point") >= 0.025, argmax("point")
    assert abs(argmax("point_tanh") - mean_t) >= 0.005 and abs(argmax("point_tanh") - median_t) >= 0.005, argmax("point_tanh")


def test_cal_constants_are_derived_and_recorded():
    import json as _json
    c = R.CONSTANTS()
    tot = n = 0
    with open(f"{ITEMS}/t3_rate_train.jsonl") as f:
        for line in f:
            d = _json.loads(line)
            tot += abs(R.base_rate("internal", d["ask"]["disease"], int(d["ask"]["horizon"])) - float(d["answer"]["rate"])); n += 1
    assert c["t3_cal_l2_scale"] == pytest.approx(R.T3_CAL_L2_SCALE_MULT * tot / n, rel=1e-9)
    assert c["t3_cal_l2_scale"] == R.cal_l2_scale() and c["t3_cal_q2_cap"] == R.T3_CAL_Q2_CAP
    assert c["t3_cal_l2_scale_mult"] == R.T3_CAL_L2_SCALE_MULT == 2.0
    if R.rank_reference_hash() == OFFICIAL_REF_SHA16:
        assert c["t3_cal_l2_scale"] == pytest.approx(0.0717544, abs=1e-6)     # 2 * MAE_b on the official set
        assert c["t3_think_ref_sha16"] == OFFICIAL_REF_SHA16
    R.cal_l2_scale.cache_clear()
    assert R.cal_l2_scale() == c["t3_cal_l2_scale"]


def test_cal_prompt_is_appended_for_the_new_variant_too():
    """There is no appended paragraph. Both think variants ARE asked the cold start's
    question, and the non-think variants keep the tool-instruction prompt every archived score used."""
    from cohortworld_env.internal import cohort_think_question, strip_disease_code, _disease_name
    def task(variant):
        return next(iter(RateTaskset(RateConfig(split="train", num_tasks=1,
                                               task=RateTaskConfig(reward=variant))).load()))
    cal, think, pt = task(CAL), task("cohort_think"), task("point_tanh")
    assert cal.data.prompt == think.data.prompt, "the question is unchanged; only what is paid changed"
    d = think.data
    assert think.data.prompt == cohort_think_question(
        think.data.prompt.split("\n\nYour task is to:")[0],
        strip_disease_code(_disease_name(d.version, d.disease)), d.horizon, d.sex)
    assert "cw_look" in pt.data.prompt, \
        "the non-think variants keep the prompt every archived score was produced under"
    assert "Reason about which larger group" not in pt.data.prompt
    assert set(R.T3_THINK_PROMPT_VARIANTS) == {"cohort_think", CAL} and set(R.T3_THINK_PROMPT_VARIANTS) <= set(R.T3_REWARD_VARIANTS)


# =========================================================================== the delta payment
# `T3_CAL_W_DELTA * delta_cal`, delta_cal = max(0, acc(d, m) - acc(c*, m)), acc(d, m) =
# exp(-|d - m| / max(|m|, s0)), c* = the best CONSTANT declaration on the PINNED reference sample of measured
# deltas (`reward.delta_reference()`, derived from the ITEM SET). The edge CHOICE is not the learning
# target, the DELTA is, so this is where the
# within-group variance GRPO needs has to come from, and the term is no longer a 0.05 garnish. The start
# reading below is 0.05 (not the fixture's entry rate, ~4e-4) so that negative deltas leave the destination
# reading in [0, 1]; the reward never reads the observation values themselves.

def _declared(dis, hor, m, d, start=0.05):
    return _episode(dis, hor, start, start + m, f"<cohort-think>I expect {d:+.10f}</cohort-think>")


@lru_cache(maxsize=None)
def _grid_means():
    """Mean `delta_accuracy(c, m)` over the WHOLE pinned reference sample, for every constant on the
    grid. One computation (~4 s), shared by the three tests that need it."""
    lo, hi, step = R.T3_CAL_DELTA_GRID
    grid = sorted({0.0} | {round(lo + i * step, 10) for i in range(int(round((hi - lo) / step)) + 1)})
    ms = R.delta_reference()
    return {c: sum(R.delta_accuracy(c, m) for m in ms) / len(ms) for c in grid}


def test_cal_delta_no_constant_declaration_can_beat_the_pinned_reference_constant():
    """(i) THE constant-proofness property, asserted over the whole grid.

    The exact claim "any constant earns 0" is IMPOSSIBLE for a payment that is also never negative and
    reads only (d, m): E[max(0, X)] > 0 whenever X = acc(c,m) - acc(c*,m) varies, even with E[X] <= 0. So
    what is asserted is what is true and what the term is built on: (1) the mean CENTRED accuracy of every
    constant on the pinned reference is <= 0, exactly 0 at c* and strictly negative elsewhere -- no
    template is the maximiser; (2) c* pays exactly 0 on EVERY reference edge, pointwise, not just on
    average; (3) the best constant's rectified leak is far below what a conditioned (oracle) declaration
    earns. Measured: best rectified constant -0.0048 -> 0.090, oracle 0.381."""
    means = _grid_means()
    c_star = R.cal_delta_const()
    assert max(means, key=means.get) == c_star
    assert all(means[c] - means[c_star] <= 0.0 for c in means)
    assert all(means[c] - means[c_star] < 0.0 for c in means if c != c_star), "c* must be the STRICT argmax"
    ms = R.delta_reference()
    assert all(R.delta_accuracy(c_star, m) - R.delta_accuracy(c_star, m) == 0.0 for m in ms[:2000])
    best_leak = max(sum(max(0.0, R.delta_accuracy(c, m) - R.delta_accuracy(c_star, m)) for m in ms) / len(ms)
                    for c in (-0.0048, -0.005, -0.003, -0.001, 0.0, 0.001, 0.003, 0.005))
    oracle = sum(max(0.0, R.delta_accuracy(m, m) - R.delta_accuracy(c_star, m)) for m in ms) / len(ms)
    assert 0.0 < best_leak < 0.4 * oracle, (best_leak, oracle)


def test_cal_delta_pays_a_declaration_that_beats_the_best_constant_on_one_item():
    """(ii) Beating c* on the item in front of you is the only way to be paid, and it IS paid."""
    who, dis, hor, y, e = _fixture_item()
    c_star = R.cal_delta_const()
    for m in (0.004, -0.004, 0.0008, -0.0008, 0.05):
        exact = _score(_declared(dis, hor, m, m), _commit(y), who, dis, hor, y, variant=CAL)
        assert exact["cal_delta"] > 0.0 and exact["cal_delta"] == pytest.approx(1.0 - R.delta_accuracy(c_star, m), abs=1e-12), m
        const = _score(_declared(dis, hor, m, c_star), _commit(y), who, dis, hor, y, variant=CAL)
        assert const["cal_delta"] == 0.0 and exact["cal_delta"] > const["cal_delta"], m
        assert exact["reward"] == pytest.approx(const["reward"] + R.T3_CAL_W_DELTA * exact["cal_delta"], abs=1e-12)
    # a CORRECT declaration smaller than s0 is paid: a size gate would refuse it, and in training that
    # is most of what the policy writes.
    small = _score(_declared(dis, hor, 0.001, 0.001), _commit(y), who, dis, hor, y, variant=CAL)
    assert small["cal_delta"] > 0.0 and small["cal_edge"] == 0.0, \
        "a correct sub-s0 declaration must be paid by cal_delta, not by the size-gated cal_edge"


def test_cal_delta_payment_is_never_negative_and_a_wrong_direction_pays_exactly_as_silence():
    """(iii) With c* = 0 the sign gate is a THEOREM, not a clause: a wrong-sign declaration has
    |d - m| > |m|, so it scores below the constant and the rectifier pays it what silence pays."""
    who, dis, hor, y, e = _fixture_item()
    m = 0.004
    silent = _score(_episode(dis, hor, 0.05, 0.05 + m, "<cohort-think>no number here</cohort-think>"), _commit(y), who, dis, hor, y, variant=CAL)
    assert silent["cal_delta"] == 0.0
    for d in (-m, -0.01, -0.5, +0.5, 3 * m, -R.T3_THINK_EDGE_SCALE):
        s = _score(_declared(dis, hor, m, d), _commit(y), who, dis, hor, y, variant=CAL)
        assert s["cal_delta"] == 0.0 and s["reward"] == silent["reward"], d
    for m_ in (0.0001, -0.0001, 0.0026, -0.0026, 0.004, -0.004, 0.05, -0.05, 0.0):
        for d in (-0.5, -0.01, -0.001, 0.0, 0.001, 0.01, 0.5, m_, 1.5 * m_, 0.5 * m_):
            assert _score(_declared(dis, hor, m_, d), _commit(y), who, dis, hor, y, variant=CAL)["cal_delta"] >= 0.0, (m_, d)


def test_cal_delta_pays_a_correct_small_delta_the_same_as_a_correct_large_one():
    """(iv) |m|-normalised: the same RELATIVE error pays the same at every size >= s0, both signs, so
    nothing is earned by choosing a big-|m| move -- which is the point of the correction, since which
    edge is taken is not what is being trained."""
    who, dis, hor, y, e = _fixture_item()
    s0 = R.T3_THINK_EDGE_SCALE
    pay = lambda m, d: _score(_declared(dis, hor, m, d), _commit(y), who, dis, hor, y, variant=CAL)["cal_delta"]
    want_exact = 1.0 - math.exp(-1)                     # acc(c*=0, m) = exp(-1) for every |m| >= s0
    assert all(pay(m, m) == pytest.approx(want_exact, abs=1e-12) for m in (s0, 0.003, -0.003, 0.02, -0.02, 0.3))
    want_half = math.exp(-0.5) - math.exp(-1)
    assert all(pay(m, 1.5 * m) == pytest.approx(want_half, abs=1e-12) for m in (0.003, -0.003, 0.03, 0.3))
    assert 0.0 < want_half < want_exact
    # below s0 the tolerance is s0 for BOTH terms, so the payment tapers instead of switching off
    tiny = pay(0.001, 0.001)
    assert tiny == pytest.approx(1.0 - math.exp(-0.001 / s0), abs=1e-12) and 0.0 < tiny < want_exact


def test_cal_delta_pays_nothing_for_a_copy_no_declaration_no_destination_or_a_zero_edge():
    """(v) The four gates that ARE kept, all of them `edge_prediction`'s --
    plus the zero edge, which needs no clause: acc(c* = 0, 0) = 1 is the maximum the term can reach."""
    who, dis, hor, y, e = _fixture_item()
    m = 0.004
    copied = _episode(dis, hor, 0.05, 0.05 + m, "<cohort-think>let me look first</cohort-think>", after_text=f"it moved {m:+.10f}")
    none = _episode(dis, hor, 0.05, 0.05 + m, "<cohort-think>no number</cohort-think>")
    no_dst = [_asst("", ["cw_look"]), tool_msg("cw_look", {"state_id": "A"}), _asst("", ["cw_observe"]), _obs("A", dis, hor, 0.05),
              _asst(f"<cohort-think>I expect {m:+.10f}</cohort-think>", ["cw_move"]), _move_result("B")]
    reread = no_dst + [_asst("", ["cw_observe"]), _obs("A", dis, hor, 0.05 + m)]        # the SAME cohort again
    for name, msgs in (("copied", copied), ("none", none), ("no_dst", no_dst), ("reread", reread),
                       ("zero_edge_exact", _declared(dis, hor, 0.0, 0.0)), ("zero_edge_any", _declared(dis, hor, 0.0, m)),
                       ("look_only", _look_only())):
        assert _score(msgs, _commit(y), who, dis, hor, y, variant=CAL)["cal_delta"] == 0.0, name
    assert _score(copied, _commit(y), who, dis, hor, y, variant=CAL)["think_declared"] == 0.0
    assert R.delta_cal(None, dis, hor) == 0.0


def test_cal_delta_payment_varies_across_the_rollouts_of_ONE_item_on_the_SAME_edge():
    """(vi) The mechanism the correction asks for: eight rollouts that take the SAME edge still differ,
    because their DELTA PREDICTIONS differ, and the payment separates them. In training rollouts the
    declaration varies in 55.6% of the same-edge cells that carry two declarations."""
    who, dis, hor, y, e = _fixture_item()
    m = 0.004
    same_edge = [_declared(dis, hor, m, d) for d in (m, 1.3 * m, 0.5 * m, 0.001, -m)]
    s = [_score(r, _commit(y), who, dis, hor, y, variant=CAL) for r in same_edge]
    pays = [x["cal_delta"] for x in s]
    assert pays[0] > pays[1] > pays[2] > pays[3] > 0.0 == pays[4], pays
    assert len({round(x["reward"], 9) for x in s}) == 5, "the payment must survive reward - group mean"
    assert len({round(x["think_edge"], 9) for x in s}) >= 3         # cohort_think's edge diagnostic still varies, unpaid


def test_cal_delta_constant_is_derived_and_recorded():
    """(vii) c* is DERIVED from the item set at first use and recorded, never restated as a literal:
    recomputed here from `delta_reference()` over the whole grid, and asserted equal to the shipped
    value, to CONSTANTS(), and to itself after a cache_clear."""
    c = R.CONSTANTS()
    means = _grid_means()
    assert R.cal_delta_const() == max(means, key=means.get) == c["t3_cal_delta_const"]
    assert c["t3_cal_delta_ref_sha16"] == R.delta_reference_hash() and len(c["t3_cal_delta_ref_sha16"]) == 16
    assert c["t3_cal_delta_ref_n"] == R.T3_CAL_DELTA_REF_N and c["t3_cal_delta_grid"] == list(R.T3_CAL_DELTA_GRID)
    assert c["t3_cal_w_delta"] == R.T3_CAL_W_DELTA
    ms = R.delta_reference()
    # ONE value per ITEM (`answer - rate(reference_cohort)`), not one per entry edge, so the sample
    # is bounded ABOVE by the items drawn, and what is worth asserting is that most of them publish
    # a reference at all.
    assert 0.8 * R.T3_CAL_DELTA_REF_N <= len(ms) <= R.T3_CAL_DELTA_REF_N
    assert all(abs(m) <= 1.0 for m in ms)
    R.cal_delta_const.cache_clear()
    assert R.cal_delta_const() == c["t3_cal_delta_const"]
    if R.rank_reference_hash() == OFFICIAL_REF_SHA16:                      # the official item set
        # c* is STILL 0.0 -- and that is a re-derivation, not a restatement: 41.7% of items need
        # exactly no correction, so the zero constant is pointwise exact on 41.7% of the sample.
        assert c["t3_cal_delta_const"] == 0.0 and c["t3_cal_delta_ref_sha16"] == OFFICIAL_DELTA_REF_SHA16
        assert len(ms) == OFFICIAL_DELTA_REF_N


# =========================================================================== accepted moves and the declaration parse
# (a) the REJECTED move: 35.7% of a training run's rollouts called cw_move, were REFUSED ("not a legal move from
# here" -- a fabricated id), never moved, stayed in the entry cohort and were REWARDED for it. `cohort_think_cal`
# pays `T3_CAL_W_MOVE` for an ACCEPTED move (and can floor a rollout that never moved -- swept, disabled).
# (b) the DECLARATION PARSE: `_SIGNED_NUM` cannot see a signed number that ends a sentence (20.1% vs 99.77%
# of that run's rollouts); `cohort_think_cal` reads `_SIGNED_NUM_V2`, `cohort_think` still reads v1.

def _refused_move(reason="not a legal move from here"):
    """The PRODUCTION shape of a REFUSED move (lattice.move): no `hit`, no `took`."""
    return tool_msg("cw_move", {"hit": False, "reason": reason, "moved": False})


def _episode_refused(dis, hor, e, think_text):
    """look, observe at A, THINK + move that is REFUSED, then commit -- the modal failure in training."""
    return [_asst("", ["cw_look"]), tool_msg("cw_look", {"state_id": "A"}),
            _asst("", ["cw_observe"]), _obs("A", dis, hor, e),
            _asst(think_text, ["cw_move"]), _refused_move()]


def test_accepted_move_reads_the_environments_acknowledgement_not_the_call():
    who, dis, hor, y, e = _fixture_item()
    landed = _episode(dis, hor, e, e * 1.5, "<cohort-think>I expect +0.001</cohort-think>")
    refused = _episode_refused(dis, hor, e, "<cohort-think>I expect +0.001</cohort-think>")
    assert R.accepted_move(Stub(messages=landed)) == (1.0, "04")
    assert R.accepted_move(Stub(messages=refused)) == (0.0, "")          # it CALLED cw_move and was refused
    assert R.accepted_move(Stub(messages=_look_only())) == (0.0, "")
    assert R.accepted_move(None) == (0.0, "")
    # and it is the one reader: cohort_think's diagnostic is exactly this number
    for msgs in (landed, refused, _look_only()):
        s = _score(msgs, _commit(y), who, dis, hor, y)
        assert s["think_moved"] == R.accepted_move(Stub(messages=msgs))[0]
        assert s["think_moved_code"] == R.accepted_move(Stub(messages=msgs))[1]


def test_cal_pays_w_move_only_for_an_accepted_move():
    """A refused move is worth exactly what never calling cw_move is worth, and both are worth
    W_MOVE less than the same rollout with the move accepted -- which is the whole point: in training
    the refused-move rollouts kept the entry cohort's rate and took the group's LARGEST advantages."""
    who, dis, hor, y, e = _fixture_item()
    landed = _score(_episode(dis, hor, e, e, "<cohort-think>nothing here</cohort-think>"), _commit(y), who, dis, hor, y, variant=CAL)
    refused = _score(_episode_refused(dis, hor, e, "<cohort-think>nothing here</cohort-think>"), _commit(y), who, dis, hor, y, variant=CAL)
    never = _score(_look_only(), _commit(y), who, dis, hor, y, variant=CAL)
    assert landed["think_moved"] == 1.0 and refused["think_moved"] == never["think_moved"] == 0.0
    assert refused["reward"] == never["reward"], "a refused move is worth what not calling one is worth"
    assert landed["reward"] == pytest.approx(refused["reward"] + R.T3_CAL_W_MOVE, abs=1e-12)
    assert R.T3_CAL_W_MOVE > 0.0, "the rejected move must not be profitable"


def test_cal_nomove_floor_is_disabled_but_the_code_path_works(monkeypatch):
    """The floor was SWEPT (spec §9) and DISABLED: at every value that made the non-mover advantage
    non-positive it cost 3-4x more proxy slope than the W_MOVE payment. `None` is the off switch, and
    the branch is exercised here so the constant is not a value nobody can turn on."""
    who, dis, hor, y, e = _fixture_item()
    assert R.T3_CAL_NOMOVE_FLOOR is None
    landed_msgs = _episode(dis, hor, e, e, "<cohort-think>nothing</cohort-think>")
    off = _score(_look_only(), _commit(y), who, dis, hor, y, variant=CAL)
    assert off["reward"] != 0.5
    monkeypatch.setattr(R, "T3_CAL_NOMOVE_FLOOR", 0.5)
    assert _score(_look_only(), _commit(y), who, dis, hor, y, variant=CAL)["reward"] == 0.5
    assert _score(_episode_refused(dis, hor, e, "<cohort-think>x</cohort-think>"), _commit(y), who, dis, hor, y, variant=CAL)["reward"] == 0.5
    assert _score(landed_msgs, _commit(y), who, dis, hor, y, variant=CAL)["reward"] != 0.5      # a mover is untouched


def test_the_v2_declaration_pattern_sees_a_sentence_final_number_and_v1_does_not():
    """The parsing artifact of the v1 pattern: it misses a signed number that ends a sentence."""
    assert R._SIGNED_NUM.findall("I expect -0.004.") == []
    assert R._SIGNED_NUM_V2.findall("I expect -0.004.") == ["-0.004"]
    for t in ("I expect -0.004 in the rate", "about -0.004, so", "-0.004"):          # both see these
        assert R._SIGNED_NUM.findall(t) == R._SIGNED_NUM_V2.findall(t) == ["-0.004"]
    assert R._SIGNED_NUM_V2.findall("in 2020 the rate was 5 per cent") == []          # still SIGNED only


def test_cal_reads_the_v2_declaration_and_cohort_think_still_reads_v1():
    """One trace, two parses. The declaration ends the sentence, so v1 misses it: `cohort_think`'s
    edge term and `think_declared` are 0 and its reward is EXACTLY that of a rollout with no number
    at all, while `cohort_think_cal` pays the declaration. That is the byte-identity constraint and
    the fix living side by side."""
    who, dis, hor, y, e = _fixture_item()
    m = 0.004
    final = _episode(dis, hor, 0.05, 0.05 + m, f"<cohort-think>the rate should move by {m:+.10f}.</cohort-think>")
    silent = _episode(dis, hor, 0.05, 0.05 + m, "<cohort-think>no number here</cohort-think>")
    ct_final, ct_silent = _score(final, _commit(y), who, dis, hor, y), _score(silent, _commit(y), who, dis, hor, y)
    assert ct_final["think_declared"] == 0.0 and ct_final["think_edge"] == 0.0
    assert ct_final["reward"] == ct_silent["reward"], "cohort_think must be byte-identical: it reads v1"
    cal_final, cal_silent = (_score(x, _commit(y), who, dis, hor, y, variant=CAL) for x in (final, silent))
    assert cal_final["cal_declared"] == 1.0 and ct_final["think_declared"] == 0.0
    assert cal_final["cal_delta"] > 0.0 and cal_final["cal_delta"] == pytest.approx(1.0 - R.delta_accuracy(R.cal_delta_const(), m), abs=1e-12)
    assert cal_final["reward"] > cal_silent["reward"] and cal_silent["cal_delta"] == 0.0
    assert R.edge_prediction(Stub(messages=final), dis, hor)["declared"] is None
    assert R.edge_prediction(Stub(messages=final), dis, hor, pattern=R._SIGNED_NUM_V2)["declared"] == pytest.approx(m)


# ========================================================== the reference moves with the geometry
#
# A T3 agent starts on its item's own cohort with the rate withheld, so `entry_rate` and
# `delta_reference` cannot point at the DEMOGRAPHIC entry -- a position no T3 episode occupies and a
# number no T3 agent can read -- and `edge_prediction`'s `measured` cannot need a reading BEFORE the
# move, which the mask makes impossible. Every test below is keyed on the item carrying a
# `target_cohort`: with it, the new geometry; without it, the old numbers, exactly, which is what
# lets an archive rescore to the value it was produced with.

def _t3_item_with_target():
    """A real train item whose reference cohort publishes the asked cell, with its numbers."""
    with open(f"{ITEMS}/t3_rate_train.jsonl") as f:
        for line in f:
            d = json.loads(line)
            tc = d.get("target_cohort")
            if d["version"] != "C2" or not tc:
                continue
            who = dict(version=d["version"], age_band=d["age_band"], sex=d["sex"])
            ref = R.entry_rate(disease=d["disease"], horizon=int(d["horizon"]),
                               target_cohort=tc, **who)
            if ref is not None and float(d["answer_rate"]) > 0:
                return who, d["disease"], int(d["horizon"]), float(d["answer_rate"]), tc, ref
    pytest.skip("no C2 train item whose reference cohort publishes the asked cell")


def test_the_reference_cohort_is_one_drop_away_and_the_biggest_one():
    """The rule, read off the lattice rather than restated: a drop destination, and the one with the
    largest published band. Deterministic, because a reference chosen by a POLICY stops being a
    constant the moment the policy moves."""
    from cohortworld_env.lattice import _band_lo, _shared
    _who, _dis, _hor, _y, tc, _ref = _t3_item_with_target()
    lat = _shared("C2")
    got = R.reference_cohort("C2", tc)
    mv = lat._drop_moves(tc)
    assert got in {m["to"] for m in mv}, "the reference is not reachable by one drop"
    assert _band_lo(lat.states[got]["n_band"]) == max(_band_lo(lat.states[m["to"]]["n_band"])
                                                      for m in mv)
    assert R.reference_cohort("C2", tc) == got                       # cached and deterministic
    # a cohort with nothing to drop falls back to itself rather than raising
    bare = next(s for s, st in lat.states.items()
                if not any(st["axes"].get(k) for k in ("icd", "opcs", "bnf")))
    assert R.reference_cohort("C2", bare) == bare


def test_entry_rate_follows_the_item_and_leaves_the_judge_alone():
    """With a target cohort it is the reference cohort's rate; without one it is the demographic
    entry's, unchanged. The second half is the judge's guarantee and the archive's."""
    who, dis, hor, _y, tc, ref = _t3_item_with_target()
    from cohortworld_env.lattice import _shared
    lat = _shared("C2")
    assert ref == R._published_rate(lat, R.reference_cohort("C2", tc), dis, hor)
    old = R.entry_rate(disease=dis, horizon=hor, **who)
    assert old == R._published_rate(lat, R.entry_cohort("C2", who["age_band"], who["sex"]), dis, hor)
    assert R.entry_rate(disease=dis, horizon=hor, target_cohort="", **who) == old


def test_the_declared_delta_is_now_the_correction_back_to_the_patient():
    """`measured = truth - rate(dst)` when the item names its cohort, and it no longer needs a
    reading BEFORE the move -- which is the point: that reading is the answer and is withheld, so the
    old form would have been None on every rollout and the 0.4-weighted term would have paid zero
    forever with nothing saying so."""
    dis, hor, y, dst_rate, m = "K5", 5, 0.03, 0.011, 0.019
    # an episode with NO reading before the move: think, move, read the destination, commit
    msgs = [_asst("", ["cw_look"]), tool_msg("cw_look", {"state_id": "A"}),
            _asst(f"<cohort-think>I expect {m:+.10f}</cohort-think>", ["cw_move"]),
            tool_msg("cw_move", {"state_id": "B", "hit": True, "took": "bnf.drop.03->B"}),
            _obs("B", dis, hor, dst_rate)]
    old = R.edge_prediction(Stub(messages=msgs), dis, hor)
    new = R.edge_prediction(Stub(messages=msgs), dis, hor, truth=y)
    assert old["measured"] is None, "without a truth, the form must still need both readings"
    assert new["measured"] == pytest.approx(y - dst_rate) == pytest.approx(m)
    assert new["dst"] == "B" and new["declared"] == pytest.approx(m)
    assert R.delta_payment(new["declared"], new["measured"]) > 0.0


def test_delta_reference_is_the_correction_distribution_and_c_star_is_derived_from_it():
    """One value per ITEM (`answer - rate(reference)`), not one per edge, and c* is the argmax on
    THAT sample. Recomputed here from the shipped sample, so restating the old constant would fail
    even though the old and new values happen to coincide at 0.0."""
    ms = R.delta_reference()
    assert 400 <= len(ms) <= R.T3_CAL_DELTA_REF_N, len(ms)      # one per item, most of them publish
    lo, hi, step = R.T3_CAL_DELTA_GRID
    grid = sorted({0.0} | {round(lo + i * step, 10) for i in range(int(round((hi - lo) / step)) + 1)})
    assert R.cal_delta_const() == max(grid, key=lambda c: sum(R.delta_accuracy(c, m) for m in ms))
    # the sample really is the item-level correction, checked against one item end to end
    who, dis, hor, y, tc, ref = _t3_item_with_target()
    assert (y - ref) == pytest.approx(y - R._published_rate(
        __import__("cohortworld_env.lattice", fromlist=["_shared"])._shared("C2"),
        R.reference_cohort("C2", tc), dis, hor))
    # the correction is systematically POSITIVE: widening understates this patient's risk
    assert sum(ms) / len(ms) > 0.0


def test_an_item_without_a_target_cohort_scores_exactly_as_it_did():
    """The archive's guarantee, at the level of the reward rather than the replay script: every
    archived task record predates `target_cohort`, so passing it empty must reproduce the old number
    on every variant -- and passing a real one must move at least one of them, or the re-anchoring
    is inert."""
    who, dis, hor, y, tc, _ref = _t3_item_with_target()
    msgs = _episode(dis, hor, 0.05, 0.04, "<cohort-think>I expect -0.0100000000</cohort-think>")
    for variant in ("base", "entry", "cohort_think", CAL):
        a = R.rate_reward(Stub(messages=msgs, last_reply=_commit(y)), truth=y, source="internal",
                          disease=dis, horizon=hor, variant=variant, **who)
        b = R.rate_reward(Stub(messages=msgs, last_reply=_commit(y)), truth=y, source="internal",
                          disease=dis, horizon=hor, variant=variant, target_cohort="", **who)
        assert a == b, f"{variant}: an empty target must be the no-target path"
    moved = []
    for variant in ("entry", "cohort_think", CAL):
        a = R.rate_reward(Stub(messages=msgs, last_reply=_commit(y)), truth=y, source="internal",
                          disease=dis, horizon=hor, variant=variant, **who)
        c = R.rate_reward(Stub(messages=msgs, last_reply=_commit(y)), truth=y, source="internal",
                          disease=dis, horizon=hor, variant=variant, target_cohort=tc, **who)
        moved.append(a["reward"] != c["reward"])
    assert any(moved), "no variant moved: the re-anchoring is not reaching the reward"
