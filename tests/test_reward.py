"""Acceptance tests for the frozen reward contract (docs/reward-spec.md §8).

Every test that can runs against the REAL released data -- the item set, the lattice, the shipped
reward -- rather than against hand-made fixtures. That is deliberate and it is the whole point: the
failure mode this repository keeps paying for is a guard that was written, looked correct, and was
never exercised on real data -- an abort guard that read `metrics` at the wrong nesting level killed a
healthy 120-rollout run; an audit metric that divided by the base rate manufactured its own
correlation; a data-class fix turned T1's loud crash into a silent mis-scoring. Fixtures would have
passed in all three cases.

    ./.venv/bin/python -m pytest tests/ -q
"""
import glob
import json
import os
import random
import statistics as st

import pytest

from verifiers.v1.trace import Trace
from verifiers.v1.taskset import Taskset
from verifiers.v1.utils import loaders

import cohortworld_env.reward as R
from cohortworld_env.internal import (ITEMS, DeltaTaskset, LocateData, LocateTask, LocateTaskset,
                                      RateData, RateTask, RateTaskset)
from cohortworld_env.lattice import _shared
from cohortworld_env.taskset import committed_this_turn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- contract

def test_crps_estimator_is_the_fair_one():
    """§8.1. Pins the n*(n-1) denominator AND the free-outlier property it creates.

    If someone "fixes" `_crps` back to n*n these three numbers all move, and so does every score in
    the project. The free outlier is not a bug to patch: penalising width would break strict
    properness, which is the one thing the design exists to protect.
    """
    assert R.crps([0.02] * 5, 0.02) == pytest.approx(0.0, abs=1e-12)
    assert R.crps([0.02] * 4 + [1.0], 0.02) == pytest.approx(0.0, abs=1e-12)
    assert 1 - R.crps([0.02] * 4 + [1.0], 0.50) == pytest.approx(0.712, abs=1e-9)
    assert 1 - R.crps([0.02] * 5, 0.50) == pytest.approx(0.520, abs=1e-9)


def test_two_value_report_is_a_free_lunch_and_is_therefore_rejected():
    """N1. `commit([0.0, 1.0])` has CRPS exactly 0 for EVERY truth -- maximum reward, zero reads,
    and a zero-gradient attractor once a GRPO group finds it. `MIN_REPORT` is what closes it, so the
    exploit's existence is asserted here rather than assumed: if the estimator ever changes so that
    this stops being free, the length floor can be revisited on evidence."""
    for truth in (0.0, 0.002, 0.05, 0.5, 0.9, 1.0):
        assert R.crps([0.0, 1.0], truth) == pytest.approx(0.0, abs=1e-12)
    assert not R.well_formed([0.0, 1.0])
    assert R.parse_report("ANSWER: [0.0, 1.0]") == []


@pytest.mark.parametrize("payload", [
    [1, 2, 3, 4, 5],                                  # the clean percent convention: 2 of 20 archived
    [0.005, 0.006, 0.007, 1.0, 2.0],                  # a mixed-scale runaway: 18 of 20
    [0.001, 0.0013, 0.002, 0.03, 7.0],
    [0.01, 0.15, 2.0, 3.0, 4.0],
    [0.0, 0.01, 0.5, 1.0, 3.0],
])
def test_out_of_range_rejects_the_whole_report(payload):
    """§8.2 / D2. No dropping, no clamping, no percent rescale.

    Silent dropping is not neutral -- it discards exactly the values that would have hurt, scoring
    the 20 archived out-of-range payloads at mean CRPS 0.1462 against 0.2633 for clamping, which
    makes a malformed report CHEAPER than an honest wide one.
    """
    assert not R.well_formed(payload)
    assert R.parse_report(f"ANSWER: {payload}") == []
    # and the two paths agree, which is the substance of D2
    assert R.parse_values(payload) == [float(x) for x in payload]


def test_percent_branch_is_gone():
    """D2/N11. `_parse_distribution` used to read (1, 100] as a percentage while the commit path
    dropped the same values, and `1.0` fell on the wrong side of the split from `1.5`."""
    assert R.parse_values("ANSWER: [1, 2, 3, 4, 5]") == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert R.parse_report("ANSWER: [1, 2, 3, 4, 5]") == []


def test_no_module_imports_the_deleted_sentinels():
    """§8.9. `NO_ANSWER` / `NO_ANSWER_LOSS` have no referent: no episode is unanswered."""
    assert not hasattr(R, "NO_ANSWER")
    import cohortworld_env.taskset as ts
    assert not hasattr(ts, "NO_ANSWER") and not hasattr(ts, "NO_ANSWER_LOSS")


def test_constants_are_readable_for_a_run_directory():
    """D3. A run that cannot say which constants produced its numbers cannot be compared to one.

    Asserted key by key, and the list of keys is asserted too. This used to check three of eleven
    (`reward_spec_version`, `alpha`, `budget`) and the gap is not academic: a mutation run
    moved `T2_THRESH_LOW` 0.25 -> 0.2, `T2_THRESH_HIGH` 0.75 -> 0.8, `GRACE` 3 -> 2 and
    `FLOOR_REPORT`'s top value 1.0 -> 0.9, and **all four left the suite green**. Three of them are
    written into every run's config.yaml, so four runs' worth of numbers would have been quoted
    against constants nobody had checked.
    """
    c = R.CONSTANTS()
    assert set(c) == {"env_version", "semantic_hash", "items", "phenocal", "sidecar",
                      "reward_spec_version", "alpha", "near", "min_report", "t2_thresholds",
                      "t2_prior_class", "floor_report", "budget", "grace", "per_axis", "panel",
                      "priv", "reference_rates", "t3_default_variant", "t3_step_beta",
                      "t3_tanh_gamma", "t3_tanh_scale",
                      # the cohort_think process term. They are
                      # HERE because CONSTANTS() is what a run directory's config.yaml records: the
                      # variant shipped for a few hours without them, so a run would not have said
                      # what process weight or tolerance it used.
                      "t3_think_w_rank", "t3_think_w_edge", "t3_think_edge_scale",
                      "t3_think_ref_n", "t3_think_ref_seed", "t3_think_ref_sha16",
                      # cohort_think_cal: the resolved L2 scale (derived from the item set,
                      # so a run says which value it trained under), its multiplier and the Q2 cap.
                      "t3_cal_l2_scale", "t3_cal_l2_scale_mult", "t3_cal_q2_cap",
                      # the weight of cohort_think_cal's non-negative edge payment
                      "t3_cal_w_edge",
                      # the DELTA payment -- its weight, and the pinned best constant it is
                      # centred on with the reference sample and grid that derived it. A run that
                      # did not record c* and the reference hash could not say what its process
                      # term was paying ABOVE, which is the whole content of the term.
                      "t3_cal_w_delta", "t3_cal_delta_ref_n", "t3_cal_delta_grid",
                      "t3_cal_delta_const", "t3_cal_delta_ref_sha16",
                      # what an accepted move is worth, and the (disabled) floor for a
                      # rollout that never moved. A run that did not record these could not say
                      # whether it paid for moving at all.
                      "t3_cal_nomove_floor", "t3_cal_w_move"}
    assert c["t3_cal_l2_scale"] == R.cal_l2_scale() > 0 and c["t3_cal_q2_cap"] == R.T3_CAL_Q2_CAP
    assert c["t3_cal_w_edge"] == R.T3_CAL_W_EDGE and c["t3_cal_w_delta"] == R.T3_CAL_W_DELTA
    assert c["t3_cal_delta_const"] == R.cal_delta_const() and len(c["t3_cal_delta_ref_sha16"]) == 16
    assert c["t3_cal_delta_ref_n"] == R.T3_CAL_DELTA_REF_N and c["t3_cal_delta_grid"] == list(R.T3_CAL_DELTA_GRID)
    assert c["t3_cal_w_move"] == R.T3_CAL_W_MOVE and c["t3_cal_nomove_floor"] == R.T3_CAL_NOMOVE_FLOOR
    assert c["env_version"] == R.ENV_VERSION and c["semantic_hash"] == R.semantic_hash()
    assert c["reward_spec_version"] == "2.0.0"
    assert c["alpha"] == 1.25 and c["near"] == 0.25 and c["min_report"] == 5
    assert c["t3_think_w_rank"] == R.T3_THINK_W_RANK and c["t3_think_w_edge"] == R.T3_THINK_W_EDGE
    assert len(c["t3_think_ref_sha16"]) == 16
    assert c["t2_thresholds"] == [0.25, 0.75] and c["t2_prior_class"] == "indistinguishable"
    assert c["floor_report"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert c["budget"] == 26 and c["grace"] == 3 and c["per_axis"] == 6
    assert c["reference_rates"].startswith("reference_rates_2.0.0_")
    # The three input ROOTS. `dist` and `entry_cohort` silently scored T1 against a lattice that was
    # not on disk (7 of 12 archived rollouts moved, every change UPWARD) and no run recorded which
    # panel it had read, so the run that produced the numbers could not be told from the one that
    # did not. They raise now; this is what records the root they raise about.
    # All five INPUT ROOTS, every one of them selectable by a `CW_*` environment variable and none of
    # them recorded anywhere before this. `dist` and `entry_cohort` silently scored T1 against a
    # lattice that was not on disk (7 of 12 archived rollouts moved, every change UPWARD), and the
    # run that produced the numbers could not be told from the one that did not.
    for k in ("items", "panel", "priv", "phenocal", "sidecar"):
        if c[k].startswith("CW_") and "-is-not-set-" in c[k]:
            continue          # an unset data variable is a placeholder that fails at first use, not a path
        if k == "priv" and not os.path.exists(c[k]):
            continue          # the private count store is build/score-only and not released
        assert os.path.isabs(c[k]) and os.path.isdir(c[k]), f"{k} = {c[k]!r}"


def test_the_pinned_constants_are_what_the_contract_says():
    """The pinned constants, with the REASON each has its value, checked against the module.

    This table used to live in `docs/ENVIRONMENT_SPEC.md` §2 and be parsed out of the markdown. That
    document was deleted (680 lines that had drifted from the code in 24 places), so
    the table moved here — to the executable half, where it cannot drift unnoticed.

    A contract in two places drifts; this repository has paid for that five times (`BUDGET` written
    out in five stops, `POP_ZERO_SHARE` hardcoded beside the item field carrying it, two copies of
    the 80/10/10 split, `_GRAN` and its hand-written inverse, and the deleted document itself).
    Keeping the RATIONALE beside the value is the point: a bare `ALPHA = 1.25` invites someone to
    tune it, and the reason it is 1.25 is a measurement, not a preference.
    """
    PINNED = {
        # name: (value, why it is that value)
        "ENV_VERSION": ("7.6.3",
            "bound to the code by content: reward.semantic_hash() must equal "
            "VERSIONED_SEMANTICS[ENV_VERSION], so the reward cannot change without the version "
            "changing and the version cannot change without a recorded reason"),
        "REWARD_SPEC_VERSION": ("2.0.0",
            "versions the REWARD CONTRACT separately from the environment, because they move for "
            "different reasons: a split change with an identical reward moves ENV_VERSION, not this. "
            "A score is comparable to another only if BOTH match"),
        "T3_THINK_W_RANK": (0.05,
            "weight of the rollout's centred cross-item ranking contribution. Sits at the top of the "
            "within-group T3 reward sd (0.012-0.052): visible, never able to outrank an outcome swing"),
        "T3_THINK_W_EDGE": (0.05,
            "weight of the first-edge delta prediction, stated before the destination is read"),
        "T3_THINK_EDGE_SCALE": (0.0026,
            "median |delta| of a one-step edge FROM THE ENTRY COHORT on the released C2 lattice "
            "(127,930 published first-move edges); the e-folding scale of the edge score. The first "
            "value, 0.0098, was the median over arbitrary lattice edges, 4x too wide"),
        "T3_THINK_REF_N": (512, "size of the fixed reference set the ranking term scores against"),
        "T3_THINK_REF_SEED": (42, "the seed that draws it; its content hash is in CONSTANTS()"),
        # cohort_think_cal. Its L2 scale is DERIVED (reward.cal_l2_scale(), 2 * MAE_b over the
        # train items, 0.071754 on the official set) and pinned through CONSTANTS() in
        # tests/test_cohort_think.py -- a derived value is not a literal and has no place here.
        "T3_CAL_L2_SCALE_MULT": (2.0,
            "S2 = MULT * MAE_b: the gradient of (p-y)^2 / S2 equals the L1 term's exactly at "
            "|p-y| = MAE_b, so the rank weight keeps its 'visible, never dominant' calibration"),
        "T3_CAL_Q2_CAP": (4.0,
            "Q2 = (p-y)^2 / S2 reaches 13.9 at |p-y| = 1; the cap flattens it beyond |p-y| = 0.536 "
            "(never reached by a committing policy, observed max 0.231) so one floor row cannot carry "
            "40x the group's within-group spread under mean-only advantages"),
        "T3_CAL_W_EDGE": (0.05,
            "weight of the sign-and-size-gated edge payment. UNPAID: the delta payment replaced it "
            "(edge_cal is still computed and reported as a diagnostic); the value is kept so earlier "
            "scores stay reproducible"),
        "T3_CAL_W_DELTA": (0.4,
            "weight of cohort_think_cal's DELTA payment: max(0, acc(d,m) - acc(c*,m)). The edge "
            "choice is not the learning target, the delta is -- which makes this the term that must "
            "carry the within-group variance, so it is not a 0.05 garnish. Chosen on a replay "
            "sweep over training rollouts: 0.4 is the smallest "
            "swept weight at which an exact declaration (0.632 * w = 0.253) is worth about the "
            "outcome term's own p99 within-group advantage (0.239), while corr(reward, -|p-y|) "
            "0.847 still far exceeds corr(reward, payment) 0.162"),
        "T3_CAL_DELTA_REF_N": (512,
            "train items whose ENTRY-cohort edges make the reference sample of measured deltas that "
            "c* is derived from (16,381 published edges on the official set). Drawn with "
            "T3_THINK_REF_SEED; its content hash is in CONSTANTS()"),
        "T3_CAL_W_MOVE": (0.03,
            "what cohort_think_cal pays for a move the environment ACCEPTED. In training, 35.7% of "
            "rollouts called cw_move with a fabricated id, were refused, stayed in the "
            "entry cohort and took the LARGEST advantages in their group -- GRPO was being taught "
            "to write invalid ids. Swept: 0.03 is inside the window [0.02, 0.075] where the "
            "no-move advantage is negative (-0.0071) and a rollout that moved and did WORSE still "
            "loses (-0.0176), and at the low end of it where cell AUROC is untouched (-0.001)"),
        "T3_CAL_NOMOVE_FLOOR": (None,
            "the alternative to W_MOVE -- replace a no-accepted-move rollout's reward by a constant "
            "-- swept over five quantiles of the movers' reward and DISABLED (None): at every value "
            "that made the no-move advantage non-positive it cost 3-4x more proxy slope and 5-50x "
            "more cell AUROC than the payment, because it throws away the outcome gradient of 35.7% "
            "of rollouts. Kept as a live, tested branch so the decision can be revisited with a "
            "number rather than a rewrite"),
        "T3_CAL_DELTA_GRID": ((-0.05, 0.05, 0.0001),
            "(lo, hi, step) of the constant grid c* is the argmax over, plus exactly 0. It spans "
            "every |delta| a first move plausibly measures (median 0.0026, p99 well inside 0.05) at "
            "a step finer than the declarations a policy writes"),
        "T3_THINK_PROMPT_VARIANTS": (("cohort_think", "cohort_think_cal"),
            "the variants whose prompt IS internal.cohort_think_question -- the question the cold "
            "start was trained on; internal.py reads this tuple, so a "
            "new variant that needs the chain is added here and nowhere else"),
        "ALPHA": (1.25,
            "the measured margin of the oracle over the best zero-information policy is strictly "
            "increasing in alpha (+0.0219 / +0.0696 / +0.0957 at 0 / 1.25 / 2.0); above ~2 the R "
            "term takes more than 63% of per-item reward dispersion"),
        "NEAR": (0.25,
            "T1 partial credit S = NEAR/(1+dist). A cohort one hop from the target reads literally "
            "zero on the asked cell 69.7% of the time, so exactness has to carry the score"),
        "MIN_REPORT": (5,
            "a stated report with fewer values is rejected. At m=2, commit([0.0, 1.0]) scores CRPS "
            "EXACTLY 0.0 for every truth in [0,1] -- maximum reward, zero reads, and a "
            "zero-gradient attractor once a GRPO group finds it"),
        "T2_THRESH_LOW": (0.25, "a report whose median is strictly below this reads as 'lower'"),
        "T2_THRESH_HIGH": (0.75,
            "strictly above reads as 'higher'; between the two, inclusive, is the hedge band"),
        "T2_PRIOR_CLASS": ("indistinguishable",
            "the class the ladder's floor rung produces, so 'did nothing' scores R = 0. "
            "NOT the majority class -- the frozen splits are exactly 1/3 each"),
        "FLOOR_REPORT": ((0.0, 0.25, 0.5, 0.75, 1.0), "the uninformative report on [0, 1]"),
        "LEVEL_NAMES": ({1.0: "stated_commit", 2.0: "stated_text", 3.0: "auto_observed",
                         4.0: "floor"},
            "report_level is a PUBLISHED metric, so its codes are contract, not an internal enum"),
        "BUDGET": (26,
            "the median patient sits 17 operations from any released cohort and 63.2% sit 12 or "
            "more away; the old 12 made most items unreachable by construction"),
        "GRACE": (3,
            "turns still served after navigation is refused, so the episode ends on the agent's "
            "judgement rather than on the harness's silence. out_of_budget fires at BUDGET + GRACE"),
        "PER_AXIS": (6,
            "moves LISTED per (axis, op) group; the rest are counted, not shown. It limits "
            "discoverability, not reachability -- a hidden move called by id still lands"),
    }
    for name, (want, _why) in PINNED.items():
        assert hasattr(R, name), f"the contract pins {name}, which the module does not define"
        assert getattr(R, name) == want, \
            f"the contract says {name} = {want!r}, code says {getattr(R, name)!r}"
    # Every pinned name must be IN the table, or the check is only as wide as whoever wrote it.
    assert set(PINNED) == {"ENV_VERSION", "REWARD_SPEC_VERSION", "ALPHA", "NEAR", "MIN_REPORT",
                           "T2_THRESH_LOW", "T2_THRESH_HIGH", "T2_PRIOR_CLASS", "FLOOR_REPORT",
                           "LEVEL_NAMES", "BUDGET", "GRACE", "PER_AXIS",
                           "T3_THINK_W_RANK", "T3_THINK_W_EDGE", "T3_THINK_EDGE_SCALE",
                           "T3_THINK_REF_N", "T3_THINK_REF_SEED",
                           "T3_CAL_L2_SCALE_MULT", "T3_CAL_Q2_CAP", "T3_CAL_W_EDGE",
                           "T3_CAL_W_DELTA", "T3_CAL_DELTA_REF_N", "T3_CAL_DELTA_GRID",
                           "T3_CAL_W_MOVE", "T3_CAL_NOMOVE_FLOOR",
                           "T3_THINK_PROMPT_VARIANTS"}, sorted(PINNED)
    # Every constant carries a reason. A value with no recorded reason is a value nobody can defend.
    for name, (_want, why) in PINNED.items():
        assert len(why) > 20, f"{name} has no recorded rationale"


def test_the_report_levels_are_the_numbers_the_metric_publishes():
    """`report_level` is a PUBLISHED metric -- reward-spec 6.6 requires its distribution beside any
    score, and an effect can live entirely inside one level -- so the codes are contract, not an
    internal enum. Collapsing `STATED_TEXT` 2.0 onto 1.0 left the suite green, because the only
    assertions on a level were `level in R.LEVEL_NAMES`, which cannot move when the constant moves.
    """
    assert (R.STATED_COMMIT, R.STATED_TEXT, R.AUTO_OBSERVED, R.FLOOR) == (1.0, 2.0, 3.0, 4.0)
    assert R.LEVEL_NAMES == {1.0: "stated_commit", 2.0: "stated_text",
                             3.0: "auto_observed", 4.0: "floor"}


# --------------------------------------------------------------------------- D1: the wiring

@pytest.mark.parametrize("taskset_id,taskset,task,data", [
    ("cohortworld-internal", RateTaskset, "RateTask", "RateData"),
    ("cohortworld-locate", LocateTaskset, "LocateTask", "LocateData"),
    ("cohortworld-delta", DeltaTaskset, "DeltaTask", "DeltaData"),
])
def test_each_taskset_id_rebuilds_its_own_task_class(taskset_id, taskset, task, data):
    """D1. The served worker discards the object `load()` yielded and rebuilds it as
    `type(taskset).task_type()`, so a taskset yielding three classes silently scores two of them by
    the third's rubric. That is how 120 T1 rollouts came back a constant -1.0."""
    module = loaders._import_plugin(taskset_id, "taskset", "verifiers.v1.tasksets")
    cls = loaders._plugin_class(module, Taskset, "taskset")
    assert cls is taskset
    assert cls.task_type().__name__ == task
    assert cls.task_type().data_type().__name__ == data


def test_data_classes_reject_a_row_that_is_missing_its_own_fields():
    """D1's other half. The one-data-class workaround gave every task-specific field a default, so a
    T1 row validated as a T3 row and scored as `true_rate = -1.0` forever instead of failing."""
    with pytest.raises(Exception):
        RateData(idx=0, prompt="x", version="C2")            # no disease/horizon/true_rate
    with pytest.raises(Exception):
        LocateData(idx=0, prompt="x", version="C2")          # no target_cohort


# --------------------------------------------------------------------------- T1 geometry

def test_dist_is_zero_only_for_the_same_cohort():
    """§8.4. `dist == 0` iff the state IS the target, on every released cohort of every version and
    on 1000 random pairs each. Coarseness must be in the axis set or two cohorts recorded at
    different granularity read as distance 0."""
    random.seed(42)
    for ver in ("C1", "C2", "default"):
        lat = _shared(ver)
        if lat is None:
            continue
        keys = {}
        for sid, d in lat.states.items():
            keys.setdefault(R._axset(d), []).append(sid)
        assert len(keys) == len(lat.states), f"{ver}: colliding axis sets"
        ids = list(lat.states)
        for _ in range(1000):
            a, b = random.choice(ids), random.choice(ids)
            assert (R.dist(a, b) == 0) == (a == b)


# --------------------------------------------------------------------------- the other defects

def test_reference_rate_is_defined_for_every_unseen_item():
    """N8. 104 of the OLD set's 304 internal unseen items had no train (disease, horizon) base rate,
    and the old code returned 0.0 -- a value indistinguishable from a genuinely perfect reference. It
    inflated |skill| by about 2.4x.

    The count is the released item set's (docs/ENVIRONMENT_SPEC.md). It is
    pinned rather than read off the file, because the property under test is "defined for EVERY
    item" and a loop over a truncated or empty file passes it vacuously.
    """
    n = 0
    for line in open(f"{ITEMS}/t3_rate_unseen.jsonl"):
        d = json.loads(line)
        b = R.base_rate("internal", d["disease"], int(d["horizon"]))
        assert 0.0 <= b <= 1.0
        n += 1
    assert n == 943   # the unseen share of a 90:5:5 split


def test_disease_resolution_refuses_a_code_it_does_not_have():
    """N2. A one-character query used to match the first disease whose English NAME contained that
    letter, and `observe` then served a real rate for the wrong outcome under `status: published`."""
    lat = _shared("C2")
    for bad in ("H", "M", "N", "K", "C56", "zzz"):
        assert lat._resolve_disease(bad) is None
    assert lat._resolve_disease("K5") == "K5"
    assert lat._resolve_disease("digestive [K5]") == "K5"


def test_legal_move_summary_truncates_by_default():
    """N4. The truncation was written, documented and never switched on, because neither caller
    passed the argument: 0 of 1,017 served move groups in one run were cut."""
    from cohortworld_env import legal_moves as LM
    lat = _shared("C2")
    entry = lat.entry_for("55-59", "Male")
    full = LM.summary(entry, per_axis=0)
    cut = LM.summary(entry)
    assert len(json.dumps(cut)) < len(json.dumps(full))
    for group in cut["legal_moves"].values():
        assert len(group["moves"]) <= 7            # 6 plus the explicit "N more not shown" marker


def test_delta_taskset_serves_only_answerable_items():
    """N3/N12/N13. A T2 item names its edge in one lattice and its outcome at one granularity, and on
    the OLD item set 417 of 625 were different versions -- an agent standing on the edge cannot read
    the outcome -- while 32 more named a cross-version edge the action set withholds. The taskset
    silently dropped them, so the three splits served different tasks.

    The item builder repairs it at the SOURCE (`gen/t2_compose.py`): every emitted item satisfies the
    clause, so the served count now EQUALS the file's length and the skip count is zero. That is the
    stronger property and it is what is asserted -- a served count alone would still pass if the
    builder regressed and the taskset went back to dropping a third of the file.
    """
    from cohortworld_env.internal import DeltaConfig
    for split, expect in (("train", 297), ("seen", 60), ("unseen", 108)):
        tasks = list(DeltaTaskset(DeltaConfig(id="t2", split=split)).load())
        assert len(tasks) == expect
        assert len(tasks) == sum(1 for _ in open(f"{ITEMS}/t2_delta_{split}.jsonl")), \
            f"{split}: the taskset is still dropping items the builder emitted"
        for t in tasks:
            lat = _shared(t.data.version)
            assert t.data.start_cohort in lat.states
            assert t.data.disease in lat.didx
            assert t.data.answer_direction in ("up", "down", "indistinguishable")


def test_t2_reference_class_is_the_hedge_the_floor_produces():
    """C7/S4. `T2_PRIOR_CLASS` was justified as "the train-split majority, 0.3823 of 429" -- and on
    the OLD item set the SERVED train majority was `down` (46/125), not `indistinguishable`
    (43/125). The constant was right and the reason was not: it is the reference because it is what
    the ladder's floor rung produces, so "did nothing" scores R = 0.

    On the released item set the justification cannot even be restated, and that is the point of
    keeping this test: T2 is composed with the three directions EXACTLY balanced in every split, so there is no
    majority class to appeal to. `T2_PRIOR_CLASS` still names the hedge, still because the floor rung
    produces it. The assertion therefore moved from "the majority is `down`" to "there is no majority
    at all", which is the same invariant -- the constant is not the majority -- stated against the
    item set that actually ships.
    """
    import collections
    from cohortworld_env.internal import DeltaConfig
    assert R.t2_decision(R.delta_report(type("T", (), {"messages": [], "last_reply": ""})(),
                                        "K5", 5)[0]) == R.T2_PRIOR_CLASS
    balance = {}
    for split in ("train", "seen", "unseen"):
        c = collections.Counter(t.data.answer_direction
                                for t in DeltaTaskset(DeltaConfig(id="t2", split=split)).load())
        balance[split] = dict(c)
        assert set(c) == {"up", "down", "indistinguishable"}, split
        assert len(set(c.values())) == 1, f"{split} is no longer direction-balanced: {dict(c)}"
    assert balance["train"] == {"down": 99, "indistinguishable": 99, "up": 99}
    assert balance["seen"] == {"down": 20, "indistinguishable": 20, "up": 20}
    assert balance["unseen"] == {"down": 36, "indistinguishable": 36, "up": 36}
    assert R.T2_PRIOR_CLASS == "indistinguishable"


def test_t2_degenerate_optimum_exists_and_alpha_demotes_it():
    """§4.1/§4.3, with the correction the arithmetic forces.

    Reproduces Prop. 9(b) in our own item set: with a PURE proper score an honest reporter whose
    pick is systematically WRONG (q = 0) scores 0.83805, above every policy with q < 0.75, and the
    minimum sits at chance. That degenerate optimum is real and is why T3's argument does not excuse
    T2.

    What `ALPHA * R` does about it, measured: it lifts the ceiling from 1.000 to 1.772 while leaving
    the q = 0 policy at 0.83805, so the degenerate optimum stops competing with competence, and the
    reward is monotone in accuracy over [1/3, 1] -- the region a trained policy occupies -- for both
    a point-mass and an honest reporter.

    What it does NOT do, contrary to reward-spec 4.3: it does not make `r(q)` increasing BELOW
    chance. That derivation assumes the decision tracks the pick, which is true of a point mass and
    false of the honest reporter that creates the degeneracy -- below chance the honest report's
    median sits on the hedge class, so `R` is constant there and no alpha can tilt it. Asserted here
    rather than papered over, because a test that claimed monotonicity everywhere would fail the
    moment anyone re-derived it.
    """
    prior = {"indistinguishable": 0.3823, "down": 0.3543, "up": 0.2634}
    Y = {"down": 0.0, "indistinguishable": 0.5, "up": 1.0}

    def honest(pick, q, m=100):
        v = []
        for k in prior:
            w = q if k == pick else (1 - q) / 2
            v += [Y[k]] * max(0, round(w * m))
        return v or [Y[pick]]

    def point(pick, q):
        return [Y[pick]] * 5

    def expected(q, alpha, report):
        tot = 0.0
        for truth, pt in prior.items():
            for pick in prior:
                pr = q if pick == truth else (1 - q) / 2
                v = report(pick, q)
                r = ((1.0 if R.t2_decision(v) == truth else 0.0)
                     - (1.0 if R.T2_PRIOR_CLASS == truth else 0.0))
                tot += pt * pr * (alpha * r + 1.0 - R.crps(v, Y[truth]))
        return tot

    full = [i / 40 for i in range(41)]
    pure = [expected(q, 0.0, honest) for q in full]
    assert pure[0] == pytest.approx(0.83805, abs=1e-4)          # the degenerate optimum
    assert min(pure) == pytest.approx(0.78820, abs=1e-4)        # ... and it is not the minimum
    assert min(pure) < pure[0] and pure.index(min(pure)) in range(11, 16)   # minimum near chance

    above_chance = [1 / 3 + i * (2 / 3) / 40 for i in range(41)]
    for report in (honest, point):
        e = [expected(q, R.ALPHA, report) for q in above_chance]
        assert all(b >= a - 1e-9 for a, b in zip(e, e[1:])), "not monotone above chance"
    assert expected(1.0, R.ALPHA, honest) == pytest.approx(1.77212, abs=1e-4)
    assert expected(0.0, R.ALPHA, honest) == pytest.approx(0.83805, abs=1e-4)


def test_commit_accepts_a_string_payload():
    """N6. FastMCP validates against the annotation before the tool body runs, so a `list[float]`-only
    signature turned a complete answer into an error message carrying neither the values nor the word
    "committed" -- 3 of 360 archived rollouts never recovered and were scored the flat worst case."""
    import asyncio
    from cohortworld_env.toolset import CohortWorldToolset, CohortWorldToolsetConfig
    ts = CohortWorldToolset(CohortWorldToolsetConfig())
    out = asyncio.run(ts.commit("[0.001, 0.0015, 0.0025, 0.004, 0.]"))
    assert out["committed"] is True
    assert out["rates"] == [0.001, 0.0015, 0.0025, 0.004, 0.0]
    assert R.well_formed(out["rates"])


# ------------------------------------------------------- the version is bound to what it names

def test_the_version_names_the_code_it_ships_with():
    """THE hole this file existed to have and did not close.

    Mutating `ENV_VERSION` to "9.9.9" once left all 276 tests GREEN. A version nothing checks
    is not a version, it is a comment that happens to be written into every run directory -- and the
    whole of D3 is that `NO_ANSWER` moved 0.0 -> -1.0 and `BUDGET` 12 -> 26 on one day with nothing
    on disk saying so. `REWARD_SPEC_VERSION` was caught only by ACCIDENT: `_reference_path` globs the
    table by that string and raises when the glob comes back empty, so renaming the table alongside a
    bump would have turned it green again.

    `reward.semantic_hash()` is a content hash of every module whose code can change a reward, with
    comments, docstrings and formatting normalised away by `ast.unparse`. Both directions are red:

      * change ALPHA, a threshold, a ladder rung, a branch -> the hash moves under an unchanged
        version -> the first assert fires;
      * bump ENV_VERSION with nothing else changed -> the version is not in the table -> the second.

    Verified by mutation, both directions: `ALPHA 1.25 -> 1.30` and
    `ENV_VERSION -> "9.9.9"` each fail this test, with `git diff --numstat` checked non-zero
    first so a mutation that silently did nothing could not be read as a pass.
    """
    got = R.semantic_hash()
    assert R.ENV_VERSION in R.VERSIONED_SEMANTICS, (
        f"ENV_VERSION is {R.ENV_VERSION!r} and no code is recorded under it.\n"
        f"Either the version was bumped with no semantic change -- put it back -- or the change is\n"
        f"real: add {R.ENV_VERSION!r}: {got!r} to reward.VERSIONED_SEMANTICS, and say what moved in\n"
        f"docs/ENVIRONMENT_SPEC.md.")
    assert R.VERSIONED_SEMANTICS[R.ENV_VERSION] == got, (
        f"the frozen surface changed under ENV_VERSION {R.ENV_VERSION!r}: recorded "
        f"{R.VERSIONED_SEMANTICS[R.ENV_VERSION]!r}, code hashes to {got!r}.\n"
        f"A reward change is a MAJOR bump. Raise ENV_VERSION (and REWARD_SPEC_VERSION if any value\n"
        f"in docs/ENVIRONMENT_SPEC.md moved), add the new pair to reward.VERSIONED_SEMANTICS, and\n"
        f"record which scores it invalidates. Diff the surface with:\n"
        f"    python -c 'import cohortworld_env.reward as R; print(R._semantic_source())'")


def test_the_hash_covers_every_module_that_can_change_a_reward():
    """The binding is only as wide as its file list, and a list is the easy thing to forget to grow.

    Every module the frozen packages import must be in `_SEMANTIC_MODULES` or the hash is blind to
    it -- which is how `legal_moves.summary(per_axis=6)` sat outside the contract through the freeze.
    `cohortworld_patient` is excluded on purpose and asserted excluded, so "not in the list" is a
    decision rather than an oversight.
    """
    import pathlib
    here = pathlib.Path(REPO)
    shipped = {str(p.relative_to(here)) for pkg in ("cohortworld_env", "cohortworld_phenocal")
               for p in (here / pkg / pkg).glob("*.py") if p.name != "__init__.py"}
    assert shipped <= set(R._SEMANTIC_MODULES), (
        f"not hashed, so not frozen: {sorted(shipped - set(R._SEMANTIC_MODULES))}")
    assert not any("patient" in m for m in R._SEMANTIC_MODULES)
    for rel in R._SEMANTIC_MODULES:
        assert os.path.exists(f"{REPO}/{rel}"), rel


def test_the_hash_ignores_comments_and_docstrings_and_not_values():
    """A hash that moved on a typo would be bypassed within a week, and one that did not move on a
    constant would be decoration. Both halves are exercised on the real source rather than asserted
    in prose: a comment and a docstring are appended to reward.py's text and must not move it; a
    digit in ALPHA must."""
    src = open(f"{REPO}/cohortworld_env/cohortworld_env/reward.py").read()
    assert "ALPHA = 1.25" in src
    noise = src.replace("\n\n\n", "\n\n\n# a comment nobody should have to justify\n")
    assert _surface(noise) == _surface(src), "a comment moved the hash"
    assert _surface(src.replace('"""Weight on the discrimination term', '"""REWRITTEN', 1)) \
        == _surface(src), "a docstring moved the hash"
    assert _surface(src.replace("ALPHA = 1.25", "ALPHA = 1.30", 1)) != _surface(src), \
        "a pinned constant did NOT move the hash"


_surface = R._normalise


# ------------------------------------------------------- constants with a semantic consequence
#
# Everything below was a MUTATION SURVIVOR: the value moved, the tests stayed green.
# A literal assertion pins the number; each test here pins what the number DOES, because a table of
# literals and a self-referential `== R.THING` are the same amount of protection, which is none.

@pytest.mark.parametrize("median,expect", [
    (0.0, "down"), (0.2499, "down"), (0.25, "indistinguishable"), (0.2501, "indistinguishable"),
    (0.5, "indistinguishable"), (0.7499, "indistinguishable"), (0.75, "indistinguishable"),
    (0.7501, "up"), (1.0, "up"),
])
def test_the_t2_decision_boundaries_are_the_pinned_thresholds(median, expect):
    """`T2_THRESH_LOW` 0.25 -> 0.2 and `HIGH` 0.75 -> 0.8 both survived, because the only test that
    fed `t2_decision` anything fed it medians of exactly 0, 0.5 and 1.0 -- and every pair in
    (0, 0.5) x (0.5, 1) agrees on those three. The boundary is the contract: both comparisons are
    STRICT, so a median sitting exactly on a threshold is the hedge."""
    assert R.t2_decision([median] * 5) == expect


def test_an_unresolvable_id_is_scored_as_far_not_near():
    """`dist`'s sentinel is 99 and the number is load-bearing: `NEAR / (1 + 99)` = 0.0025 against
    `NEAR / (1 + 1)` = 0.125 for a genuine one-hop miss. Mutating 99 -> 0 survived the suite and
    turns a garbage commit into a PERFECT score, since `dist == 0` is what "same cohort" means."""
    lat = _shared("C2")
    real = next(iter(lat.states))
    assert R.dist("C2:0000000000000000", real) == 99
    assert R.locate_score("C2:0000000000000000", real) == pytest.approx(R.NEAR / 100.0)
    assert R.locate_score(real, real) == 1.0
    # An id from no CohortWorld lattice at all resolves the same way -- and does NOT raise, which is
    # what keeps one malformed commit from voiding a whole eval.
    assert R.dist("nosuchversion:abc", real) == 99


def test_a_missing_panel_raises_at_scoring_time_instead_of_paying_more():
    """The determinism hole: scoring was deterministic given (item, trajectory) and NOT given the
    files on disk. `lattice._shared` swallows `FileNotFoundError` and returns None -- right for a
    rollout, wrong for a score -- so `dist` returned its no-such-cohort sentinel for every id in the
    missing version and `entry_cohort` returned `""`. Rescoring a 12-rollout T1 run with one
    `CohortWorld_default_*_states.jsonl` hidden moved **7 of 12 rollouts, max |delta| 0.1225, every
    change UPWARD**: a missing input made the agent look better, silently, with `report_level`
    unchanged. Run in a subprocess because the fix is about what the process inherits."""
    import subprocess
    import sys
    import tempfile
    code = ("import cohortworld_env.reward as R\n"
            "for call in (lambda: R.dist('C2:aaaaaaaaaaaaaaaa', 'C1:bbbbbbbbbbbbbbbb'),\n"
            "             lambda: R.entry_cohort('C2', '55-59', 'Male')):\n"
            "    try:\n"
            "        call(); print('SILENT')\n"
            "    except FileNotFoundError as e:\n"
            "        print('RAISED')\n")
    with tempfile.TemporaryDirectory() as empty:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env={**os.environ, "CW_PANEL": empty, "CW_PRIV": empty})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split() == ["RAISED", "RAISED"], out.stdout + out.stderr[-2000:]


def test_the_move_listing_is_truncated_at_the_pinned_width():
    """`per_axis` was a DEFAULT ARGUMENT of `legal_moves.summary` through the first freeze: in no
    config.yaml, no `CONSTANTS()`, no line of §2 -- so 6 could become 3, moving the achievable policy
    set (which §8 calls MAJOR) while every score stayed formally comparable. It withholds 13,179
    legal moves across the archive and **no action reveals them**, so the width is contract."""
    from cohortworld_env import legal_moves as LM
    # SIX, written out. Reading `R.PER_AXIS` on both sides is self-referential -- the expected width
    # moves with the constant -- and `PER_AXIS 6 -> 3` survived the first version of this test for
    # exactly that reason. The whole point is that the width is a NUMBER an agent's action set
    # depends on, so the number belongs in the assertion.
    assert R.PER_AXIS == 6
    lat = _shared("C2")
    entry = lat.entry_for("55-59", "Male")
    assert LM.summary(entry) == LM.summary(entry, per_axis=6)
    cut, full = LM.summary(entry), LM.summary(entry, per_axis=0)
    truncated = 0
    for k, group in cut["legal_moves"].items():
        shown = [m for m in group["moves"] if m["id"] != "..."]
        assert len(shown) <= 6, f"{k} lists {len(shown)} moves, over the pinned width of 6"
        if group["n"] > 6:
            truncated += 1
            assert len(shown) == 6
            assert group["moves"][-1]["note"] == f"{group['n'] - 6} more not shown"
    assert truncated, "no group was truncated here; the fixture no longer exercises the width"
    assert len(json.dumps(cut)) < len(json.dumps(full))


def test_the_taskset_stops_at_num_tasks_not_one_past_it():
    """`served >= c.num_tasks` -> `>` survived, because every test used the default 1,000,000 and the
    limiter's boundary was never reached. Measured under the mutation: `num_tasks=3` served 4."""
    from cohortworld_env.internal import DeltaConfig
    for n in (1, 3, 7):
        assert len(list(DeltaTaskset(DeltaConfig(id="t2", split="train", num_tasks=n)).load())) == n


def test_the_outcome_table_is_this_versions_and_not_another(capsys):
    """`lattice._GRAN` maps version -> outcome granularity, and swapping its C1 and C2 codes survived
    the suite twice over: nothing asserted an `at_risk` band and nothing asserted a T2 disease name.

    Both are live. `observe` reads the at-risk store at `_GRAN[version]`, and the store is keyed by
    granularity, so a wrong code reads a DIFFERENT disease's cell. `internal._GRAN_TABLE` inverts the
    same mapping to name the disease a T2 prompt asks about, and a wrong code finds nothing and
    renders the bare letter -- the question then asks about `"H" (ICD-10 H)`, which is the exact
    defect `_GRAN_TABLE` was introduced to fix.
    """
    from cohortworld_env.lattice import _GRAN, _atrisk
    from cohortworld_env.internal import DeltaConfig, _GRAN_TABLE
    assert _GRAN == {"C1": 1, "C2": 2, "default": 3}
    assert _GRAN_TABLE == {1: "C1", 2: "C2", 3: "default"}

    # A cell whose answer DIFFERS between granularities, so the code is exercised and not just typed.
    lat = _shared("C2")
    sid, disease, horizon = "C2:000990ef1cf8884b", lat.diseases[21], lat.horizons[0]
    assert lat.goto(sid)
    assert lat.observe(disease, horizon)["at_risk"] == "10-12"
    assert _atrisk("C2", sid, 1, 0, 21) == "<10 still at risk"      # C1's granularity: wrong cell

    # 110 of the 429 T2 train items are granularity 1. Under a swapped table 79 of the 125 served
    # items render the disease as its own bare code; under the right one, none do.
    import re
    bare = [t.data.idx for t in DeltaTaskset(DeltaConfig(id="t2", split="train")).load()
            if (m := re.search(r'"([^"]+)" \(ICD-10 ([^)]+)\)', t.data.prompt))
            and m.group(1) == m.group(2)]
    assert bare == [], f"{len(bare)} T2 prompts name the disease by its own code: {bare[:5]}"
