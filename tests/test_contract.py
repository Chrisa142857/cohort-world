"""The freeze gate: what must hold for RL training to be run against this environment unchanged.

`tests/test_reward.py` pins the DERIVATION -- the numbers `docs/reward-spec.md` argues for. This
file pins the WIRING and the training precondition, which is a different question: not "is the
reward the one we designed" but "does the reward this code actually produces carry a gradient, on
the right key, for every task".

D1 was not a wrong number; it was a CONSTANT one -- 120 T1 rollouts on 120 different items all
scored -1.0, because `vf.Taskset`'s first type parameter rebuilds the served task as the taskset's
declared class and `LocateTask` was being rebuilt as `InternalTask` and scored by T3's rubric. GRPO
normalises advantage within a group, so a constant reward is an advantage of exactly zero: the run
looks healthy, burns a GPU for hours, and teaches the policy nothing. Nothing in the harness reports
it. `degenerate` below is the detector, and `test_the_degeneracy_detector_can_fail` is its control.

Everything here is GPU-free.
"""
import collections
import glob
import json
import os
import re
import statistics as st
import subprocess
import sys
import textwrap

import pytest

if not (os.environ.get("CW_ITEMS") and os.environ.get("CW_OFFICIAL")):
    pytest.skip("CW_ITEMS or CW_OFFICIAL is unset: the environment contract tests need the released item set and "
                "official/ data (see data/README.md)", allow_module_level=True)

from verifiers.v1.trace import Trace

import cohortworld_env.reward as R
from cohortworld_env.internal import (DeltaData, DeltaTask, DeltaTaskset, LocateData, LocateTask,
                                      RateData, RateTask)
from cohortworld_env.taskset import _spent
from cohortworld_phenocal.taskset import PhenoCalData, PhenoCalTask

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASK_CLASSES = [RateTask, LocateTask, DeltaTask, PhenoCalTask]


# --------------------------------------------------------------------------- THE GRPO CONDITION

def degenerate(rewards) -> str | None:
    """Why this group of rollouts carries no learning signal, or None if it does.

    A GRPO group's advantage is `(r - mean(r)) / sd(r)`. At sd = 0 every advantage is 0 and the
    update is a no-op no matter how well or badly the rollouts went, so a constant reward is not a
    bad reward -- it is the absence of one. The second clause catches the near-miss the first cannot:
    a rubric that resolves 120 distinct behaviours onto two values is technically non-constant and
    still tells the policy almost nothing.

    Groups smaller than 8 are not judged: a real GRPO group is 8-16 rollouts and a handful of
    smoke rollouts can legitimately tie.
    """
    if len(rewards) < 8:
        return None
    if st.pvariance(rewards) <= 0.0:
        return f"zero variance across {len(rewards)} rollouts (reward is the constant {rewards[0]})"
    if len(set(round(r, 9) for r in rewards)) < 3:
        return f"only {len(set(round(r, 9) for r in rewards))} distinct rewards in {len(rewards)}"
    return None


def test_the_degeneracy_detector_can_fail():
    """T05, the positive control. A guard nobody has watched fail is the bug class this repo keeps
    paying for -- so the detector is exercised on a group it MUST reject and on one it must accept,
    in the same test, before anything downstream is allowed to trust a green result."""
    assert degenerate([-1.0] * 120) is not None                  # D1, exactly as it happened
    assert degenerate([0.0] * 8) is not None
    assert degenerate([0.0] * 4 + [1.0] * 4) is not None         # non-constant, still 2 values
    assert degenerate([i / 8 for i in range(8)]) is None
    assert degenerate([-1.0] * 4) is None                        # too small to judge


# The zero-information constant each task's reward form dictates when NOTHING was done. Only T1 has
# one: `locate_reward` is `S(final) - S(entry)`, so an agent that never leaves the entry cohort
# scores exactly 0.0 on every item, by construction and not by coincidence. T2 and T3 answer through
# the report ladder, whose level-4 `FLOOR_REPORT` is a constant REPORT but not a constant REWARD --
# it is scored against each item's own truth -- so a constant reward there is unexplained and stays a
# failure.
NULL_POLICY_CONSTANT = {"LocateTask": 0.0}


def null_policy_defect(task_name, rewards) -> str | None:
    """Why a group in which NO rollout acted is still a defect, or None if the constant is the
    expected one. Split out of the test so it has a positive control -- see the test below."""
    want = NULL_POLICY_CONSTANT.get(task_name)
    if want is None:
        return (f"no rollout acted and there is no known zero-information constant for "
                f"{task_name}, so this constant reward is UNEXPLAINED")
    if not all(abs(r - want) < 1e-12 for r in rewards):
        return (f"no rollout acted, so the reward must be exactly {want} (S(final) - S(entry) at "
                f"the entry cohort); it is {sorted(set(rewards))}")
    return None


def test_the_null_policy_branch_can_fail():
    """The positive control for the branch above, in the same shape as
    `test_the_degeneracy_detector_can_fail`. A branch that only ever returns None is an exemption,
    not a check, and this file's own docstring calls that the bug class this repo keeps paying for.
    """
    assert null_policy_defect("LocateTask", [0.0] * 12) is None        # a real null-policy case
    assert null_policy_defect("LocateTask", [0.5] * 12) is not None    # constant, but WRONG constant
    assert null_policy_defect("LocateTask", [0.0] * 11 + [1e-6]) is not None
    assert null_policy_defect("RateTask", [1.0] * 12) is not None      # no constant is expected here
    assert null_policy_defect("DeltaTask", [1.0] * 12) is not None


# --------------------------------------------------------------------------- reward key and range

@pytest.mark.parametrize("cls,key", [(RateTask, "rate"), (LocateTask, "locate"),
                                     (DeltaTask, "delta"), (PhenoCalTask, "rate")])
def test_each_task_declares_exactly_one_reward_and_it_is_the_expected_one(cls, key):
    """T01. `@vf.reward` sets a `reward` attribute on the method; the served worker reads it off the
    task class it rebuilt. A task scored under another task's key is D1's visible symptom -- the D1
    records carried `crps` and no `locate` -- so the key is asserted, not assumed."""
    names = sorted(n for n in dir(cls) if getattr(getattr(cls, n, None), "reward", False))
    assert names == [key]


# --------------------------------------------------------------------------- the mandatory report

class Stub:
    """The smallest thing the reward's trace readers accept. They go through `getattr`, so a real
    `Trace` is not needed to exercise the ladder -- and must not be, since the point of these tests
    is the case where the harness produced nothing at all."""

    def __init__(self, messages=(), last_reply=""):
        self.messages = list(messages)
        self.last_reply = last_reply
        self.info = {}


def tool_msg(name, payload):
    return {"role": "tool", "name": name, "content": json.dumps(payload)}


@pytest.mark.parametrize("trace", [
    Stub(),                                                        # provider error before turn 1
    Stub(last_reply="I don't know."),                              # answered nothing
    Stub(last_reply="ANSWER: [1, 2, 3, 4, 5]"),                    # malformed, rejected whole
    Stub(last_reply="ANSWER: 0.4"),                                # too short
    Stub([tool_msg("cw_look", {"state_id": "C2:abc", "hit": True})]),
])
def test_there_is_no_terminal_state_without_a_scored_report(trace):
    """T31/D4, the structural half. This is what makes "the commit is a gate with no gradient behind
    it" unreintroducible: there is no input to the ladder that yields an empty report, so there is no
    branch left in which a score can be a constant standing in for silence. `crps` raises on an empty
    report rather than returning a sentinel, so a regression here is loud."""
    for report in (R.rate_report(trace, "K5", 5), R.delta_report(trace, "K5", 5)):
        vals, level, malformed = report
        assert vals, "empty report -- the ladder fell through"
        assert level in R.LEVEL_NAMES and malformed in (0.0, 1.0)
        loss = R.crps(vals, 0.02)
        assert loss == loss and -0.5 <= loss <= 1.0        # scoreable: finite and in range
    sid, level = R.locate_report(trace, "C2:entry")
    assert sid and level in R.LEVEL_NAMES


def test_the_floor_is_reached_only_when_the_agent_neither_stated_nor_read():
    """The floor must be the LAST rung, not a shortcut. An agent that read the asked cell is scored
    on its reading even in total silence -- that is the whole of "navigated correctly, read six
    cells, format broke" no longer being indistinguishable from "did nothing"."""
    read = Stub([tool_msg("cw_look", {"state_id": "C2:abc"}),
                 tool_msg("cw_observe", {"status": "published", "rate": 0.031,
                                         "disease": "K5", "horizon": 5})])
    assert R.rate_report(read, "K5", 5) == ([0.031], R.AUTO_OBSERVED, 0.0)
    assert R.rate_report(read, "K5", 1)[1] == R.FLOOR          # a DIFFERENT cell was asked
    assert R.rate_report(Stub(), "K5", 5) == (list(R.FLOOR_REPORT), R.FLOOR, 0.0)


def test_a_malformed_commit_is_flagged_and_still_scored():
    """D2 + D4 together. The report is rejected AS A WHOLE (no dropping, no clamping) and the episode
    still gets a score off the rung below -- rejecting the payload must not resurrect the constant."""
    tr = Stub([tool_msg("cw_look", {"state_id": "C2:abc"}),
               tool_msg("cw_observe", {"status": "published", "rate": 0.031,
                                       "disease": "K5", "horizon": 5}),
               tool_msg("cw_commit", {"committed": True, "cohort_id": "C2:abc",
                                      "rates": [1.0, 2.0, 3.0, 4.0, 5.0]})])
    vals, level, malformed = R.rate_report(tr, "K5", 5)
    assert malformed == 1.0 and level == R.AUTO_OBSERVED and vals == [0.031]


# --------------------------------------------------------------------------- the one parse rule

@pytest.mark.parametrize("text,expect", [
    ("ANSWER: [0.02, 0.03, 0.04, 0.05, 0.06]", [0.02, 0.03, 0.04, 0.05, 0.06]),
    ("ANSWER: [2%, 3%, 4%, 5%, 6%]", [0.02, 0.03, 0.04, 0.05, 0.06]),
    ("ANSWER: [0., .6%, .8%, .8%, .8%]", [0.0, 0.006, 0.008, 0.008, 0.008]),
    ("ANSWER: 0.5% 0.6% 0.7% 0.8% 0.9%", [0.005, 0.006, 0.007, 0.008, 0.009]),
    ('ANSWER: {"samples": [0.01, 0.02, 0.03, 0.04, 0.05]}', [0.01, 0.02, 0.03, 0.04, 0.05]),
    ("ANSWER: .05 .06 .07 .08 .09", [0.05, 0.06, 0.07, 0.08, 0.09]),
])
def test_a_percent_sign_is_the_only_thing_that_rescales(text, expect):
    """The `%` half of D2, and the 100x hole it closes.

    `commit` accepts a string so that a payload FastMCP would have rejected still reaches the trace,
    and its docstring cites `'[0., .6%, .8%, .8%, .8%]'` as one it rescues. That string parsed to
    `[0.0, 0.6, 0.8, 0.8, 0.8]`: a model that wrote 0.8% scored as a confident 80%, `well_formed`,
    `report_level = stated_commit`, `report_malformed = 0`. `well_formed`'s [0, 1] rule catches
    `[2%, ..., 6%]` and by construction CANNOT catch a percent convention already below 1.

    Reading `%` is not the "clever percent heuristic" D2 forbids -- it does not infer a convention
    from the magnitudes, it reads the unit the model wrote. Exercised on the whole archive before it
    shipped: of 2,753 answer strings, 63 parse differently and 8 flip from rejected to accepted, all
    8 being genuine five-value percent reports (`[2%, 4%, 5%, 7%, 10%]`); none flips the other way.
    """
    got = R.parse_values(text.split(":", 1)[1])
    assert got == pytest.approx(expect, abs=1e-12)
    assert R.well_formed(got)


def test_absence_of_a_percent_sign_never_rescales():
    """The other half, and the one that keeps the rule from becoming a heuristic. `[1, 2, 3, 4, 5]`
    against a truth of 0.0084 is unambiguously 1-5% to a human and is STILL rejected whole, because
    18 of the 20 archived out-of-range payloads are mixed-scale runaways a magnitude rule gets
    wrong. If this ever loosens, D2 is back."""
    assert R.parse_values("[1, 2, 3, 4, 5]") == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert not R.well_formed([1.0, 2.0, 3.0, 4.0, 5.0])
    assert R.parse_report("[0.005, 0.006, 0.007, 1.0, 2.0]") == []


# --------------------------------------------------------------------------- budget, commit, stop

@pytest.mark.asyncio
async def test_a_commit_is_free_and_a_failed_move_is_not():
    """T28. The commit used to cost budget, so a commit made on the last allowed call pushed `_spent`
    to BUDGET, `out_of_budget` fired on the same step, and the episode ended before the commit's
    result was appended: **44 of 120 rollouts in one run produced a well-formed answer that was thrown
    away.** Ending the episode is what commit is FOR."""
    calls = [tool_msg("cw_look", {"hit": True}), tool_msg("cw_move", {"hit": False}),
             tool_msg("cw_observe", {"status": "published", "rate": 0.01})]
    assert _spent(Stub(calls)) == 3
    assert _spent(Stub(calls + [tool_msg("cw_commit", {"committed": True})])) == 3


@pytest.mark.asyncio
async def test_navigation_is_refused_at_budget_but_the_answer_is_not():
    """T29/D4. The stop used to end the rollout the instant the budget ran out -- in one run that was
    11 of the first 26 rollouts taking a flat worst case that the agent never chose. Refusing further
    NAVIGATION while leaving turns to answer measures estimation, not patience."""
    from cohortworld_env.toolset import CohortWorldToolset, CohortWorldToolsetConfig
    ts = CohortWorldToolset(CohortWorldToolsetConfig(budget=3))
    for _ in range(3):
        assert (await ts.look()).get("refused") is not True
    for refused in (await ts.look(), await ts.move("whatever"),
                    await ts.observe("K5", 5)):
        assert refused["refused"] is True and refused["moves_left"] == 0
    assert (await ts.commit([0.01] * 5))["committed"] is True     # ... but answering still works

    task = RateTask(RateData(idx=0, prompt="p", version="C2", disease="K5", horizon=5,
                             true_rate=0.01, zero_stratum=False, pop_zero_share=0.727,
                             budget=3), None)
    at_budget = Stub([tool_msg("cw_look", {})] * 3)
    at_grace = Stub([tool_msg("cw_look", {})] * (3 + R.GRACE))
    assert not await task.out_of_budget(at_budget)
    assert await task.out_of_budget(at_grace)


@pytest.mark.asyncio
async def test_every_tool_result_shows_the_remaining_budget():
    """T30. Over 120 rollouts the agent burned a mean 18.77 of 20 calls and 109 of 120 never answered
    at all. An agent cannot budget what it cannot see."""
    from cohortworld_env.toolset import CohortWorldToolset, CohortWorldToolsetConfig
    ts = CohortWorldToolset(CohortWorldToolsetConfig(budget=5))
    seen = []
    for _ in range(5):
        seen.append(await ts.look())
    assert [o["moves_left"] for o in seen] == [4, 3, 2, 1, 0]
    assert all("warning" in o for o in seen if o["moves_left"] <= 3)
    assert "warning" not in seen[0]


# --------------------------------------------------------------------------- properness of R

def stub_commit(vals):
    """A trace whose only content is a well-formed `cw_commit` carrying `vals`.

    Policy tests go through `rate_reward` on one of these rather than re-deriving the formula. A test
    that recomputes `ALPHA * R + (1 - CRPS)` inline passes whatever the shipped reward does -- which
    is not hypothetical: the first version of `test_R_is_invariant_to_the_spread_of_the_report` did
    exactly that and SURVIVED a mutation that made `R` depend on `max(vals) - min(vals)`. The
    mutation run is the only reason that was caught, and it is why every policy number below is read
    off the reward rather than recomputed beside it.
    """
    return Stub([tool_msg("cw_commit", {"committed": True, "cohort_id": "C2:abc",
                                        "rates": list(vals)})])


def _rate_task(disease, horizon, truth):
    return RateTask(RateData(idx=0, prompt="p", version="C2", disease=disease, horizon=horizon,
                             true_rate=truth, zero_stratum=(truth == 0.0),
                             pop_zero_share=0.727), None)


def test_R_is_invariant_to_the_spread_of_the_report():
    """T21, and the single property the whole `ALPHA * R` design rests on (reward-spec 2.2).

    `R` may depend on the report's LOCATION and on the outcome, never on its DISPERSION. If it ever
    does, `dR/d(spread)` is non-zero, the added term enters the coordinate CRPS elicits, and the
    reward stops being strictly proper -- at which point honest uncertainty is punished and the whole
    reason for using a proper score is void. The first thing any future additive term has to satisfy.
    """
    truth = 0.02
    for centre in (0.0, 0.01, 0.02, 0.2, 1.0):
        seen = {}
        for half in (0.0, 0.001, 0.05, 0.4):
            vals = [max(0.0, min(1.0, centre + f * half)) for f in (-2, -1, 0, 1, 2)]
            got = _rate_task("K5", 5, truth)._score(stub_commit(vals))
            if got["point"] != pytest.approx(centre, abs=1e-12):
                continue                   # clipped at the edge; the LOCATION moved, not the spread
            seen[half] = round(got["R"], 12)
            assert got["report_level"] == R.STATED_COMMIT
        assert len(seen) >= 2, f"no spread was actually varied at centre {centre}"
        assert len(set(seen.values())) == 1, f"R moved with the spread at centre {centre}: {seen}"


@pytest.fixture(scope="module")
def unseen_items():
    from cohortworld_env.internal import ITEMS
    return [(d["disease"], int(d["horizon"]), float(d["answer_rate"]))
            for d in (json.loads(l) for l in open(f"{ITEMS}/t3_rate_unseen.jsonl"))]


def _mean_reward(items, vals_of, alpha, monkeypatch):
    """Mean SHIPPED reward of a synthetic policy over the internal unseen items.

    Every value comes out of `RateTask._score`, so a change to the reward form, to the ladder or to
    the parse rule moves these numbers. `alpha` is patched onto the module because it is a pinned
    constant and what happens as it varies is the point of the test.
    """
    monkeypatch.setattr(R, "ALPHA", alpha)
    return st.mean(_rate_task(d, h, y)._score(stub_commit(vals_of(d, h, y)))["reward"]
                   for d, h, y in items)


def _trivial_policies():
    bs = lambda d, h: R.base_rate("internal", d, h)
    return {
        "zeros": lambda d, h, y: [0.0] * 5,
        "base +-50%": lambda d, h, y: [bs(d, h) * f for f in (0.5, 0.75, 1.0, 1.25, 1.5)],
        "tuned hack": lambda d, h, y: [min(1.0, 0.60 * bs(d, h) * f) for f in (0., 1., 2., 3., 4.)],
        "floor": lambda d, h, y: list(R.FLOOR_REPORT),
    }


def test_alpha_makes_competence_beat_every_zero_information_policy(unseen_items, monkeypatch):
    """D5 on T3, stated as the property that actually matters and measured on the real item set.

    Prop. 9's degeneracy is a claim about the reward's ARGMAX: under a pure proper score a policy can
    gain by degrading accuracy and honestly reporting the degradation. The check is therefore
    "oracle vs the best trivial policy", not "the current model vs the best trivial policy" -- the
    second is a statement about the MODEL, and conflating the two is what made reward-spec 2.5's
    alpha bound read as unsatisfiable in the adversarial review.

    On the 304 internal unseen items the competence margin (oracle minus the best of four
    zero-information policies) is strictly INCREASING in alpha: +0.0219 at alpha = 0, +0.0696 at the
    pinned 1.25, +0.0957 at 2.0. Alpha does exactly what it is for. The DIRECTION of the inequality
    is the assertion; the magnitudes are not pinned, because they move with the item set.

    What this does NOT claim is that an untrained policy clears that bar. That is training headroom,
    not a degenerate reward.
    """
    oracle = lambda d, h, y: [y] * 5
    margins = []
    for alpha in (0.0, 1.25, 2.0):
        best = max(_mean_reward(unseen_items, f, alpha, monkeypatch)
                   for f in _trivial_policies().values())
        margins.append(_mean_reward(unseen_items, oracle, alpha, monkeypatch) - best)
        assert margins[-1] > 0.0, f"a zero-information policy beats the oracle at alpha={alpha}"
    assert margins[0] < margins[1] < margins[2], f"alpha does not widen the margin: {margins}"


# --------------------------------------------------------------------------- wiring hazards

def test_every_module_level_path_is_absolute_from_a_foreign_cwd():
    """T42. A relative default path has killed two runs here (CW_SYS, then legal_moves) and the
    tool-server subprocess makes it unusually easy to reintroduce: verifiers chdirs the runtime into
    a cache directory, so a path that resolves in the repo root resolves to nothing at rollout time
    and the failure surfaces as an unrelated-looking ToolsetError on every rollout.

    Run in a subprocess from `/` with the `CW_*` overrides cleared, because the check is exactly
    about what the process inherits.
    """
    code = textwrap.dedent("""
        import os, sys
        os.chdir("/")
        for k in list(os.environ):
            if k.startswith("CW_"):
                del os.environ[k]
        sys.path[:0] = [%r + "/cohortworld_env", %r + "/cohortworld_phenocal"]
        import cohortworld_env.lattice, cohortworld_env.legal_moves, cohortworld_env.toolset
        import cohortworld_env.internal, cohortworld_env.taskset, cohortworld_env.reward
        import cohortworld_phenocal.taskset
        # An absolute path can still be absolute to the WRONG REPOSITORY, which is what
        # `os.path.isabs` alone could never see. In a fresh clone, `reward` derived `official/` from
        # __file__ while `lattice`, `legal_moves` and `phenocal.taskset` spelled out
        # the author's checkout by absolute path -- so the clone's reward read one tree's reference table
        # and its lattice came from another, and twelve tests asserted "the lattice is the rebuild"
        # against a repository they were not in. A default must resolve inside THIS checkout.
        EXTERNAL = {
            # UKB PhenoCal's published panel: another project, read-only, and its official split and
            # leaderboard were produced on exactly those bytes. There is no copy to put in this repo;
            # it is declared here and in docs/ENVIRONMENT_SPEC.md rather than left to look like a bug.
            "cohortworld_phenocal.taskset.PHENOCAL",
            # An item root outside the checkout (the released item set), hashed into every run's inputs.txt by
            # newrun.sh and named in config.yaml by CONSTANTS()["items"].
            "cohortworld_env.internal.ITEMS",
        }
        bad = []
        for m in (cohortworld_env.lattice, cohortworld_env.legal_moves, cohortworld_env.toolset,
                  cohortworld_env.internal, cohortworld_env.taskset, cohortworld_env.reward,
                  cohortworld_phenocal.taskset):
            for name, v in vars(m).items():
                # A path has no whitespace in it. PROMPT text does, and prompt text can carry a
                # slash -- internal._T3_THINK_FINISH holds a closing </cohort-think> tag, which
                # tripped this check as a "relative path". Excluding strings that
                # contain whitespace narrows the heuristic to what it is actually about; a genuine
                # relative default like "official/CohortWorld_C2_states.jsonl" contains none and is
                # still caught, which the sibling test below PROVES rather than assumes.
                if (name.isupper() and isinstance(v, str) and ("/" in v)
                        and not v.startswith("http") and not any(c.isspace() for c in v)):
                    q = f"{m.__name__}.{name}"
                    if not os.path.isabs(v):
                        bad.append(f"{q} = {v!r} (relative)")
                    elif not v.startswith(%r) and q not in EXTERNAL:
                        bad.append(f"{q} = {v!r} (outside this checkout)")
        print("BAD:" + "|".join(bad))
    """) % (REPO, REPO, REPO)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd="/")
    assert out.returncode == 0, out.stderr[-3000:]
    bad = out.stdout.strip().split("BAD:")[-1]
    assert bad == "", f"relative module-level paths: {bad}"


def test_the_path_checker_above_can_actually_fail():
    """The narrowing above -- skip strings containing whitespace, so prompt text carrying
    a closing `</cohort-think>` tag stops reading as a relative path -- is exactly the kind of change
    that silently turns a guard into a no-op. So run the SAME predicate over synthetic values and
    require it to flag a genuine relative default and a genuine foreign-absolute one.

    A clean guard proves nothing until it is shown it CAN trip.
    """
    REPO_ = "/srv/checkout/cohortworld"  # a synthetic absolute checkout path for the predicate

    def flags(name, v):
        if not (name.isupper() and isinstance(v, str) and ("/" in v)
                and not v.startswith("http") and not any(c.isspace() for c in v)):
            return None
        if not os.path.isabs(v):
            return "relative"
        return None if v.startswith(REPO_) else "outside"

    assert flags("OFFICIAL", "official/CohortWorld_C2_states.jsonl") == "relative"
    assert flags("ITEMS", "../items_elsewhere/items") == "relative"
    assert flags("DEEP", "a/b/c/d/e/f/g.jsonl") == "relative"
    assert flags("TABLE", "/some/other/repo/official/x.jsonl") == "outside"
    assert flags("OFFICIAL", REPO_ + "/official") is None
    assert flags("URL", "https://example.org/a/b") is None
    assert flags("_T3_THINK_FINISH", "write <cohort-think>entry=1 k=2</cohort-think> now") is None
    assert flags("lowercase_name", "official/x.jsonl") is None


def test_the_packages_read_the_checkout_they_are_in_and_not_the_one_they_were_written_in(tmp_path):
    """T42's blind spot, and the reason a fresh clone read two repositories at once.

    The test above cannot see a path hardcoded to the author's checkout while it is RUNNING in that
    checkout: the string is absolute and it is inside this
    checkout, so both of its rules pass. Verified by mutation -- restoring the old
    hardcoded `lattice.PANEL` left that test green. It only ever failed somewhere else, which is
    precisely where nobody was running it.

    So run the packages from somewhere else. Copy the two package directories into a tmp tree laid
    out the way the repo lays them out, import them with every `CW_*` cleared, and require every
    module-level path to resolve inside the TMP tree. A default derived from `__file__` follows the
    copy; one that names a repository does not.
    """
    import shutil
    pkgs = tmp_path / "pkgs"
    for pkg in ("cohortworld_env", "cohortworld_phenocal"):
        shutil.copytree(f"{REPO}/{pkg}/{pkg}", pkgs / pkg,
                        ignore=shutil.ignore_patterns("__pycache__"))
    code = textwrap.dedent("""
        import os, sys
        os.chdir("/")
        for k in list(os.environ):
            if k.startswith("CW_"):
                del os.environ[k]
        sys.path.insert(0, %r)
        import cohortworld_env.lattice as L, cohortworld_env.legal_moves as LM
        import cohortworld_env.reward as R, cohortworld_phenocal.taskset as P
        print("BAD:" + "|".join(
            f"{n}={v!r}" for n, v in (("lattice.PANEL", L.PANEL), ("lattice.PRIV", L.PRIV),
                                      ("legal_moves.PANEL", LM.PANEL),
                                      ("phenocal.SIDECAR", P.SIDECAR),
                                      ("reward.reference_root", R._reference_root()))
            if not v.startswith(%r)))
    """) % (str(pkgs), str(tmp_path))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd="/")
    assert out.returncode == 0, out.stderr[-3000:]
    bad = out.stdout.strip().split("BAD:")[-1]
    assert bad == "", f"these roots did not follow the checkout: {bad}"


@pytest.mark.asyncio
async def test_setup_task_binds_the_lattice_to_the_items_own_version():
    """T40. `Task.toolsets(config)` is handed a TaskConfig that carries no `version`, so the toolset
    starts on its default lattice and only `setup_task` can correct it. When it did not, a C2
    question was answered inside the C1 lattice -- `no disease matching 'injury / poisoning [T8]' in
    the C1 disease table` -- and the agent burned its whole budget hunting a cohort that could not
    exist."""
    from cohortworld_env.toolset import CohortWorldToolset, CohortWorldToolsetConfig
    ts = CohortWorldToolset(CohortWorldToolsetConfig(version="C1"))
    await ts.setup_task(RateData(idx=0, prompt="p", version="C2", disease="K5", horizon=5,
                                 true_rate=0.01, zero_stratum=False, pop_zero_share=0.727,
                                 age_band="55-59", sex="Male"))
    assert ts.lat.version == "C2"
    assert ts.lat.current.startswith("C2:")


@pytest.mark.skipif(not os.path.isdir(os.path.join(REPO, 'official')), reason='needs official/ inside this checkout to hide a states file from')
def test_setup_task_refuses_the_wrong_lattice_when_the_items_own_is_missing(tmp_path):
    """T40's silent half, and the last of the "a reward that depends on which files happen to exist"
    family. An earlier fix made a missing panel loud at SCORING time (`reward.dist` / `entry_cohort`
    raise) and left the ROLLOUT side quiet: `setup_task` did `if lat is not None: self.lat = lat`, so
    an item whose version had no states file kept whatever lattice the toolset was built with.

    Measured by hiding `CohortWorld_default_*_states.jsonl`: `RateTaskset` served its
    151 `default` items unchanged, every one of them was bound to the C2 lattice, and a rollout then
    scored **reward 0.9425 at report_level 2** -- an ordinary number from an agent that spent its
    whole budget navigating cohorts the question was not about. T1 would have raised through
    `entry_cohort`; T3's reward never touches the lattice, so nothing anywhere could notice.

    Only a PARTIAL panel gets in -- a wholly absent one raises in `Lattice.__init__` -- which is why
    it survived the freeze, and why §9 could claim "a missing panel is now loud" and be wrong.

    A symlink farm, so no panel data is copied and `official/` is never touched. Run in a subprocess
    because the check is about what the process resolves `CW_PANEL` to.
    """
    farm = tmp_path / "panel"
    farm.mkdir()
    hidden = 0
    for f in glob.glob(f"{REPO}/official/*"):
        if re.match(r"CohortWorld_default_\w+_states\.jsonl$", os.path.basename(f)):
            hidden += 1
        else:
            os.symlink(f, farm / os.path.basename(f))
    assert hidden, "no default states file to hide -- the panel's file names moved"
    code = textwrap.dedent("""
        import asyncio
        from cohortworld_env.internal import RateData
        from cohortworld_env.toolset import CohortWorldToolset, CohortWorldToolsetConfig
        ts = CohortWorldToolset(CohortWorldToolsetConfig(version="C2"))
        item = RateData(idx=0, prompt="p", version="default", disease="K5", horizon=5,
                        true_rate=0.01, zero_stratum=False, pop_zero_share=0.727,
                        age_band="55-59", sex="Male")
        try:
            asyncio.run(ts.setup_task(item))
            print("SILENT:" + ts.lat.version)
        except FileNotFoundError:
            print("RAISED:" + ts.lat.version)
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd="/",
                         env={**os.environ, "CW_PANEL": str(farm)})
    assert out.returncode == 0, out.stderr[-3000:]
    # `RAISED:C2`: it must refuse, and the version it names must be the one it was still standing on.
    assert out.stdout.strip() == "RAISED:C2", out.stdout + out.stderr[-2000:]


# ------------------------------------------------------------------ the item's own rate is masked
#
# Every internal T3 item IS a lattice cohort: the row carries `target_cohort`, `source.kind ==
# "lattice"`, and `answer_rate` is exactly that cohort's published rate. All 18,593 targets are still
# served, so an agent that walked there and called `cw_observe` was handed the label -- 1.49% of the
# answering rollouts in one training run did, for +0.43 mean reward. `Lattice.masked` withholds the
# RATE at that one state for the episode. It withholds nothing else: the state stays advertised and
# reachable on purpose, because the intended T3 procedure is to STAND on the item's own small cohort
# and retreat to a larger one that can be read.

def _t3_rows(split="unseen", n=600):
    """Real items, from the item set the environment is bound to. Never a hand-made stand-in: the
    leak is a property of the ITEM SET meeting the lattice, and a synthetic row cannot exhibit it."""
    from cohortworld_env.internal import ITEMS
    rows = []
    with open(f"{ITEMS}/t3_rate_{split}.jsonl") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rows.append(json.loads(line))
    return rows


def _rate_data(d, idx=0):
    return RateData(idx=idx, prompt="p", version=d["version"], disease=d["disease"],
                    horizon=int(d["horizon"]), true_rate=float(d["answer_rate"]),
                    zero_stratum=bool(d["zero_stratum"]),
                    pop_zero_share=float(d["pop_zero_share"]),
                    age_band=str(d.get("age_band", "")), sex=str(d.get("sex", "")),
                    target_cohort=str(d.get("target_cohort", "") or ""))


def _readable_rows(k=2):
    """Items whose target really does publish the answer, on DIFFERENT targets.

    An item whose target reads `NO_FOLLOWUP_WINDOW` would let every test below pass without the mask
    doing anything, which is the "green check that cannot fail" failure. These rows are the positive
    control: the unmasked read returns `answer_rate` exactly.

    All rows come from ONE lattice version, so a test may point one item's toolset at another item's
    target: across versions that state simply does not exist and the test would raise for a reason
    that has nothing to do with masking.
    """
    from cohortworld_env.lattice import _shared
    out, seen, ver = [], set(), None
    for d in _t3_rows():
        tc = d.get("target_cohort")
        lat = _shared(d["version"])
        if not tc or lat is None or tc in seen or tc not in lat.states:
            continue
        if ver is not None and d["version"] != ver:
            continue
        lat.current = tc
        got = lat.observe(d["disease"], int(d["horizon"]))
        if got.get("status") == "published" and abs(got["rate"] - float(d["answer_rate"])) < 1e-12:
            out.append(d)
            seen.add(tc)
            ver = d["version"]
            if len(out) >= k:
                break
    assert len(out) >= k, f"only {len(out)} item(s) whose target publishes the answer -- the tests " \
                          f"below would pass on an environment that masks nothing"
    return out


@pytest.mark.asyncio
async def test_the_items_own_cohort_never_returns_its_rate():
    """The leak, and its closure, on a real item. The control in `_readable_rows` proves the read
    WAS the answer, so this is not a check that cannot fail."""
    from cohortworld_env.toolset import CohortWorldToolset, CohortWorldToolsetConfig
    d = _readable_rows(1)[0]
    ts = CohortWorldToolset(CohortWorldToolsetConfig(version=d["version"]))
    await ts.setup_task(_rate_data(d))
    ts.lat.current = d["target_cohort"]                      # the agent walked here; that is allowed
    got = await ts.observe(d["disease"], int(d["horizon"]))
    assert got["status"] == "RATE_WITHHELD_HERE", got
    assert "rate" not in got and "at_risk" not in got, got
    # every OTHER horizon at that state is withheld too -- the answer is one cell, but the
    # neighbouring cells of the same cohort are most of it, and the zero stratum is 50% of the set.
    for h in (1, 2, 3, 5, 10):
        # the STATUS is asserted, not just the absence of a rate: a budget refusal also has no
        # `rate`, and taking that for a working mask has already cost one wrong audit run.
        assert (await ts.observe(d["disease"], h))["status"] == "RATE_WITHHELD_HERE"


@pytest.mark.asyncio
async def test_the_masked_cohort_is_still_advertised_and_still_reachable():
    """The mask is NOT a wall. It withholds the rate and nothing else, because the T3
    procedure this environment is being shaped towards puts the agent ON its item's own cohort and
    asks it to retreat to a bigger one it can read -- a clinician falling back to a reference class.
    A target that could not be stood on would forbid the intended policy, not the exploit."""
    from cohortworld_env.legal_moves import moves
    from cohortworld_env.toolset import CohortWorldToolset, CohortWorldToolsetConfig
    d = _readable_rows(1)[0]
    tc = d["target_cohort"]
    ts = CohortWorldToolset(CohortWorldToolsetConfig(version=d["version"]))
    await ts.setup_task(_rate_data(d))
    # somewhere with an edge INTO the target: that edge must still be listed, and taking it must land
    src = next((sid for sid in ts.lat.states if any(m["to"] == tc for m in moves(sid))), None)
    if src is None:
        pytest.skip(f"no in-edge to {tc} in {d['version']} -- reachability cannot be exercised here")
    ts.lat.current = src
    listed = [mv["id"] for grp in (await ts.look())["action_space"]["legal_moves"].values()
              for mv in grp["moves"]]
    want = [f"{m['axis']}.{m['op']}.{m['code']}->{tc}" for m in moves(src) if m["to"] == tc]
    assert any(w in listed for w in want) or any("..." in i for i in listed), \
        "the move into the target vanished from cw_look; the mask withholds the RATE, not the state"
    assert (await ts.move(want[0]))["state_id"] == tc, "cw_move into the target was refused"
    assert ts.lat.current == tc


@pytest.mark.asyncio
async def test_a_second_task_in_the_same_process_masks_only_its_own_target():
    """THE SHARED-LATTICE TRAP. `_shared()` caches ONE Lattice per version per process and hands the
    same object to every rollout after this one. A mask written onto it would follow the object into
    the next episode and withhold an unrelated cohort's rates -- invisibly, since nothing about a
    mask reaches the trace. `setup_task` therefore masks a per-episode SHALLOW COPY.

    SHOWN TO BE ABLE TO FAIL by monkeypatching `copy.copy` to the identity inside
    `setup_task` so the mask lands on the shared object. Under that mutation the FIRST assertion
    still passes -- B's own mask overwrites A's, so B can still read A's target -- and it is the last
    two that catch it: A's own target becomes readable again (`True` where the shipped code gives
    `False`), and `_shared(v).masked` comes back `'C2:21ea5b912db6564f'` instead of `''`. Worth
    recording, because a version of this test with only the first assertion would have been green on
    the exact bug it names."""
    from cohortworld_env.lattice import _shared
    from cohortworld_env.toolset import CohortWorldToolset, CohortWorldToolsetConfig
    a, b = _readable_rows(2)
    assert a["target_cohort"] != b["target_cohort"] and a["version"] == b["version"]
    # Start both toolsets on a DIFFERENT version, so `setup_task` rebinds through `_shared()` and the
    # object under test really is the cached one. Constructed on the item's own version instead, each
    # toolset gets a private `Lattice` from its own `__init__` and the trap is never sprung.
    other = next(v for v in ("C1", "C2", "default") if v != a["version"])
    ts_a = CohortWorldToolset(CohortWorldToolsetConfig(version=other))
    await ts_a.setup_task(_rate_data(a))
    ts_b = CohortWorldToolset(CohortWorldToolsetConfig(version=other))
    await ts_b.setup_task(_rate_data(b))
    assert ts_a.lat.version == ts_b.lat.version == a["version"]

    ts_b.lat.current = a["target_cohort"]
    got = await ts_b.observe(a["disease"], int(a["horizon"]))
    assert got.get("status") == "published" and "rate" in got, \
        f"task B cannot read task A's target: the mask escaped onto the shared lattice ({got})"
    ts_b.lat.current = b["target_cohort"]
    assert "rate" not in await ts_b.observe(b["disease"], int(b["horizon"]))
    # ... and A is still masked after B ran, which a shared object would also have broken
    ts_a.lat.current = a["target_cohort"]
    assert "rate" not in await ts_a.observe(a["disease"], int(a["horizon"]))
    # the cached object itself was never touched
    for v in {a["version"], b["version"]}:
        assert _shared(v).masked == "", "_shared()'s Lattice was mutated; every later rollout inherits it"


@pytest.mark.asyncio
async def test_a_task_that_names_no_target_is_untouched():
    """The external judge must be provably unaffected. PhenoCal items are not lattice cohorts, carry
    no `target_cohort`, and `mask_cohort` is "" for them -- so `setup_task` does not even take the
    per-episode copy, and the advertised move set and every observation are the ones a bare `Lattice`
    gives. T1 and T2 ride on the same guarantee: `LocateData.target_cohort` is the answer the agent
    must WALK to, and masking it would have broken T1 outright."""
    from cohortworld_env.lattice import Lattice, _shared
    from cohortworld_env.toolset import CohortWorldToolset, CohortWorldToolsetConfig
    item = PhenoCalData(idx=0, prompt="p", true_rate=0.01, version="C2", age_band="55-59",
                        sex="Female", disease="K5", horizon=5, n=1000)
    ts = CohortWorldToolset(CohortWorldToolsetConfig(version="C2"))
    await ts.setup_task(item)
    assert ts.lat.masked == "" and _shared("C2").masked == "", \
        "a task with no target must not be given a mask, and must not leave one behind"
    ref = Lattice("C2")                                    # a lattice that has never heard of masks
    ref.enter("55-59", "Female")
    assert ref.current == ts.lat.current
    assert (await ts.look())["action_space"] == ref.state_of(ref.current)["action_space"]
    for h in (1, 2, 3, 5, 10):
        for dis in ref.diseases[:3]:                       # real codes, so the loop cannot be vacuous
            assert await ts.observe(dis, h) | {"moves_left": 0} == \
                   ref.observe(dis, h) | {"moves_left": 0}
    # and the T1 half: its target stays readable, because T1's target is not an answer to read
    lt = CohortWorldToolset(CohortWorldToolsetConfig(version="C2"))
    await lt.setup_task(LocateData(idx=0, prompt="p", version="C2",
                                   target_cohort=_readable_rows(1)[0]["target_cohort"],
                                   age_band="55-59", sex="Female"))
    assert lt.lat.masked == "", "T1's target_cohort must never be masked -- it is where the agent goes"


@pytest.mark.asyncio
async def test_a_t3_episode_starts_on_the_items_own_cohort_and_a_phenocal_one_does_not():
    """The mask's other half. T3's entry WAS `enter(age_band, sex, prefer_coarsest=True)` -- the
    largest cohort of the right demographics -- from which every available move specialises, so the
    only chain the environment could express was "start somewhere reliable, walk into a 10-49 cohort,
    copy its reading". The order the design wants is the reverse: start on the patient's own
    subgroup, find it unreadable, and retreat to a reference class worth believing.

    Nothing else moves. PhenoCal carries no target, so `mask_cohort` is "" and its entry is the
    demographic one it has always had -- the judge's baselines stay comparable on this axis. (Its
    ACTION SPACE is covered separately, by `test_the_judge_never_sees_the_generalising_op`.)"""
    from cohortworld_env.lattice import _shared
    from cohortworld_env.toolset import CohortWorldToolset, CohortWorldToolsetConfig
    d = _readable_rows(1)[0]
    ts = CohortWorldToolset(CohortWorldToolsetConfig(version=d["version"]))
    await ts.setup_task(_rate_data(d))
    assert ts.lat.current == d["target_cohort"]
    assert (await ts.look())["state_id"] == d["target_cohort"]
    # ... and standing there still cannot read the answer, which is what makes the entry safe
    assert "rate" not in await ts.observe(d["disease"], int(d["horizon"]))

    pc = CohortWorldToolset(CohortWorldToolsetConfig(version="C2"))
    await pc.setup_task(PhenoCalData(idx=0, prompt="p", true_rate=0.01, version="C2",
                                     age_band="55-59", sex="Female", disease="K5", horizon=5,
                                     n=1000))
    assert pc.lat.current == _shared("C2").entry_for("55-59", "Female")


# ---------------------------------------------------------------------- the generalising `drop` op
#
# The five op groups the edge table serves -- bnf.add, opcs.add, icd.add, age_band.shift, sex.switch
# -- all specialise or move the person; there was no way to make the reference class BIGGER along a
# history axis. `zoom coarser` exists in `lattice.py` but is available at 5% of item cohorts (15/300)
# and reaches nothing. An exact-match "remove one code" edge is a dead end too: 41% of targets have
# no such released neighbour and 84% can never reach a 200+ cohort, because the intermediate rungs
# were never released. `drop` is defined as GENERALISATION instead -- give up the criterion, land on
# the most specific released cohort whose axes are a subset of what remains -- and that always has a
# destination, because the demographic-only cohort is a subset of everything.

def test_every_drop_move_generalises_to_a_released_cohort():
    """The rule, checked rather than asserted in prose: the destination is released, it is in the
    same stratum, and its codes are a strict subset of the origin's minus the dropped one."""
    from cohortworld_env.lattice import _shared

    def codes(st):
        ax = st["axes"]
        return frozenset((k, tuple(e)) for k in ("icd", "opcs", "bnf") for e in (ax.get(k) or []))

    lat = _shared("C2")
    checked = 0
    for d in _readable_rows(6):
        sid = d["target_cohort"]
        if sid not in lat.states:
            continue
        here = codes(lat.states[sid])
        for m in lat._drop_moves(sid):
            assert m["to"] in lat.states, f"{m['to']} is not a released cohort"
            there = codes(lat.states[m["to"]])
            dropped = next(e for e in here if e[0] == m["axis"] and e[1][0] == m["code"])
            assert there <= here - {dropped}, "a drop landed somewhere that is not a generalisation"
            assert lat.states[m["to"]]["axes"]["age_band"] == lat.states[sid]["axes"]["age_band"]
            assert lat.states[m["to"]]["axes"]["sex"] == lat.states[sid]["axes"]["sex"]
            checked += 1
    assert checked, "no drop move inspected -- the check cannot fail as written"


@pytest.mark.asyncio
async def test_the_drop_move_is_advertised_and_executes():
    """Advertised, taken by the same id grammar as every other move, and it lands. `cw_look`'s
    listing and `cw_move`'s resolution read ONE set (`legal_moves` is handed the drops as `extra`),
    so an advertised drop that could not be taken is not expressible."""
    from cohortworld_env.toolset import CohortWorldToolset, CohortWorldToolsetConfig
    d = _readable_rows(1)[0]
    ts = CohortWorldToolset(CohortWorldToolsetConfig(version=d["version"]))
    await ts.setup_task(_rate_data(d))                       # the entry IS the item's own cohort
    groups = (await ts.look())["action_space"]["legal_moves"]
    drops = [g for k, g in groups.items() if k.endswith(".drop")]
    assert drops, f"no drop group advertised at the item's own cohort ({sorted(groups)})"
    mid = drops[0]["moves"][0]["id"]
    assert mid.split(".")[1] == "drop" and "->" in mid
    got = await ts.move(mid)
    assert got["hit"] is True and got["state_id"] == mid.split("->", 1)[1]
    assert ts.lat.current == got["state_id"]


def test_a_cohort_with_one_code_generalises_to_the_demographic_cohort():
    """The floor of the rule. With one criterion dropped there is nothing left, so the destination is
    the cohort with no recorded history at all -- which is why a legal generalising move always
    exists and the agent can never be stranded on an unreadable cohort."""
    from cohortworld_env.lattice import _shared
    lat = _shared("C2")
    one = next((sid for sid, st in lat.states.items()
                if sum(len(st["axes"].get(k) or []) for k in ("icd", "opcs", "bnf")) == 1), None)
    assert one, "no single-code cohort in C2 -- the floor cannot be exercised"
    mv = lat._drop_moves(one)
    assert len(mv) == 1
    dest = lat.states[mv[0]["to"]]["axes"]
    assert not any(dest.get(k) for k in ("icd", "opcs", "bnf")), dest
    assert dest["age_band"] == lat.states[one]["axes"]["age_band"]


@pytest.mark.asyncio
async def test_the_judge_never_sees_the_generalising_op():
    """`drop` is served ONLY where the item named a target cohort. The judge is frozen rather than
    its baselines re-run, so the external PhenoCal action space must be byte-equal to the edge table
    alone -- `legal_moves.summary(sid)`.

    Checked AT A CODED COHORT REACHED AFTER A MOVE, not only at the entry. At the entry the check is
    nearly free: all 18 C2 demographic entries are zero-code cohorts, so `_drop_moves` returns []
    there whatever the gate does, and a test that stopped at the entry would pass with the gate
    deleted. One `cw_move` lands somewhere with codes, and that is where the gate has to hold.

    SHOWN TO BE ABLE TO FAIL: with `Lattice._served_drops = Lattice._drop_moves` (the
    gate deleted) the judge's post-move listing at `C2:aed95067c457f1be` gains `bnf.drop` --
    `['age_band.shift', 'bnf.add', 'bnf.switch', 'bnf.zoom_coarser']` becomes that plus `bnf.drop` --
    and the byte-equality below goes False."""
    from cohortworld_env import legal_moves as LM
    from cohortworld_env.lattice import _shared
    from cohortworld_env.toolset import CohortWorldToolset, CohortWorldToolsetConfig
    pc = CohortWorldToolset(CohortWorldToolsetConfig(version="C2"))
    await pc.setup_task(PhenoCalData(idx=0, prompt="p", true_rate=0.01, version="C2",
                                     age_band="55-59", sex="Female", disease="K5", horizon=5,
                                     n=1000))
    look = await pc.look()
    assert look["action_space"] == LM.summary(look["state_id"]), "the judge's entry listing moved"
    mid = next(mv["id"] for g in look["action_space"]["legal_moves"].values()
               for mv in g["moves"] if mv["id"] != "...")
    got = await pc.move(mid)
    dst = got["state_id"]
    lat = _shared("C2")
    coded = sum(len(lat.states[dst]["axes"].get(k) or []) for k in ("icd", "opcs", "bnf"))
    assert coded, f"{dst} has no codes -- this test would pass with the gate deleted"
    assert lat._drop_moves(dst), f"{dst} has no drop move -- ditto"      # the gate has work to do
    assert {k: v for k, v in got.items() if k.startswith("legal_move") or k == "note"} == \
        dict(LM.counts(dst)), "the judge's post-move counts moved"
    assert (await pc.look())["action_space"] == LM.summary(dst), \
        "the judge saw a `drop` at a coded cohort: the target-only gate is not holding"
    # ... and the same cohort DOES advertise drops to an episode that named a target, so the gate is
    # a gate and not an off switch.
    t3 = CohortWorldToolset(CohortWorldToolsetConfig(version="C2"))
    await t3.setup_task(_rate_data(_readable_rows(1)[0]))
    t3.lat.current = dst
    assert (await t3.look())["action_space"] != LM.summary(dst)


def test_every_item_cohort_has_a_generalising_move():
    """The property the T3 procedure depends on: an agent placed on its item's own cohort, unable to
    read it, must always have somewhere to retreat to. Measured over a sample of the served items --
    100% on 500 train items, 1-12 moves each, median 3, and 58.6% reach a 200+ or 1000+ cohort in
    that single move."""
    from cohortworld_env.lattice import _shared
    n = stranded = 0
    for d in _t3_rows(n=200):
        lat = _shared(d["version"])
        tc = d.get("target_cohort")
        if lat is None or not tc or tc not in lat.states:
            continue
        n += 1
        stranded += not lat._drop_moves(tc)
    assert n >= 100, f"only {n} items resolved -- the sample is too small to mean anything"
    assert stranded == 0, f"{stranded}/{n} item cohorts have no way to generalise"


@pytest.mark.slow
@pytest.mark.parametrize("version", ["C1", "C2", "default"])
def test_move_ids_are_unique_within_a_cohort(version):
    """T38. The served id used to be built from the axis and code alone, so two moves that differed
    only in destination collided -- 46 ids on one state -- and `move` resolved whichever came first.
    The destination is in the id now (`bnf.add.0103->C2:8590...`), which is what makes it unique."""
    from cohortworld_env import legal_moves as LM
    from cohortworld_env.lattice import _shared
    lat = _shared(version)
    if lat is None:
        pytest.skip(f"{version} lattice not built")
    checked = 0
    for sid in list(lat.states)[:200]:
        # `extra` is passed, so the `drop` moves are inside the uniqueness check rather than
        # beside it: a drop id is built by the same `_mid` and can collide with an edge-table id or
        # with another drop the same way.
        ids = [m["id"] for g in LM.summary(sid, per_axis=0,
                                           extra=lat._drop_moves(sid))["legal_moves"].values()
               for m in g["moves"]]
        assert len(ids) == len(set(ids)), f"{version} {sid}: {len(ids) - len(set(ids))} collisions"
        for i in ids:
            assert "->" in i and i.split("->", 1)[1].startswith(f"{version}:")
        checked += 1
    assert checked, "no states inspected"


def commit_call_msg(rates):
    """An assistant turn that CALLS `cw_commit` and gets no result -- the shape every healthy
    committed rollout now has, because the `committed` stop ends the episode on this very turn."""
    return {"role": "assistant", "content": "Committing.",
            "tool_calls": [{"name": "cw_commit", "arguments": json.dumps({"rates": rates})}]}


def test_a_commit_is_read_from_the_call_when_its_result_never_lands():
    """The regression only a LIVE run could produce, and the reason Part B exists.

    N9 moved the `committed` stop onto the assistant's tool CALL, because reading the commit's RESULT
    made it fire a turn late or never. But verifiers holds a tool result in
    `Session.handle_tool`'s `prepared_tool_results` until the NEXT model request, and after a stop
    there is no next request -- so the fix removed the commit result from the trace of every healthy
    rollout, and `commit_payload` (which read results) returned None for all of them.

    Effect on the first live smoke: 11 of 12 stopped `committed`, 0 commit results
    on the trace, **12 of 12 scored at `auto_observed`** -- every stated report in the run silently
    discarded and scored on the rung below, with `report_malformed = 0` throughout because a payload
    that is not there is not malformed. Nothing in the summary said anything was wrong.

    No archive replay could have found it: every archived run predates the stop change and still
    carries its commit results. Two fixes each correct in isolation, composing into a silent one.
    """
    tr = Stub([tool_msg("cw_look", {"state_id": "C2:abc"}),
               tool_msg("cw_observe", {"status": "published", "rate": 0.031,
                                       "disease": "K5", "horizon": 5}),
               commit_call_msg([0.02, 0.03, 0.04, 0.05, 0.06])])
    vals, level, malformed = R.rate_report(tr, "K5", 5)
    assert level == R.STATED_COMMIT and malformed == 0.0
    assert vals == [0.02, 0.03, 0.04, 0.05, 0.06]
    # the string form the model actually emits, and the T1 position it implies
    tr2 = Stub([tool_msg("cw_look", {"state_id": "C2:abc"}),
                commit_call_msg("[0.0, 0.001, 0.0025, 0.004, 0.01]")])
    assert R.rate_report(tr2, "K5", 5)[1] == R.STATED_COMMIT
    assert R.locate_report(tr2, "C2:entry") == ("C2:abc", R.STATED_COMMIT)
    # a result, when there IS one, still wins -- the archive is scored by the same function
    tr3 = Stub([commit_call_msg([0.9] * 5),
                tool_msg("cw_commit", {"committed": True, "cohort_id": "C2:xyz",
                                       "rates": [0.02] * 5})])
    assert R.rate_report(tr3, "K5", 5)[0] == [0.02] * 5


def test_a_serialised_score_memo_is_never_reused_across_processes():
    """The archive-quoting-itself trap, found in the same live smoke as the commit regression.

    `_memo` caches the score in `trace.info`, and verifiers SERIALISES `info` into the archived
    trace -- so every archived rollout carries a frozen copy of the score its own run computed. A
    rescoring that trusted it would return the OLD contract's number while appearing to recompute
    one, and it would AGREE with the archive, which reads as corroboration. It is the exact shape of
    "a result matching your prediction gets less scrutiny".

    Rollouts recorded before `_memo` reached a serialised field carry no such entry, which is why the
    offline rescoring done for D1 was sound. The guard
    is a per-process token: an entry this process did not write is ignored and recomputed, and the
    stale entry is left in place rather than deleted, so reading an archive never edits it.
    """
    task = _rate_task("K5", 5, 0.02)
    tr = stub_commit([0.02] * 5)
    tr.info = {"_cw_score": {"reward": -999.0, "crps_loss": 0.0, "R": 0.0, "point": 0.0,
                             "base": 0.0, "report_level": R.FLOOR, "report_malformed": 0.0,
                             "m": 5.0},
               "_cw_score_by": "another-process-entirely"}
    got = task._score(tr)
    assert got["reward"] != -999.0 and got["report_level"] == R.STATED_COMMIT
    assert tr.info["_cw_score_by"] != "another-process-entirely"
    # ... and within one process the memo still works, or metrics recompute the score eight times
    assert task._score(tr) is got


def test_the_external_task_uses_the_one_memo_implementation():
    """The same trap reached PhenoCal through a hand-copied memo. One implementation, imported."""
    import inspect
    from cohortworld_phenocal.taskset import PhenoCalTask
    src = inspect.getsource(PhenoCalTask._score)
    assert "_memo" in src and "_cw_score" not in src


def test_t2_measures_one_operation_and_the_composition_is_pinned():
    """T2's honest limit on the released item set, pinned so no T2 number is ever quoted without it.

    THE OLD DEFECT. `DeltaTaskset` served only the items whose edge lattice and
    outcome granularity agreed -- 417 of 625 failed that -- and which items survived depended on the
    OPERATION, so the three splits were three different tasks: train 46% `bnf.switch`, seen 68%
    `sex.switch`, unseen 63% `age_band.shift`. The test that stood here asserted the modal operation
    DIFFERED in all three, i.e. it pinned the defect.

    THE CURRENT SITUATION IS THE OPPOSITE ONE AND IS ALSO A LIMIT. The item builder repaired emission
    (every item is servable, the directions are exact thirds) and dropped the `age_band` stratum, whose clause
    would have had to state an age more precisely than the source supports. What survives is
    overwhelmingly `sex.switch`: 264/297, 54/60 and **105/108**. Nine of the thirteen non-age strata
    have ZERO direction-balanced supply at any granularity, and `bnf switch` -- the other stratum
    measured load-bearing (+0.286) -- has none at all in `unseen`.

    So T2-unseen measures ONE operation. That is a property of the release, not of the composition,
    and no resampling reaches past it. It is pinned here rather than left in a report because a T2
    headline read as "the agent can reason about cohort edges" would be reading 105 sex switches.
    """
    from cohortworld_env.internal import DeltaConfig
    want = {"train": (297, "sex.switch", 264), "seen": (60, "sex.switch", 54),
            "unseen": (108, "sex.switch", 105)}
    modes = []
    for split, (n, top, k) in want.items():
        tasks = list(DeltaTaskset(DeltaConfig(id="t2", split=split)).load())
        assert len(tasks) == n, split
        ops = collections.Counter(f"{t.data.axis}.{t.data.op}" for t in tasks)
        assert ops.most_common(1)[0] == (top, k), f"{split}: {ops.most_common(3)}"
        assert not any(o.startswith("age_band.") for o in ops), \
            f"{split}: the age stratum is back -- its clause states an age the source does not"
        dirs = collections.Counter(t.data.answer_direction for t in tasks)
        assert len(set(dirs.values())) == 1 and len(dirs) == 3, f"{split}: {dict(dirs)}"
        modes.append(ops.most_common(1)[0][0])
    # The splits have CONVERGED on one operation. Asserted in that direction on purpose: the old
    # test asserted they diverged, and reading a converged set through it would have gone red for
    # the wrong reason and been "fixed" by deleting the assertion.
    assert set(modes) == {"sex.switch"}, f"the composition moved off one operation: {modes}"


# --------------------------------------------------------------------------- the unexecuted rungs
#
# Every test in this section closes a MUTATION SURVIVOR. The shape of the gap was one
# thing: whole branches of the ladder were never driven. Deleting T3's `stated_text` rung, flipping
# T2's auto rung, accepting an empty `cohort_id` on T1 and reversing `_T2_Y` all left the suite
# green, because no test ever fed the ladder a WELL-FORMED answer line, two readings at different
# cohorts, or an empty id -- and no test anywhere called `delta_reward`.

def test_a_well_formed_answer_line_is_scored_on_its_own_rung():
    """T3/T2 rung 2. `stated_text` existed and was never exercised: the ladder tests fed only
    MALFORMED answer lines (rung 2 falls through by design) and the parse tests stopped at
    `parse_values` without ever reaching the ladder. Deleting the rung outright stayed green, and it
    is the rung an agent that answers in prose instead of calling `commit` lands on."""
    said = Stub(last_reply="Reasoning.\nANSWER: [0.01, 0.02, 0.03, 0.04, 0.05]")
    for report in (R.rate_report(said, "K5", 5), R.delta_report(said, "K5", 5)):
        assert report == ([0.01, 0.02, 0.03, 0.04, 0.05], R.STATED_TEXT, 0.0)
    # A commit outranks it, and a MALFORMED commit does not promote the text to rung 1.
    both = Stub([tool_msg("cw_commit", {"committed": True, "cohort_id": "C2:abc",
                                        "rates": [0.2] * 5})],
                last_reply="ANSWER: [0.01, 0.02, 0.03, 0.04, 0.05]")
    assert R.rate_report(both, "K5", 5) == ([0.2] * 5, R.STATED_COMMIT, 0.0)
    bad = Stub([tool_msg("cw_commit", {"committed": True, "cohort_id": "C2:abc",
                                       "rates": [7.0] * 5})],
               last_reply="ANSWER: [0.01, 0.02, 0.03, 0.04, 0.05]")
    assert R.rate_report(bad, "K5", 5) == ([0.01, 0.02, 0.03, 0.04, 0.05], R.STATED_TEXT, 1.0)


def _two_readings(first, second, disease="K5", horizon=5):
    """A T2 trace that did the experiment: read the cell, moved, read it again."""
    return Stub([tool_msg("cw_look", {"state_id": "C2:aaa"}),
                 tool_msg("cw_observe", {"status": "published", "rate": first,
                                         "disease": disease, "horizon": horizon}),
                 tool_msg("cw_move", {"hit": True, "state_id": "C2:bbb"}),
                 tool_msg("cw_observe", {"status": "published", "rate": second,
                                         "disease": disease, "horizon": horizon})])


@pytest.mark.parametrize("first,second,expect", [
    (0.010, 0.020, [1.0]),        # the rate went UP after the move
    (0.020, 0.010, [0.0]),        # ...and down
    (0.015, 0.015, [0.5]),        # no difference at all is the hedge, not a direction
])
def test_the_t2_auto_rung_reads_the_sign_between_two_cohorts(first, second, expect):
    """T2 rung 3, and the rung the prompt actually prescribes: "read its rate, make the change, read
    it again". No test built a T2 trace with two readings at different cohorts, so SWAPPING the pair
    (which reverses every sign) and loosening `d > 0` to `d >= 0` (which reads a zero delta as "up")
    both survived. The zero case is not a corner: "not distinguishable" is one of the three labels
    and it is the one a constant answer would hide in."""
    assert R.delta_report(_two_readings(first, second), "K5", 5) == (expect, R.AUTO_OBSERVED, 0.0)
    # Two readings at the SAME cohort are not an experiment: the ladder falls to the hedge.
    same = Stub([tool_msg("cw_look", {"state_id": "C2:aaa"}),
                 tool_msg("cw_observe", {"status": "published", "rate": first,
                                         "disease": "K5", "horizon": 5}),
                 tool_msg("cw_observe", {"status": "published", "rate": second,
                                         "disease": "K5", "horizon": 5})])
    assert R.delta_report(same, "K5", 5) == ([0.5], R.FLOOR, 0.0)


def test_a_commit_that_names_no_cohort_falls_to_the_rung_below():
    """T1 rung 1's guard. `payload.get("cohort_id")` is truthy-checked, not `is not None`, and the
    difference is a commit whose reconstructed id is EMPTY -- which is exactly what `commit_payload`
    builds when `last_state_id` finds nothing, i.e. when the agent committed without ever having a
    tool result print a position. Accepting `""` as the answer scores the empty string against the
    target: `dist` returns 99 and the rollout collects partial credit for standing nowhere."""
    nowhere = Stub([tool_msg("cw_commit", {"committed": True, "cohort_id": "", "rates": None})])
    assert R.locate_report(nowhere, "C2:entry") == ("C2:entry", R.FLOOR)
    somewhere = Stub([tool_msg("cw_look", {"state_id": "C2:walked"}),
                      tool_msg("cw_commit", {"committed": True, "cohort_id": "", "rates": None})])
    assert R.locate_report(somewhere, "C2:entry") == ("C2:walked", R.AUTO_OBSERVED)
    named = Stub([tool_msg("cw_commit", {"committed": True, "cohort_id": "C2:chosen"})])
    assert R.locate_report(named, "C2:entry") == ("C2:chosen", R.STATED_COMMIT)


def _delta_task(direction):
    return DeltaTask(DeltaData(idx=0, prompt="p", version="C2", disease="K5", horizon=5,
                               answer_direction=direction, start_cohort="C2:aaa"), None)


@pytest.mark.parametrize("direction,vals,reward", [
    ("up", [1.0] * 5, 2.25), ("up", [0.0] * 5, 0.0), ("up", [0.5] * 5, 0.5),
    ("down", [0.0] * 5, 2.25), ("down", [1.0] * 5, 0.0), ("down", [0.5] * 5, 0.5),
    ("indistinguishable", [0.5] * 5, 1.0),
    ("indistinguishable", [1.0] * 5, -0.75), ("indistinguishable", [0.0] * 5, -0.75),
])
def test_the_t2_reward_is_read_off_delta_reward_and_not_recomputed(direction, vals, reward):
    """**No test in the suite ever called `delta_reward`.** The one T2 policy test defines its own
    local `Y = {"down": 0.0, ...}` beside the implementation instead of using `R._T2_Y` -- which is
    exactly the "recompute it beside the reward" trap `stub_commit`'s docstring warns about, and the
    reason it does not bite: reversing `_T2_Y` and dropping the reference-class term from `R` both
    left the suite green.

    The whole 3 x 3 table is pinned, off the shipped reward, because the two mutations move
    DIFFERENT cells: reversing the outcome encoding turns 2.25 into 1.25 on the diagonal, and
    dropping the `T2_PRIOR_CLASS` term turns the hedge-on-hedge cell from 1.0 into 2.25. Either one
    alone is invisible in a single number.
    """
    got = _delta_task(direction)._score(stub_commit(vals))
    assert got["reward"] == pytest.approx(reward, abs=1e-12)
    assert got["correct"] == (1.0 if R.t2_decision(vals) == direction else 0.0)
    assert got["R"] == pytest.approx(
        (1.0 if R.t2_decision(vals) == direction else 0.0)
        - (1.0 if R.T2_PRIOR_CLASS == direction else 0.0), abs=1e-12)
    with pytest.raises(ValueError):
        R.delta_reward(stub_commit(vals), direction="sideways", disease="K5", horizon=5)


def test_the_floor_rung_scores_the_value_the_spec_names():
    """`FLOOR_REPORT`'s top value 1.0 -> 0.9 survived, because the only test on the floor compared
    the report to `list(R.FLOOR_REPORT)` -- self-referential, and it cannot move when the constant
    moves. 5 to 18 rollouts per archived arm reach this rung, so its REWARD is a real number in a
    real headline: pin that instead."""
    got = _rate_task("K5", 5, 0.02)._score(Stub())
    assert got["report_level"] == R.FLOOR and got["m"] == 5.0
    # The VALUES, written out. `== list(R.FLOOR_REPORT)` is what the old test did and it is
    # self-referential: the assertion moves with the constant, so it cannot see the constant move.
    assert R.rate_report(Stub(), "K5", 5)[0] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert got["point"] == 0.5
    assert got["crps_loss"] == pytest.approx(0.238, abs=1e-12)
    assert got["reward"] == pytest.approx(R.ALPHA * got["R"] + 1.0 - 0.238, abs=1e-12)
    # A second truth, because at a LOW truth the floor's CRPS is invariant to its top value:
    # 1.0 -> 0.9 lowers term1 by 0.02 and the dispersion term by exactly 0.02, so `crps_loss` stays
    # 0.238 to the last bit and the mutation survived a test that checked only this item. Measured,
    # not reasoned: at truth 0.95 the same change moves it 0.220 -> 0.240.
    assert R.crps(list(R.FLOOR_REPORT), 0.95) == pytest.approx(0.220, abs=1e-12)


# --------------------------------------------------------------------------- stops and metrics
#
# Three more mutation survivors, all of the same shape: a value or a sign that decides a published
# number, checked only for being a finite float.

def _n_tool_calls(n):
    return Stub([tool_msg("cw_look", {"state_id": "C2:aaa"}) for _ in range(n)])


@pytest.mark.asyncio
@pytest.mark.parametrize("task", [
    RateTask(RateData(idx=0, prompt="p", version="C2", disease="K5", horizon=5, true_rate=0.02,
                      zero_stratum=False, pop_zero_share=0.727), None),
    PhenoCalTask(PhenoCalData(idx=0, prompt="p", true_rate=0.02, version="C2",
                              variant="PhenoCal_C2", gp_linked=True, disease="K5", horizon=5,
                              n=100), None),
], ids=["internal", "phenocal"])
async def test_the_budget_stop_fires_at_budget_plus_grace_and_not_before(task):
    """`GRACE` 3 -> 2 survived on BOTH halves of the environment. The one test that mentions the
    grace period exercises the TOOLSET's refusal (`_exhausted`), never the stop -- and they are
    different objects: the toolset stops serving MOVES at `BUDGET`, and the episode ends `GRACE`
    turns later so the agent can still answer. Shrinking the gap ends the episode while an agent
    that has just been told "state your answer now" is doing so, which is the exact failure
    `_exhausted` was written to remove (11 of 26 rollouts scoring a flat worst case)."""
    # 26 + 3, WRITTEN OUT. `task.data.budget + R.GRACE` is what this said first and it is
    # self-referential twice over: both terms move with the constants, so `GRACE 3 -> 2` shifted the
    # expected boundary along with the real one and the test stayed green. A boundary test has to
    # name the boundary.
    n = 26 + 3
    assert task.data.budget == 26 and R.GRACE == 3
    assert _spent(_n_tool_calls(n - 1)) == n - 1
    assert await task.out_of_budget(_n_tool_calls(n - 1)) is False
    assert await task.out_of_budget(_n_tool_calls(n)) is True


class Cut:
    """A trace whose last model call reports why generation ended."""

    def __init__(self, reason):
        self.messages, self.last_reply, self.info = [], "", {}
        self.calls = [{"finish_reason": "stop"}, {"finish_reason": reason}]


@pytest.mark.asyncio
async def test_truncated_is_one_only_when_the_last_turn_was_cut():
    """Inverting `truncated` survived: the only assertion on it is "every metric is a real number",
    and 1.0 and 0.0 are both real numbers. It is not decoration -- a cut-off turn emits no tool call
    and lands in the stop table as `agent_completed`, indistinguishable from an agent that finished
    and chose not to answer, and in one discarded run, 86-92% of every worst-case item was one. An
    inverted flag reports the confound as absent exactly when it is present."""
    from cohortworld_env.internal import _truncated
    assert _truncated(Cut("length")) == 1.0
    assert _truncated(Cut("stop")) == 0.0
    assert _truncated(Cut("tool_calls")) == 0.0
    assert _truncated(Stub()) == 0.0                       # no calls recorded at all
    pc = PhenoCalTask(PhenoCalData(idx=0, prompt="p", true_rate=0.02, version="C2",
                                   variant="PhenoCal_C2", gp_linked=True, disease="K5",
                                   horizon=5, n=100), None)
    assert await pc.truncated(Cut("length")) == 1.0 and await pc.truncated(Cut("stop")) == 0.0


def test_the_record_source_stratum_is_read_off_the_item_it_describes():
    """`gp_linked` and the `matched_stratum` metric that publishes it, both invertible with the suite
    green. This is the split every PhenoCal headline must be reported by: **38.6% of PhenoCal_C2 test
    items ask about "hospital records only" patients**, a population the lattice does not contain
    (it is GP-linked only, because OPCS and BNF history need primary-care linkage). Inverted, the
    metric labels the 61.4% the environment CAN answer as the ones it cannot, and a split headline
    reads exactly backwards."""
    from cohortworld_phenocal.taskset import PHENOCAL, PhenoCalConfig, PhenoCalTaskset
    raw = {}
    for i, line in enumerate(open(f"{PHENOCAL}/PhenoCal_C2_test.jsonl")):
        if i >= 200:
            break
        raw[i] = "hospital records only" in json.loads(line)["prompt"]
    tasks = list(PhenoCalTaskset(
        PhenoCalConfig(id="pc", variant="PhenoCal_C2", split="test", num_tasks=200)).load())
    assert len(tasks) == 200
    for t in tasks:
        assert t.data.gp_linked is not raw[t.data.idx], f"idx {t.data.idx}"
    n_linked = sum(t.data.gp_linked for t in tasks)
    assert (n_linked, len(tasks) - n_linked) == (47, 153), "the fixture's stratum mix moved"


@pytest.mark.asyncio
async def test_matched_stratum_publishes_the_stratum_and_not_its_negation():
    """The metric half of the same defect: `matched_stratum` is what lets a score be SPLIT by whether
    the environment holds the item's population at all, and inverting it survived the suite."""
    def pc(gp_linked):
        return PhenoCalTask(PhenoCalData(idx=0, prompt="p", true_rate=0.02, version="C2",
                                         variant="PhenoCal_C2", gp_linked=gp_linked, disease="K5",
                                         horizon=5, n=100), None)
    assert await pc(True).matched_stratum(Stub()) == 1.0
    assert await pc(False).matched_stratum(Stub()) == 0.0
