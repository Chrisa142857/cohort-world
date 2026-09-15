"""The three internal CohortWorld tasks, on complete five-axis data: T3 rate, T1 locate, T2 delta.

**Three tasksets, one task class each.** This module used to hold one `InternalTaskset` declared
`vf.Taskset[InternalTask, InternalConfig]` and selected between the three tasks with a `which` field.
That does not work, and the way it failed is the reason this file is shaped as it is:

`Taskset`'s first type parameter fixes the TASK CLASS at runtime, not just for a type checker. On the
served path -- the default, and the path every CohortWorld run has taken -- the client ships only the
task DATA over the wire and `verifiers/v1/serve/server.py` rebuilds the object as
`type(self.env.taskset).task_type()`. The `LocateTask` that `load()` yielded was discarded and
reconstructed as `InternalTask`, so **all 120 T1 rollouts of one measured run were scored by T3's
CRPS rubric against a `true_rate` of -1.0 and came back a constant -1.0** -- zero variance, i.e. zero
group-normalised advantage, i.e. unlearnable by construction. The archived traces record it plainly:
`task.type == "InternalTask"` on a T1 prompt.

The same parameter fixes the DATA class, and that half had already been noticed and "fixed" by giving
all three tasks one `InternalData` with every task-specific field optional. That workaround is what
turned a LOUD failure (T1 crashed on `horizon: Field required`) into a silent mis-scoring, and it is
deleted here: three data classes with required fields, so a malformed row fails at validation instead
of scoring as `""` forever.

verifiers documents the rule this now obeys, in `verifiers/v1/taskset.py`'s own module docstring:
"there is a one-to-one mapping between taskset and task type, i.e. a taskset may only yield one task
type." One implementation module, three shim packages (`cohortworld-internal`, `cohortworld-locate`,
`cohortworld-delta`), because `utils/loaders.py` refuses a package exporting more than one Taskset.

Two things differ from the external PhenoCal items and both ride on the item rather than being
assumed away:

* The reference forecast is built from the INTERNAL train split. PhenoCal's (disease, horizon) base
  rates are means over a different population and would not be a fair reference here.
* Items are drawn 50/50 from the zero and non-zero strata, so `zero_stratum` and `pop_zero_share`
  ride on every item. Any aggregate meant to describe the population must be reweighted; an
  unweighted mean over this set describes the set, not the population.
"""
from __future__ import annotations
import asyncio
import json
import random
import re as _re, os, sys
from typing import ClassVar, Iterator

import verifiers.v1 as vf
from pydantic import field_validator

from . import reward as R
from .lattice import _GRAN
from .reward import BUDGET, GRACE, REWARD_SPEC_VERSION
from .taskset import _spent, committed_this_turn
from .toolset import CohortWorldToolset, CohortWorldToolsetConfig

# The item directory is an argument, never a literal: CW_ITEMS names the released item set. Left
# unset, the placeholder below makes the first open fail with a path that says what is missing,
# rather than a path that names a machine.
ITEMS = os.environ.get("CW_ITEMS") or "CW_ITEMS-is-not-set-point-it-at-the-item-directory"

_SYS = ""
if os.environ.get("CW_SYS"):
    _p = os.environ["CW_SYS"]
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _c in (_p, os.path.join(_repo, _p)):
        if os.path.exists(_c):
            _SYS = open(_c).read().strip()
            break
    else:
        raise FileNotFoundError(f"CW_SYS={_p!r} not found (cwd={os.getcwd()!r}, repo={_repo!r})")

# Absolute, resolved from THIS file. tests/test_contract.py::test_every_module_level_path_is_absolute_from_a_foreign_cwd
# guards this: verifiers' subprocess runtime chdirs to a cache dir, and a loader that tried a RELATIVE
# candidate first would, from a foreign cwd that happens to hold a prompts/ dir, read someone else's file and
# report nothing.
_THINK_SYS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "prompts", "sys_cohort_think_twostage.txt")
_THINK_SYS = ""


def think_system_prompt() -> str:
    """The system prompt for the two-stage variants, and the documented lever for model behaviour.

    `docs/v1/gepa.md:9` names `Task.system_prompt` as the thing that shapes a policy's behaviour, and
    until now the T3 think variants shipped `_SYS = ""` -- the policy saw the tool schema the chat
    template renders and nothing else: no statement of the procedure, and no statement that an answer
    without a reading is worth nothing.

    Structural, not content-specific, in DeepSeek-R1's sense (arXiv:2501.12948 §2.2.3: "We intentionally
    limit our constraints to this structural format, avoiding any content-specific biases"). It says
    reason-then-act and what the committed number is made of; it never says which group to choose or
    which way the risk should go.

    The last line is R1's economics rather than a new reward term. R1's format reward checks only the
    think tags (§2.2.2); what forces the ANSWER to exist is the accuracy reward returning zero when no
    parseable answer is there. Here the population metric is that accuracy reward: without a reading
    there is no prediction, and the rollout earns nothing from it. The prompt states that plainly,
    because a policy that is never told cannot be said to have chosen.
    """
    global _THINK_SYS
    if not _THINK_SYS:
        if not os.path.exists(_THINK_SYS_FILE):
            raise FileNotFoundError(f"{_THINK_SYS_FILE} not found")
        _THINK_SYS = open(_THINK_SYS_FILE).read().strip()
    return _THINK_SYS


_DNAME: dict = {}


def _disease_name(version: str, code: str) -> str:
    """The clinical label for a disease code, as the external items carry it.

    Without it an internal item asks about `"H91" (ICD-10 H91)` while an external one asks about
    `"eye / ear disorder [H2]" (ICD-10 H2)`. That is a difference in how hard the QUESTION is to
    read, not in the science, and the two halves of the design are meant to be comparable.
    """
    if version not in _DNAME:
        from .lattice import _shared
        lat = _shared(version)
        _DNAME[version] = dict(lat.dname) if lat is not None else {}
    return _DNAME[version].get(code, code)


def _toolsets(config) -> list[vf.Toolset]:
    """The tool server for one rollout.

    `config` here is a TaskConfig -- it carries neither `version` nor `tools`, because
    `Task.toolsets(config)` is handed the task's own config and not the TasksetConfig. That is not
    something to work around here: `CohortWorldToolset.setup_task` reads the version, the entry
    demographics and the budget off the task DATA, which is the only object that has them.
    """
    return [CohortWorldToolset(getattr(config, "tools", CohortWorldToolsetConfig()))]


def _truncated(trace) -> float:
    """1.0 if the last model turn was cut at the generation cap.

    A cut-off turn emits no tool call, so the harness ends the episode as `agent_completed` --
    indistinguishable in the stop table from an agent that finished and chose not to answer. On one
    discarded run, 86-92% of every worst-case item was a truncated final turn, and since truncation
    rate follows prompt verbosity the arms were separating on that rather than on reasoning.
    """
    for src in ("calls", "llm_calls"):
        calls = getattr(trace, src, None) or []
        if calls:
            last = calls[-1]
            fr = getattr(last, "finish_reason", None) or (
                last.get("finish_reason") if isinstance(last, dict) else None)
            return 1.0 if fr == "length" else 0.0
    return 0.0


_MEMO_TOKEN = f"{REWARD_SPEC_VERSION}:{os.urandom(8).hex()}"
"""Identifies THIS process's memo entries. A random token, not a pid, because pids are reused."""


def _memo(trace, compute):
    """Score once per trace: metrics run before rewards and both want the same components.

    The cache lives in `trace.info` because that is the only mutable per-trace bag the task hooks
    receive -- and verifiers SERIALISES `info` into the archived trace. So an archived trace carries
    a frozen copy of the score its own run computed, and any later rescoring that trusted it would
    return the OLD contract's number while looking like it had recomputed one. That is a
    confirming-result trap of the worst kind: the rescoring agrees with the archive, which reads as
    corroboration, and is instead the archive quoting itself. Caught when a fixed `commit_payload`
    changed `rate_report` to `stated_commit` on 10 of 12 live rollouts while `RateTask._score` on the
    same traces still said `auto_observed` for all 12.

    Only one archived run was affected on disk -- every earlier archive predates `_memo` reaching a
    serialised field -- but "only one file so far" is how this class of defect always starts.

    `_MEMO_TOKEN` is what makes the cache process-local: an entry this process did not write is
    ignored and recomputed. The stale entry is left in place rather than deleted, so reading an
    archive never silently edits it.
    """
    info = getattr(trace, "info", None)
    if isinstance(info, dict) and info.get("_cw_score_by") == _MEMO_TOKEN:
        return info["_cw_score"]
    got = compute()
    if isinstance(info, dict):
        info["_cw_score"], info["_cw_score_by"] = got, _MEMO_TOKEN
    return got


# --------------------------------------------------------------------------- shared shape

class _CWData(vf.TaskData):
    """What every internal item carries, and nothing that only one task uses.

    `version`, `age_band` and `sex` are NOT decoration: `CohortWorldToolset.setup_task` reads them
    off the task data to bind the right lattice and place the agent in a cohort of the right age and
    sex before it acts. Without them the placement never fires and every rollout starts wherever the
    state dictionary happens to begin -- "a question about 40-44 year old men answered from inside a
    65-69 year old women's cohort", after which every navigation misses.

    This base holds only the fields all three tasks genuinely share. It is NOT the old
    `InternalData`, which carried every task's fields with defaults so that any row validated against
    any task -- that is what let a T1 item be scored by T3's rubric without anything raising.
    """
    version: str
    age_band: str = ""
    sex: str = ""
    budget: int = BUDGET
    """Carried on the DATA because that is the only object `setup_task` receives.

    `InternalConfig.tools.budget` was dead: `Task.toolsets(config)` gets a TaskConfig, so setting
    `tools.budget` in a TOML changed the resolved-config printout and nothing else, while four
    hardcoded copies of `26` did the actual work. One value, `reward.BUDGET`, now reaches the toolset
    through here.
    """


class _CWTask(vf.Task):
    """Stops and toolsets shared by all three internal tasks. Scoring is per task, deliberately."""

    @vf.stop
    async def committed(self, trace: vf.Trace) -> bool:
        return committed_this_turn(trace)

    @vf.stop
    async def out_of_budget(self, trace: vf.Trace) -> bool:
        return _spent(trace) >= self.data.budget + GRACE

    @classmethod
    def toolsets(cls, config) -> list[vf.Toolset]:
        return _toolsets(config)


# --------------------------------------------------------------------------- T3: rate

class RateTaskConfig(vf.TaskConfig):
    """Per-task knobs for T3. `reward` names the form in `reward.T3_REWARD_VARIANTS` a run trains
    on; the default is the scored contract. Set from an RL config as `env.taskset.task.reward`."""
    reward: str = R.DEFAULT_T3_VARIANT

    patients_per_population: int = 1
    """M: how many patients ONE episode answers, so that one episode IS one population.

    1 keeps the historical shape -- one episode, one patient -- and every archived run used it.

    Above 1, `CohortWorldEnv.run` draws M-1 further patients from this taskset at ROLLOUT TIME and
    answers all M concurrently in the same episode, each in its own context with its own
    <cohort-think> turn, its own action turns and its own committed number. One episode is then a
    complete population and a group of N episodes is a true M x N matrix, entirely present in
    `Algorithm.score_group`.

    WHY IT IS DONE HERE AND NOT IN THE DATASET OR THE DISPATCHER. A dataset row stays ONE patient, so
    `item_id` keeps its meaning (it is the GRPO group and CS-GRPO's problem key) and no split, no
    curriculum bookkeeping and no archived comparison moves. The dispatcher cannot deliver M problems
    in one group even with a cs_grpo-only subclass: `curriculum/base.py:57-58` raises "A finalized
    group contains multiple task keys" and `train_source.py:31` constructs a Curriculum
    unconditionally, so a multi-problem group cannot be reported at all.

    Minting tasks at rollout time and fanning one seat out concurrently is verifiers' own documented
    pattern, not a workaround: `environments/proposer_solver/proposer_solver/taskset.py:144-149` runs
    `agents.solver.run(solve_task)` in an `asyncio.TaskGroup` `self.config.n` times on a task minted
    in `run()`, and `episode.py:104` makes `traces: list[Trace]` first-class.

    Note what it does NOT fix: a cell still needs a published reading before it can carry a
    prediction, so a population only forms once cells act. The matrix removes the STALE comparison
    set (members were 1-2 policy versions old) and the cross-batch bookkeeping; it does not remove
    the requirement that every cell answers."""

    @field_validator("patients_per_population")
    @classmethod
    def _sane_m(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"patients_per_population must be >= 1, got {v}")
        return v

    @field_validator("reward")
    @classmethod
    def _known_variant(cls, v: str) -> str:
        if v not in R.T3_REWARD_VARIANTS:
            raise ValueError(f"unknown T3 reward variant {v!r}; one of {sorted(R.T3_REWARD_VARIANTS)}")
        return v


class _CWConfig(vf.TasksetConfig):
    """The knobs all three internal tasksets share. No `which`: the task is the taskset id now, see
    the module docstring."""
    split: str = "unseen"                    # train / seen / unseen -- profile-disjoint
    num_tasks: int = 1_000_000               # read the whole file; the eval's own shuffle samples
    tools: CohortWorldToolsetConfig = CohortWorldToolsetConfig()


class RateConfig(_CWConfig):
    # `task` narrows the TaskConfig vf.TasksetConfig already carries (the one every Task is
    # constructed with) to the T3 subclass; it stays a TaskConfig, which is what `yield
    # RateTask(..., c.task)` needs -- an earlier shadowing with a STRING is what broke that. Only
    # T3 has it: a `task.reward` on the locate or delta source is an unknown key there, not a no-op.
    task: RateTaskConfig = RateTaskConfig()

    cells_per_cohort: int = 1
    """How many published (horizon, disease) cells to serve per cohort. 1 is the frozen item set,
    byte for byte -- the item's own cell and nothing else. Above 1, each cohort additionally yields
    that many DRAWN cells, expanding the training set without writing a new item file.

    TRAIN ONLY, enforced below. The seen and unseen splits are the held-out measurement and are served
    exactly as the frozen file has them; drawing extra questions on their cohorts would change what the
    words "seen" and "unseen" mean in every number already recorded against them.
    """

    cells_seed: int = 42
    """Seeds the draw. Recorded in the run config, because a training set that cannot be regenerated is
    not a training set anyone can compare against."""


class RateData(_CWData):
    item_id: str = ""
    """The item this task came from, and -- when a cohort serves several drawn cells -- WHICH cell.

    Not decoration: anything that groups rollouts by problem keys on it. Two questions about the same
    cohort are two problems, and a key that could not tell them apart would put a rollout answering
    about one disease into the group being compared on another.
    """

    disease: str
    horizon: int
    true_rate: float
    zero_stratum: bool
    pop_zero_share: float
    n_band: str = ""
    hops_from_entry: int = -1
    target_cohort: str = ""
    """The lattice cohort this item's `true_rate` IS. Carried so the environment can withhold it.

    Every T3 item is a lattice cohort -- `source.kind == "lattice"` and `answer_rate` is that
    cohort's published rate -- and all 18,593 items' targets are still served. The field was in the
    items file from the start and simply never reached the environment, so `cw_observe` on that one
    state returned the label. Default "" and nothing is masked, which is what an item set without
    the field gets.
    """

    @property
    def mask_cohort(self) -> str:
        """Which cohort `setup_task` withholds the rates of. T3's target, and nothing else.

        Read through a NAMED property rather than off `target_cohort` directly, because
        `LocateData.target_cohort` is the opposite kind of thing: T1's target is the answer the agent
        must WALK TO and its rates were never the label, so a `setup_task` that masked any
        `target_cohort` it found would silently change T1's environment as well. The two fields share
        a name because the items file does; they do not share a meaning.
        """
        return self.target_cohort


class RateTask(_CWTask, vf.Task[RateData, vf.State, RateTaskConfig]):
    """T3: put a number on one patient's risk, scored exactly as the external PhenoCal items are.

    `r = ALPHA * R + (1 - CRPS)`. `cohortworld_env.reward` holds the derivation of both halves and
    the report ladder that guarantees there is something to score on every episode. Under a
    non-default `config.reward` the trained reward is the named variant's; every metric below keeps
    its contract meaning.
    """

    def _score(self, trace) -> dict:
        return _memo(trace, lambda: R.rate_reward(
            trace, truth=self.data.true_rate, source="internal",
            disease=self.data.disease, horizon=self.data.horizon,
            variant=self.config.reward, version=self.data.version,
            age_band=self.data.age_band, sex=self.data.sex,
            # The item's own cohort is what re-anchors the reference and the declared delta. Absent
            # (an item set without the field, or the external judge) the demographic-entry geometry
            # and reward stand, which is how an archive rescores to its own number.
            target_cohort=self.data.target_cohort))

    @vf.reward
    async def rate(self, trace: vf.Trace) -> float:
        return self._score(trace)["reward"]

    @vf.metric
    async def crps_loss(self, trace: vf.Trace) -> float:
        """The CRPS itself, so the reward's two halves can be reported apart. Pair with
        `crps_ref_loss` for the skill score, which must be computed from the AGGREGATE losses and
        never by averaging a per-item ratio -- per item that ratio is unbounded (a 10x error on one
        question scored -11.8) and its mean is decided by whichever item had the smallest reference
        loss, not by the model."""
        return self._score(trace)["crps_loss"]

    @vf.metric
    async def discrimination(self, trace: vf.Trace) -> float:
        """`R = |b - y| - |point - y|`: how much closer to the truth than the reference forecast."""
        return self._score(trace)["R"]

    @vf.metric
    async def side_vs_base(self, trace: vf.Trace) -> float:
        """1.0 when the report's median and the truth lie on the same side of the base rate `b`
        (0 at median == b). Logged for every variant, so a `base` run and an `entry_step` run
        read the same monitor."""
        return self._score(trace)["side"]

    @vf.metric
    async def crps_ref_loss(self, trace: vf.Trace) -> float:
        """CRPS of the reference forecast. Defined for 100% of items now; see `reward.base_rate` for
        the cascade and for what the fabricated 0.0 it replaces was doing to every skill score."""
        return R.crps([self._score(trace)["base"]], self.data.true_rate)

    @vf.metric
    async def crps_zero_loss(self, trace: vf.Trace) -> float:
        """The other trivial floor: predict 0 for everything. On zero-heavy internal items it is the
        STRONGER of the two, so a headline quoted only against the base rate flatters itself."""
        return R.crps([0.0], self.data.true_rate)

    @vf.metric
    async def zero_stratum(self, trace: vf.Trace) -> float:
        """1.0 if this item's truth is exactly zero. Report split by it, and reweight with the item's
        own `pop_zero_share` -- the set is 50/50 by construction and the population is not."""
        return 1.0 if self.data.zero_stratum else 0.0

    @vf.metric
    async def pop_zero_share(self, trace: vf.Trace) -> float:
        """Read off the item rather than the module constant it used to be. Every item already
        carried this field and nothing read it, while `POP_ZERO_SHARE = 0.727` sat hardcoded a few
        lines above -- two sources for one number, only one of which travels with the item set."""
        return self.data.pop_zero_share

    @vf.metric
    async def report_level(self, trace: vf.Trace) -> float:
        """1 stated_commit / 2 stated_text / 3 auto_observed / 4 floor. Publish its distribution with
        any score: the reward has no non-answer branch any more, so this is what replaces the
        answer-rate column, and an effect can live entirely inside one level."""
        return self._score(trace)["report_level"]

    @vf.metric
    async def report_malformed(self, trace: vf.Trace) -> float:
        return self._score(trace)["report_malformed"]

    @vf.metric
    async def truncated(self, trace: vf.Trace) -> float:
        return _truncated(trace)

    @vf.metric
    async def budget_used(self, trace: vf.Trace) -> float:
        return float(_spent(trace))


    # The `cohort_think` process term, reported every step. Without these none of the term's own
    # values reaches `monitors/file/metrics.jsonl`, and the run's stop rules cannot be read while it
    # is running. `_score` is memoised, so they are free; they are 0.0 under every other variant.
    @vf.metric
    async def think_rank(self, trace: vf.Trace) -> float:
        return float(self._score(trace).get("think_rank", 0.0))

    @vf.metric
    async def think_edge(self, trace: vf.Trace) -> float:
        return float(self._score(trace).get("think_edge", 0.0))

    @vf.metric
    async def think_declared(self, trace: vf.Trace) -> float:
        return float(self._score(trace).get("think_declared", 0.0))

    @vf.metric
    async def think_moved(self, trace: vf.Trace) -> float:
        return float(self._score(trace).get("think_moved", 0.0))

    @vf.metric
    async def think_relevant(self, trace: vf.Trace) -> float:
        """The code the first move added lies in the target cohort's code set: the move parameter is
        the part of this patient's own history the lattice can express. UNPAID; reported so a run
        can be read for whether the chosen edge tracks the question or ignores it."""
        s = self._score(trace)
        code = s.get("think_moved_code") or ""
        if not code:
            return 0.0
        try:
            from .lattice import _shared
            target = getattr(self.data, "target_cohort", "") or ""
            lat = _shared(self.data.version)
            st = lat.states.get(target) or lat.states.get(f"{self.data.version}:{target}")
            if not st:
                return 0.0
            codes = {str(c[0]) if isinstance(c, (list, tuple)) else str(c)
                     for ax in ("icd", "opcs", "bnf") for c in (st["axes"].get(ax) or [])}
            return 1.0 if code in codes else 0.0
        except Exception:
            return 0.0

    @vf.metric
    async def think_edge_true(self, trace: vf.Trace) -> float:
        return float(self._score(trace).get("think_edge_true", 0.0))

    @vf.metric
    async def cal_declared(self, trace: vf.Trace) -> float:
        """Did this rollout declare a delta BEFORE its move, under the `_SIGNED_NUM_V2` parse?
        `think_declared` is the same question under `_SIGNED_NUM`, which cannot see a number that ends
        a sentence and so reported 0.20 where the truth is 0.997 (reward._SIGNED_NUM_V2's docstring).
        Both are emitted: a run whose metrics.jsonl carried only the `_SIGNED_NUM` number would be read as a policy that
        stopped declaring when it had not. Reporting only."""
        return float(self._score(trace).get("cal_declared", 0.0))

    @vf.metric
    async def cal_delta(self, trace: vf.Trace) -> float:
        """The DELTA payment, before its weight (0 for every variant that does not pay it).

        Here for the same reason the `cohort_think` process metrics above are: without it the ONE term `cohort_think_cal`
        pays for the process never reaches `monitors/file/metrics.jsonl`, so the smoke gate
        (the format gate) and the step-25 readouts would be measuring
        `think_edge` -- a diagnostic that is no longer paid -- while the run was learning from
        something else. Reporting only: no reward value moves."""
        return float(self._score(trace).get("cal_delta", 0.0))


# --------------------------------------------------------------------------- T1: locate

class LocateTaskConfig(vf.TaskConfig):
    """Per-task knobs for T1. `think` turns on the two-turn protocol of `sec:csgrpo`: the question
    asks for the match first, the system prompt becomes the one the cold start was trained on, and
    `CohortWorldEnv` follows up with `ASKS_LOCATE`. Default False, so every archived T1 score is
    produced by the same path as before. Set from an eval config as `env.taskset.task.think`.

    `budget` caps the tool calls of ONE cell. It has to live here rather than in `tools.budget`,
    which is dead for the reason its own docstring gives: `Task.toolsets(config)` is handed a
    TaskConfig that carries no `tools`, so the value never arrives and `_CWData.budget` is what the
    server reads. 0 means the environment default. It exists because the two-turn T1 cell needs a
    STOP: at the default the trained policy averaged 34.4 cw_move and 19.4 cw_observe per episode and
    committed in 0.06, so 365 of 392 scored episodes ended on out_of_budget having wandered, while
    the T3 two-turn cell spends a median of 3 calls. T1 scores the cohort the agent stands on, so
    look, move and commit is the whole intended shape."""
    think: bool = False
    budget: int = 0
    nudges: int = 0
    """How many times the environment may ask on this cell. 0 means `NUDGES_PER_EPISODE`. It only
    matters together with `think`, because the ask loop is what it feeds."""


class LocateConfig(_CWConfig):
    """The shared knobs; a separate class because the type parameter is what selects the task."""
    task: LocateTaskConfig = LocateTaskConfig()


class LocateData(_CWData):
    target_cohort: str
    hops_from_entry: int = -1


class LocateTask(_CWTask, vf.Task[LocateData]):
    """T1: stand on the cohort whose recorded history best speaks for this patient.

    `r = S(final) - S(entry)`, with `S = 1.0` on the target and `NEAR / (1 + dist)` elsewhere, where
    `dist` is the symmetric difference of the two cohorts' axis sets. No proper-score term: T1's
    answer is a POSITION and the target is deterministic given the item, so there is no outcome
    variable, no belief to elicit and nothing for a scoring rule to be proper about.

    Subtracting `S(entry)` is invisible to GRPO -- within a group every rollout shares the item and
    hence the constant -- but it makes the contract readable: a policy that never moves scores
    exactly 0.0 on every item, by construction.

    There is no "self-quote" attack to gate against: `cw_commit` reports the cohort the agent is
    STANDING on, and since `cw_move` only accepts ids the environment advertised, an id cannot be
    typed without walking to it.
    """

    def _score(self, trace) -> dict:
        return _memo(trace, lambda: R.locate_reward(
            trace, target=self.data.target_cohort, version=self.data.version,
            age_band=self.data.age_band, sex=self.data.sex))

    @vf.reward
    async def locate(self, trace: vf.Trace) -> float:
        return self._score(trace)["reward"]

    @vf.metric
    async def exact(self, trace: vf.Trace) -> float:
        return self._score(trace)["exact"]

    @vf.metric
    async def axis_distance(self, trace: vf.Trace) -> float:
        """Features between the committed cohort and the target. Replaces `hops_to_target`, which
        BFS'd a strictly DIRECTED graph and returned "further than 3" for 71% of near-misses, so most
        of the partial credit collapsed onto one constant."""
        return self._score(trace)["dist"]

    @vf.metric
    async def report_level(self, trace: vf.Trace) -> float:
        return self._score(trace)["report_level"]

    @vf.metric
    async def truncated(self, trace: vf.Trace) -> float:
        return _truncated(trace)

    @vf.metric
    async def budget_used(self, trace: vf.Trace) -> float:
        return float(_spent(trace))


# --------------------------------------------------------------------------- T2: delta

class DeltaTaskConfig(vf.TaskConfig):
    """Per-task knobs for T2. `think` and `budget` are the T1 flags' counterparts, see
    `LocateTaskConfig`."""
    think: bool = False
    budget: int = 0


class DeltaConfig(_CWConfig):
    """The shared knobs; a separate class because the type parameter is what selects the task."""
    task: DeltaTaskConfig = DeltaTaskConfig()


class DeltaData(_CWData):
    disease: str
    horizon: int
    answer_direction: str
    start_cohort: str
    """Where the agent is placed: the cohort the edge leaves FROM.

    T2 used to place the agent nowhere at all -- its rows carried no `age_band`/`sex`, so
    `setup_task`'s `enter()` never fired and all 625 items began at `next(iter(states))`, the same
    arbitrary cohort, while the prompt told the agent to "stand on a group without that feature".
    That is exactly the failure `Lattice.enter` was written to prevent, reintroduced on the one task
    that had never been run and so had nobody to notice it.
    """
    axis: str = ""
    op: str = ""


class DeltaTask(_CWTask, vf.Task[DeltaData]):
    """T2: is one operation's effect on the rate distinguishable from noise, and which way.

    NOT "predict the delta". `rate(C_k) = rate(C_0) + sum(deltas)` telescopes exactly -- verified on
    4,640 random 2-4 step paths with zero exceptions -- so the delta is the difference of two
    published rates and there is nothing to learn. The question the identity does not answer is
    whether the difference is separable from sampling noise at these denominators; only 0.196-0.219%
    of edge cells are.

    Scored on the same report channel and the same CRPS as T3, against `y in {0, 0.5, 1}`. T2 is the
    task where a PURE proper score is provably degenerate: expected `1 - CRPS` over the train class
    prior is NON-MONOTONE in accuracy -- 0.838 at q = 0, a minimum of 0.788 at chance, recovering only
    past q ~= 0.75 -- so a systematically-wrong-but-honest policy beats every policy below 75%
    accuracy. That is arXiv:2608.00301 Prop. 9(b) realised in our own item set, and it is what the
    `ALPHA * R` term exists to remove.

    The class balance MUST be published beside any T2 number: a three-way label with an unbalanced
    prior flatters any constant answer, and the unseen split's majority class is not the train
    split's.
    """

    def _score(self, trace) -> dict:
        return _memo(trace, lambda: R.delta_reward(
            trace, direction=self.data.answer_direction,
            disease=self.data.disease, horizon=self.data.horizon))

    @vf.reward
    async def delta(self, trace: vf.Trace) -> float:
        return self._score(trace)["reward"]

    @vf.metric
    async def correct(self, trace: vf.Trace) -> float:
        return self._score(trace)["correct"]

    @vf.metric
    async def crps_loss(self, trace: vf.Trace) -> float:
        return self._score(trace)["crps_loss"]

    @vf.metric
    async def discrimination(self, trace: vf.Trace) -> float:
        return self._score(trace)["R"]

    @vf.metric
    async def report_level(self, trace: vf.Trace) -> float:
        return self._score(trace)["report_level"]

    @vf.metric
    async def report_malformed(self, trace: vf.Trace) -> float:
        return self._score(trace)["report_malformed"]

    @vf.metric
    async def truncated(self, trace: vf.Trace) -> float:
        return _truncated(trace)

    @vf.metric
    async def budget_used(self, trace: vf.Trace) -> float:
        return float(_spent(trace))


# --------------------------------------------------------------------------- prompts
#
# Prompt text is part of the reward contract: it changes the achievable policy set, so a change here
# is a MAJOR bump under REWARD_SPEC_VERSION. The sentence that had to go is "a reply with neither
# scores zero" -- it is now false. There is no unanswered episode; the agent's most recent reading is
# its answer until it states a better one.

_T3_FINISH = (
    "\n\nStart with cw_look: it shows the group you are standing in and every move that group "
    "accepts, each with an id. Take a move with cw_move(id) -- you cannot compose one, and an id "
    "that is not listed has no group behind it. cw_observe reads a rate where you stand."
    "\n\nKeep your reasoning short -- a few sentences per step. A reply that runs past the "
    "generation limit is cut off mid-thought, so say less and act more."
    f"\n\nYou have {BUDGET} tool calls in total and every call costs one, including observe; each "
    "result tells you how many are left. Your most recent reading is your answer until you replace "
    "it. To state a better one, call commit with at least five plausible values for that "
    "probability, each a fraction between 0 and 1 (write 2% as 0.02) -- a stated answer is the only "
    "way to express how uncertain you are, and the spread across your values is what that is scored "
    "on. If you would rather not call a tool, end your reply with a line "
    "'ANSWER: [v1, v2, v3, v4, v5]'.")

_T1_FINISH = (
    "\n\nStart with cw_look: it shows the group you are standing in and every move that group "
    "accepts, each with an id. Take a move with cw_move(id) -- you cannot compose one, and an id "
    "that is not listed has no group behind it."
    "\n\nKeep your reasoning short -- a few sentences per step. A reply that runs past the "
    "generation limit is cut off mid-thought, so say less and act more."
    f"\n\nYou have {BUDGET} tool calls in total and every call costs one, including observe. "
    "Whichever group you are standing on is your answer; call commit with no arguments once you are "
    "standing on the one you have chosen, so walk to it first.")

_T2_FINISH = (
    "\n\nMeasure it rather than reasoning about it: you are standing on a group WITHOUT the second "
    "patient's difference. Read its rate with cw_observe, make the change with cw_move, and read it "
    "again. Do it from more than one starting group if the budget allows. A difference that reverses "
    "sign between starting groups is not distinguishable from noise, and saying so is the correct "
    "answer."
    "\n\nKeep your reasoning short -- a few sentences per step. A reply that runs past the "
    "generation limit is cut off mid-thought, so say less and act more."
    f"\n\nYou have {BUDGET} tool calls in total and every call costs one, including observe. Your "
    "two readings are your answer until you replace them. To state a better one, call commit with "
    "at least five values, each 1.0 for higher, 0.0 for lower or 0.5 for not distinguishable -- "
    "mixing them is how you say how sure you are, and the spread is what that is scored on. If you "
    "would rather not call a tool, end your reply with a line 'ANSWER: [v1, v2, v3, v4, v5]'.")

_GRAN_TABLE = {v: k for k, v in _GRAN.items()}
"""granularity -> lattice version, derived from `lattice._GRAN` rather than re-typed.

The two used to be hand-written inverses of one mapping in two modules. It decides which disease
table NAMES the code and nothing else: a T2 item carries `granularity` and no `version`, so looking
the name up in C2's table left a g1 code rendering as its own bare letter -- the question asked about
`"H" (ICD-10 H)`. Which lattice the agent WALKS is a different question, answered by the edge's own
`_audit.from` prefix; conflating the two is what `DeltaTaskset.load` refuses to do.
"""

_AXIS_WORD = {"opcs": "procedure", "bnf": "medicine", "icd": "diagnosis"}


def _t2_clause(axis: str, op: str, code) -> str:
    """How the one difference between the two patients reads.

    Per (axis, operation), because one template cannot cover them: `age_band shift` is **79 of the
    118** items and rendered as "had instead 70-74 (age_band)", which is not English and not a
    clinical statement. `sex switch` is worse -- "had instead Male (sex)" reverses who is who.
    """
    w = _AXIS_WORD.get(axis, axis)
    if axis == "age_band":
        return f"the second is in the {code} age band instead"
    if axis == "sex":
        return f"the second is {str(code).lower()} and the first is not"
    if op == "add":
        verb = {"bnf": "was also prescribed", "opcs": "also had",
                "icd": "was also diagnosed with"}.get(axis, "also had")
        return f"the second {verb} {code}"
    if op == "switch":
        return f"the second has {code} on the {w} axis where the first has something else"
    if op.startswith("zoom"):
        finer = op.endswith("finer")
        return (f"the second's {w} history is recorded at a "
                f"{'finer' if finer else 'coarser'} level of detail")
    return f"the second differs on the {w} axis ({code})"


_CODE_IN_BRACKETS = _re.compile(r"\s*\[[A-Z0-9]{1,6}\]")


def cohort_think_question(patient: str, disease_words: str, horizon, sex: str) -> str:
    """THE question, for both stages. The SFT dataset builder imports this rather than
    carrying its own copy, because a second copy of a prompt is how the two stages drift apart.

    Byte-exact to the cold-start demonstration set, quirks included: the disease is named WITHOUT its code (the
    demonstrations never show codes, and a policy asked about `"H91" (ICD-10 H91)` is being asked a
    different question), and the horizon reads "in the next 1 years" at one year because that is what
    1,845 demonstrations were written against. Fixing the grammar here would silently change the prompt
    the cold start knows; it is a separate decision with a separate retraining.
    """
    low = str(sex or "").lower()
    pron = "she is" if low.startswith("f") else ("he is" if low.startswith("m") else "they are")
    Cap = "She" if low.startswith("f") else ("He" if low.startswith("m") else "They")
    return (f"{patient}\n\n"
            f"Your task is to: estimate the probability that {pron} diagnosed with \"{disease_words}\" for the "
            f"first time in the next {horizon} years. "
            f"{Cap} has never "
            f"been diagnosed with it.\n\n"
            f"Reason about which larger group of patients you would compare this one against, and by how "
            f"much this patient's own risk differs from that group's.")


def locate_think_question(patient: str, sex: str) -> str:
    """T1 in the two-turn form: state the match first, act on it second.

    Same shape as `cohort_think_question` and deliberately not the same words, because the thing to
    reason about is which group the history matches, not by how much a rate moves."""
    low = str(sex or "").lower()
    him = "her" if low.startswith("f") else ("him" if low.startswith("m") else "them")
    his = "her" if low.startswith("f") else ("his" if low.startswith("m") else "their")
    return (f"{patient}\n\n"
            f"Your task is to: find the recorded group that best speaks for {him} -- the one whose "
            f"recorded history differs from {his} own on the fewest features.\n\n"
            "Reason about which recorded group you would match this patient to, and which of this "
            "patient's features that group does not carry.")


def delta_think_question(stem: str) -> str:
    """T2 in the two-turn form. `stem` is the item's own question without its finish instructions."""
    return (f"{stem}\n\nReason about which way that one difference moves the risk, and by how "
            "much.")


def strip_disease_code(name: str) -> str:
    """The disease label as the demonstrations render it: no bracketed code."""
    return _CODE_IN_BRACKETS.sub("", name or "")


# --------------------------------------------------------------------------- the rollout loop
NUDGE = (
    "You have said which group you would compare this patient against and by how much their risk "
    "differs. Now carry it out. The moves have ids you cannot guess, so look first, and reply with "
    "nothing but this function call:\n\n"
    "<tool_call>\n<function=cw_look>\n</function>\n</tool_call>\n\n"
    "The result will list every move this group accepts. Then take the one that reaches the group you "
    "named, read the rate there with cw_observe, and call cw_commit with that reading minus the "
    "difference you stated."
)
"""What the environment says when a reply reasons but takes no action.

FORM A, chosen on evidence after form B was measured and failed. Form B is DeepSeek-R1's shape -- one
reply carries the reasoning and the action, and nothing but the reward requires the action
(arXiv:2501.12948 Table 1; the format reward checks only the think tags, §2.2.2, and what forces the
answer to exist is the accuracy reward scoring zero without one). It was tried with the system prompt
stating the procedure and the economics, and produced ZERO tool calls in ~190 rollouts. R1's economics
punishes not acting only when acting sometimes happens: with every rollout in a group failing to act,
the advantage of acting and of not acting are both zero, and there is nothing to prefer.

The tool literature says the same thing from the other side. R1-Searcher pays an explicit invocation
reward in stage 1 (retrieval 0.5 if n>=1, 0 if n=0, answer correctness ignored) and drops it in stage 2 --
but it starts from a base model that already searches sometimes. The systems that start from a cold start
which never calls (ReTool, ParaVT, ASTER) all put tool calls in the COLD-START DATA, a path fixed shut
here by design: the demonstrations teach the reasoning, and the actions are learned in the environment.

So the environment asks. ONCE per episode, only when the reply made no tool call and nothing is
committed, naming only what the reply already said and one no-argument call -- the lowest-barrier
emission there is. It is the documented mechanism, not a workaround: `docs/v1/agent.md:14-19` and
`skills/create-environments/SKILL.md:191` make the env's `run()` and `interaction.turn()` THE way a
caller drives turns. Measured against no ask at all: num_turns 1.0 -> 2.2, calls 0% -> 2.6%.

What the ask must NOT do is become the thing that is trained. `credit_turn = "first"` gives every turn
after the first exactly zero advantage, so the policy is never paid for answering the ask -- which it
does by repeating its own reasoning almost verbatim.
"""

NUDGE_AFTER_LOOK = (
    "You now have the list of moves this group accepts. Take the one that reaches the group you named: "
    "reply with nothing but a cw_move call whose move_id is copied EXACTLY from that list, then read the "
    "rate there with cw_observe, then call cw_commit with that reading minus the difference you stated."
)
"""The SECOND ask, and it must differ from the first or the second ask is not a second question.

`NUDGE`'s operative sentence is "look first, and reply with nothing but this function call:
<tool_call><function=cw_look></function></tool_call>". Sending it twice asks an episode that has ALREADY
looked to look again -- and the population the second ask exists for is exactly the episodes that obeyed
the first one literally. A literally-obedient policy would look twice, spend another budget unit, and end
in the state the second ask was added to escape.

This one names the move instead, and it is asked of the state where the move works: `cw_move` is accepted
120/155 = 77.4% when a listing is already in context and 0/122 when it is not (Fisher p = 9.3e-13). It
still asks for a call whose argument is COPYABLE rather than guessable -- the earlier version that asked
for `cw_move` with an id produced 0 calls in 23 nudged episodes, because at that point no listing had ever
been printed. The difference is not the wording, it is that the listing is now there.
"""

ASKS = (NUDGE, NUDGE_AFTER_LOOK)
"""One entry per ask, so the loop asks a different question each time rather than repeating itself."""

NOTE_NO_RATE = (
    "This task asks for a GROUP, not a rate, so there is no cell to read here and cw_observe returns nothing you can use. Navigate with cw_look, which lists where you are and every move you may take, and commit where you stand. "
)
"""Said in every T1 ask, because the trained policy's habit is invalid here.

CS-GRPO trains on T3, where an episode HAS an asked cell and `cw_observe` with no argument reads it.
That is the cheapest read the policy knows and it is what the trained policy emits. T1 has no asked
cell, so the call returns `no disease/horizon given and this episode has no asked cell to default to`
and costs a budget unit. Measured over 247 episodes of the handoff protocol: 19.2 observes per
episode of which 49% errored that way, against 3.0 and 9% for the untrained agent on the same cell.
The policy was navigating with a sense that returns nothing, which is why it moved 35.1 times per
episode and committed in 11 of 181."""

NUDGE_LOCATE = (
    "You have said which recorded group this patient's history matches. Now carry it out. " + NOTE_NO_RATE +
    "The moves "
    "have ids you cannot guess, so look first, and reply with nothing but this function call:\n\n"
    "<tool_call>\n<function=cw_look>\n</function>\n</tool_call>\n\n"
    "The result will list every move this group accepts. Then take the ONE move that reaches the group "
    "you named and call cw_commit there."
)
NUDGE_LOCATE_AFTER_LOOK = (
    "You now have the list of moves this group accepts. " + NOTE_NO_RATE +
    "Take the one that reaches the group you named: "
    "reply with nothing but a cw_move call whose move_id is copied EXACTLY from that list, then call "
    "cw_commit."
)
NUDGE_DELTA = (
    "You have said which way that one difference moves the risk. Now measure it rather than arguing "
    "it. The moves have ids you cannot guess, so look first, and reply with nothing but this function "
    "call:\n\n<tool_call>\n<function=cw_look>\n</function>\n</tool_call>\n\n"
    "The result will list every move this group accepts. Then read the rate where you stand with "
    "cw_observe, take the ONE move that makes the change, read the rate again, and "
    "call cw_commit with AT LEAST FIVE values, each 1.0 for higher, 0.0 for lower or 0.5 for not distinguishable. Mixing them is how you say how sure you are, and fewer than five is refused."
)
NUDGE_DELTA_AFTER_LOOK = (
    "You now have the list of moves this group accepts. Read the rate where you stand with cw_observe, "
    "then reply with a cw_move call whose move_id is copied EXACTLY from that list, read the rate "
    "again, and call cw_commit with AT LEAST FIVE values, each 1.0 for higher, 0.0 for lower or 0.5 for not distinguishable. Mixing them is how you say how sure you are, and fewer than five is refused."
)
NUDGE_LOCATE_AGAIN = (
    "You have moved. This task asks for a GROUP, not a rate, so there is no cell to read here and "
    "cw_observe returns nothing you can use. Navigate with cw_look, which lists where you are and "
    "every move you may take. "
        "If the group you are standing on still differs from this patient's recorded history on any "
    "feature, you have not arrived, so take the listed move that removes the next one. Commit only "
    "when nothing remains to remove, and copy every move_id EXACTLY from the list."
)
"""The repeatable T1 ask. It states when the walk is FINISHED, which is the one thing the trained
policy has no signal for.

Four earlier wordings were measured and none of them is this. Two asks at the default budget let the
policy wander, 841 of 919 episodes out_of_budget. A budget of 4 made it take its one move and stop,
committing in 789 and landing at a median axis distance of 6.0 against the untrained agent's 7.0,
exactly right on 0.6%. Asking once per move to name the remaining difference wandered again. Handing
over `_T1_FINISH`, the instruction the untrained agent commits under in 606 of 919, produced 11
commits in 181, because the policy was navigating with `cw_observe` and every call errored.

Once the environment answered that call rather than erroring, the behaviour became orderly at once: 3.6 moves an episode, 905 of
917 committing, a median of 3 budget units, a median distance of 6.0 and 2.2% exact. What remains is
that it stops after three moves while the target is three to eight away, and `_T1_FINISH` tells it
that whichever group it stands on is its answer, which is an invitation to stop. This wording removes
that invitation and names the stopping condition instead. It reveals nothing: the residual difference
is visible in the cohort description `look` and `observe` already print beside the patient's own
history."""

ASKS_LOCATE = (NUDGE_LOCATE, NUDGE_LOCATE_AFTER_LOOK, NUDGE_LOCATE_AGAIN)
ASKS_DELTA = (NUDGE_DELTA, NUDGE_DELTA_AFTER_LOOK)
"""T1 and T2 asks. `NUDGE` names a rate and a stated difference, which neither task has, because T1
commits a COHORT and T2 commits a direction. Sending the T3 text to them asks for something the task
cannot accept, so each task gets the same two-step shape, look and then act on the listing, worded for
what it actually commits.

FIVE VALUES ON T2. `reward.MIN_REPORT` is 5 and a shorter commit is refused, after which the answer
falls through the ladder. The first wording here asked for "a number", and every one of the 169 scored
two-turn T2 episodes committed exactly one value, so 160 of them were scored on the last rate read
rather than on anything the policy stated. The wording now asks for five, as the one-turn closing
instructions already do.

ONE MOVE, like the T3 pair. The first wording here said "take the moves that reach the group you
named", and the plural read as an invitation to walk: 838 of 919 T1 episodes exhausted the 26-call
budget at a mean of 28.3 calls and only 69 committed, against a median of 3 calls in the T3 two-turn
cell, where 889 of 943 episodes spend exactly look, move, observe. CohortThink moves once by design
and these asks now say so."""


def asks_for(task) -> tuple:
    """Which asks this task answers. Defaults to the T3 pair, so nothing archived changes."""
    return {"LocateTask": ASKS_LOCATE, "DeltaTask": ASKS_DELTA}.get(type(task).__name__, ASKS)


def nudges_for(task) -> int:
    """How many times the environment may ask. `NUDGES_PER_EPISODE` for everything except a T1 cell
    that sets `nudges`, whose last ask is repeatable and whose target is several moves away. The
    budget still bounds the walk."""
    if type(task).__name__ == "LocateTask":
        return int(getattr(getattr(task, "config", None), "nudges", 0) or 0) or NUDGES_PER_EPISODE
    return NUDGES_PER_EPISODE

NUDGES_PER_EPISODE = 2
"""Two, and the second one is the whole point.

It was one, on the argument that "a second ask would be teaching by repetition rather than by reward".
That argument is sound and is NOT what changed. What changed is a measurement: the first ask is spent
buying the menu, so with one ask the episode ends holding it.

The ask names `cw_look` because move ids cannot be guessed. An episode that obeys therefore spends its
single ask on `cw_look`, receives the full legal-move listing -- and then the loop is out of iterations.
Measured on one training run: `cw_move` is accepted 120/155 = 77.4% when a listing is already in context and 0/122
when it is not (Fisher p = 9.3e-13). So the one configuration in which the action is reliably available
is exactly the one the episode was ending in.

The second ask lands on that context. It is not repetition: the first ask and the second are asked of
different states -- one with no ids in context, one with all of them."""


def acted(trace) -> bool:
    """Did this episode take an action, rather than only talk?

    A named predicate because the inline version was wrong in a way reading it could not reveal:
    `reward.tool_results` is a GENERATOR, so `if reward.tool_results(trace)` tested a generator object's
    truthiness -- always true -- and the ask never fired once. The first run with it reproduced the run
    without it, rollout for rollout.
    """
    from . import reward as _R
    return any(True for _ in _R.tool_results(trace)) or committed_this_turn(trace)


class CohortWorldEnv(vf.SingleAgentEnv):
    """The single-agent env with two additions: a reply that reasons but does not act is asked to act,
    and -- when `patients_per_population > 1` -- one episode answers M patients, so one episode IS one
    population and a group of N episodes is a true M x N matrix in `Algorithm.score_group`."""

    _pool: ClassVar[list] = []
    """The taskset, materialised once per process, to draw an episode's other patients from.

    Materialised rather than re-iterated: `Taskset.__iter__` re-runs `load()`, which reads the item file.
    One list per process is the same trade the lattice memoisation makes, and for the same reason -- the
    per-rollout cost is paid inside the env-server worker's event loop, where it blocks every other
    episode multiplexed onto that worker."""

    def _materialise_pool(self) -> list:
        """Build the draw pool ONCE, and do it in `setup()` rather than in the first `run()`.

        Measured on the real train taskset: `list(self.taskset)` is 16,770 tasks and takes 67.7 s.
        `run()` executes inside the env-server worker's asyncio loop, so paying it there would block every
        other episode multiplexed onto that worker for over a minute -- the exact shape of an earlier
        defect, where a per-rollout `Lattice(...)` build inside the same loop held worker CPU
        at 88% and left the GPUs idle 77% of the time. `setup` is verifiers' documented hook for work an
        env owns and cannot rebuild per rollout (`verifiers/v1/env.py:82`: "optional overrides: `setup`,
        `finalize`, `start`/`stop`"). The lazy path is kept as a fallback so a caller that never runs
        `setup` still works rather than drawing an empty population."""
        cls = type(self)
        if not cls._pool:
            cls._pool = list(self.taskset)
        return cls._pool

    async def setup(self, agents) -> None:
        """`agents` is NOT optional: the base is `Env.setup(self, agents: Agents)` and `env.py:271` calls
        it as `await self.setup(agents)`. The first version of this override took only `self`, every
        episode died with `TypeError: setup() takes 1 positional argument but 2 were given`, and the run
        crashed 4 minutes in with "10 consecutive zero-output batch equivalents" -- 40 episodes, 0 traces,
        0 requests to vLLM. It was not caught because the pre-launch check asserted the override was a
        coroutine function and stopped there; arity was never compared. tests/test_population_episode.py
        now compares the SIGNATURE against the base, which is the assertion that would have caught it."""
        await super().setup(agents)
        task_cfg = getattr(getattr(self, "taskset", None), "config", None)
        m = getattr(getattr(task_cfg, "task", None), "patients_per_population", 1) or 1
        if int(m) > 1:
            self._materialise_pool()

    def _peers(self, task, m: int) -> list:
        """M-1 OTHER patients for this episode's population, drawn from this taskset.

        Seeded by the anchor's own key, so the population a given anchor is scored against is the same
        on a replay as it was on the run -- a population drawn from an unseeded RNG would make the
        metric unreproducible, which is the one property a scored artifact cannot lose.

        Drawn from `self.taskset`, so they are the split the run is training on: the taskset is already
        split-scoped, and drawing from anywhere else would put val or test patients in a train population.
        """
        if m <= 1:
            return []
        pool = type(self)._pool or self._materialise_pool()
        if len(pool) <= 1:
            return []
        anchor = getattr(task, "key", None) or getattr(getattr(task, "data", None), "item_id", "") or ""
        rng = random.Random(f"population:{anchor}")
        want = min(m - 1, len(pool) - 1)
        peers, seen = [], {id(task)}
        while len(peers) < want:
            cand = pool[rng.randrange(len(pool))]
            if id(cand) in seen:
                continue
            seen.add(id(cand))
            peers.append(cand)
        return peers

    async def _answer(self, task, agents) -> None:
        """One patient: the opening reply, then up to NUDGES_PER_EPISODE asks. One trace."""
        asks = asks_for(task)
        n_asks = nudges_for(task)
        async with agents.agent.interaction(task) as interaction:
            # a prompted task speaks first: the bare turn takes its opening reply
            segment = await interaction.turn()
            for _i in range(n_asks):
                # `segment.terminated` ONLY -- `acted()` used to gate this too, and that was the bug.
                # `acted()` is true after ANY tool call, so an episode that did exactly what the ask
                # requested (one `cw_look`) was recorded as having acted and never asked again, ending
                # with the legal-move listing sitting unused in its context. The predicate that belongs
                # here is "is this episode finished", which is what `terminated` means: the `committed`
                # and `out_of_budget` stops (`_CWTask`) both raise it. `acted()` is kept and still
                # tested -- it is a true statement about a trace -- it just is not this loop's question.
                if segment.terminated:
                    return
                segment = await interaction.turn(asks[min(_i, len(asks) - 1)])

    async def run(self, task, agents) -> None:
        _c = getattr(task, "config", None)
        # T3 selects the two-turn loop by its reward variant, which is what every archived T3 score
        # was produced under. T1 and T2 have no reward variants, so they select it with `think`, and
        # both default to the single-turn loop.
        if (getattr(_c, "reward", None) not in R.T3_THINK_PROMPT_VARIANTS
                and not bool(getattr(_c, "think", False))):
            await agents.agent.run(task)        # every other variant keeps the loop its scores used
            return
        m = int(getattr(getattr(task, "config", None), "patients_per_population", 1) or 1)
        peers = self._peers(task, m)
        if not peers:
            await self._answer(task, agents)    # M == 1: byte-identical to every archived run
            return
        # ONE EPISODE, M TRACES, ONE POPULATION. The same seat is run concurrently on the anchor and on
        # its drawn peers; `episode.traces` collects all M. This is `proposer_solver`'s shape
        # (environments/proposer_solver/proposer_solver/taskset.py:144-149) -- a task minted in `run()`
        # fanned out to concurrent runs of one seat -- applied to patients instead of solvers.
        async with asyncio.TaskGroup() as tg:
            for one in (task, *peers):
                tg.create_task(self._answer(one, agents))


# --------------------------------------------------------------------------- cell resampling
ZERO_FRACTION = 0.5
"""The zero / non-zero split of the drawn cell, reproducing `build/render_spirit.py`'s `ZERO_FRACTION`.

Two copies of a rule drift, so `tests/test_frozen_items.py` asserts this value against the builder's
own. Sampling uniformly from the published cells instead would be a different task: most cells are zero
(the builder stratifies for exactly that reason), and a training set drawn uniformly would be dominated
by items whose answer is 0.
"""


def estimable_cells(state: dict, granularity: int) -> list[tuple]:
    """Every (horizon index, disease index, rate) this cohort actually publishes.

    `not_estimable` is the panel's own k>=10 certificate per cell, and honouring it is what makes the
    draw legal -- the same gate `lattice.observe` applies. Mirrors `render_spirit._cells`' candidate
    construction; the two are checked against each other in the tests.
    """
    ne = state.get("not_estimable") or {}
    mat = state.get(f"rates_g{granularity}") or state.get("rates")
    if not mat:
        return []
    out = []
    for hi in range(len(mat)):
        m = ne.get(str(hi), ne.get(hi))
        if m == "all":
            continue
        bad = set(m or ())
        out += [(hi, di, mat[hi][di]) for di in range(len(mat[hi])) if di not in bad]
    return out


def draw_cells(state: dict, granularity: int, k: int, rng) -> list[tuple]:
    """Up to `k` DISTINCT published cells, each drawn with the builder's zero / non-zero stratification.

    Why this exists. The frozen item set uses ONE cell per cohort -- `render_spirit.build_t1_t3`'s
    docstring says so, and `_cells` enumerates the whole matrix and then throws all but one away. A full
    census of the released panel counted 42,711,644 published cells against 21,352 in the
    item set: 0.05%, one cell in two thousand, with every publishable item on a distinct cohort. The
    training signal is therefore one question per cohort, and a policy can meet it by memorising a
    cohort's answer rather than by learning that a reference class transfers across diseases.

    Drawing more cells here EXPANDS THE TRAINING SET WITHOUT WRITING A NEW ITEM FILE: the frozen set on
    disk is untouched, every score already recorded stays comparable, and the first cell drawn is the
    item's own so the frozen items remain a subset of what is served.
    """
    cand = estimable_cells(state, granularity)
    if not cand:
        return []
    nz = [c for c in cand if c[2] > 0]
    zc = [c for c in cand if c[2] == 0]
    out, seen = [], set()
    for _ in range(k * 4):                      # a few extra tries: the draw can repeat a cell
        if len(out) >= k:
            break
        want_zero = rng.random() < ZERO_FRACTION
        pick = (zc if want_zero else nz) or (nz if want_zero else zc)
        if not pick:
            break
        c = pick[rng.randrange(len(pick))]
        if (c[0], c[1]) in seen:
            continue
        seen.add((c[0], c[1]))
        out.append(c)
    return out


# --------------------------------------------------------------------------- tasksets

class RateTaskset(vf.Taskset[RateTask, RateConfig]):
    """T3 on CohortWorld's own cohorts. Served as `cohortworld-internal`."""

    def load(self) -> Iterator[RateTask]:
        c: RateConfig = self.config
        if c.cells_per_cohort > 1 and c.split != "train":
            raise ValueError(
                f"cells_per_cohort={c.cells_per_cohort} on split {c.split!r}: the draw is TRAIN ONLY. "
                "seen and unseen are the held-out measurement and are served exactly as the frozen file "
                "has them; drawing extra questions on their cohorts changes what those numbers mean.")
        served = 0
        with open(f"{ITEMS}/t3_rate_{c.split}.jsonl") as f:
            for i, line in enumerate(f):
                if served >= c.num_tasks:
                    break
                d = json.loads(line)
                # the item's own cell first, so the frozen set is a SUBSET of what is served
                cells = [(None, d["disease"], int(d["horizon"]), float(d["answer_rate"]),
                          bool(d["zero_stratum"]))]
                cells += _drawn_cells(d, c.cells_per_cohort - 1, c.cells_seed)
                for suffix, disease, horizon, rate, zero in cells:
                    if served >= c.num_tasks:
                        break
                    served += 1
                    yield _rate_task(d, i, c, disease, horizon, rate, zero, suffix)


def _drawn_cells(d: dict, k: int, seed: int) -> list[tuple]:
    """`k` further published cells for this item's cohort, drawn reproducibly.

    Empty when k <= 0, when the cohort is not in the lattice, or when it publishes nothing else -- an
    item that cannot be expanded is served once, not dropped. The RNG is seeded from the item id, so
    the same item draws the same questions on every run and on every worker.
    """
    if k <= 0:
        return []
    from .lattice import _shared
    # granular=True: this is the ONE runtime path that reads `rates_g{1,2,3}` (via `estimable_cells`
    # below). It runs on the taskset, once per env-server worker -- not in the per-rollout tool server,
    # which is why the fields are opt-in and default off. Asking for them here is not an optimisation
    # detail: `estimable_cells` falls back with `state.get(f"rates_g{g}") or state.get("rates")`, so a
    # lattice without them would draw cells from the WRONG matrix and raise nothing.
    lat = _shared(d.get("version"), granular=True)
    sid = str(d.get("target_cohort") or "")
    if lat is None or sid not in lat.states:
        return []
    g = int(d.get("granularity") or 2)
    rng = random.Random(f"{seed}:{d.get('item_id')}")
    names = _disease_table(d.get("version"), g)
    if not names:
        return []
    out = []
    for hi, di, rate in draw_cells(lat.states[sid], g, k + 1, rng):
        if di >= len(names):
            continue
        horizon = lat.horizons[hi] if hi < len(lat.horizons) else None
        if horizon is None or (names[di] == d["disease"] and int(horizon) == int(d["horizon"])):
            continue                                  # the item's own cell is already served
        out.append((f"c{hi}-{di}", names[di], int(horizon), float(rate), rate == 0.0))
        if len(out) >= k:
            break
    return out


_DTABLE: dict = {}


def _disease_table(version: str, g: int) -> list:
    """The disease list the cohort's rate matrix is indexed by, at this granularity."""
    key = (version, g)
    if key not in _DTABLE:
        # Read from the panel CW_PANEL names, like every other released file, and refuse to run on a
        # missing table: an empty list here used to drop every extra served cell of a T3 item with no
        # error, so a clone whose data sat anywhere but <repo>/official changed the episode silently.
        from .lattice import PANEL
        f = os.path.join(PANEL, f"CohortWorld_{version}_diseases_g{g}.json")
        if not os.path.exists(f):
            raise FileNotFoundError(f"T3 needs the disease table {f}; set CW_PANEL to the released panel")
        _DTABLE[key] = json.load(open(f))["diseases"]
    return _DTABLE[key]


def _rate_task(d: dict, idx: int, c: RateConfig, disease: str, horizon: int, rate: float,
               zero: bool, suffix: str | None) -> RateTask:
    """One T3 task. `suffix` is None for the item's own cell and names the drawn cell otherwise, so a
    cohort's several questions are several DISTINCT items -- anything keyed on `item_id` (the GRPO
    group, CS-GRPO's problem key) would otherwise treat two different questions as one."""
    he = "he" if str(d["sex"]).lower().startswith("m") else "she"
    He = he.capitalize()
    yr = "year" if int(horizon) == 1 else f"{horizon} years"
    name = _disease_name(d["version"], disease)
    ask = (f"\n\nYour task is to: estimate the probability that {he} is diagnosed with "
           f"\"{name}\" (ICD-10 {disease}) for the first time in the next "
           f"{yr}. {He} has never been diagnosed with it.")
    if c.task.reward in R.T3_THINK_PROMPT_VARIANTS:
        # the two-stage pair: the QUESTION is exactly what the cold start was trained on, and the
        # SYSTEM prompt carries the procedure and the functions -- the documented behaviour lever,
        # which these variants shipped empty until now
        prompt = cohort_think_question(d["patient"], strip_disease_code(name), horizon,
                                       str(d.get("sex", "")))
        system = think_system_prompt()
    else:
        prompt = d["patient"] + ask + _T3_FINISH
        system = _SYS                      # every archived score was produced under CW_SYS, or under ""
    return RateTask(
        RateData(idx=idx, prompt=prompt, system_prompt=system,
                 version=d["version"], disease=disease, horizon=int(horizon),
                 true_rate=float(rate), zero_stratum=bool(zero),
                 pop_zero_share=float(d["pop_zero_share"]),
                 age_band=str(d.get("age_band", "")), sex=str(d.get("sex", "")),
                 n_band=str(d.get("n_band", "")),
                 hops_from_entry=int(d.get("hops_from_entry", -1)),
                 item_id=(str(d.get("item_id", "")) + (f":{suffix}" if suffix else "")),
                 # `.get`, not `["target_cohort"]`: an item set without the field masks nothing rather
                 # than failing to load. Read off the item exactly as `LocateTaskset` reads its own, so
                 # the two cannot drift.
                 target_cohort=str(d.get("target_cohort", "") or "")),
        c.task)


class LocateTaskset(vf.Taskset[LocateTask, LocateConfig]):
    """T1 on CohortWorld's own cohorts. Served as `cohortworld-locate`."""

    def load(self) -> Iterator[LocateTask]:
        c: LocateConfig = self.config
        with open(f"{ITEMS}/t1_locate_{c.split}.jsonl") as f:
            for i, line in enumerate(f):
                if i >= c.num_tasks:
                    break
                d = json.loads(line)
                him = "him" if str(d["sex"]).lower().startswith("m") else "her"
                his = "his" if him == "him" else "her"
                ask = (f"\n\nYour task is to: find the recorded group that best speaks for {him} -- "
                       f"the one whose recorded history differs from {his} own on the fewest "
                       "features.")
                think = bool(getattr(c.task, "think", False))
                bud = int(getattr(c.task, "budget", 0) or 0) or BUDGET
                yield LocateTask(
                    LocateData(idx=i, budget=bud,
                               prompt=(locate_think_question(d["patient"], str(d.get("sex", "")))
                                       if think else d["patient"] + ask + _T1_FINISH),
                               system_prompt=think_system_prompt() if think else _SYS,
                               version=d["version"], target_cohort=d["target_cohort"],
                               age_band=str(d.get("age_band", "")), sex=str(d.get("sex", "")),
                               hops_from_entry=int(d.get("hops_from_entry", -1))),
                    c.task)


class DeltaTaskset(vf.Taskset[DeltaTask, DeltaConfig]):
    """T2 on CohortWorld's own cohorts. Served as `cohortworld-delta`.

    **Only the items the environment can actually answer are served, and the rest are counted on
    stderr rather than dropped in silence.** A T2 item names its edge in one lattice (`_audit.from`)
    and its outcome at one granularity, and for 417 of 625 items those are DIFFERENT versions: the
    edge exists in C2 while the asked disease code only exists in C1's table, or the reverse. An
    agent standing on the edge then cannot read the outcome, and an agent that can read the outcome
    is not standing on the edge. A further 32 items name a cross-version edge, which `legal_moves`
    withholds from the served action set (`same_version_only=True`), so the one move whose direction
    is the label cannot be taken at all.

    Serving those items would not make the task hard, it would make it unanswerable -- and T2 has
    never been run, so nobody has yet been misled by them. Survivors: 125 train / 31 seen / 41 unseen.
    **That is below the 120 rollouts the acceptance run needs, and the fix is the item builder
    emitting edges whose lattice and outcome granularity agree, not a change here.** Until then, run
    the acceptance eval on `train`.
    """

    def load(self) -> Iterator[DeltaTask]:
        from .lattice import _shared
        c: DeltaConfig = self.config
        served = skipped = 0
        with open(f"{ITEMS}/t2_delta_{c.split}.jsonl") as f:
            for i, line in enumerate(f):
                if served >= c.num_tasks:
                    break
                d = json.loads(line)
                start = d["_audit"]["from"]
                version = start.split(":", 1)[0]
                lat = _shared(version)
                # Answerable means all three at once: the edge's endpoints are released in this
                # lattice, the move is one the action set advertises, and the asked disease is in
                # THIS version's table.
                # The `cross_version` clause is currently REDUNDANT and is kept deliberately. A
                # mutation run replacing this `or` chain's `cross_version` test with an `and` left
                # the served counts at 125 / 31 / 41 -- an equivalent mutant, not an untested branch:
                # on this item set every cross-version edge also fails one of the other three tests.
                # It is the clause that states the INTENT (`legal_moves` withholds cross-version
                # edges from the served action set, so the one move whose direction is the label
                # cannot be taken), and it stops being redundant the moment an item set carries a
                # cross-version edge whose endpoints are both released. Do not "simplify" it away on
                # the evidence that no test can see it.
                if (lat is None or d.get("cross_version")
                        or start not in lat.states or d["_audit"]["to"] not in lat.states
                        or d["disease"] not in lat.didx):
                    skipped += 1
                    continue
                served += 1
                # No patient narrative here, deliberately: rendering one pins the exact answer_delta
                # on 625/625 items given the from-state, and composed with narrative->state recovery
                # that is answer recovery with zero navigation. No age, no sex, no dated codes --
                # those are the join keys against the edge table.
                clause = _t2_clause(d["axis"], d["op"], d["code"])
                name = _disease_name(_GRAN_TABLE.get(int(d.get("granularity", 2)), "C2"),
                                     d["disease"])
                yr = "year" if int(d["horizon"]) == 1 else f"{d['horizon']} years"
                # "of the same age and sex" is false when the difference IS age or sex.
                alike = ("Two patients alike in every recorded respect but one"
                         if d["axis"] in ("age_band", "sex") else
                         "Two patients of the same age and sex, whose records are alike in every "
                         "recorded respect but one")
                stem = (f"{alike}: {clause}."
                        f"\n\nYour task is to: decide whether the second patient's "
                        f"probability of being diagnosed with \"{name}\" (ICD-10 {d['disease']}) "
                        f"for the first time within {yr} is higher than the first's, lower, or "
                        f"not distinguishable from it.")
                think = bool(getattr(c.task, "think", False))
                prompt = delta_think_question(stem) if think else stem + _T2_FINISH
                axes = lat.states[start]["axes"]
                yield DeltaTask(
                    DeltaData(idx=i, prompt=prompt,
                              budget=int(getattr(c.task, "budget", 0) or 0) or BUDGET,
                              system_prompt=think_system_prompt() if think else _SYS, version=version,
                              disease=d["disease"], horizon=int(d["horizon"]),
                              answer_direction=d["answer_direction"], start_cohort=start,
                              axis=d["axis"], op=d["op"],
                              age_band=str(axes["age_band"]), sex=str(axes["sex"])),
                    c.task)
        print(f"[cohortworld-delta] split={c.split}: served {served}, skipped {skipped} items the "
              f"environment cannot answer (edge lattice != outcome granularity, or a cross-version "
              f"edge the action set withholds)", file=sys.stderr)
