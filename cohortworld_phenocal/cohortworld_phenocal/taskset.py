from __future__ import annotations
import json, os, re
from typing import Iterator

import verifiers.v1 as vf

from cohortworld_env import reward as R
from cohortworld_env.reward import BUDGET, GRACE
from cohortworld_env.internal import (cohort_think_question, strip_disease_code,
                                      think_system_prompt)
from cohortworld_env.taskset import _spent, committed_this_turn
from cohortworld_env.toolset import CohortWorldToolset, CohortWorldToolsetConfig

# The CohortCal directory (PhenoCal_C2_{train,val,test}.jsonl) is an argument: CW_PHENOCAL. Left
# unset, the placeholder makes the first open fail with a path that says what is missing.
PHENOCAL = os.environ.get("CW_PHENOCAL") or "CW_PHENOCAL-is-not-set-point-it-at-the-CohortCal-directory"
# SIDECAR is repo-relative, like `lattice.PANEL`.
SIDECAR = os.environ.get(
    "CW_SIDECAR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "official"))
_AXES: dict[str, dict] = {}
_BASE: dict[str, dict] = {}


def _base_rates(variant: str) -> dict:
    """(disease, horizon) -> mean rate, from the TRAIN split. Kept for the analysis tools in `gen/`.

    NOT what the reward reads. `cohortworld_env.reward.base_rate` does, off a pinned file, and it
    falls back through disease -> horizon -> grand mean so it is defined for 100% of items; this
    returns a plain dict whose `.get()` is `None` for a missing key, and callers that turned that
    `None` into `0.0` fabricated a perfect reference on ~40% of items and inflated every skill score
    they quoted by about 2.4x. Anything scoring a rollout should use the pinned table.

    `1 - CRPS` on raw probabilities cannot rank anything at these rates. Measured on a true rate of
    0.0021: an exact answer scores 1.0000, a 10x overprediction scores 0.9811, and **a constant
    "always predict 0" scores 0.9979** -- above every model score seen so far. The ordering it
    produces is essentially "who guessed closest to zero", which is not calibration.

    Forecast verification handles this with a skill score against a reference (climatology):
    1 - CRPS_model / CRPS_reference. The reference has to be held FIXED across every system being
    compared, so it is taken from `train` and never from the split being scored.
    """
    if variant not in _BASE:
        acc: dict = {}
        path = f"{PHENOCAL}/{variant}_train.jsonl"
        if os.path.exists(path):
            for line in open(path):
                d = json.loads(line)
                m = d["metadata"]
                k = (m["disease_icd10_3char"], int(m["horizon_years"]))
                acc.setdefault(k, []).append(float(d["ground_truth"]["rate"]))
        _BASE[variant] = {k: sum(v) / len(v) for k, v in acc.items()}
    return _BASE[variant]


def _axes_for(variant: str, prefix: str) -> dict | None:
    """The cohort's procedure and prescription history, from the sidecar.

    PhenoCal's own eighteen factors are all diagnoses, so its items carry no OPCS or BNF history --
    and those are the two axes CohortWorld's actions edit. Without them an agent can only zoom, and
    `add` and `switch` have nothing to work on. The sidecar supplies them by recomputing each
    cohort's membership; PhenoCal's files are never modified, because its official split and the
    published leaderboard were produced on exactly those bytes.
    """
    if variant not in _AXES:
        table = {}
        path = f"{SIDECAR}/{variant}_axes_sidecar.jsonl"
        if os.path.exists(path):
            for line in open(path):
                d = json.loads(line)
                table[d["prompt_prefix"]] = d
        _AXES[variant] = table
    return _AXES[variant].get(prefix)


class PhenoCalTaskConfig(vf.TaskConfig):
    """Per-TASK knobs. It exists because `CohortWorldEnv.run` gates its two-stage nudge loop on
    `task.config.reward`, and `task.config` is the TASK config -- `verifiers/v1/task.py:143` -- not the
    taskset's. `RateTaskConfig` is where the internal half puts the same field, read as
    `env.taskset.task.reward`.

    Getting this wrong is silent, and it already cost a full cell twice over. Without the field the
    getattr returns None, the gate refuses, the run takes the single-turn path, and 2 x 31,784 external
    rollouts came back 100% `floor` with BOTH arms at 0.3422 -- the floor constant. The model had closed
    its <cohort-think> block and was never asked to act: 0 cw_commit calls in 300 sampled traces, while
    271 of them had already written a number in prose. My first fix put this field on PhenoCalConfig
    (the TASKSET config), where the gate cannot see it either -- the same failure, one class over."""

    reward: str = "phenocal"
    """`cohort_think_cal` runs the two-stage loop the RL policy was trained under. The default keeps
    every archived PhenoCal cell byte-identical."""


class PhenoCalConfig(vf.TasksetConfig):
    task: PhenoCalTaskConfig = PhenoCalTaskConfig()
    """Declared HERE or the CLI never sees it. `verifiers.taskset_config_type` reads the `task` field's
    annotation off the TASKSET config -- not off the Task's generic parameters -- so a TaskConfig that is
    only bound as `vf.Task[Data, State, Config]` is invisible to `--env.taskset.task.*` and pydantic
    rejects the knob with "Extra inputs are not permitted". `RateConfig` declares `task: RateTaskConfig`
    for exactly this reason; this line is the same statement for PhenoCal."""

    variant: str = "PhenoCal_C2"     # PhenoCal | PhenoCal_C1 | PhenoCal_C2 | BioPhenoCal*
    split: str = "test"
    """PhenoCal's own official profile-disjoint split (seed 42, 50/10/40).

    Train on `train` and `val`, evaluate on `test`, and never mix them: the published PhenoCal
    leaderboard and the R1/R2 results were produced on exactly these files, so a score is comparable
    only if it comes off the same split. The files are read, never written -- editing them would
    silently invalidate every number already published against them.
    """
    num_tasks: int = 1_000_000
    """Rows to read from the benchmark file. Leave it reading the WHOLE file.

    This truncates the file to its HEAD, and the eval's own `shuffle` runs afterwards, over whatever
    survived. With the old default of 500 the eval shuffled 500 rows and took 120 of them -- and the
    first 500 rows of PhenoCal_C2_test are 84% "hospital records only" against 38.6% for the file,
    because the file is not in random order (GP-linked share: 43.8% over rows 0-2,000, 98.2% over the
    last 784). The run sampled exactly the unrepresentative head that `shuffle` was added to avoid.
    Sample with the eval-level `num_tasks` + `shuffle`, not with this.
    """
    version: str = "C2"              # the CohortWorld lattice the agent navigates
    tools: CohortWorldToolsetConfig = CohortWorldToolsetConfig()



_DECADE = {20: "twenties", 30: "thirties", 40: "forties", 50: "fifties",
           60: "sixties", 70: "seventies", 80: "eighties"}


def _stage(band: str, poss: str) -> str:
    """An onset band as a clinician would say it, never as a range.

    "50-60" must not render as "between 50 and 60": a range is a group, and it can also contradict
    the definite age drawn below. "from his early fifties" carries the same information at the same
    precision without either problem.
    """
    m = re.match(r"(\d+)\s*-\s*(\d+)", band.strip())
    if not m:
        return band.strip()
    lo = int(m.group(1))
    return f"from {poss} early {_DECADE.get(lo // 10 * 10, str(lo) + 's')}"


def _render_individual(prompt_text: str, idx: int) -> dict | None:
    """PhenoCal's population question, re-asked about one person.

    The published prompt is a stratum: "Consider UK adults aged 60-64 (male; ...) ... Among such
    people ... what percentage are diagnosed ...". Every one of those elements names a group, and an
    agent trained on group questions has to be re-taught at deployment, where the thing in front of
    it is a patient. So: a definite age drawn inside the published band, a pronoun, onset stated as
    a stage of life, and the question asked about him.

    The age is drawn deterministically from `idx` and stays inside the published band, so it is no
    more precise than the item already was, and re-running the eval asks the same question.
    Verified over the whole PhenoCal_C2 test file: 31,784/31,784 render, with zero occurrences of
    "Consider UK adults", "aged ", "Among such people", "what percentage" or "age band".
    """
    m = re.search(r"aged\s+(\d+)\s*-\s*(\d+)\s*\((male|female);\s*([^)]*)\)", prompt_text)
    q = re.search(r'diagnosed with "([^"]+)"\s*\(ICD-10\s*([A-Za-z0-9]+)\)[^?]*?within the '
                  r'next\s+(\d+)\s+years?', prompt_text, re.S)
    if not m or not q:
        return None
    lo, hi, sex, src = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
    age = lo + (idx * 7919) % (hi - lo + 1)
    he, his, He = ("he", "his", "He") if sex == "male" else ("she", "her", "She")
    who = "man" if sex == "male" else "woman"
    him = "him" if sex == "male" else "her"
    dis_name, dis_code, horizon = q.group(1), q.group(2), int(q.group(3))

    hist = re.search(r"first appeared\):\n(.*?)(?:\n\n|\nCommon procedures|\nAmong such)",
                     prompt_text, re.S)
    lines = []
    if hist:
        for ln in hist.group(1).splitlines():
            ln = ln.strip().lstrip("-").strip()
            if not ln or ln.lower().startswith("no recorded conditions"):
                continue
            band, _, what = ln.partition(":")
            if what.strip():
                lines.append(f"  {_stage(band, his)}: {what.strip()}")

    head = f"A {age}-year-old {who}."
    body = (f"{head} {his.capitalize()} records show:\n" + "\n".join(lines)) if lines else \
           f"{head}\nNothing is recorded for {him} before today."
    note = ("" if "hospital records only" not in src else
            f"\nOnly {his} hospital records are available; {he} has no primary-care prescribing "
            f"record, so any medicine {he} takes will not appear above.")
    when = "in the next year" if horizon == 1 else f"in the next {horizon} years"
    ask = (f"\n\nYour task is to: estimate the probability that {he} is diagnosed with "
           f'"{dis_name}" (ICD-10 {dis_code}) for the first time {when}. '
           f"{He} has never been diagnosed with it.")
    return {"text": body + note + ask, "age": age, "sex": sex}


_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_sys(path: str) -> str:
    """Load the arm's system prompt, resolving a relative path against the REPO, not the cwd.

    This module is imported inside the `cw` tool server's subprocess runtime, whose cwd is not the
    repo. `open("prompts/sys_clinician_contrast_v1.txt")` therefore raised FileNotFoundError at
    import, the module never loaded, the tool server never served, and **every arm-B rollout died
    with `tool server 'cw' not serving in runtime` before a single LLM call** -- 15/15 with zero
    nodes. Arm A survived only because CW_SYS="" short-circuits the open().

    A missing file RAISES rather than falling back to "". Falling back would make the clinician arm
    silently identical to the no-system-prompt arm and produce a clean-looking null result, which is
    worse than a crash.
    """
    if not path:
        return ""
    for cand in (path, os.path.join(_REPO, path)):
        if os.path.exists(cand):
            return open(cand).read().strip()
    raise FileNotFoundError(f"CW_SYS={path!r} not found (cwd={os.getcwd()!r}, repo={_REPO!r})")


_SYS_PATH = os.environ.get("CW_SYS", "")
# Re-run a SUBSET of a split by row index (a JSON list), e.g. the rows a finished run lost to an
# outage. Unset = the whole split. The resolved path belongs in the run's config.yaml.
_IDX = set(json.load(open(os.environ["CW_IDX_FILE"]))) if os.environ.get("CW_IDX_FILE") else None
_SYS = _read_sys(_SYS_PATH)
"""The system prompt for this arm, or "" for the no-system-prompt arm.

Read from CW_SYS so the arms differ in exactly one file and nothing else. The format and budget
contract lives in the USER prompt, byte-identical in every arm, so the arm without a system prompt
does not silently lose it.
"""


class PhenoCalData(vf.TaskData):
    true_rate: float
    version: str = "C2"          # which lattice this question's disease codes live in
    variant: str = "PhenoCal_C2" # which benchmark file, so the base-rate reference matches it
    gp_linked: bool = True       # False = a "hospital records only" item; see `matched_stratum`
    age_band: str = ""
    sex: str = ""
    disease: str
    horizon: int
    n: int
    budget: int = BUDGET
    """Carried on the DATA because `setup_task` is the only hook that receives it; see
    `cohortworld_env.internal._CWData.budget` for the config field it replaces."""


class PhenoCalTask(vf.Task[PhenoCalData, vf.State, PhenoCalTaskConfig]):
    """A PhenoCal question, answered by navigating CohortWorld.

    Scored by CRPS on a predicted distribution, so the number is comparable to the single-turn
    PhenoCal leaderboard while the model is free to act. Two things make the comparison honest and
    must travel with any result:

      * the vocabularies differ. PhenoCal describes a cohort by 18 curated risk factors; CohortWorld
        by ICD/OPCS/BNF codes. The agent has to cross that gap, so this is a transfer test, not a
        lookup, and a model that simply reads a rate off a matching cohort will not find one.
      * the group sizes differ by two orders of magnitude. PhenoCal's cohorts hold a median of 3,125
        people, CohortWorld's 12-13, so rates read inside the environment are far noisier than the
        target. Error has to be split into "found the wrong group" and "found the right group whose
        rate is itself unstable", or the environment's sampling noise is scored as the agent's
        reasoning error.
    """

    def _score(self, trace) -> dict:
        """The reward and its components, computed once per trace.

        Identical to the internal T3 task's, on purpose: the transfer comparison is only honest if
        the two halves are scored by the same object rather than by two copies of the same intention.
        Only `source` differs, which is which train split the reference forecast comes from.
        """
        # One memo implementation, in `cohortworld_env.internal`: this used to be a second copy and
        # the copy is what carried the serialised-cache trap (see `_memo`'s docstring) into the
        # external half as well.
        from cohortworld_env.internal import _memo
        # `variant` was never passed, so this fell to DEFAULT_T3_VARIANT="base" while training and the
        # internal cell used `cohort_think_cal` -- i.e. the two halves were NOT "scored by the same
        # object" that this docstring promises. Guarded because PhenoCalTaskConfig's default reward is
        # "phenocal", which is a taskset name and not a key of T3_REWARD_VARIANTS.
        # A typo in the knob must NOT score as `base`. The first version of this guard fell back
        # silently for anything unrecognised, so `cohort_think_call` would have produced a complete,
        # plausible, wrong cell -- the failure this repo's own rule names ("a default must fail loudly
        # on a mismatch instead of proceeding quietly"). Caught by preflight_reward.py --selfcheck.
        _v = getattr(self.config, "reward", None)
        if _v in R.T3_REWARD_VARIANTS:
            _variant = _v
        elif _v in (None, "", "phenocal"):          # the taskset's own default means the base contract
            _variant = R.DEFAULT_T3_VARIANT
        else:
            raise ValueError(
                f"reward={_v!r} is neither a T3 variant {sorted(R.T3_REWARD_VARIANTS)} "
                f"nor this taskset's default 'phenocal'")
        # `version`/`age_band`/`sex` are keyword-only requirements of the non-base variants (the entry
        # cohort needs them); `base` ignores them, which is why omitting them was invisible until the
        # variant was actually passed. `target_cohort` is deliberately NOT passed: a PhenoCal item has
        # no lattice cohort of its own, and the parameter defaults to absent.
        return _memo(trace, lambda: R.rate_reward(
            trace, truth=self.data.true_rate, source=self.data.variant,
            disease=self.data.disease, horizon=self.data.horizon, variant=_variant,
            version=self.data.version, age_band=self.data.age_band, sex=self.data.sex))

    @vf.reward
    async def rate(self, trace: vf.Trace) -> float:
        """`ALPHA * R + (1 - CRPS)` on the always-emitted report. See `cohortworld_env.reward`.

        Renamed from `crps`, deliberately and not for tidiness: the number this returns is no longer
        a CRPS, and a reward key that still said `crps` would let every downstream table keep
        treating it as one. The CRPS is still published, as the `crps_loss` metric.
        """
        return self._score(trace)["reward"]

    @vf.metric
    async def crps_loss(self, trace: vf.Trace) -> float:
        """The CRPS itself. Pair with `crps_ref_loss` to get the skill score."""
        return self._score(trace)["crps_loss"]

    @vf.metric
    async def discrimination(self, trace: vf.Trace) -> float:
        """`R = |b - y| - |point - y|`: how much closer to the truth than the reference forecast."""
        return self._score(trace)["R"]

    @vf.metric
    async def report_level(self, trace: vf.Trace) -> float:
        """1 stated_commit / 2 stated_text / 3 auto_observed / 4 floor -- publish its distribution
        with any score. It replaces the answered/unanswered column, which no longer exists because
        no episode is unanswered."""
        return self._score(trace)["report_level"]

    @vf.metric
    async def report_malformed(self, trace: vf.Trace) -> float:
        return self._score(trace)["report_malformed"]

    @vf.metric
    async def crps_ref_loss(self, trace: vf.Trace) -> float:
        """CRPS of the reference forecast: the (disease, horizon) base rate from the TRAIN split.

        The skill score is  1 - mean(crps_loss) / mean(crps_ref_loss)  over the eval, and it must be
        computed that way -- from the AGGREGATE scores, not by averaging a per-item ratio. Per item
        the ratio is unbounded (a 10x error on one question scored -11.8) and its mean is decided by
        whichever item had the smallest reference loss, not by the model.

        Why a reference is needed at all: `1 - CRPS` on raw probabilities cannot rank anything at
        these rates. On a true rate of 0.0021 an exact answer scores 1.0000, a 10x overprediction
        scores 0.9811, and a constant "always predict 0" scores **0.9979** -- above every model
        score measured so far. The reference is taken from `train`, never from the split being
        scored, and is held fixed across every model compared, as a skill score requires.
        """
        return R.crps([self._score(trace)["base"]], self.data.true_rate)

    @vf.metric
    async def crps_zero_loss(self, trace: vf.Trace) -> float:
        """CRPS of the other trivial baseline: predict 0 for everything.

        Reported alongside `crps_ref_loss` because on PhenoCal_C2 test it is the STRONGER of the two
        floors -- always-zero scores 0.612 skill against the base rate, since the true rates are
        right-skewed and the (disease, horizon) mean sits well above the typical item. A headline
        quoted only against the base rate would be flattered by the weaker baseline, which is the
        trap the environment spec already warns about. Beat both, or say which one was not beaten.
        """
        return R.crps([0.0], self.data.true_rate)

    @vf.metric
    async def matched_stratum(self, trace: vf.Trace) -> float:
        """1 if CohortWorld actually holds the record-source stratum this question is about.

        The lattice is built on the GP-linked subset only -- OPCS and BNF history require primary
        care linkage -- so all 7,113 C2 states render as "linked primary-care and hospital records"
        and record source is not one of the five axes. **38.6% of PhenoCal_C2 test items ask about
        "hospital records only" patients**, a population the environment does not contain: the agent
        answers them from GP-linked cohort rates, which is a different population with systematically
        different recorded history (those people have no BNF record at all).

        Emitted so every score can be split by it. A headline over all items silently averages the
        61.4% the environment can answer with the 38.6% it structurally cannot.
        """
        return 1.0 if self.data.gp_linked else 0.0

    @vf.stop
    async def committed(self, trace: vf.Trace) -> bool:
        return committed_this_turn(trace)

    @vf.stop
    async def out_of_budget(self, trace: vf.Trace) -> bool:
        return _spent(trace) >= self.data.budget + GRACE

    @vf.metric
    async def truncated(self, trace: vf.Trace) -> float:
        """1.0 if the LAST model turn was cut off at the generation cap.

        A cut-off turn emits no tool call, so the harness ends the episode as `agent_completed` --
        indistinguishable, in the stop table, from an agent that finished and chose not to answer.
        Measured on the first three-arm run: **86-92% of every worst-case (crps 1.0) item was a
        truncated final turn**, 19 of 21 in arm A, 11 of 13 in B, 31 of 36 in C. Since truncation
        rate follows prompt verbosity, the arms were separating on that rather than on reasoning.
        Reported so the confound can never hide inside `agent_completed` again.
        """
        for src in ("calls", "llm_calls"):
            calls = getattr(trace, src, None) or []
            if calls:
                last = calls[-1]
                fr = getattr(last, "finish_reason", None) or (
                    last.get("finish_reason") if isinstance(last, dict) else None)
                return 1.0 if fr == "length" else 0.0
        return 0.0

    @vf.metric
    async def budget_used(self, trace: vf.Trace) -> float:
        return float(_spent(trace))

    @classmethod
    def toolsets(cls, config) -> list[vf.Toolset]:
        """Declared on the TASK, not the taskset, so each rollout gets its own tool server.

        A taskset-scoped server starts once for the whole eval and never sees `setup_task`, so the
        per-question entry placement silently did nothing -- verifiers' own docstring says
        "taskset-scoped servers skip this hook". Per-task servers cost a little more to start and
        are what makes each rollout begin in the cohort its own question is about.

        The version, the entry demographics and the budget all arrive through `setup_task`, which
        reads them off the task DATA. They cannot arrive through `config`: this classmethod is handed
        a TaskConfig, which carries none of the three, so the `model_copy(update={"version": ver})`
        that used to sit here copied a field off an object that never had it and silently did
        nothing -- every rollout then walked the C1 default while the resolved config said C2.
        """
        return [CohortWorldToolset(getattr(config, "tools", CohortWorldToolsetConfig()))]


class PhenoCalTaskset(vf.Taskset[PhenoCalTask, PhenoCalConfig]):
    def load(self) -> Iterator[PhenoCalTask]:
        c = self.config
        if c.split not in ("train", "val", "test"):
            raise ValueError(f"split must be train, val or test; got {c.split!r}")
        path = f"{PHENOCAL}/{c.variant}_{c.split}.jsonl"
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= c.num_tasks:
                    return
                if _IDX is not None and i not in _IDX:
                    continue
                d = json.loads(line)
                m = d["metadata"]
                # The published prompt is a POPULATION question. It is re-asked about one person:
                # a definite age drawn inside the published band, a pronoun, onset as a stage of
                # life. An agent trained on "what percentage of such people" must be re-taught at
                # deployment, where what stands in front of it is a patient, not a stratum.
                ind = _render_individual(d["prompt"], i)
                if ind is None:
                    continue
                # The sidecar splice ("Common procedures and prescriptions in this group
                # (percentage of them with any such record)") is deliberately gone: it is a group
                # statistic, it says "this group" twice, and it was the largest surviving piece of
                # population framing left in the item.
                mm = re.search(r"aged ([\d\-+]+) \((male|female)", d["prompt"])
                sex = ("Female" if mm and mm.group(2) == "female" else "Male") if mm else ""
                if c.task.reward in R.T3_THINK_PROMPT_VARIANTS:
                    # THE TWO-STAGE PAIR, identical to internal.py's branch and for the same reason:
                    # `Task.system_prompt` is the documented behaviour lever, and this taskset shipped
                    # it EMPTY while serving a procedural question of its own. The policy was therefore
                    # tested on CohortCal under a protocol it was never trained or validated under --
                    # no cohort-think contract in the system turn, and every mechanic crammed into the
                    # user turn (1,360 chars here vs 534 on the internal cell). Measured on one
                    # training run: 2823/2974 rollouts scored on the `auto_observed` fallback and 0 on
                    # `stated_commit`, so what was reported as accuracy was where the agent stopped.
                    # The question is built by `cohort_think_question`, never restated here: it is
                    # byte-exact to the demonstration set the cold start was trained on.
                    # THE PATIENT BODY IS THE PANEL'S OWN TEXT, not `_render_individual`'s prose.
                    # The cold start was trained on "Consider UK adults about N age (sex; linkage) with
                    # the following recorded history:" followed by dashed lines, and the PANEL ITEM IS
                    # ALREADY IN THAT SHAPE ("Consider UK adults aged 55-59 (female; primary-care +
                    # hospital records) with the following recorded medical history ...\n- 40-50: excess
                    # alcohol use"). `_render_individual` rewrites it into prose the model never saw
                    # ("A 40-year-old man.\nNothing is recorded for him before today."), and on that
                    # surface 0 of 1,418 rollouts produced a number. The two
                    # corpora differ only in how the dashed lines are grouped -- by onset age here, by
                    # axis internally -- and that is real information in the item, not a rendering choice
                    # to invent. The individualisation stays on the single-turn board path below, where
                    # it was introduced and where its rationale holds.
                    patient = d["prompt"].split("Among such people")[0].strip()
                    prompt = cohort_think_question(patient,
                                                   strip_disease_code(m.get("disease_name", "")),
                                                   int(m["horizon_years"]), sex)
                    system = think_system_prompt()
                else:
                    system = _SYS
                    prompt = (ind["text"] + "\n\nStart with cw_look: it shows the group you are "
                              "standing in and every move that group accepts, each with an id. Take a "
                              "move with cw_move(id) -- you cannot compose one, and an id that is not "
                              "listed has no group behind it. cw_observe reads a rate where you stand."
                              "\n\nKeep your reasoning short -- a few sentences per step. A reply that "
                              "runs past the generation limit is cut off mid-thought, so say less and "
                              "act more.\n\n"
                              f"You have {BUDGET} tool calls in total and every call costs one, "
                              "including observe; each result tells you how many are left. Your most "
                              "recent reading is your answer until you replace it. To state a better "
                              "one, call commit with at least five plausible values for that "
                              "probability, each a fraction between 0 and 1 (write 2% as 0.02) \u2014 a "
                              "stated answer is the only way to express how uncertain you are, and the "
                              "spread across your values is what that is scored on. If you would rather "
                              "not call a tool, end your reply with a line "
                              "'ANSWER: [v1, v2, v3, v4, v5]'.")
                yield PhenoCalTask(
                    PhenoCalData(idx=i, prompt=prompt, system_prompt=system, true_rate=float(d["ground_truth"]["rate"]),
                                 disease=m["disease_icd10_3char"], horizon=int(m["horizon_years"]),
                                 n=int(m.get("n", 0)), version=c.version, variant=c.variant,
                                 gp_linked="hospital records only" not in d["prompt"],
                                 age_band=mm.group(1) if mm else "", sex=sex),
                    c.task)
