"""CohortWorld actions, exposed to the harness as MCP tools.

Written against the verifiers v1 API (docs/v1/tasksets.md): a `vf.Toolset` bundles `@vf.tool`
methods the same way a `vf.Taskset` bundles `vf.Task`. This module HAS been executed against the
real library, so its signatures are verified by execution, not a contract left to check.

Action-space degrees of freedom are expressed in the TOOL SIGNATURES, not in prose. verifiers
registers each `@vf.tool` with FastMCP, which builds its JSON Schema from the type annotations
(`mcp.add_tool` -> `func_metadata` -> `arg_model.model_json_schema`), so a `Literal` becomes a JSON
Schema `enum` the model sees as the allowed values. The axes an agent may edit, the zoom directions,
and the horizons are therefore constrained at the schema level rather than by asking politely in the
prompt. Codes stay `str`: the candidate list runs to hundreds per axis and differs per cohort, so it
travels in the state's `action_space` block, where a miss is a legal move that costs budget.

Budget: every tool here increments `state.spent`, whether or not it hits. A failed `select_cohort`
still charges, which is what makes probing for an upper bound a real strategy instead of a free
oracle.

The legal candidate codes are NOT a tool. They are part of the state -- they are the degrees of
freedom the action space has at this point, not something to be discovered by spending a turn. They
are handed to the agent in the initial observation (see `Lattice.state_of`), which also keeps them
out of the trace as a repeated tool result.
"""
import copy, json
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator
import verifiers.v1 as vf                                    # noqa: F401  (contract under test)

from .lattice import PANEL, Lattice, _shared
from .reward import BUDGET, parse_values


def _digits_to_int(v):
    """`horizon: Literal[...] | None` has an anyOf schema with no top-level `type`, so vLLM's
    qwen3_xml parser cannot tell it is an integer and passes "10"; pydantic's Literal check then rejected
    every explicit horizon (literal_error in 138/142 base arm-B rows of a measured run without this coercion).
    Coerce a digit string before validation. BeforeValidator leaves the JSON schema -- the tool block the
    model reads -- byte-identical; out-of-set values ("7") are still rejected by the Literal."""
    return int(v) if isinstance(v, str) and v.strip().isdigit() else v


class CohortWorldToolsetConfig(vf.ToolsetConfig):
    """Task-scoped config: one tool server per rollout.

    The official guidance (verifiers `skills/create-environments`) pairs the scope with the config
    class -- `Task.toolsets` takes a `vf.ToolsetConfig`, `Taskset.toolsets` takes a
    `vf.SharedToolsetConfig`. This was a `SharedToolsetConfig` constructed from `Task.toolsets`, and
    the mismatch killed the env server before any rollout began:
    `AttributeError: 'CohortWorldToolsetConfig' object has no attribute 'colocated'`.

    Per-rollout scope is what this environment needs: each question places the agent in a cohort of
    its own age band and sex, and a server shared across the whole eval cannot hold per-question
    position.
    """
    version: str = "C2"     # C1 | C2 | default. A STARTING value only: `setup_task` rebinds the
                            # lattice per task from the task data, because `Task.toolsets(config)`
                            # is handed a TaskConfig that does not carry `version`. Left at "C1" it
                            # showed up in every resolved config as `tools.version: "C1"` next to
                            # `taskset.version: "C2"`, which reads like the bug that was just fixed.
    budget: int = BUDGET
    """A STARTING value, like `version`. `setup_task` rebinds it from the task data.

    Setting `tools.budget` in a TOML used to change the resolved-config printout and nothing else:
    `Task.toolsets(config)` is handed a TaskConfig, which carries no `tools`, so the copy made there
    never arrived -- and meanwhile four hardcoded `26`s did the real work. `reward.BUDGET` is the one
    value now, and `_CWData.budget` is how it reaches this object.
    """


class CohortWorldToolset(vf.Toolset[CohortWorldToolsetConfig]):
    TOOL_PREFIX = "cw"

    def __init__(self, config: CohortWorldToolsetConfig):
        self.asked_disease: str | None = None
        self.asked_horizon: int | None = None
        super().__init__(config)
        # MEMOIZED, because unmemoized this line is 6.4 s of blocked event loop per rollout.
        #
        # `verifiers/v1/rollout.py:240` builds the toolsets SYNCHRONOUSLY inside the env-server worker's
        # asyncio loop -- `toolsets = self.task.toolsets(self.task.config)` -- and the worker then ships
        # only `config.model_dump_json()` to the tool-server subprocess, which builds its own. So a bare
        # `Lattice(config.version)` here parsed 320 MB of states TWICE per rollout, and the first of the
        # two was pure waste that also stalled every other episode multiplexed onto that worker.
        #
        # Measured: `_toolsets()` called exactly as
        # rollout.py calls it cost wall=6.39/6.40/5.95 s, cpu=the same, maxrss 2.57 GB, `memoized=False`
        # on every call. The 8 pool workers sat at 88.0-88.8% CPU sustained, 89% loop occupancy, 18.8
        # core-seconds of worker CPU per tool server. 9 loops / 18.8 core-s = 25.5 rollouts/min predicted;
        # 25.5/min spawned and 22.3/min completed were observed -- throughput WAS the loop CPU budget.
        # Downstream: vLLM idle in 230 of 300 one-second samples, `num_requests_waiting` 0 in every
        # sample, KV cache 0.7% at its peak. The GPUs were not slow, they were unfed.
        #
        # `_shared` is the module's own answer to this and was already used by `setup_task` for the
        # cross-version case (line ~108); the constructor simply never used it. The shallow copy is the
        # discipline `Lattice.masked` documents: it gives this episode its own `masked` and `current`
        # while sharing the states dict, the edge index and the caches, which are read-only. Copying
        # unconditionally here (rather than only under a mask, as `setup_task` does) is what makes the
        # sharing safe for T1/T2/PhenoCal too, whose `mask` is "" and which would otherwise navigate the
        # shared object -- today harmless because a tool server serves one rollout, a real cross-episode
        # bug the moment `toolsets` moves to the Taskset and one server is reused across rollouts.
        #
        # Behaviour on a missing panel is unchanged: `_shared` caches None for a version it cannot load,
        # and `Lattice(...)` then raises FileNotFoundError exactly as this line always did.
        self.lat = copy.copy(_shared(config.version) or Lattice(config.version))
        self.spent = 0
        self.budget = int(config.budget)

    async def setup_task(self, task) -> None:
        """Place the agent in a cohort matching this question's age and sex, before it acts.

        Called by the MCP server at the start of each task. Doing this in `Task.setup` did not work:
        the task object has no handle on the toolset, so the hook ran and changed nothing, and the
        agent kept starting wherever the state dictionary happened to begin -- a question about
        40-44 year old men answered from inside a 65-69 year old women's cohort. Every navigation
        missed: two consecutive rollouts logged 13 and 11 attempted moves with zero hits.
        """
        d = getattr(task, "data", task)
        get = (lambda k: d.get(k, "")) if isinstance(d, dict) else (lambda k: getattr(d, k, ""))
        band, sex, ver = get("age_band"), get("sex"), get("version")
        # Bind the lattice to the version this question's codes live in. The version cannot be
        # taken from the toolset config: `Task.toolsets(config)` is handed a TaskConfig, not the
        # TasksetConfig that carries `version`, so the copy made there never arrived and every
        # rollout walked the C1 default. A C2 question then asked for a two-character code the C1
        # disease table does not contain -- `no disease matching 'injury / poisoning [T8]' in the
        # C1 disease table` -- and the agent burned its whole budget hunting a cohort that could
        # not exist. This hook fires per task and does carry the data, so it is where version belongs.
        # ...and it RAISES when that version's lattice is not on disk, rather than keeping the one it
        # already had. `_shared` returns None for a missing panel file, so `if lat is not None`
        # silently left a `default` item walking the C2 lattice for its whole episode. Measured
        # by hiding `CohortWorld_default_*_states.jsonl`: `RateTaskset` still served its
        # 151 `default` items, `setup_task` bound C2 to every one of them, and the rollout scored
        # reward 0.9425 at report_level 2 -- an ordinary-looking number from an agent that navigated
        # the wrong lattice, with no exception, no warning and no metric saying so.
        #
        # This is the ROLLOUT half of the hole `reward.dist` / `entry_cohort` closed at scoring time,
        # and it hides better than that half did: T1 still raises through `entry_cohort`,
        # but T3's reward never touches the lattice and so cannot notice. A PARTIAL panel is the only
        # way in -- a wholly absent one raises in `Lattice.__init__` -- which is why it survived, and
        # why docs/ENVIRONMENT_SPEC.md could say "a missing panel is now loud" while this line kept a
        # half-missing one quiet.
        if ver and ver != self.lat.version:
            lat = _shared(ver)
            if lat is None:
                raise FileNotFoundError(
                    f"this item is version {ver!r} and that lattice is not in {PANEL}. Refusing to "
                    f"run it on the {self.lat.version!r} lattice: the agent would navigate cohorts "
                    f"the question is not about and the reward would come out looking ordinary")
            # COPY, like line 97 and line 186 do. `_shared` returns ONE Lattice per version, cached
            # across every rollout in this worker, and `move`/`zoom` write `self.lat.current`. Binding
            # it uncopied makes one rollout's navigation visible to every other rollout on the same
            # version in the same worker. T3 happens to escape because its answer mask always forces
            # the `copy.copy` at line 186, but T1, T2 and PhenoCal have `mask == ""` and do not -- so
            # the bug was latent, not absent, and it is a shared-mutable-state bug, which is the kind
            # that produces an ordinary-looking number rather than an exception.
            self.lat = copy.copy(lat)
        self.budget = int(get("budget") or self.budget)
        # The cell this episode is ASKED about, so `observe()` can default to it. The agent is told the
        # disease in words ("blood/neoplasm") because the demonstrations never show codes -- that is the
        # design -- and `observe` wants the code, so without this default the two halves cannot meet.
        # Measured on the form-A smoke: of the cw_observe calls that followed a successful move, three of
        # four failed with `no disease matching '<name>' in the <version> disease table`, and the one
        # that succeeded was the only rollout in ~300 to produce a usable prediction.
        self.asked_disease = str(get("disease") or "") or None
        _h = get("horizon")
        self.asked_horizon = int(_h) if _h not in ("", None) else None
        # WITHHOLD THE ANSWER'S OWN RATES, for this episode only.
        #
        # Every internal T3 item IS a lattice cohort -- the item carries `target_cohort` and its
        # `answer_rate` is that cohort's published rate -- and all 18,593 items' targets are still
        # served, so `cw_observe` on that one state returned the label -- 500/500 probed train items
        # returned their own answer exactly (`gen/leak_audit.py --no-mask`). Measured on one run's 2,597
        # training rollouts: 33 (1.27%) took a published reading while standing on the target, worth
        # +0.43 mean reward. The one-move prompt was the only thing limiting it, which is a habit,
        # not a safeguard.
        #
        # ONLY the rates are withheld. The state stays advertised by `cw_look` and reachable by
        # `cw_move`, because standing there is the point: the T3 entry is the item's own small,
        # specific cohort, and the intended move is to RETREAT to a larger reference cohort the agent
        # can read and correct back by the difference -- which is the clinician's order of
        # operations, and the reverse of copying a 10-49 cohort's reading.
        #
        # `mask_cohort`, not `target_cohort`: `LocateData` carries a `target_cohort` that is the
        # answer the agent must WALK TO, so masking every field of that name would change T1's
        # environment too. Only `RateData` answers to `mask_cohort`; PhenoCal and T2 return "" and
        # nothing at all happens here for them.
        #
        # THE SHARED OBJECT IS NOT MUTATED. `_shared(ver)` caches ONE Lattice per version per
        # process and hands the same object to every rollout that follows, so `lat.masked = ...` on
        # it would carry this item's mask into the next item's episode -- a mask that is invisible,
        # is not in the trace, and silently withholds an unrelated cohort's rates. A shallow copy
        # gets this episode its own `masked` (and its own `current`) while sharing the states dict,
        # the edge index and the caches, which are read-only here: the whole cost is copying a
        # ten-key `__dict__`. Test: `test_a_second_task_in_the_same_process_masks_only_its_own`.
        mask = str(get("mask_cohort") or "")
        if mask:
            if mask not in self.lat.states:
                raise ValueError(
                    f"target_cohort {mask!r} is not a released cohort in {self.lat.version}; the "
                    f"item set and the lattice disagree, and the answer would stay readable")
            self.lat = copy.copy(self.lat)
            self.lat.masked = mask
        # A named starting cohort wins over the demographic entry. T2 asks about ONE edge, so its
        # agent has to begin at that edge's tail; T1/T3 ask about a person, so theirs begins at the
        # most generic cohort of the right age and sex. T2 used to carry neither, and all 625 of its
        # items therefore started at `next(iter(states))` -- the same arbitrary cohort for every
        # question, while the prompt said "stand on a group without that feature".
        #
        # T3 NOW STARTS ON THE ITEM'S OWN COHORT, which is what `mask` is. The old entry
        # was `enter(age_band, sex, prefer_coarsest=True)` -- the largest, most reliable cohort of
        # the right demographics -- and from there every available move SPECIALISES, so the chain the
        # environment could express was "stand somewhere reliable, walk into a 10-49 cohort, copy its
        # reading". 47.7% of committed values were literally 0 under it and no one-move policy beat
        # not moving. The order the design wants is the clinician's: stand on the patient's own
        # subgroup, see that it is too small to trust (its rate is withheld, so this is not a
        # judgement call), retreat along a `drop` move to a reference class big enough to believe,
        # and correct back by the difference the history explains.
        #
        # Only T3 moves. `mask` is "" for T1, T2 and the PhenoCal judge, so all three keep the entry
        # they have always had -- the judge's baselines stay comparable on this axis.
        start = get("start_cohort") or mask
        if start:
            if not self.lat.goto(start):
                raise ValueError(f"start_cohort {start!r} is not a released cohort in "
                                 f"{self.lat.version}; the item set and the lattice disagree")
        elif band:
            self.lat.enter(band, sex)

    def _budget(self, out):
        """Every tool result carries the remaining budget.

        `_spent` charges for every call, `observe` included, and the agent was never told the count.
        Over 120 rollouts it burned a mean 18.77 of its 20 calls and **109 of 120 never answered at
        all** -- 110 hit the cap exactly. An agent cannot budget what it cannot see, and a rollout
        that is cut off mid-deliberation scores 0 for the environment's silence rather than for its
        own judgement.
        """
        if not isinstance(out, dict):
            return out
        left = max(0, self.budget - self.spent)
        out = dict(out)
        out["moves_left"] = left
        if left <= 3:
            out["warning"] = (f"{left} moves left. State your answer now with commit -- your most "
                              f"recent reading stands as your answer until you replace it.")
        return out

    def _exhausted(self) -> dict[str, Any] | None:
        """Refuse a move once the budget is spent, instead of ending the episode unanswered.

        The `out_of_budget` stop used to fire the instant `_spent` reached BUDGET, terminating the
        rollout before the agent could say anything. On one measured run that was **11 of the first 26 rollouts
        scoring a flat worst-case 1.0**, while the 15 that did answer scored crps 0.0729 -- a skill
        of +0.325 over the always-zero floor. The flat 1.0 is not an estimate the agent made; it is
        the harness refusing to let it make one, and averaging the two hides the signal entirely.
        Refusing further NAVIGATION while leaving turns to answer measures estimation, not patience.
        """
        if self.spent < self.budget:
            return None
        return {"refused": True, "moves_left": 0,
                "warning": "Budget exhausted -- no further moves will be served. State your answer "
                           "now: call commit, or output your ANSWER line. Your most recent reading "
                           "stands as your answer until you replace it."}

    @vf.tool
    async def look(self) -> dict[str, Any]:
        """Where you are and every move this cohort accepts. Costs 1.

        There was no way to see the action space without moving, and the legal list shipped only on
        a successful landing -- so the agent had to guess a code correctly before it could learn
        which codes existed. It took a median of **8 tool calls** to land the first time, and **30%
        of rollouts never landed at all**.
        """
        if (refused := self._exhausted()) is not None:
            return refused

        self.spent += 1
        return self._budget({"hit": True, **self.lat.state_of(self.lat.current)})

    @vf.tool
    async def move(self, move_id: str) -> dict[str, Any]:
        """Take one of the moves this cohort listed, by its `id`. Costs 1.

        Pass an `id` exactly as `look` or the last result printed it. You cannot compose a move: a
        code that is not in the list has no cohort behind it, and composing was what produced
        `{"hit": false, "bound": "n<10"}` on 40.4% of all tool calls.

        This replaces `add`, `switch` and `zoom`, which between them could not express `age_band`
        shifts (11.9% of the lattice's edges), `sex` switches, or `icd` add/switch -- 18.9% of every
        legal move had no tool at all, including the most direct way to buy generality, widening the
        age band.
        """
        if (refused := self._exhausted()) is not None:
            return refused

        self.spent += 1
        return self._budget(self.lat.move(move_id))

    @vf.tool
    async def observe(self, disease: str | None = None,
                      horizon: Annotated[Literal[1, 2, 3, 5, 10] | None,
                                         BeforeValidator(_digits_to_int)] = None) -> dict[str, Any]:
        """Read one cell of the current cohort's rate matrix. Costs 1 per (disease, horizon).

        Called with no arguments it reads THE CELL THIS QUESTION ASKS ABOUT -- the disease and horizon
        in the question -- which is what an episode almost always wants. Pass them only to read a
        different cell. The question names the disease in words and this tool takes a code, so
        defaulting is not a convenience: it is the only way an agent that was told "blood/neoplasm" can
        read the rate for `D0` without being handed the coding system.

        Resolves to exactly one of four outcomes:
          NO_FOLLOWUP_WINDOW  fewer than ten rows of this cohort sit `horizon` years before the end
                              of records -- a calendar fact, identical for every disease at that
                              horizon (31.8% of cells; 17.8% at H=1 rising to 87.0% at H=10)
          TOO_FEW_AT_RISK     enough rows, but too few are still free of this disease: the cohort
                              largely has it already (11.3%) -- disease-specific and informative
          rate = 0.0          zero events, a real measurement (38.4%)
          rate > 0.0          the rate, exact; `n` only ever banded (18.5%)
        """
        if (refused := self._exhausted()) is not None:
            return refused

        disease = disease if disease is not None else self.asked_disease
        horizon = horizon if horizon is not None else self.asked_horizon
        if disease is None or horizon is None:
            # A task with no asked cell has nothing to read, and until now this returned an error. On
            # T1 that made the tool a dead end for the one policy most likely to call it: CS-GRPO
            # trains on T3, where an argumentless observe reads the asked cell, so the trained policy
            # emits it by habit -- 19.2 times an episode on T1, 49% of them here, against 3.8 and 8%
            # for an untrained agent. Saying so in the ask did not change the rate by one point,
            # because the habit is a trained action and not a belief a sentence can correct.
            #
            # So the call answers the question it is actually being used to ask, which is "where do I
            # stand". It returns the cohort the agent is on, exactly as `look` reports it minus the
            # action space, and it costs one like every other call. It reveals nothing new: `look`
            # already prints this block, and no rate of any cell is in it.
            self.spent += 1
            st = dict(self.lat.state_of(self.lat.current))
            st.pop("action_space", None)
            return self._budget({"hit": True, "note": "this task asks for a cohort, not a rate, so "
                                 "there is no cell to read; this is where you stand", **st})
        self.spent += 1
        return self._budget(self.lat.observe(disease, int(horizon), prefer=self.asked_disease))

    @vf.tool
    async def commit(self, rates: list[float] | str | None = None) -> dict[str, Any]:
        """Finish. Ends the episode.

        For a rate question, pass `rates`: at least five plausible values for the rate, each a
        fraction between 0 and 1 (write 2% as 0.02). The spread across them is your uncertainty.

        For a question that asks WHICH cohort, pass nothing -- the cohort you are standing on when
        you call this is your answer. There is no way to name a cohort directly, by design: a cohort
        is reached by changing a history axis, the way a room is reached by walking into it.

        `rates` accepts a STRING as well as a list, and the body -- not FastMCP -- reads the numbers
        out of it. FastMCP builds this tool's JSON Schema from the annotation and rejects a mismatched
        argument BEFORE the body runs, so a `list[float]`-only signature turned a complete, well-formed
        answer into an error message containing neither the values nor the word "committed": the
        report was lost, the episode was charged a budget slot for the failure (a successful commit is
        free), and 3 of 360 archived rollouts never recovered and were scored the flat worst case with
        their five values sitting in plain sight in the trace. Real examples, verbatim from the runs:
        `input_value='[0.001, 0.0015, 0.0025, 0.004, 0. ]'` and `input_value='[0., .6%, .8%, .8%, .8%]'`.
        Whether those values are USABLE is a separate question, answered once by `reward.well_formed`
        -- the tool's job is to get them into the trace, not to judge them.
        """
        st = self.lat.states.get(self.lat.current, {})
        return {"committed": True, "cohort_id": self.lat.current,
                "cohort": st.get("prompt_body", ""),
                "rates": parse_values(rates) if rates is not None else None}


if __name__ == "__main__":
    # verifiers launches a toolset with `python -m <module>` and waits for it to report the port it
    # serves MCP on; without this the module imported cleanly, exited, and the rollout died with
    # "server did not report its port".
    CohortWorldToolset.run()
