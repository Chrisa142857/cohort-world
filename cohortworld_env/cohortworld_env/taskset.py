"""Shared trace-reading helpers, and the compatibility surface older callers still import.

**There is no Taskset in this module any more.** It used to hold three -- `RateTaskset`,
`LocateTaskset`, `LowerTaskset` -- against the previous task design (`t1_rate` / `t2_compare` /
`t3_lower`, of which `t3_lower` was deleted). They read an old `tasks/` directory
from before the lattice was rebuilt, so 88% of the cohort keys they referenced no
longer resolved and their prompts rendered as `(cohort C1:... not found)`. They had been marked
SUPERSEDED for days while still parsing, still importing, and still shadowing the live `LocateData` /
`LocateTask` names that `internal.py` defines -- 220 lines of dead code carrying two live-looking
class names (N15). Deleted rather than annotated a third time.

The live tasksets are:

    cohortworld_env.internal    RateTaskset / LocateTaskset / DeltaTaskset   (T3 / T1 / T2)
    cohortworld_phenocal        PhenoCalTaskset                              (external transfer)

`_is_commit` is gone for the same reason. It had no caller anywhere in the repo (its own `def` was
the only occurrence, and there is no star import in the project), it was left through the
freeze so that "validated bytes == frozen bytes" would hold, and it decided commit-ness by testing
for the literal `'"committed"'` anywhere in a tool message -- which is NOT how the live path decides
(`committed_this_turn` reads the assistant's tool CALL, because the result is still in
`Session.prepared_tool_results` at stop time). Unreachable today, and a bug the moment anyone revived
it as "the commit detector". Dead code inside a frozen surface is a change nobody can test for.

The file name stays `taskset.py` because modules outside this package import from it by that path,
and a rename would be churn with no reader benefit.

What lives here reads a TRACE; nothing here scores one. Scoring is `cohortworld_env.reward`, which
is the pinned contract (`REWARD_SPEC_VERSION`). The split is deliberate: the trace helpers are the
same for every task and the scoring rules are versioned.
"""
from __future__ import annotations

import verifiers.v1 as vf                                    # noqa: F401  (contract under test)

from .reward import (BUDGET, GRACE, _ID_RE, _NUM_RE, answer_line, commit_payload,  # noqa: F401
                     crps, messages, parse_report, tool_results)

# Compatibility aliases. `gen/direct_ask.py` and `cohortworld_patient` import these names; pointing
# them at `reward` IS D2's fix -- there is now one extraction rule in the project and every caller
# gets it, instead of the four disagreeing copies the audit found (N11/N14).
_answer_line = answer_line
_parse_distribution = parse_report


def _crps(samples, truth: float) -> float:
    """`1 - CRPS`, the orientation the old reward used (higher is better).

    RAISES on an empty report instead of returning a sentinel. The sentinel was `NO_ANSWER = -1.0`,
    and it is gone: no episode is unanswered any more, because `reward.rate_report` and its siblings
    always produce one. A caller arriving here with nothing has bypassed the ladder and should hear
    about it rather than have a constant averaged into a headline -- that constant was worth -0.09
    CRPS on the ISSUED metric and reversed the environment-vs-direct-asking comparison.
    """
    return 1.0 - crps(samples, truth)


def _commit_result(trace) -> dict:
    """What the agent committed, read off the environment's own record of the `commit` call.

    The agent never types an id: a cohort is somewhere you walk to, so the answer to a which-cohort
    question is where you are standing when you commit. Reading the tool result rather than the reply
    text also removes the whole class of parsing failures around a model restating ids while it
    reasons.
    """
    return commit_payload(trace) or {}


def committed_this_turn(trace) -> bool:
    """Whether the agent's most recent turn called `commit` -- the stop signal that actually fires.

    Reading the COMMIT'S RESULT to decide whether to stop cannot work: at `@stop` time that result is
    still in `Session.prepared_tool_results` and is not yet a trace node (verifiers commits a tool
    result only when the next model request arrives), so the stop was a turn late and, since the
    model usually stops talking straight after committing, it simply never fired. Every arm therefore
    ran past its own answer -- all 111 committed rollouts of one run spent one more full generation,
    only 16 were labelled `stop=committed`, and 1-9 per arm then died at `context_length`. That extra
    turn is also where the model reconsiders ("Wait, I committed too quickly. Let me reconsider."),
    and its revision was scored as nothing, because the commit had already been recorded.

    The assistant message that MADE the call is a committed node by then, so read that instead.
    """
    for m in reversed(list(getattr(trace, "messages", None) or [])):
        get = (lambda k: m.get(k)) if isinstance(m, dict) else (lambda k: getattr(m, k, None))
        if get("role") != "assistant":
            continue
        for call in (get("tool_calls") or []):
            name = getattr(call, "name", None) or \
                (call.get("name") if isinstance(call, dict) else "")
            if "commit" in (name or ""):
                return True
        return False
    return False


def _spent(trace) -> int:
    """Tool calls made so far, the commit excluded.

    Counted off the trace, not the toolset instance: a toolset can be shared across a worker's
    rollouts and its own counter would mix them.

    The commit does not cost budget. It used to, and the budget stop counts off the trace, so a
    commit made on the last allowed call pushed the count to BUDGET, `out_of_budget` fired on that
    same step, the episode was terminated before the commit's tool result was appended, and the
    answer was discarded. Measured on pc13: **44 of 120 rollouts called `cw_commit` with a
    well-formed payload, received no reply, ran on to 20/20 and took the flat worst-case loss** --
    more than a third of the run, every one of them an answer the agent had actually produced.
    Ending the episode is what commit is FOR; charging for the privilege and then discarding it is
    the worst of both.
    """
    tool_msgs = sum(1 for role, _n, _c in messages(trace) if role == "tool")
    commits = sum(1 for _n, d in tool_results(trace) if d.get("committed") is True)
    return max(0, tool_msgs - commits)
