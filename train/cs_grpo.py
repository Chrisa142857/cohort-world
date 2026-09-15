"""CS-GRPO: cross-sample group relative policy optimization.

GRPO compares N rollouts of ONE problem and credits each by its reward minus the group mean. That works
when the reward is a property of a single rollout. AUROC and calibration slope are not: they are
properties of a POPULATION of patients, and a single patient's prediction has no AUROC. Scoring one
rollout at a time therefore cannot optimise the metric the benchmark is judged on.

CS-GRPO samples a matrix instead: M problems x N rollouts each. It then forms POPULATIONS by taking one
rollout per problem, scores each population with the frozen judge's own AUROC and slope, and gives each
rollout the mean score of the populations it took part in. That mean is a real estimate of "how much does
this rollout help the population metric", and it is comparable across the N rollouts of its own problem,
which is what the GRPO baseline then subtracts.

Two rewards, two token spans, because they are claims about different parts of the answer:

  * FORMAT (per rollout, from ``demo_quality.cohort_think_format_reward``) -- the same five requirements
    the demonstrations were built to satisfy, so the cold start and the RL objective agree on the shape.
    Assigned to EVERY sampled token: the whole answer is what has the shape.
  * POPULATION METRIC (per rollout, via the populations it joined) -- assigned ONLY to the tokens of the
    number the answer commits to. The reasoning is not what moves an AUROC; the number is. Crediting the
    prose with the metric would reward whatever wording happened to accompany a lucky matrix.

Both are centred within the rollout's own problem, so the baseline stays group-relative.

No barrier, and why. ``Algorithm.score_group`` is awaited on the orchestrator's own main loop
(``orchestrator.py:573`` -> ``train_sink.add``), so a group that waits there for its siblings blocks the
only loop that could deliver them. The first attempt did exactly that and sat through its whole timeout
to score a 1-problem "matrix". A population needs OTHER patients, not other patients finished at the same
instant, so the other members come from a rolling context of the most recently scored problems. Credit is
assigned only to the rollouts of the group being scored; the context supplies the population the metric is
measured over, the way a comparison set does for any AUROC. The context is at most one step old, so its
members are near-on-policy, and nothing about them is trained on twice.
"""

from __future__ import annotations

import math
import importlib.util
import os
import re


def _cw_repo():
    """The CohortWorld checkout the self-checks import gen/binary_metrics.py from. This file is
    installed inside prime-rl's tree, so the checkout cannot be derived from __file__."""
    repo = os.environ.get("CW_REPO")
    if not repo:
        raise RuntimeError("set CW_REPO to the CohortWorld checkout (it provides gen/binary_metrics.py)")
    return repo
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

import torch
import verifiers.v1 as vf

from prime_rl.configs.algorithm import CSGRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.echo import EchoAlgorithm
from prime_rl.orchestrator.algo.routing import assign_advantages
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


# --------------------------------------------------------------------------- the frozen definitions
def _load(path: str, name: str):
    """Import a module by file path. The AUROC, the slope and the format rule are DEFINED in the
    benchmark's own repository; importing them keeps one implementation instead of a second copy here
    that drifts from the judge the results are quoted against."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"cannot import {name} from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def population_schedule(num_problems: int, num_samples: int) -> list[list[int]]:
    """Which rollout each problem contributes to each population.

    ``schedule[p][m] = j`` means population ``p`` takes problem ``m``'s rollout ``j``. The design is
    ``j = (a + m * b) mod N`` over all ``(a, b)``, which gives N^2 populations in which every rollout
    appears exactly N times -- balanced, so no rollout's mean is estimated from more populations than
    another's, and reproducible, so a rerun credits the same rollouts the same way. Enumerating all N^M
    populations instead is the exact quantity and is not computable past a handful of problems."""
    n = max(int(num_samples), 1)
    return [[(a + m * b) % n for m in range(num_problems)] for a in range(n) for b in range(n)]


def population_metric(preds, truths, thresholds, *, auroc_fn, w_auroc: float, w_slope: float,
                      slope_target: float = 1.0) -> float | None:
    """Score one population the way the frozen judge scores a leaderboard row.

    AUROC is over the binarised question the benchmark asks -- is this stratum's rate above its own
    (disease, horizon) base rate -- with the prediction relative to that same threshold as the score.
    The slope is ``polyfit(truth, pred, 1)[0]``, and is rewarded for being near 1: a slope of 3 is as
    miscalibrated as a slope of 1/3, so the reward is on ``-|log slope|`` rather than on the slope."""
    import numpy as np

    keep = [i for i, t in enumerate(thresholds) if t is not None and t > 0]
    if len(keep) < 3:
        return None
    p = np.array([preds[i] for i in keep], dtype=float)
    t = np.array([truths[i] for i in keep], dtype=float)
    f = np.array([thresholds[i] for i in keep], dtype=float)
    labels = (t > f).astype(int).tolist()
    if len(set(labels)) < 2:            # AUROC is undefined with one class in the population
        return None
    au = auroc_fn((p / f).tolist(), labels)
    if au != au:                        # nan
        return None
    total, weight = w_auroc * float(au), w_auroc
    if w_slope:
        var = float(((t - t.mean()) ** 2).sum())
        if var > 0:
            slope = float(np.polyfit(t, p, 1)[0])
            # a slope of 0 or a negative slope is as far from calibrated as this reward can express
            score = 0.0 if slope <= 0 else float(np.exp(-abs(np.log(slope / slope_target))))
            total += w_slope * score
            weight += w_slope
    return total / weight if weight else None


def contribution_metric(index, preds, truths, thresholds, *, w_auroc: float, w_slope: float,
                        slope_target: float = 1.0):
    """One patient's OWN contribution to the population statistics, not the statistics themselves.

    Why this exists. The population AUROC moves in steps of 1/(n_pos * n_neg) -- about 0.012 in a
    population of 18 -- so one rollout's prediction changes it only when it crosses another patient's
    rank. Measured on an early run's first 18 groups: 66.7% received NO metric gradient at all, and the
    magnitude fell as the context grew (0.0748 at 5-9 context problems, 0.0266 at 10-14, 0.0214 at
    15-19). A reward that a rollout cannot move is not a reward.

    Mann-Whitney decomposes exactly: AUROC is the mean over patients of their own pairwise concordance
    against the opposite-label members. That per-patient term is still a POPULATION quantity -- it does
    not exist without the other patients -- but its steps are 1/n_opposite (about 0.11 in the same
    population), and a rollout's own prediction always moves its own term. Calibration is handled the
    same way: the slope score of the population as it is, against the slope score of the same population
    with this patient's prediction replaced by the population mean. That difference is this patient's
    contribution to calibration, and it is continuous in the prediction.
    """
    import numpy as np

    keep = [i for i, t in enumerate(thresholds) if t is not None and t > 0]
    if index not in keep or len(keep) < 3:
        return None
    p = np.array([preds[i] for i in keep], dtype=float)
    t = np.array([truths[i] for i in keep], dtype=float)
    f = np.array([thresholds[i] for i in keep], dtype=float)
    k = keep.index(index)
    y = (t > f).astype(int)
    s = p / f
    total, weight = 0.0, 0.0
    if w_auroc:
        opp = np.where(y != y[k])[0]
        # NO OPPOSITE LABEL MEANS NO MEASUREMENT, not a smaller one. `weight += w_auroc` below sits
        # inside this branch, so skipping it silently dropped the AUROC half from the NUMERATOR AND THE
        # DENOMINATOR and returned the slope term on a DOUBLE scale -- a different quantity wearing the
        # same name, carrying zero discrimination information, and pooled into the same `values` dict as
        # properly averaged two-term values. Verified by execution: preds
        # [0.12,0.10,0.18,0.25], truths [0.12,0.15,0.20,0.30]; thresholds [0.08]*4 (labels 1,1,1,1) gave
        # 0.6649933065595717, BIT-IDENTICAL to the same call with w_auroc=0, while thresholds
        # [0.08,0.30,0.08,0.08] (labels 1,0,1,1) gave 0.8324966532797858 = (1.0 + 0.6649933)/2.
        #
        # `population_metric` has always refused this case (its own guard: "AUROC is undefined with one
        # class in the population"). The two credit modes disagreed, and only this one was in use.
        # Refusing costs gradient on unmeasurable columns, which is the honest price: the alternative --
        # crediting a fabricated 0.5 concordance -- lets the policy be moved by which labels the success
        # buffer happened to hold rather than by anything it did.
        #
        # Found by a pre-launch adversarial review and reproduced independently before acting.
        if not len(opp):
            return None
        if len(opp):
            better = (s[k] > s[opp]) if y[k] == 1 else (s[k] < s[opp])
            ties = s[k] == s[opp]
            total += w_auroc * float((better.sum() + 0.5 * ties.sum()) / len(opp))
            weight += w_auroc
    if w_slope:
        var = float(((t - t.mean()) ** 2).sum())
        if var > 0:
            def slope_score(pred_vec):
                slope = float(np.polyfit(t, pred_vec, 1)[0])
                return 0.0 if slope <= 0 else float(np.exp(-abs(np.log(slope / slope_target))))
            neutral = p.copy()
            neutral[k] = p.mean()          # this patient contributing nothing of its own
            # centred on 0.5 so the two terms live on the same scale as the concordance above
            total += w_slope * (0.5 + slope_score(p) - slope_score(neutral))
            weight += w_slope
    return total / weight if weight else None


def numeric_token_positions(token_ids: list[int], decode) -> list[bool]:
    """Which sampled tokens carry the number the answer commits to.

    Marks the tokens of the LAST ``Adjustment: ...`` span. Decoding token by token and accumulating the
    text is deliberate: a token can straddle the boundary ("-0" then ".0808"), and a mask built from a
    regex over the joined string would then credit half a number."""
    pieces = [decode([t]) for t in token_ids]
    text = "".join(pieces)
    m = None
    for m in re.finditer(r"Adjustment:\s*[+-]?\d+\.\d+", text):
        pass
    if m is None:
        return [False] * len(token_ids)
    lo, hi, out, at = m.start(), m.end(), [], 0
    for piece in pieces:
        end = at + len(piece)
        out.append(at < hi and end > lo)
        at = end
    return out


_NUM = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?%?")


def commit_rate_token_positions(token_ids: list[int], decode) -> list[bool]:
    """Which sampled tokens carry the numbers of the `rates` argument of a cw_commit call (pred_source=commit).

    The rendering is not assumed: Qwen3.5's qwen3_xml writes `<parameter=rates>[...]</parameter>`, a JSON tool
    call writes `"rates": [...]`. Both put the literal `rates` before the values, so the span starts after the
    LAST `rates` in the node and ends at the first `</parameter>`, `</tool_call>` or `}` after it. Token by
    token, for the same straddling reason as `numeric_token_positions`."""
    pieces = [decode([t]) for t in token_ids]
    text = "".join(pieces)
    k = text.rfind("rates")
    if k < 0:
        return [False] * len(token_ids)
    start, end = k + len("rates"), len(text)
    for stop in ("</parameter>", "</tool_call>", "}"):
        j = text.find(stop, start)
        if j >= 0:
            end = min(end, j)
    spans = [(start + mm.start(), start + mm.end()) for mm in _NUM.finditer(text[start:end])]
    out, at = [], 0
    for piece in pieces:
        e = at + len(piece)
        out.append(any(at < hi and e > lo for lo, hi in spans))
        at = e
    return out


def _calls_commit(node) -> bool:
    for tc in (getattr(getattr(node, "message", None), "tool_calls", None) or []):
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        if name is None:
            fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
            name = (fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)) if fn is not None else None
        if name and "commit" in str(name):
            return True
    return False


# --------------------------------------------------------------------------- the algorithm
def delta_accuracy(reads: list[dict], scale: float) -> tuple[list[float], list[bool]]:
    """(credit, measurable) per cell for the DECLARED difference. One definition, two callers.

    credit = exp(-|declared - (reference - truth)| / scale), in (0, 1]; 0 and False where any of the
    three is missing. The error is computed WITHOUT the clip that `pred` carries: `pred` is a rate and
    cannot be negative, but an overshooting declaration must still be priced as overshooting, and a
    clipped error saturates at `truth` exactly there.

    EXPONENTIAL, not `1 - min(err/scale, 1)`. The linear-with-cap form was written first and its own
    control test killed it: at scale 0.0359 a declaration of 0.0002 against a true difference of 0.05
    (error 1.39 scales) and a wild overshoot of 0.20 (error 4.18 scales) BOTH saturate at exactly 0.0,
    so the term is flat over the whole region the trained policy occupies and supplies no gradient where
    it is needed most. The decay has no flat region, and it is the shape the environment already prices
    declarations with (`reward.delta_accuracy`), so the two cannot disagree about what "close" means.

    Lives at module level so the self-check exercises the SAME arithmetic the trainer uses rather than a
    second copy of the formula -- a second copy is how the two silently disagree.
    """
    acc, has = [], []
    for d in reads:
        dec, ref, tru = d.get("declared"), d.get("reference"), d.get("truth")
        ok = dec is not None and ref is not None and tru is not None and scale > 0
        has.append(bool(ok))
        acc.append(math.exp(-abs(float(dec) - (float(ref) - float(tru))) / scale) if ok else 0.0)
    return acc, has


class CSGRPOAlgorithm(Algorithm):
    """Score each problem's rollouts by the population metrics they produce against a rolling context
    of the most recently finished problems."""

    def __init__(self, config: CSGRPOAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.cfg = config
        repo = config.benchmark_repo or os.environ.get("CW_REPO", "")
        if not repo:
            raise ValueError("cs_grpo needs `benchmark_repo` (or $CW_REPO): the AUROC, the slope and the "
                             "format rule are imported from it rather than restated here")
        self._metrics = _load(os.path.join(repo, "gen", "binary_metrics.py"), "cw_binary_metrics")
        self._quality = _load(os.path.join(repo, "gen", "demo_quality.py"), "cw_demo_quality")
        from cohortworld_env import reward as cw_reward      # the env is already importable at runtime
        self._reward = cw_reward
        # the other patients a population is measured over: one entry per problem, newest last
        self._context: deque[tuple[str, list[dict]]] = deque(maxlen=max(config.num_problems - 1, 1))
        # SPEAR's exploitation half: the usable cells of past steps, newest last. maxlen 0 makes every
        # append a no-op and every read empty, so `success_buffer = 0` needs no branch anywhere below.
        self._successes: deque[dict] = deque(maxlen=max(int(config.success_buffer), 0))
        self._topped = 0
        self._decode = None
        self._warned_cold = False
        self._logger = get_logger()
        # ECHO's observation CE, borrowed rather than reimplemented. `_weight_observations` needs exactly
        # these two attributes; giving it them makes `EchoAlgorithm`'s vetted selection rule available to
        # an algorithm that does not inherit from it. `roles` is the vetted default (tool bodies only).
        self.role_weights = {"tool": float(config.echo_alpha)} if config.echo_alpha else {}
        self.filter_fn = None

    async def score_episode(self, episode: vf.Episode) -> None:
        """Cross-entropy on what the ENVIRONMENT said, when `echo_alpha` is set.

        The population term credits the number the policy commits to; the ladder credits the turn that
        acted. Neither teaches the policy what a legal move id LOOKS like -- and that is not a matter of
        preference: the id is `axis.op.code->version:16-hex` and appears only inside a tool result. All
        220 accepted moves across two earlier runs copied one verbatim out of the context; 0 were composed.
        So the shape of the environment's own replies is the thing to learn, and CE on those tokens is
        prime-rl's documented way to learn it (`docs/algorithms.md:77`, :191).
        """
        if not self.role_weights:
            return
        for trace in episode.traces:
            if not trace.has_error and trace.agent.trainable:
                EchoAlgorithm._weight_observations(self, trace)

    def _read(self, trace: vf.Trace) -> dict:
        text = trace.completion_text if hasattr(trace, "completion_text") else _completion_text(trace)
        # the block of the turn being credited: with `credit_turn = "first"` the number under
        # optimisation is the one the cold start's own turn committed to, not one restated later
        # `action` scores the FIRST block, like `first`, because that is the block it CREDITS.
        # demo_quality.py:59-62 states the invariant -- "the number under optimisation must be the number
        # in the tokens that carry the advantage" -- and adding "action" to a two-way expression silently
        # routed it to "last". Reproduced: with a completion whose turn 1 declares -0.0100 and whose turn 3
        # revises to -0.0900, `_read` returned declared = -0.09 while `_assign_trace` wrote metric_adv on
        # the tokens of -0.0100. The policy was paid for the accuracy of the post-action commit on the
        # tokens of the pre-action guess.
        fmt = self._quality.cohort_think_format(
            text, allow_multiple=True,
            block="first" if self.cfg.credit_turn in ("first", "action") else "last")
        data = getattr(trace.task, "data", None)
        truth = _get(data, "true_rate")
        disease, horizon = _get(data, "disease"), int(_get(data, "horizon") or 0)
        # the judge's own binarising threshold, called rather than restated
        threshold = self._reward.base_rate(self.cfg.rate_source, disease, horizon)
        # WHAT THIS ROLLOUT ACTUALLY READ. `readings` is the environment's record of every published
        # cw_observe, tagged with the cohort the agent was standing in at the time; the last one for the
        # asked (disease, horizon) is the number its commit is built on.
        reference, ref_cohort = None, ""
        target = _get(data, "target_cohort") or ""
        for r in self._reward.readings(trace):
            if r.get("disease") == disease and int(r.get("horizon") or -1) == horizon:
                if r.get("state_id") and r["state_id"] == target:
                    continue          # its own cohort is masked in T3; a reading there is not a move
                reference, ref_cohort = float(r["rate"]), r.get("state_id") or ""
        declared = fmt["declared"]
        # the three rungs, read off the trace rather than inferred from each other: a rollout can call
        # and not move, and can move and not read, and those are different states to pay differently
        results = list(self._reward.tool_results(trace))
        called = bool(results)
        moved = any(d.get("hit") and d.get("state_id") and d["state_id"] != target
                    for _name, d in results)
        # commit = what the agent READ where it moved to, minus the correction it argues for. A rollout
        # that never moved and never read has no prediction -- not a zero one, and not the item's own
        # entry rate handed to it for free.
        pred = None if (declared is None or reference is None) else max(float(reference) - float(declared), 0.0)
        committed_number = None
        if getattr(self.cfg, "pred_source", "read_minus_declared") == "commit":
            # EXPERIMENTAL, negative: on a policy trained 50 steps with the delta term every sampled
            # commit was empty (256/256), so every rollout scored the floor and the metric advantage was
            # exactly zero -- this option cannot teach committing a number to a policy that never commits one.
            # Pay the number the rollout STATES, i.e. what evaluation scores as stated_commit.
            # Under read_minus_declared the commit's arguments never reached the credit, and the policy
            # learned `cw_commit {}`. No commit / empty / unparseable -> the floor report's median.
            payload = self._reward.commit_payload(trace)
            vals = []
            if payload is not None and payload.get("rates") is not None:
                # EVALUATION'S OWN RULE (reward.parse_report = parse_values + well_formed: >= MIN_REPORT values,
                # all in [0, 1], else nothing). A malformed commit is scored as a malformed commit, not rescued.
                vals = [float(v) for v in self._reward.parse_report(payload["rates"])]
            committed_number = bool(vals)
            pred = (sorted(vals)[len(vals) // 2] if len(vals) % 2 else
                    0.5 * (sorted(vals)[len(vals) // 2 - 1] + sorted(vals)[len(vals) // 2])) if vals else \
                   float(sorted(self._reward.FLOOR_REPORT)[len(self._reward.FLOOR_REPORT) // 2])
        return {"format": self._quality.cohort_think_format_reward(text, allow_multiple=True), "pred": pred,
                "committed_number": committed_number,
                "truth": truth, "threshold": threshold, "declared": declared,
                # `reference` is returned, not recomputed from `pred`: `pred` is CLIPPED at 0, so
                # `pred + declared` differs from the reading exactly when the declaration overshoots --
                # which is the case the delta term must be able to see.
                "reference": reference,
                "called": called, "moved": moved,
                "read": reference is not None, "ref_cohort": ref_cohort}

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        if not traces:
            return
        # ONE EPISODE, ONE POPULATION -- when the env put M patients in it. `iter_trainable_traces`
        # yields (episode, trace), so the episodes ARE the columns of the matrix and the patients within
        # an episode are its rows. When every episode carries a single trace this is the historical
        # shape and the rolling-context path below runs unchanged, so archived behaviour does not move.
        cells: dict[int, list] = defaultdict(list)
        for episode, trace in iter_trainable_traces(episodes):
            cells[id(episode)].append((trace, self._read(trace)))
        per_episode = list(cells.values())
        if max((len(c) for c in per_episode), default=0) > 1:
            self._assign_matrix(per_episode)
            return
        rows = [(trace, self._read(trace)) for trace in traces]
        context = [entry for entry in self._context if entry[0] != _problem_key(traces[0])]
        self._assign(rows, context)
        self._context.append((_problem_key(traces[0]), [data for _trace, data in rows]))

    def _assign_matrix(self, per_episode: list[list[tuple[vf.Trace, dict]]]) -> None:
        """The real M x N matrix: every population is one episode, every baseline is one patient.

        The rolling context existed because `score_group` only ever saw N rollouts of ONE problem, so a
        population had to borrow its other patients from problems finished 1-2 steps earlier. With M
        patients inside each episode the comparison set is same-step, same-policy-version, and complete:
        no deque, no cross-batch bookkeeping, no members generated by a policy that no longer exists.

        Two axes, and they are different quantities:
          * POPULATION (down a column) -- the M cells of one episode are scored together, because AUROC
            and calibration slope are properties of a cohort and do not exist for one patient.
          * BASELINE (across a row) -- a cell's advantage is centred against the SAME patient's cells in
            the other episodes. That is GRPO's baseline, still group-relative, and it is what makes the
            credit comparable: two cells of one patient differ only in what the policy did.
        """
        values: dict[int, float] = {}
        scored = 0
        topped = 0
        unmeasurable = 0
        for column in per_episode:
            datas = [d for _t, d in column]
            usable = [i for i, d in enumerate(datas) if d["pred"] is not None]
            if not usable:
                continue
            # TOP UP FROM PAST SUCCESSES. The real cells stay at positions 0..len(usable)-1, which is
            # what lets `contribution_metric(k, ...)` keep pointing at the cell being credited; the
            # borrowed members only ever sit behind them and only ever widen the comparison set.
            extra = self._top_up([datas[i] for i in usable],
                                 self.cfg.min_population - len(usable))
            if len(usable) + len(extra) < self.cfg.min_population:
                continue
            topped += 1 if extra else 0
            # AND the gap must actually have CLOSED. `_top_up` is best-effort: it prefers the missing
            # label but returns whatever the buffer holds, and the count gate above only counts members.
            # Without this, a single-label column reached `contribution_metric` and (before the guard
            # there) was scored on the slope half alone. Skipped and counted, so the cost is visible
            # rather than inferred.
            if self.cfg.auroc_weight and len({_label(d) for d in
                                              [datas[i] for i in usable] + extra} - {None}) < 2:
                unmeasurable += 1
                continue
            preds = [datas[i]["pred"] for i in usable] + [d["pred"] for d in extra]
            truths = [datas[i]["truth"] for i in usable] + [d["truth"] for d in extra]
            thrs = [datas[i]["threshold"] for i in usable] + [d["threshold"] for d in extra]
            for k, i in enumerate(usable):
                if self.cfg.credit == "contribution":
                    v = contribution_metric(k, preds, truths, thrs, w_auroc=self.cfg.auroc_weight,
                                            w_slope=self.cfg.slope_weight,
                                            slope_target=self.cfg.slope_target)
                else:
                    v = population_metric(preds, truths, thrs, auroc_fn=self._metrics.auroc,
                                          w_auroc=self.cfg.auroc_weight, w_slope=self.cfg.slope_weight,
                                          slope_target=self.cfg.slope_target)
                if v is not None:
                    values[id(column[i][0])] = v
                    scored += 1
        by_patient: dict[str, list] = defaultdict(list)
        for column in per_episode:
            for trace, data in column:
                by_patient[_problem_key(trace)].append((trace, data))
        borrowed = (f", {topped} populations topped up from a buffer of {len(self._successes)}"
                    if self._successes.maxlen else "")
        # stored, not just logged: the gate's only observable effect is this count (the metric's own guard
        # already refuses such a column), so a self-check that does not read it cannot constrain the gate.
        # The first version of `_check_single_label_refusal` asserted "no cell was credited", which the
        # metric guard satisfies on its own -- removing the gate left the check GREEN.
        self._unmeasurable = unmeasurable
        if unmeasurable:
            borrowed += f", {unmeasurable} skipped for a single-label population"
        self._logger.info(
            f"cs_grpo[matrix]: {len(per_episode)} populations x "
            f"{max((len(c) for c in per_episode), default=0)} patients, {scored} cells scored, "
            f"{len(by_patient)} baseline rows{borrowed}")
        # AFTER the loop above, never during it: a population must not be measured against its own cells.
        for column in per_episode:
            for _t, d in column:
                if d["pred"] is not None:
                    self._successes.append({k: d[k] for k in ("pred", "truth", "threshold")})
        for rows in by_patient.values():
            self._assign(rows, [], metric_values=values)

    def _top_up(self, real: list[dict], need: int) -> list[dict]:
        """Comparison-set members from past successes: newest first, and close the label gap first.

        Label order matters and is not cosmetic. `contribution_metric` scores a patient by concordance
        against the OPPOSITE-label members of its population, so a column whose real readings all carry
        the same label has no opposite to be concordant with and the AUROC half is undefined however
        many members are added. Closing the gap before filling by recency is what makes a topped-up
        population measurable rather than merely large.
        """
        if need <= 0 or not self._successes:
            return []
        pool = list(reversed(self._successes))
        have = {_label(d) for d in real} - {None}
        out: list[dict] = []
        taken: set[int] = set()
        for d in pool:
            if len(have) >= 2 or len(out) >= need:
                break
            lab = _label(d)
            if lab is not None and lab not in have:
                out.append(d); taken.add(id(d)); have.add(lab)
        for d in pool:
            if len(out) >= need:
                break
            if id(d) not in taken:
                out.append(d); taken.add(id(d))
        return out

    def _assign(self, rows: list[tuple[vf.Trace, dict]], context: list[tuple[str, list[dict]]],
                metric_values: dict[int, float] | None = None) -> None:
        n = len(rows)
        repeats = max(int(self.cfg.populations_per_rollout), 1)
        totals, counts, scored = [0.0] * n, [0] * n, 0
        if metric_values is not None:
            # the matrix path already scored every cell against its own episode's population; this call
            # only has to centre them, so the context machinery is bypassed rather than fed an empty deque
            for j, (trace, _d) in enumerate(rows):
                v = metric_values.get(id(trace))
                if v is not None:
                    totals[j], counts[j] = v, 1
                    scored += 1
        elif len(context) + 1 >= self.cfg.min_population:
            for j in range(n):
                for r in range(repeats):
                    members = [rows[j][1]]          # index 0 is the rollout being credited
                    # one rollout from each context problem, rotated by r so the same rollout is not
                    # the population's representative every time
                    for m, (_key, others) in enumerate(context):
                        members.append(others[(r + m) % len(others)])
                    usable = [d for d in members if d["pred"] is not None]
                    if len(usable) < self.cfg.min_population or members[0]["pred"] is None:
                        continue
                    if self.cfg.credit == "contribution":
                        value = contribution_metric(
                            0, [d["pred"] for d in usable], [d["truth"] for d in usable],
                            [d["threshold"] for d in usable], w_auroc=self.cfg.auroc_weight,
                            w_slope=self.cfg.slope_weight, slope_target=self.cfg.slope_target)
                    else:
                        value = population_metric(
                            [d["pred"] for d in usable], [d["truth"] for d in usable],
                            [d["threshold"] for d in usable], auroc_fn=self._metrics.auroc,
                            w_auroc=self.cfg.auroc_weight, w_slope=self.cfg.slope_weight,
                            slope_target=self.cfg.slope_target)
                    if value is None:
                        continue
                    scored += 1
                    totals[j] += value
                    counts[j] += 1
        elif not self._warned_cold:
            self._warned_cold = True
            self._logger.info(
                f"cs_grpo: the rolling context is still filling ({len(context) + 1} of "
                f"{self.cfg.min_population} problems); these first groups train on the format term only.")

        fmts = torch.tensor([d["format"] for _t, d in rows], dtype=torch.float32)
        fmt_adv = (fmts - fmts.mean()) * self.cfg.format_weight
        rungs = (("called", self.cfg.call_weight), ("moved", self.cfg.move_weight),
                  ("read", self.cfg.read_weight))
        read_adv = torch.zeros_like(fmt_adv)
        for field, weight in rungs:
            if not weight:
                continue
            hit = torch.tensor([1.0 if d.get(field) else 0.0 for _t, d in rows], dtype=torch.float32)
            read_adv = read_adv + (hit - hit.mean()) * weight
        # NOT merged when the ladder has its own destination. Under `credit_turn = "action"` the ladder
        # is a claim about the turn that ACTED, and the format term is a claim about the turn that has
        # the shape; summing them here would put both on turn 1, which is what made the action
        # untrainable (800/800 trainable branches had nonzero advantages confined to turn 1, including
        # all 37 containing a tool call).
        ladder_adv = read_adv
        if self.cfg.credit_turn != "action":
            fmt_adv = fmt_adv + read_adv
            ladder_adv = torch.zeros_like(read_adv)
        metric = torch.tensor([(totals[j] / counts[j]) if counts[j] else float("nan") for j in range(n)],
                              dtype=torch.float32)
        seen = ~torch.isnan(metric)
        metric_adv = torch.zeros_like(metric)
        # Set FIRST, unconditionally. Assigning it only inside the branch below left the previous row's
        # value in place for every row that scored nothing, so the log's `[baseline ...]` label was stale
        # for most rows: step 1 of one run logged `[baseline group]` on 30 of 112 rows while only 4 of them
        # actually used a group baseline. A label that is right 4 times in 30 is worse than no label,
        # because it is the evidence the fix gets judged on.
        self._baseline_src = "none"
        if seen.any():
            # the baseline is the mean over the rollouts of THIS problem that a population could score;
            # a rollout that produced no number is left at zero rather than given the group mean, which
            # would pay it for the others' work
            #
            # ONE SCORED CELL IS ITS OWN MEAN, and that is the SECOND arithmetic wall. Fixing the column
            # (the success buffer) made populations scorable; it cannot make a patient's ROW scorable,
            # because a row's cells live in different columns and a cell is scored only if its own
            # rollout read. Measured on step 1 of an earlier run: 86 baseline rows, `mean |metric adv| 0.0000` in all
            # 86, `metric advantage on 1 rollouts` in 7 and on 0 in 79 -- cells were being scored and the
            # gradient was still identically zero. At p = 2.2%, P(>= 2 of N=8 rollouts of one patient
            # read) = 0.011, so ~2 of 192 rows per step would ever clear a within-row baseline. The wall
            # is not reachable by raising N either: N=32 only gets it to 0.17.
            #
            # So when the row cannot supply a baseline, borrow the GROUP's scored cells -- same step,
            # same policy version, passed in already as `metric_values`. This is weaker than the
            # within-row baseline it replaces, because two patients differ in difficulty and not only in
            # what the policy did. It is defensible here and not in general: `contribution_metric`
            # returns a NORMALISED quantity (a concordance fraction in [0, 1] plus a slope-difference
            # term centred on 0.5), not a raw rate, so cells of different patients are already on one
            # scale. The within-row baseline is still preferred whenever it exists.
            if int(seen.sum()) > 1:
                baseline = float(metric[seen].mean())
                self._baseline_src = "row"
            elif metric_values and len(metric_values) > 1:
                baseline = sum(metric_values.values()) / len(metric_values)
                self._baseline_src = "group"
            else:
                baseline = None
                self._baseline_src = "none"
            if baseline is not None:
                metric_adv[seen] = (metric[seen] - baseline) * self.cfg.metric_weight

        # DELTA ACCURACY, on the declaration itself. The identity that makes this cheap:
        #     |declared - (reference - truth)|  ==  |(reference - declared) - truth|
        # i.e. "how wrong was the difference it argued for" IS "how wrong its prediction would be",
        # computed WITHOUT the clip. `pred` keeps its `max(..., 0)` floor because a rate cannot be
        # negative; the error below deliberately does not, because a clipped error saturates at `truth`
        # exactly when the declaration overshoots, which is the direction this term exists to price.
        # Centred across the row like every other term, so it pays for declaring better than the peers
        # answering the same patient rather than for declaring at all.
        delta_adv = torch.zeros_like(metric_adv)
        self._delta_seen = 0            # set unconditionally: leaving the previous group's value in place
                                        # is the stale-label bug `_baseline_src` above documents
        # READ DEFENSIVELY. A config that predates these fields -- an older run's config object, a
        # lightweight stub, anything constructed before this term existed -- must behave exactly as it
        # did before, not raise. The nine self-checks below construct such a stub, and reading
        # `self.cfg.delta_weight` directly made all eight of them fail with AttributeError, which is the
        # same failure any pre-delta caller would have hit at step 1 of a resumed run.
        delta_weight = float(getattr(self.cfg, "delta_weight", 0.0) or 0.0)
        if delta_weight:
            scale = getattr(self.cfg, "delta_scale", None)
            if scale is None:
                scale = float(self._reward.cal_l2_scale()) / float(self._reward.T3_CAL_L2_SCALE_MULT)
            acc, has = delta_accuracy([d for _t, d in rows], scale)
            m = torch.tensor(has, dtype=torch.bool)
            # CENTRING IS WHAT MAKES THIS TERM VANISH IN PRACTICE, and the smoke measured it: over 32
            # groups the median |delta adv| was 0.0019 against the metric's 0.0829, with
            # corr(within-group declaration spread, |delta adv|) = 0.721. Subtracting the group mean turns
            # an ABSOLUTE accuracy into a RELATIVE one, and the RL policy declares nearly the same number
            # in every rollout of a group (median spread 0.0016, the 256x compression), so the centred
            # values are ~0 no matter how wrong that shared number is. Raising delta_weight scales ~0.
            #   delta_center = True  -> previous behaviour, byte-identical, the default.
            #   delta_center = False -> absolute: `a - 1.0` is <= 0, zero only at a perfect declaration,
            #                          so a whole group agreeing on a wrong number is still penalised.
            # The trade is real and is the reason this is a flag rather than a fix: dropping the baseline
            # removes its variance reduction and adds a constant negative drift on those tokens.
            #   None  -> subtract the GROUP MEAN: previous behaviour, byte-identical, the default.
            #   float -> subtract that FIXED accuracy, making the term absolute.
            # 0.5 rather than 1.0 for the fixed case, and the difference matters: `a - 1.0` is <= 0
            # everywhere, so every declared-number token would carry a negative advantage and IPO would
            # push the model to stop emitting the Adjustment span at all -- training it out of the very
            # behaviour the term exists to shape. With 0.5, accuracy above it earns POSITIVE credit.
            # With a = exp(-err/scale) and scale 0.035877, a = 0.5 at err = 0.0249, so a declaration within
            # ~0.025 of the true delta is rewarded and one further away is charged. The RL-after policy
            # (declares 0.0002 against a true delta of ~0.05) sits at a = 0.25, i.e. a real negative push
            # with a reachable path to positive credit.
            baseline = getattr(self.cfg, "delta_baseline", None)
            # An absolute term needs only ONE scoreable rollout; a relative one needs two to differ.
            if int(m.sum()) >= (2 if baseline is None else 1):
                a = torch.tensor(acc, dtype=torch.float32)
                base = a[m].mean() if baseline is None else float(baseline)
                delta_adv[m] = (a[m] - base) * delta_weight
            self._delta_seen = int(m.sum())
        # The decoder is what finds the declared number's tokens. It used to be built only when the
        # metric was nonzero, which would have silently dropped this term in any ablation that sets
        # metric_weight = 0 and delta_weight > 0.
        decode = (self._decoder()
                  if (float(metric_adv.abs().sum()) > 0 or float(delta_adv.abs().sum()) > 0) else None)
        # the span the ladder is normalised against: the group's mean turn-1 sampled-token count, which is
        # where the ladder used to land and therefore what "the same gradient mass as before" means
        ref_lens = []
        for trace, _d in rows:
            first = next((nd for br in trace.branches if br.trainable for nd in br.nodes if any(nd.mask)), None)
            if first is not None:
                ref_lens.append(sum(first.mask))
        ref_len = (sum(ref_lens) / len(ref_lens)) if ref_lens else 1.0
        self._commit_landed = 0        # pred_source=commit: rollouts whose metric advantage landed on commit tokens
        for j, (trace, _data) in enumerate(rows):
            self._assign_trace(trace, float(fmt_adv[j]), float(ladder_adv[j]), float(metric_adv[j]),
                               decode, ref_len, float(delta_adv[j]))
        declared = [d["pred"] for _t, d in rows if d["pred"] is not None]
        spread = (max(declared) - min(declared)) if len(declared) > 1 else 0.0
        n_read = sum(1 for _t, d in rows if d.get("read"))
        n_call = sum(1 for _t, d in rows if d.get("called"))
        n_move = sum(1 for _t, d in rows if d.get("moved"))
        n_decl = sum(1 for _t, d in rows if d.get("declared") is not None)
        n_commit = sum(1 for _t, d in rows if d.get("committed_number"))
        # VISIBLE, not inferred afterwards. Under `action` the format term is turn 1's only tenant, and
        # an earlier run's own log shows it contributing exactly nothing in 1388 of 1586 groups (87.5%) -- 32/32
        # rollouts declare and score identically on the five format checks. In that state every turn-1
        # token of every rollout that ACTED is assigned 0.0 and pruned, so the cohort-think reasoning the
        # paradigm exists to train receives no gradient at all. It resolves itself once populations score
        # (the metric lands on turn 1's number), so this is a counter to watch, not an error to raise.
        mute = (self.cfg.credit_turn == "action" and float(fmt_adv.abs().sum()) == 0.0
                and not bool(seen.any()))
        self._logger.info(
            f"cs_grpo[{self.cfg.credit}]{' TURN-1 MUTE' if mute else ''}: "
            f"{n} rollouts x {len(context)} context problems, {scored} "
            f"populations scored, metric advantage on {int(seen.sum())} rollouts "
            f"[baseline {getattr(self, '_baseline_src', 'none')}] "
            f"(mean |metric adv| {float(metric_adv.abs().mean()):.4f}, "
            # REPORTED SEPARATELY, never summed into the metric's figure. Delta accuracy and the
            # AUROC/slope the metric optimises can disagree -- that trade is the reason the term was left
            # out originally -- and a single merged number would hide whichever one is losing. The
            # `on N` count is what a smoke run checks: the term can be configured and still land on
            # nothing if no rollout declares, reads and has a measurable truth.
            f"|delta adv| {float(delta_adv.abs().mean()):.4f} on "
            f"{getattr(self, '_delta_seen', 0)} rollouts, "
            f"|fmt adv| {float(fmt_adv.abs().mean()):.4f}, "
            f"|ladder adv| {float(read_adv.abs().mean()):.4f} on "
            f"{'the acting turn' if self.cfg.credit_turn == 'action' else 'turn 1'}); "
            f"{n_decl}/{n} declared, {n_call}/{n} called, {n_move}/{n} moved, "
            f"{n_read}/{n} read a rate elsewhere, "
            + (f"{n_commit}/{n} committed a number (pred_source=commit), "
               f"metric advantage on commit tokens in {getattr(self, '_commit_landed', 0)}/{n}, "
               if getattr(self.cfg, "pred_source", "read_minus_declared") == "commit" else "")
            + f"{len(declared)}/{n} usable, within-group prediction spread {spread:.4f}")

    def _assign_trace(self, trace: vf.Trace, fmt_adv: float, ladder_adv: float,
                      metric_adv: float, decode, ref_len: float = 1.0, delta_adv: float = 0.0) -> None:
        """Put each term on the tokens its claim is about.

        `first` / `all`: unchanged. `ladder_adv` is 0.0 for them -- `_assign` has already folded the
        ladder into `fmt_adv`, so their arithmetic is byte-identical to before this parameter existed.

        `action`: format on turn 1, the ladder on the turn that CARRIES a tool call, the metric on the
        committed number wherever it is. The action node is found by `message.tool_calls`, which is how
        verifiers records it (a real one from an earlier run: node 2 assistant 165 sampled tokens, node 4
        assistant tool_calls=True 15 sampled tokens, node 6 assistant 165 sampled tokens -- the action is
        a 15-token span and the restatement is the 165-token one).

        A rollout that never acted has no action node, and its ladder advantage -- negative, because the
        term is centred within the group -- falls back to turn 1. That fallback is not a detail: drop it
        and the group pays for acting without charging for not acting, which leaves the comparison with
        no baseline.
        """
        # BUILT THE WAY `assign_advantages` CONSUMES IT, deliberately. That function walks
        # `[n for n in trace.nodes if id(n) in trainable and any(n.mask)]` (routing.py:13-14), i.e.
        # trace.nodes order. An earlier version here walked branch-concatenation order and de-duped by
        # first appearance; on a multi-branch trace the two orders differ, the TOTAL token count still
        # matches so routing.py never raises, and every advantage silently lands on the wrong node.
        # Training so far logs `Branches 1.0` so it does not bite yet -- which is exactly why it had to be fixed
        # before it does.
        trainable = {id(n) for b in trace.branches if b.trainable for n in b.nodes}
        sampled = [n for n in trace.nodes if id(n) in trainable and any(n.mask)]
        # EVERY acting node, not the first. Change B's design is that ask 1 buys the move listing with
        # `cw_look` and ask 2 spends it with `cw_move`/`cw_observe`/`cw_commit`, so "the first node with
        # tool_calls" is the LOOK -- the whole ladder would land on a ~2-token look and the `cw_move` that
        # the run exists to train would take literal 0.0 and be pruned. That is the `first`-mode defect moved one
        # turn later, and it is guaranteed rather than incidental: the two changes produce it together.
        action_nodes = {i for i, node in enumerate(sampled)
                        if getattr(getattr(node, "message", None), "tool_calls", None)}
        if self.cfg.credit_turn == "action" and not action_nodes:
            action_nodes = {0}               # never acted: the ladder's charge lands on the only turn there is
        commit_mode = getattr(self.cfg, "pred_source", "read_minus_declared") == "commit"
        commit_node = None
        if commit_mode:
            for i, node in enumerate(sampled):
                if _calls_commit(node):
                    commit_node = i          # the LAST commit, the one evaluation reads
        values = []
        for i, node in enumerate(sampled):
            ids = [t for t, s in zip(node.token_ids, node.mask, strict=True) if s]
            if self.cfg.credit_turn == "first" and i > 0:
                values.extend(0.0 for _ in ids)  # a later turn: no gradient, and pruned from the batch
                continue
            if self.cfg.credit_turn == "action":
                # Eligibility is POSITIONAL, not "did any term happen to be nonzero". The first version
                # of this line fell through to the metric whenever `metric_adv` was nonzero, so a third
                # turn that merely RESTATES the answer collected the population credit on its copy of the
                # number -- training the policy to say the same thing twice, which is the exact objection
                # `credit_turn = "first"` was created to answer. The paradigm trains the cohort-think
                # reasoning to be more accurate, so the metric belongs to the number that reasoning
                # committed to in turn 1, and nowhere else.
                if i != 0 and i not in action_nodes:
                    values.extend(0.0 for _ in ids)   # neither turn 1 nor an action: a restatement
                    continue
                # LENGTH-NEUTRAL. The rl loss is a per-token SUM over ONE global divisor
                # (trainer/rl/loss.py:156, trainer/rl/train.py:325-339), so a rollout's gradient mass is
                # advantage x tokens-it-lands-on, not advantage. Under `first` every rollout's ladder sat
                # on turn 1 and the centred term cancelled across the group -- that cancellation IS the
                # baseline. Moving the caller's POSITIVE ladder onto a ~15-token action node while every
                # non-caller's NEGATIVE ladder stays on a ~165-token turn 1 correlates sign with span by
                # ~11:1. Measured on stub traces shaped like a real trace: sum(advantage x tokens)
                # per group +0.000000 under `first`, -14.531250 under `action`; over 200 groups with
                # variable lengths, +0.043 +/- 5.59 vs -14.59 +/- 1.18. Scaling by ref_len/len restores it.
                share = len(action_nodes)
                base = (fmt_adv if i == 0 else 0.0)
                if i in action_nodes:
                    base += ladder_adv * (ref_len / max(len(ids), 1)) / share
            else:
                base = fmt_adv
            if commit_mode:
                # pred_source=commit: the metric is a claim about the COMMITTED rates, so it lands on
                # their tokens; the delta term is still a claim about turn 1's declared number. An empty or
                # malformed commit has no number tokens, so its (floor) contribution lands on the whole commit
                # turn -- otherwise committing nothing would be free. A rollout that never called cw_commit is
                # charged at its declared number, as before.
                adj = numeric_token_positions(ids, decode) if (i == 0 and decode) else [False] * len(ids)
                rates = [False] * len(ids)
                if i == commit_node and decode:
                    rates = commit_rate_token_positions(ids, decode)
                    if not any(rates):
                        rates = [True] * len(ids)
                    if metric_adv:
                        self._commit_landed = getattr(self, "_commit_landed", 0) + 1
                m_adj = metric_adv if commit_node is None else 0.0
                values.extend(base + ((delta_adv + m_adj) if a else 0.0) + (metric_adv if r else 0.0)
                              for a, r in zip(adj, rates, strict=True))
                continue
            on_number = metric_adv + delta_adv        # both claims are about the declared number
            numeric = numeric_token_positions(ids, decode) if (on_number and decode) else [False] * len(ids)
            values.extend(base + (on_number if is_num else 0.0) for is_num in numeric)
        assign_advantages(trace, values)

    def _decoder(self):
        if self._decode is not None:
            return self._decode
        for owner in (self.clients, getattr(self.clients, "renderer", None)):
            tok = getattr(owner, "tokenizer", None)
            if tok is not None and hasattr(tok, "decode"):
                self._decode = tok.decode
                return self._decode
        name = self.cfg.tokenizer_name or getattr(self.clients, "model_name", None)
        if name:
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(name)
                self._logger.info(f"cs_grpo: loaded the tokenizer for the numeric mask from {name}")
                self._decode = lambda ids: tok.decode(ids, skip_special_tokens=False)
                return self._decode
            except Exception as exc:                     # noqa: BLE001 - reported, never swallowed
                raise RuntimeError(
                    f"cs_grpo could not load a tokenizer from {name!r} to find the declared number's "
                    f"tokens: {exc}. Set `tokenizer_name` to a local path."
                ) from exc
        raise RuntimeError(
            "cs_grpo needs a tokenizer to find the tokens of the declared number: the metric reward is "
            "assigned to those tokens only. Without one it would have to fall back to crediting the whole "
            "answer, which is a different algorithm wearing this one's name."
        )


def _label(cell: dict) -> bool | None:
    """The binary label `contribution_metric` actually uses: `truth > threshold`.

    Not `bool(truth)`. `truth` is an incidence RATE, so `bool` is True for every cell that is not exactly
    zero and a label-gap check written that way can never find a gap. Written that way first; caught by
    reading `contribution_metric`, which computes `y = (t > f)` and keeps only `threshold > 0`.
    """
    t, f = cell.get("truth"), cell.get("threshold")
    if t is None or f is None or f <= 0:
        return None
    return bool(t > f)


def _completion_text(trace: vf.Trace) -> str:
    for attr in ("completion", "text", "output_text"):
        value = getattr(trace, attr, None)
        if isinstance(value, str):
            return value
    parts = []
    for branch in trace.branches:
        for message in getattr(branch, "messages", []) or []:
            if getattr(message, "role", None) == "assistant" and isinstance(getattr(message, "content", None), str):
                parts.append(message.content)
    return "\n".join(parts)


def _problem_key(trace: vf.Trace) -> str:
    data = getattr(trace.task, "data", None)
    for field in ("item_id", "idx", "id"):
        value = getattr(data, field, None) if data is not None else None
        if value is not None:
            return str(value)
    return str(getattr(trace.task, "id", id(trace)))


def _get(data: Any, field: str) -> Any:
    return getattr(data, field, None) if data is not None else None


# --------------------------------------------------------------------------- runnable check
def demo() -> None:
    """The maths, checked without a GPU, a model or prime-rl's runtime.

    Three assertions, each with the thing it would catch:
      1. the schedule is balanced -- every rollout joins exactly N of the N^2 populations, so no
         rollout's mean is averaged over more evidence than its neighbour's;
      2. a population of confident, correctly ordered predictions scores above a shuffled one -- the
         metric is not blind to the ordering it exists to reward;
      3. the numeric mask covers the number and nothing else, on a tokenizer that splits it.
    """
    n, m = 4, 6
    sched = population_schedule(m, n)
    assert len(sched) == n * n
    counts = defaultdict(int)
    for pick in sched:
        for problem, j in enumerate(pick):
            counts[(problem, j)] += 1
    assert set(counts.values()) == {n}, f"unbalanced schedule: {sorted(set(counts.values()))}"

    import numpy as np
    from prime_rl.orchestrator.algo import cs_grpo as _self  # noqa: F401  (import path sanity)

    auroc = _load(os.path.join(_cw_repo(),
                               "gen", "binary_metrics.py"), "cw_binary_metrics").auroc
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.01, 0.2, 40)
    thr = np.full(40, 0.1)
    good = population_metric(truth.tolist(), truth.tolist(), thr.tolist(), auroc_fn=auroc,
                             w_auroc=1.0, w_slope=1.0)
    bad = population_metric(rng.permutation(truth).tolist(), truth.tolist(), thr.tolist(),
                            auroc_fn=auroc, w_auroc=1.0, w_slope=1.0)
    assert good is not None and bad is not None and good > bad, (good, bad)

    pieces = {1: "the answer is ", 2: "Adjustment: ", 3: "-0", 4: ".0808", 5: ". done"}
    mask = numeric_token_positions([1, 2, 3, 4, 5], lambda ids: "".join(pieces[i] for i in ids))
    assert mask == [False, True, True, True, False], mask     # ". done" starts after the number ends
    # a token that STRADDLES the end of the number must still be credited: it carries the last digit
    straddle = {1: "Adjustment: ", 2: "-0.08", 3: "08. done"}
    mask2 = numeric_token_positions([1, 2, 3], lambda ids: "".join(straddle[i] for i in ids))
    assert mask2 == [True, True, True], mask2
    # and an answer with no number at all must credit nothing, rather than the whole sequence
    empty = {1: "no number here at all"}
    assert numeric_token_positions([1], lambda ids: "".join(empty[i] for i in ids)) == [False]
    _check_credit()
    _check_ladder()
    _check_first_turn_only()
    _check_action_turn()
    _check_matrix()
    _check_success_buffer()
    _check_single_cell_baseline()
    _check_single_label_refusal()
    print("cs_grpo self-check: schedule balanced, metric orders correctly, numeric mask exact, "
          "credit lands on the number and favours the better prediction")


class _Msg:
    def __init__(self, tool_calls=None):
        self.role, self.tool_calls = "assistant", tool_calls


class _Node:
    def __init__(self, token_ids, mask, tool_calls=None):
        self.token_ids, self.mask, self.advantages = token_ids, mask, None
        # `_assign_trace` finds the acting turn by `message.tool_calls`, which is how verifiers records
        # it; a node without a message is simply not the acting turn.
        self.message = _Msg(tool_calls)


class _Branch:
    def __init__(self, nodes):
        self.nodes, self.trainable = nodes, True


class _Trace:
    """The smallest object `_assign_trace` and `assign_advantages` actually touch."""
    def __init__(self, nodes):
        self.id, self.nodes, self.branches = "t", nodes, [_Branch(nodes)]


def _check_first_turn_only() -> None:
    """`credit_turn = "first"` must leave every later turn at exactly zero, and must still pay the first.

    The failure it guards is silent in both directions: a version that credits every turn trains the
    policy to answer the environment's ask (and, since it answers by repeating itself, to say the same
    thing twice), while a version that zeroes everything trains nothing at all and looks identical in the
    loss curve. So the assertion is that the two turns DISAGREE -- the first nonzero, the second exactly
    zero -- which neither failure can satisfy.
    """
    class _Cfg:
        credit = "contribution"
        credit_turn = "first"
        call_weight = move_weight = read_weight = 0.0
        auroc_weight = slope_weight = 0.0
        metric_weight, format_weight, slope_target = 0.0, 1.0, 1.0
        min_population, populations_per_rollout, num_problems = 99, 1, 2

    algo = CSGRPOAlgorithm.__new__(CSGRPOAlgorithm)
    algo.cfg, algo._metrics, algo._logger = _Cfg(), None, get_logger()
    algo._warned_cold, algo._decode = True, lambda ids: "x"

    def two_turn_trace():
        return _Trace([_Node([1, 2, 3], [True, True, True]),      # turn 1: the cold start's reasoning
                       _Node([4, 5], [False, False]),             # the environment's ask: not sampled
                       _Node([6, 7, 8], [True, True, True])])     # turn 2: answering the ask
    def row(fmt):
        return {"format": fmt, "pred": None, "truth": 0.1, "threshold": 0.1, "declared": None,
                "called": False, "moved": False, "read": False}

    traces = [two_turn_trace(), two_turn_trace()]
    algo._assign([(traces[0], row(1.0)), (traces[1], row(0.0))], [])
    for t in traces:
        first, second = t.nodes[0].advantages, t.nodes[2].advantages
        assert first and any(abs(v) > 0 for v in first), f"the trained turn carries no gradient: {first}"
        assert second is not None and all(v == 0.0 for v in second), (
            f"a later turn carries credit ({second}); with credit_turn='first' the policy would be "
            f"trained to answer the environment's ask")
    a, b = traces[0].nodes[0].advantages[0], traces[1].nodes[0].advantages[0]
    assert a > b, f"the better-formatted rollout was not paid more on the trained turn: {a} vs {b}"


def _check_matrix() -> None:
    """The M x N matrix scores populations where the rolling context scores none, on the SAME cells.

    This is the whole point of putting M patients in one episode, so it is asserted as a DIFFERENCE
    rather than as "the matrix works": the identical data is run through both paths, and the rolling
    context must score 0 (its deque is empty on the first group -- which is exactly the state an earlier run never
    left, 0 populations in ~600 consecutive groups) while the matrix scores every usable cell.

    Also asserted: the baseline is per PATIENT across episodes, not per episode. Two cells of the same
    patient that produced different predictions must receive different advantages; if the code centred
    per episode instead, cells of DIFFERENT patients would be compared and the difference would vanish.
    """
    import numpy as np

    class _Agent:
        trainable = True
        name = "agent"

    class _Ep:
        def __init__(self, traces): self.traces = traces

    class _Data:
        def __init__(self, key): self.item_id = key

    class _Task:
        def __init__(self, key): self.data, self.id = _Data(key), key

    class _MTrace(_Trace):
        def __init__(self, nodes, key, pred, truth):
            super().__init__(nodes)
            self.agent, self.has_error = _Agent(), False
            self.task = _Task(key)          # `_problem_key` reads trace.task.data.item_id
            self._key, self._pred, self._truth = key, pred, truth

    class _Cfg:
        credit = "contribution"
        credit_turn = "first"
        call_weight = move_weight = read_weight = 0.0
        auroc_weight, slope_weight, slope_target = 1.0, 0.0, 1.0
        metric_weight, format_weight = 1.0, 0.0
        min_population, populations_per_rollout, num_problems = 8, 1, 32
        rate_source = "default"

    M, N = 10, 4
    rng = np.random.default_rng(7)
    truths = rng.uniform(0.01, 0.20, M)
    algo = CSGRPOAlgorithm.__new__(CSGRPOAlgorithm)
    algo.cfg, algo._metrics, algo._logger = _Cfg(), None, get_logger()
    algo._warned_cold = True
    # maxlen 0: this check is about the matrix alone, so the success buffer must contribute nothing.
    # `_check_success_buffer` is where the top-up is asserted.
    algo._successes, algo._topped = deque(maxlen=0), 0
    # the decoder must render a NUMBER, or `numeric_token_positions` marks nothing and the metric lands
    # on no token at all -- the advantages come back all-zero and the check reads as "the matrix scored
    # nothing" when the matrix scored fine. Caught exactly that way while writing this.
    _pieces = {1: "Adjustment: ", 2: "-0", 3: ".0400."}
    algo._decode = lambda ids: "".join(_pieces[i] for i in ids)

    episodes, cells = [], []
    for j in range(N):
        col = []
        for m in range(M):
            # The N cells of a patient must span GOOD to BAD predictions, or every cell scores the same
            # concordance, the per-patient centring returns 0, and the check reads as "the matrix scored
            # nothing" while the log says "40 cells scored". That is what the first version of this data
            # did: pred = truth + N(0, 0.02) is accurate enough that every cell is perfectly concordant.
            # Cell j of every patient is progressively worse, so the baseline has something to separate.
            pred = float(truths[m]) if j == 0 else float(truths[(m + j) % M])
            t = _MTrace([_Node([1, 2, 3], [True, True, True])], f"patient-{m}",
                        pred, float(truths[m]))
            col.append(t)
            cells.append((t, {"format": 1.0, "pred": t._pred, "truth": t._truth, "threshold": 0.08,
                              "declared": 0.01, "called": False, "moved": False, "read": True}))
        episodes.append(_Ep(col))

    per_episode = [[(t, d) for t, d in cells if t in ep.traces] for ep in episodes]

    # THE MATRIX PATH
    algo._assign_matrix(per_episode)
    matrix_adv = {id(t): (t.nodes[0].advantages or [0.0])[0] for ep in episodes for t in ep.traces}
    n_nonzero = sum(1 for v in matrix_adv.values() if abs(v) > 1e-9)
    assert n_nonzero > 0, "the matrix scored no cell at all; a population of M=10 with min_population 8 must score"

    # THE ROLLING-CONTEXT PATH on the same cells: an empty deque cannot reach min_population
    for ep in episodes:
        for t in ep.traces:
            t.nodes[0].advantages = None
    algo._assign([(t, d) for t, d in per_episode[0]], [])
    ctx_nonzero = sum(1 for t in episodes[0].traces
                      if t.nodes[0].advantages and any(abs(v) > 1e-9 for v in t.nodes[0].advantages))
    assert ctx_nonzero == 0, (
        f"the rolling-context path scored {ctx_nonzero} cells on data where it should score none; then "
        f"the comparison above proves nothing about what the matrix added")

    # THE BASELINE IS PER PATIENT. Same patient, different predictions -> different advantage.
    per_patient = defaultdict(list)
    for ep in episodes:
        for t in ep.traces:
            per_patient[t._key].append(matrix_adv[id(t)])
    varied = [k for k, v in per_patient.items() if len(set(round(x, 9) for x in v)) > 1]
    assert varied, (
        "every cell of every patient got the identical advantage: the baseline is not separating the "
        "N rollouts of a patient, so there is no group-relative signal for the policy to climb")


def _check_success_buffer() -> None:
    """A column with too few readings of its own must score once past successes are available, and not before.

    This is a measured training state reproduced exactly: every population has ONE usable cell and
    `min_population` is above one, so `_assign_matrix` scored 0 cells in 12 of 12 groups. Asserted as a
    DIFFERENCE across two calls on identical data, because each half alone is satisfiable by a bug --
    a version that never tops up scores 0 twice, and a version that tops up from the population's OWN
    cells scores on the first call. Only feeding the buffer after the scoring loop gives 0 then nonzero.
    """
    class _Agent:
        trainable = True
        name = "agent"

    class _Ep:
        def __init__(self, traces): self.traces = traces

    class _Data:
        def __init__(self, key): self.item_id = key

    class _Task:
        def __init__(self, key): self.data, self.id = _Data(key), key

    class _MTrace(_Trace):
        def __init__(self, nodes, key):
            super().__init__(nodes)
            self.agent, self.has_error = _Agent(), False
            self.task = _Task(key)

    class _Cfg:
        credit = "contribution"
        credit_turn = "first"
        call_weight = move_weight = read_weight = 0.0
        auroc_weight, slope_weight, slope_target = 1.0, 0.0, 1.0
        metric_weight, format_weight = 1.0, 0.0
        min_population, populations_per_rollout, num_problems = 4, 1, 32
        success_buffer = 64

    def fresh(algo):
        """Four populations, one usable cell each -- and the four span both labels and both qualities."""
        episodes, per_episode = [], []
        # threshold 0.08: truth 0.12 is a positive, 0.04 a negative. pred near truth is concordant.
        spec = [(0.12, 0.12), (0.04, 0.04), (0.12, 0.02), (0.04, 0.20)]
        for j, (truth, pred) in enumerate(spec):
            col = []
            for m in range(3):                       # M = 3 cells, only cell 0 has a prediction
                t = _MTrace([_Node([1, 2, 3], [True, True, True])], f"patient-{m}")
                d = {"format": 1.0, "pred": pred if m == 0 else None, "truth": truth,
                     "threshold": 0.08, "declared": 0.01,
                     "called": False, "moved": False, "read": m == 0}
                col.append((t, d))
            episodes.append(_Ep([t for t, _ in col]))
            per_episode.append(col)
        return episodes, per_episode

    def nonzero(episodes):
        return sum(1 for ep in episodes for t in ep.traces
                   if t.nodes[0].advantages and any(abs(v) > 1e-9 for v in t.nodes[0].advantages))

    algo = CSGRPOAlgorithm.__new__(CSGRPOAlgorithm)
    algo.cfg, algo._metrics, algo._logger = _Cfg(), None, get_logger()
    algo._warned_cold, algo._topped = True, 0
    algo._successes = deque(maxlen=64)
    _pieces = {1: "Adjustment: ", 2: "-0", 3: ".0400."}
    algo._decode = lambda ids: "".join(_pieces[i] for i in ids)

    eps1, pe1 = fresh(algo)
    algo._assign_matrix(pe1)
    first = nonzero(eps1)
    assert first == 0, (
        f"a population with 1 usable cell and min_population 4 scored {first} cells on the FIRST call: "
        f"the buffer is being fed before the scoring loop, so a population is being measured against "
        f"its own cells")
    assert len(algo._successes) == 4, f"the four readings did not reach the buffer: {len(algo._successes)}"

    eps2, pe2 = fresh(algo)
    algo._assign_matrix(pe2)
    second = nonzero(eps2)
    assert second > 0, (
        "with 4 past successes in the buffer the same populations still scored nothing; the top-up is "
        "not reaching the metric")

    # The label gap must be closed, or the concordance half has no opposite to score against.
    labels = {_label(d) for _t, d in pe2[0]} - {None}
    picked = algo._top_up([d for _t, d in pe2[0] if d["pred"] is not None], 3)
    assert len({_label(d) for d in picked} | labels) == 2, (
        f"the top-up returned one label only ({[_label(d) for d in picked]}); AUROC is undefined on a "
        f"population with no opposite-label member however large it is")


def _check_single_cell_baseline() -> None:
    """One scored cell in a row must still get gradient -- and a row that CAN centre itself must not lose it.

    Both halves, because each alone is passed by a bug. A version with no fallback scores 0 on the first
    half (that is step 1 of an earlier run: 86 rows, `mean |metric adv| 0.0000` in all 86). A version that always
    uses the group baseline passes the first half and silently throws away the within-row baseline, which
    is the only one that compares two cells differing ONLY in what the policy did.
    """
    class _Cfg:
        credit = "contribution"
        credit_turn = "first"
        call_weight = move_weight = read_weight = 0.0
        auroc_weight, slope_weight, slope_target = 1.0, 0.0, 1.0
        metric_weight, format_weight = 1.0, 0.0
        min_population, populations_per_rollout, num_problems = 4, 1, 32
        success_buffer = 0

    def algo_and_rows(n_rows):
        algo = CSGRPOAlgorithm.__new__(CSGRPOAlgorithm)
        algo.cfg, algo._metrics, algo._logger = _Cfg(), None, get_logger()
        algo._warned_cold, algo._topped = True, 0
        algo._successes = deque(maxlen=0)
        _pieces = {1: "Adjustment: ", 2: "-0", 3: ".0400."}
        algo._decode = lambda ids: "".join(_pieces[i] for i in ids)
        traces = [_Trace([_Node([1, 2, 3], [True, True, True])]) for _ in range(n_rows)]
        rows = [(t, {"format": 1.0, "pred": 0.05, "truth": 0.12, "threshold": 0.08, "declared": 0.01,
                     "called": True, "moved": True, "read": True}) for t in traces]
        return algo, traces, rows

    def adv(trace):
        return sum(trace.nodes[0].advantages or [0.0])

    # HALF 1: one scored cell in the row, a group that has several. Must be nonzero.
    algo, traces, rows = algo_and_rows(2)
    group = {id(traces[0]): 0.90, 9991: 0.10, 9992: 0.20}      # row has 1; group has 3
    algo._assign(rows, [], metric_values=group)
    got = adv(traces[0])
    assert abs(got) > 1e-9, (
        f"a row with exactly one scored cell got advantage {got}; it centred against itself and the "
        f"population metric contributes nothing, which is the measured step-1 failure exactly")
    assert algo._baseline_src == "group", f"expected the group fallback, used {algo._baseline_src!r}"
    # direction: 0.90 against a group mean of 0.40 must be POSITIVE
    assert got > 0, f"a cell well above the group mean was paid {got}"

    # HALF 2: two scored cells in the row -> the ROW baseline, not the group's.
    algo, traces, rows = algo_and_rows(2)
    group = {id(traces[0]): 0.60, id(traces[1]): 0.40, 9991: 0.00, 9992: 0.00}
    algo._assign(rows, [], metric_values=group)
    assert algo._baseline_src == "row", (
        f"a row that can centre itself used the {algo._baseline_src!r} baseline; the within-row "
        f"comparison is the only one where two cells differ ONLY in what the policy did")
    a0, a1 = adv(traces[0]), adv(traces[1])
    # row mean 0.50 -> +0.10 / -0.10 and they must SUM to zero. The group mean is 0.25, which would give
    # +0.35 / +0.15 -- both positive and not summing to zero, so this assertion separates the two.
    assert a0 > 0 > a1, f"the row baseline did not separate the two cells: {a0}, {a1}"
    assert abs(a0 + a1) < 1e-6, f"a within-row baseline must sum to zero; got {a0} + {a1}"


def _check_single_label_refusal() -> None:
    """A population with one label must be REFUSED, not scored on the slope half at double weight.

    The defect this pins, found by a pre-launch review and reproduced by execution before the
    fix: `weight += w_auroc` sat inside `if len(opp):`, so a population with no opposite-label member
    dropped the AUROC half from the NUMERATOR AND THE DENOMINATOR and returned the slope term on a
    different scale. `population_metric` had always refused this case; the two credit modes disagreed and
    only the contribution one was in use. With the group-mean baseline fallback in place, one such value
    shifted every other cell's baseline, so the metric partly measured which labels the buffer held.

    Asserted three ways, because the single-label refusal alone is satisfiable by a metric that refuses
    EVERYTHING: the two-label case must still score, the single-label case must return None, and the
    two-label value must be the MEAN of the two terms (which is what pins the denominator).
    """
    preds = [0.12, 0.10, 0.18, 0.25]
    truths = [0.12, 0.15, 0.20, 0.30]
    W = dict(w_auroc=1.0, w_slope=1.0, slope_target=1.0)

    one = contribution_metric(0, preds, truths, [0.08] * 4, **W)            # labels 1,1,1,1
    two = contribution_metric(0, preds, truths, [0.08, 0.30, 0.08, 0.08], **W)   # labels 1,0,1,1
    assert one is None, (
        f"a single-label population scored {one}; that value is the slope term at double weight with the "
        f"discrimination half silently dropped, and it pools into the same baseline as two-term values")
    assert two is not None, "the two-label population must still be measurable, or the guard refuses all"

    slope = contribution_metric(0, preds, truths, [0.08, 0.30, 0.08, 0.08],
                                w_auroc=0.0, w_slope=1.0, slope_target=1.0)
    auroc = contribution_metric(0, preds, truths, [0.08, 0.30, 0.08, 0.08],
                                w_auroc=1.0, w_slope=0.0, slope_target=1.0)
    assert abs(two - (slope + auroc) / 2) < 1e-12, (
        f"the two-label value {two} is not the mean of its two terms ({slope}, {auroc}); the denominator "
        f"is not counting both weights and the scale is again ambiguous")
    # and the slope term alone must NOT equal the two-term value -- that equality was the whole defect
    assert abs(two - slope) > 1e-9, "the two-term value equals the slope term; the AUROC half is not landing"

    # the matrix gate must skip such a column rather than hand it to the metric
    class _Cfg:
        credit = "contribution"
        credit_turn = "first"
        call_weight = move_weight = read_weight = 0.0
        auroc_weight, slope_weight, slope_target = 1.0, 1.0, 1.0
        metric_weight, format_weight = 1.0, 0.0
        min_population, populations_per_rollout, num_problems = 3, 1, 32
        success_buffer = 0

    class _Agent:
        trainable = True
        name = "agent"

    class _Data:
        def __init__(self, k): self.item_id = k

    class _Task:
        def __init__(self, k): self.data, self.id = _Data(k), k

    class _MT(_Trace):
        def __init__(self, k):
            super().__init__([_Node([1, 2, 3], [True, True, True])])
            self.agent, self.has_error, self.task = _Agent(), False, _Task(k)

    algo = CSGRPOAlgorithm.__new__(CSGRPOAlgorithm)
    algo.cfg, algo._metrics, algo._logger = _Cfg(), None, get_logger()
    algo._warned_cold, algo._topped = True, 0
    algo._successes = deque(maxlen=0)
    _pieces = {1: "Adjustment: ", 2: "-0", 3: ".0400."}
    algo._decode = lambda ids: "".join(_pieces[i] for i in ids)

    # one column, three usable cells, ALL label 1 (truth > threshold for every one)
    col = [(_MT(f"p{m}"), {"format": 1.0, "pred": 0.05 + 0.01 * m, "truth": 0.20, "threshold": 0.08,
                           "declared": 0.01, "called": True, "moved": True, "read": True})
           for m in range(3)]
    algo._assign_matrix([col])
    nz = sum(1 for t, _d in col if t.nodes[0].advantages
             and any(abs(v) > 1e-9 for v in t.nodes[0].advantages))
    assert nz == 0, (
        f"{nz} cells of a single-label column were credited; nothing refused the column")
    # AND the gate must be what skipped it. Asserting "no cell credited" alone is satisfied by the
    # metric's own guard, so removing the gate left that assertion green -- verified, and the reason this
    # line exists. The counter is the gate's only observable effect.
    assert getattr(algo, "_unmeasurable", 0) == 1, (
        f"the matrix gate did not count the single-label column (_unmeasurable="
        f"{getattr(algo, '_unmeasurable', 'unset')}); the run would log `0 cells scored` without saying "
        f"whether the cause was no readings or an unmeasurable population")


def _check_action_turn() -> None:
    """`credit_turn = "action"` must put the ladder on the turn that ACTED, and `first` must not.

    What this guards is the defect it was written for, measured on an earlier run: under `first`, nonzero
    advantages were confined to assistant turn 1 in 800/800 trainable branches INCLUDING all 37 that
    contained a tool call, so the tool-call tokens received literal 0.0 and were then stripped from the
    loss and its denominator. The ladder was being paid to turn-1 prose, which can only upweight wordings
    that co-occurred with acting and can never upweight a tool-call token.

    The assertion is a DISAGREEMENT between the two modes on the same traces. A version that credited the
    action turn under both modes, and a version that credited it under neither, each satisfy "the action
    node is nonzero under action" or "is zero under first" -- but only a working implementation satisfies
    both at once. The non-caller's fallback is asserted separately, because dropping it would leave the
    group paying for acting without charging for not acting.
    """
    class _Cfg:
        credit = "contribution"
        credit_turn = "action"
        call_weight, move_weight, read_weight = 0.1, 0.2, 0.3
        auroc_weight = slope_weight = metric_weight = 0.0
        format_weight, slope_target = 0.0, 1.0          # the ladder alone is under test
        min_population, populations_per_rollout, num_problems = 99, 1, 2

    def caller():
        return _Trace([_Node([1, 2, 3], [True, True, True]),                 # turn 1: the reasoning
                       _Node([4, 5], [False, False]),                        # the ask: not sampled
                       _Node([6, 7], [True, True], tool_calls=[{"name": "cw_look"}]),   # THE ACTION
                       _Node([8, 9, 10], [True, True, True])])               # a restatement
    def talker():
        return _Trace([_Node([1, 2, 3], [True, True, True]),
                       _Node([4, 5], [False, False]),
                       _Node([6, 7, 8], [True, True, True])])                # never acted
    def row(called):
        return {"format": 1.0, "pred": None, "truth": 0.1, "threshold": 0.1, "declared": None,
                "called": called, "moved": called, "read": called}

    def run(mode):
        cfg = _Cfg(); cfg.credit_turn = mode
        algo = CSGRPOAlgorithm.__new__(CSGRPOAlgorithm)
        algo.cfg, algo._metrics, algo._logger = cfg, None, get_logger()
        algo._warned_cold, algo._decode = True, lambda ids: "x"
        a, t = caller(), talker()
        algo._assign([(a, row(True)), (t, row(False))], [])
        return a, t

    acted_trace, talk_trace = run("action")
    action_node = acted_trace.nodes[2].advantages
    assert action_node and any(abs(v) > 0 for v in action_node), (
        f"the acting turn carries no gradient under credit_turn='action': {action_node}. The tool-call "
        f"tokens are the only ones whose upweighting can make the policy act.")
    assert all(v == 0.0 for v in acted_trace.nodes[3].advantages), (
        f"a restatement after the action was credited: {acted_trace.nodes[3].advantages}")
    assert all(v > 0 for v in action_node), f"the caller was charged rather than paid: {action_node}"
    # A RESTATEMENT MUST NOT COLLECT THE METRIC. `_assign_trace` is called DIRECTLY with a nonzero
    # metric, because driving it through `_assign` needs a scoreable population and a version of this
    # check that did produced "0 populations scored" -- i.e. metric_adv 0, i.e. it asserted nothing.
    # The decoder returns a string with a number in it, so `numeric_token_positions` marks real tokens.
    algo_m = CSGRPOAlgorithm.__new__(CSGRPOAlgorithm)
    algo_m.cfg, algo_m._logger = _Cfg(), get_logger()
    algo_m.cfg.credit_turn = "action"
    algo_m._warned_cold = True
    # The same per-token piece table the numeric-mask checks above use: the decoder is handed ONE node's
    # sampled ids and must render exactly them, so a decoder that ignores its argument (the first version
    # of this line) breaks the prefix arithmetic and marks nothing. Note tokens 8-10 spell a number too --
    # the restatement HAS a copy of the answer, which is the whole point of asserting it gets zero.
    _pieces = {1: "Adjustment: ", 2: "-0", 3: ".0400.", 6: "cw_", 7: "look",
               8: "Adjustment: ", 9: "-0", 10: ".0400."}
    algo_m._decode = lambda ids: "".join(_pieces[i] for i in ids)
    m = caller()
    algo_m._assign_trace(m, fmt_adv=0.5, ladder_adv=0.3, metric_adv=0.7, decode=algo_m._decode)
    turn1, action, restated = m.nodes[0].advantages, m.nodes[2].advantages, m.nodes[3].advantages
    assert any(v > 0.5 for v in turn1), (
        f"the metric did not reach turn 1's number: {turn1}. If this is vacuous the next assertion "
        f"proves nothing, so it is checked FIRST.")
    assert all(v == 0.0 for v in restated), (
        f"a restatement collected credit while the metric path was live: {restated}. Positional "
        f"eligibility is what stops the policy being paid to say the same thing twice.")
    assert any(abs(v) > 1e-9 for v in action), f"the ladder did not reach the acting turn: {action}"

    # THE BASELINE INVARIANT, and the reason the ladder is length-scaled at all. The rl loss is a
    # per-token SUM over one global divisor, so a rollout's gradient mass is advantage x tokens. Under
    # `first` the ladder sat on turn 1 for every rollout and the centred term cancelled across the group;
    # moving the caller's POSITIVE ladder to a short tool-call node while the non-callers' NEGATIVE ladder
    # stays on long prose correlates sign with span. Reproduced before the fix on stubs shaped like a real trace:
    # sum(advantage x tokens) per group was +0.000000 under `first` and -14.531250 under `action`.
    # This asserts the sum is restored, on the same shape: turn 1 = 165 sampled tokens, action node = 15.
    def spans(n_first, n_action=None, n_rest=None):
        nodes = [_Node(list(range(n_first)), [True] * n_first)]
        nodes.append(_Node([900, 901], [False, False]))                       # the ask: not sampled
        if n_action:
            nodes.append(_Node(list(range(100, 100 + n_action)), [True] * n_action,
                               tool_calls=[{"name": "cw_move"}]))
            nodes.append(_Node(list(range(200, 200 + n_rest)), [True] * n_rest))
        return _Trace(nodes)

    # 0.01, not 1e-6: the advantages are float32 and 32 traces x ~345 sampled tokens accumulate ~2e-6 of
    # rounding. The defect this guards measured -14.53 on exactly this shape, so the tolerance sits 1450x
    # below the failure and 5000x above the noise -- it discriminates, which a tolerance chosen to make
    # the test pass would not.
    for mode, tol in (("first", 0.01), ("action", 0.01)):
        cfg = _Cfg(); cfg.credit_turn = mode
        alg = CSGRPOAlgorithm.__new__(CSGRPOAlgorithm)
        alg.cfg, alg._metrics, alg._logger = cfg, None, get_logger()
        alg._warned_cold, alg._decode = True, lambda ids: "x"
        traces = [spans(165, 15, 165)] + [spans(165) for _ in range(31)]
        rows_ = [(traces[0], row(True))] + [(t, row(False)) for t in traces[1:]]
        alg._assign(rows_, [])
        mass = sum(sum(nd.advantages) for t in traces
                   for nd in t.nodes if nd.advantages is not None)
        assert abs(mass) < tol, (
            f"credit_turn={mode!r}: the group's gradient mass is {mass:+.6f}, not 0. The centred ladder "
            f"is only a baseline if sum(advantage x tokens) cancels across the group; a short positive "
            f"span against long negative ones is a standing net charge on all sampled prose.")

    fallback = talk_trace.nodes[0].advantages
    assert fallback and all(v < 0 for v in fallback), (
        f"the rollout that never acted was not charged on its only turn: {fallback}. Without this the "
        f"group pays for acting and does not charge for not acting, so the comparison has no baseline.")

    # THE CONTROL: the same traces under `first` must leave the acting turn at exactly zero. If this
    # passes too, `action` changed nothing and the assertions above are measuring the old behaviour.
    old_acted, _ = run("first")
    assert all(v == 0.0 for v in old_acted.nodes[2].advantages), (
        f"credit_turn='first' credited the acting turn ({old_acted.nodes[2].advantages}); then the two "
        f"modes do not differ and this check proves nothing")


def _check_ladder() -> None:
    """The rungs must be ORDERED, or the ladder is decoration.

    The defect it guards: `read_weight` alone paid only the last step, so a rollout that called cw_look --
    the only way to learn a move id -- was given exactly the advantage of one that did nothing (measured:
    34 of 1,338 called, 1 read). A test that merely checked "the advantage is nonzero" would pass on a
    ladder whose rungs were in the wrong order, so what is asserted here is the ORDER.
    """
    class _Cfg:
        credit = "contribution"
        credit_turn = "all"
        call_weight, move_weight, read_weight = 0.1, 0.2, 0.3
        auroc_weight = slope_weight = metric_weight = 0.0    # the ladder alone is under test
        slope_target = 1.0
        format_weight = 0.0
        min_population = 99                                   # no population can be scored
        populations_per_rollout = 1
        num_problems = 2

    algo = CSGRPOAlgorithm.__new__(CSGRPOAlgorithm)
    algo.cfg, algo._metrics, algo._logger = _Cfg(), None, get_logger()
    algo._warned_cold, algo._decode = True, lambda ids: "x"

    def row(**kw):
        base = {"format": 1.0, "pred": None, "truth": 0.1, "threshold": 0.1, "declared": None,
                "called": False, "moved": False, "read": False}
        base.update(kw)
        return base

    traces = [_Trace([_Node([1], [True])]) for _ in range(4)]
    rows = [(traces[0], row()),
            (traces[1], row(called=True)),
            (traces[2], row(called=True, moved=True)),
            (traces[3], row(called=True, moved=True, read=True))]
    algo._assign(rows, [])
    adv = [t.nodes[0].advantages[0] for t in traces]
    assert adv[0] < adv[1] < adv[2] < adv[3], (
        f"the rungs are not ordered: nothing={adv[0]:.4f} called={adv[1]:.4f} moved={adv[2]:.4f} "
        f"read={adv[3]:.4f}. A rollout that took the first step must be paid more than one that took none.")
    assert abs(sum(adv)) < 1e-6, f"the ladder must stay group-relative (mean zero), got {sum(adv):.6f}"


def _check_credit() -> None:
    """The path the smoke could NOT exercise: a 2B base model never declares a number, so no population
    is ever formed and the metric advantage is identically zero. That is a true statement about the
    model, not evidence about this code, so the population path is checked here against rollouts that do
    declare -- and the check fails if the metric ever lands anywhere but the number's tokens."""
    class _Cfg:
        credit = "contribution"
        credit_turn = "all"
        call_weight = move_weight = read_weight = 0.0
        auroc_weight = slope_weight = metric_weight = 1.0
        slope_target = 1.0
        format_weight = 0.2
        min_population = 4
        populations_per_rollout = 4
        num_problems = 6

    repo = _cw_repo()
    metrics = _load(os.path.join(repo, "gen", "binary_metrics.py"), "cw_binary_metrics")

    algo = CSGRPOAlgorithm.__new__(CSGRPOAlgorithm)      # no clients, no env: only the scoring is under test
    algo.cfg, algo._metrics, algo._logger = _Cfg(), metrics, get_logger()
    algo._warned_cold, algo._decode = True, lambda ids: "".join(
        {1: "reasoning ", 2: "Adjustment: ", 3: "-0.05", 4: ". "}[i] for i in ids)

    def rollout(pred, truth=0.12, threshold=0.1, fmt=1.0):
        return {"format": fmt, "pred": pred, "truth": truth, "threshold": threshold, "declared": None}

    # a context of five other patients, half above their threshold and half below
    context = [(f"p{i}", [rollout(0.05 + 0.03 * i, truth=0.05 + 0.03 * i)]) for i in range(5)]
    traces = [_Trace([_Node([1, 2, 3, 4], [True, True, True, True])]) for _ in range(2)]
    # rollout 0 predicts this patient correctly high, rollout 1 predicts it low
    rows = [(traces[0], rollout(0.19, truth=0.19)), (traces[1], rollout(0.01, truth=0.19))]
    algo._assign(rows, context)

    adv = [n.advantages for t in traces for n in t.nodes]
    assert all(a is not None and len(a) == 4 for a in adv), adv
    NUM, PROSE = (1, 2), (0, 3)          # "Adjustment: ", "-0.05" | "reasoning ", ". "
    for a in adv:
        for i in PROSE:
            assert abs(a[i]) < 1e-9, (
                f"prose token {i} carries credit; with equal formats the format advantage is zero and "
                f"the metric must not reach it: {a}")
        assert any(abs(a[i]) > 0 for i in NUM), f"the metric never reached the number: {a}"
    for i in NUM:
        assert adv[0][i] > adv[1][i], (
            f"the worse-ordered prediction was paid more on token {i}: {adv[0][i]} vs {adv[1][i]}")

    # THE REGRESSION THIS CREDIT SCHEME EXISTS TO PREVENT: paying the whole population statistic makes a
    # rollout's advantage vanish as the population grows. Check that the contribution credit still
    # separates the two rollouts in a population five times larger.
    big = [(f"q{i}", [rollout(0.02 + 0.006 * i, truth=0.02 + 0.006 * i)]) for i in range(25)]
    traces2 = [_Trace([_Node([1, 2, 3, 4], [True, True, True, True])]) for _ in range(2)]
    rows2 = [(traces2[0], rollout(0.19, truth=0.19)), (traces2[1], rollout(0.01, truth=0.19))]
    algo._assign(rows2, big)
    a0, a1 = traces2[0].nodes[0].advantages, traces2[1].nodes[0].advantages
    assert a0[2] > a1[2], (
        f"in a population of 27 the credit no longer separates the rollouts: {a0} vs {a1}. This is the "
        f"dilution the `population` credit showed in training and the reason `contribution` is the default.")


if __name__ == "__main__":
    demo()
