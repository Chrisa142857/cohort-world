"""The pinned CohortWorld reward contract -- `docs/reward-spec.md`, implemented.

The document is the derivation; this file is the implementation, and the two are versioned together
by `REWARD_SPEC_VERSION`. Any change to a value in the constants block below is a MAJOR bump and
invalidates every score produced under a previous version.

Why one module instead of a method on each Task: the same three rules have to hold for the three
internal tasks and for the external PhenoCal task, and the last time they were "the same contract"
they were the same contract by being COPIED. The copies drifted: `_parse_distribution` read a value
in (1, 100] as a percentage while the commit path silently dropped it, and by the time anyone
looked `gen/results_table.py` had grown a third rule again. Four call sites, one implementation.

The three things this replaces, and why each was a defect rather than a preference:

* **The commit was a gate.** Scoring asked "did the agent commit?" and answered a constant when it
  had not -- 5.8-18.3% of rollouts, all scored the same -1.0, so "navigated correctly, read six
  cells, format broke" was indistinguishable from "did nothing". A constant is zero advantage
  variance under GRPO, i.e. unlearnable by construction. Fixed by `rate_report` / `locate_report` /
  `delta_report`: a report exists for EVERY episode, produced by a ladder that terminates in
  something the environment can always construct.
* **A pure proper score has a degenerate optimum.** arXiv:2608.00301 Prop. 9: with a pure proper
  score the model gains by degrading accuracy on hard items and honestly reporting the degradation.
  Realised in our own data -- on T2 the expected `1 - CRPS` is NON-MONOTONE in accuracy, minimised
  at chance, and a systematically-wrong-but-honest policy (q = 0) scores 0.838, beating every policy
  below 75% accuracy. Fixed by `ALPHA * R`, a correctness term that depends on the report's LOCATION
  and the outcome but never on its DISPERSION, so strict properness in the dispersion coordinate
  survives untouched (reward-spec 2.2).
* **Nothing was pinned.** `NO_ANSWER` moved 0.0 -> -1.0 and `BUDGET` 12 -> 26 at the same time,
  and no run directory recorded either, so scores either side of that change are not comparable and
  nothing says so. Fixed by `CONSTANTS`, which a run dir writes into its config.yaml.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
from functools import lru_cache

# --------------------------------------------------------------------------- pinned constants

ENV_VERSION = "7.6.3"
"""Semantic version of the WHOLE environment, as `docs/ENVIRONMENT_SPEC.md` describes it.

Scope: the cohort and disease IDENTIFIERS, the item sets and splits, the action space (tool
signatures), the observation space (every field a tool result carries, `at_risk` included), the
episode and its termination rules, and the reward contract below.

MAJOR: any of those changed -- every prior score is invalidated and no score may be compared across
the boundary. MINOR: additive only, e.g. a metric no reward reads. PATCH: docs and comments.
A reward change is by construction a MAJOR bump HERE as well; `REWARD_SPEC_VERSION` exists so a run
can also name the reward alone, which is the narrower thing a rescoring needs.

It lives in this module rather than in `__init__.py` because that file imports nothing on purpose,
and because a version that is not written into every run directory names nothing a run can be
checked against.

A missing panel is loud at SCORING time and at ROLLOUT time alike. `dist`, `entry_cohort` and
`reference_cohort` RAISE on an absent lattice rather than scoring one silently, and
`CohortWorldToolset.setup_task` raises rather than keeping the lattice it already had when the
item's own is absent: with `CohortWorld_default_*_states.jsonl` hidden, a quiet rollout side served
`RateTaskset`'s 151 `default` items bound to C2 -- reward 0.9425 at report_level 2 from an agent
that navigated the wrong cohorts for its whole episode. A run with a complete panel behaves
identically either way.

**The version is bound to the code by content, not by discipline.** A version string that no check
reads can be mutated with every test still GREEN, which means it names nothing. See `semantic_hash`
and `VERSIONED_SEMANTICS` below: changing the reward without bumping the version and bumping the
version without changing the reward both fail, with different messages.
"""

REWARD_SPEC_VERSION = "2.0.0"
"""Semantic version of THIS CONTRACT, not of the code around it.

MAJOR: any value below, any reward form, or any prompt sentence that changes the achievable policy
set -- every prior score is invalidated. MINOR: an additive metric no reward reads. PATCH: docs.
Written into every run's config.yaml and into every scored artifact's filename, because a constant
change that no run records makes scores on either side of it silently incomparable.
"""

ALPHA = 1.25
"""Weight on the discrimination term `R`, for T2 and T3.

Bounded below at 1.247 by the only reachable degenerate policy we could measure -- a zero-information
report whose level is a fixed multiple of the base rate (`disease` and `horizon` are both in the
prompt, so `b` is a memorisable table), tuned directly on the eval items: CRPS 0.01820, meanR
+0.00322. Arm B needs alpha > 1.247 to beat it, arm A > 1.148. Bounded above near 2 by the R term
taking 63% of per-item reward dispersion already at 1.25.

NOT the paper's `alpha > 1`, which does not transfer: Prop. 9 needs it because its accuracy and
report coordinates live in different spaces and `q = 0` is perfectly forecastable, while CRPS scores
a distribution over the outcome's own space, where that state does not exist. Treat the agreement
with the paper's number as a coincidence.

What alpha does on T2, measured rather than derived (`tests/test_reward.py`): it lifts a perfectly
accurate policy from 1.000 to 1.772 while leaving the degenerate q = 0 policy at 0.838, and makes
the reward monotone in accuracy over [1/3, 1]. It does NOT make `r(q)` increasing BELOW chance --
reward-spec 4.3's `alpha > 0.3371` assumes the decision tracks the pick, which is true of a point
mass and false of the honest reporter that creates the degeneracy in the first place, whose median
sits on the hedge class there so `R` is constant and no alpha can tilt it. That region is not one a
trained policy occupies, and it no longer competes with competence, which is what alpha is for.
"""

DEFAULT_T3_VARIANT = "base"
"""Which T3 reward FORM a run trains on. `base` is the `REWARD_SPEC_VERSION` contract, byte for byte; every scored
artifact and every leaderboard number is produced with it, and the external (PhenoCal) scorer never
passes anything else. The other names in `T3_REWARD_VARIANTS` are TRAINING-TIME rewards under study
(the item set is fixed; only the reward moves), selected per run by `env.taskset.task.reward`,
which the resolved orchestrator config records. A reward mean produced under any other variant is
not comparable to a `base` reward mean; the scored metrics (CRPS, R, AUROC on the external set) are.
"""

T3_STEP_BETA = 0.25
"""Weight of the side term in the `entry_step` variant: `+BETA` when the report's median and the
truth lie on the same side of the pinned base rate `b`, else 0 (0 at median == b). Small against
`ALPHA * R` so it breaks the look-once tie without outweighing accuracy. See `_t3_entry` for what
this term does and does not change under GRPO."""

T3_TANH_GAMMA = 0.5
T3_TANH_SCALE = 0.0512

T3_THINK_W_RANK = 0.05
T3_THINK_W_EDGE = 0.05
T3_THINK_EDGE_SCALE = 0.0026
T3_THINK_REF_N = 512
T3_THINK_REF_SEED = 42
"""The `cohort_think` variant. Its specification, verbatim in intent:
`<cohort-think>` fixes a TASK, not a structure -- as CoT's think region fixes "thinking" -- and RL
must make the policy, on meeting it, reason about which single edge's delta is most reliably
estimable from the patient's five-axis history and what that delta's sign and size are. So nothing
here pays for reproducing a template. Two things are paid:

* `T3_THINK_W_RANK` weights the rollout's CONTRIBUTION TO THE CROSS-ITEM RANKING -- the board's
  AUROC is a cross-item statistic, GRPO's advantage is within-item, and the bridge is to score each
  rollout by how its own prediction ranks against a FIXED reference set of (truth, base rate) pairs.
  That varies with the rollout's prediction, so it survives `reward - group mean`, and it is the
  per-rollout form of the metric the study is judged on. The reference set is `T3_THINK_REF_N`
  train items drawn with `T3_THINK_REF_SEED`; its content hash is in CONSTANTS().
* `T3_THINK_W_EDGE` weights the EDGE PREDICTION: the delta the policy states for its first move
  BEFORE it observes the destination, against the delta its own two readings then measure, scored
  `exp(-|stated - measured| / SCALE) - exp(-|measured| / SCALE)`: the IMPROVEMENT over declaring zero.
  The first form, `exp(-|stated - measured| / SCALE)` alone, was maximised by the constant string
  `+0` (measured in an audit): on 127,930 published first-move edges from the entry cohorts the median
  |delta| is 0.0026 and 23.4% are exactly 0, so `+0` scored 0.63-0.74 while a sign-correct informed
  prediction scored 0.585. Subtracting the zero-declaration's own score makes the template worth
  exactly 0 and pays only for information. SCALE is the median |delta| of an ENTRY edge (0.0026),
  not of an arbitrary lattice edge (the earlier 0.0098 was 4x too wide and widened the plateau).
  RE-CHECKED UNDER THE CURRENT DIRECTION AND KEPT: the delta a declaration must now predict is
  `answer_rate - rate(reference_cohort)`, whose median |delta| over the pinned sample is **0.00279**
  (494 items, mean +0.0288, sd 0.0499, 41.7% exactly 0). The old number came from a distribution the
  environment no longer produces and lands within 7% of the new one, so the value is retained -- on
  the NEW measurement, not on the old sentence. Re-tuning it is a sweep like `T3_CAL_W_DELTA`'s.
  Stating before observing is enforced by message ORDER on the trace; a number written after the
  reading is a copy and is not paid.

Both weights sit at the top of the within-group T3 reward sd (0.012-0.052): visible, never dominant.
What was measured before choosing this form, on 4,000 internal train items (global-median AUROC):
not moving 0.879; the biggest-neighbour shortcut 0.936; the one-step "oracle" 0.987 -- but that
oracle coincides with the biggest neighbour only 4.9% of the time, shares a code with the target only
2.9%, and lands in 10-49-person cohorts 90% of the time, i.e. it is the luckiest of 36 quantised
candidates, not a learnable choice. Choosing a well-estimated cohort IS legitimate reasoning (a
clinician prefers a reliable reference class), so the ranking term may reward it; the reasoning the
user wants trained -- delta from five-axis difference -- lives in the edge term, where conditioning
on five-axis similarity was measured to raise sign-consistency from 35% to 88%.
"""

T3_CAL_L2_SCALE_MULT = 2.0
T3_CAL_Q2_CAP = 4.0
T3_CAL_W_EDGE = 0.05
T3_CAL_W_DELTA = 0.4
T3_CAL_DELTA_REF_N = 512
T3_CAL_DELTA_GRID = (-0.05, 0.05, 0.0001)
T3_CAL_NOMOVE_FLOOR = None
T3_CAL_W_MOVE = 0.03
T3_THINK_PROMPT_VARIANTS = ("cohort_think", "cohort_think_cal")
"""The `cohort_think_cal` variant: the outcome term of `cohort_think` with the L1 point loss replaced
by a CAPPED, SCALED L2 loss, the tanh covariance term removed, the `cohort_think` edge term computed
and reported but UNPAID, and NON-NEGATIVE process payments for the delta declared before the move
and for an accepted move (both below). The outcome part is `ALPHA * R2 + (1 - Q2) +
T3_THINK_W_RANK * rank` with `Q2 = min((p - y)^2 / S2, T3_CAL_Q2_CAP)`, `R2 = min((b - y)^2 / S2, CAP) - Q2`, and
`S2 = cal_l2_scale() = T3_CAL_L2_SCALE_MULT * MAE_b`, where MAE_b is the pinned base-rate
forecaster's mean absolute error over the internal T3 TRAIN items (16,770 rows: 0.035877, so
S2 = 0.071754). S2 is DERIVED from the item set at first use (lru_cached, like
`rank_reference`), never a literal, and `CONSTANTS()["t3_cal_l2_scale"]` records the resolved value.

WHY L2. The frozen judge's calibration slope
is the OLS of the committed POINT on the truth, cov(P,Y)/var(Y) = corr * sd(P)/sd(Y)
(np.polyfit(Y, P, 1)[0]). On every RL screen the slope FELL while corr ROSE:
the `point` run (from the base model) sd(P)/sd(Y) 0.83 -> 0.49, judge slope 0.385 -> 0.339;
the `point_tanh` run: 0.83 -> 0.40, 0.385 -> 0.277; Cox slope (truth on point) 1.44 / 1.74, i.e.
under-dispersed and biased low, not overconfident. The L1 point loss is consistent for the
conditional MEDIAN (Gneiting 2011, JASA 106:746-762, doi:10.1198/jasa.2011.r10138), the squared
loss for the conditional MEAN; the train truth is right-skewed (mean 0.0448, median 0.000), so the
L1 optimum under weak information sits below the mean-calibrated forecast. Mean-only GRPO advantages
are an unbiased estimator of grad E[r] (Dr GRPO, arXiv:2503.20783 App. A), so the population optimum
of this term is the conditional mean. A competing explanation for the observed shrinkage is the
policy-class attractor "commit the entry reading r0" (an SFT cold-start checkpoint commits r0 on
98.5% of items with no RL at all); early training readouts are pre-registered to separate the two.

WHY no covariance term. A proposed |y-b|-normalised capped `cov` term and the existing `tanh`
term both move the weak-information optimum away from the cell mean: on the train items the
per-cell optimum of `cal`-with-cov sits -0.028 / +0.030 (p10 / p90) from the cell mean, point_tanh
-0.033 / +0.033, the pure L2 form -0.0004 / +0.0003;
the full cal-with-cov ranked the oracle cell MEDIAN above the cell MEAN (0.956 vs 0.926 on the
judge items) -- the non-properness the change exists to remove. Cost, measured on 323 replayed
groups of a training run (hard-argmax selection proxy, paired vs cohort_think): slope +0.004 [+0.000,+0.013],
cell-AUROC -0.007 [-0.016,-0.001] -- the same -0.007 the `point` argmax showed, i.e.
the price of the tanh term's dispersion payment; Spearman(group advantage sd, |y - mean p|) 0.587 ->
0.766; mean advantage on right-side overshoot 0.0025 < undershoot 0.0080 (cohort_think 0.0146 =
0.0146). The proxy is a one-step within-group ceiling and cannot see the cross-group advantage
magnitude the L2 mechanism relies on.

WHY the cap. Q2 reaches 1 / S2 = 13.9 at |p - y| = 1; a FLOOR row (p = 0.5) against y ~ 0 would
carry ~40x the normal within-group spread under mean-only advantages. `T3_CAL_Q2_CAP` = 4 flattens
the term beyond |p - y| = sqrt(4 * S2) = 0.536, which a committing policy never reaches (measured
on a training run: max |p - y| 0.231). Both squared terms are capped so `R2 >= 0 iff |p - y| <= |b - y|` still holds.

WHY the `cohort_think` edge is unpaid here, and what `edge_cal` computes. Under the |m|-normalised
form (edgeB) the policy's actual declarations paid -0.053 on a measured training run and the best
constant template is exactly 0, so that term's gradient points to silence; under a sign-only form the
constant string "-0.001" earns +0.176 on the same 336 declared-and-measured rollouts (a perfect
sign-and-size oracle 0.455) because 48% of measured first-move deltas are negative and 17% positive
-- a sign prior, the `+0` exploit in a new coat. But a term that pays nothing is ground away under
GRPO -- on a run whose edge term paid NEGATIVE on average because most declarations are wrong, the
declared share fell from ~0.25 to 0.14-0.22 -- and the standing requirement is that RL KEEP the
chain (read the start rate -> declare a signed delta BEFORE the one move -> move -> read -> commit).
`edge_cal`, a gated payment now computed and REPORTED UNPAID (see below), is

    edge_cal = [sign(d) == sign(m) != 0 and |d| >= s0]
               * max(0, exp(-|d - m| / max(|m|, s0)) - exp(-1)) / (1 - exp(-1))

with d the declared delta, m the measured one (both from `edge_prediction`, unchanged) and
s0 = T3_THINK_EDGE_SCALE, weighted by `T3_CAL_W_EDGE` = 0.05 (the rank term's weight: visible,
never able to outrank the outcome term). Properties, each pinned by tests/test_cohort_think.py:
(a) never negative, so declaring never costs reward -- a wrong declaration is paid exactly as
silence; (b) |m|-normalised: a correct delta pays 1 whether it is 0.003 or 0.05, and nothing is paid
for dispersion or for choosing a big-|m| move; (c) 0 when nothing was declared before the move,
when no DIFFERENT cohort was read at the asked cell after it, or when the number is a copy of an
observation; (d) the gate: a declaration smaller than s0 (the median entry-edge |delta|), one with
the wrong sign, or any declaration on an edge that measured exactly 0 (35% of the measured run's
first moves) pays 0. The gate is what that data demanded: the ungated form pays the constant string `+0`
0.457 on the 336 rollouts (every m == 0 edge scores 1) against 0.408 for the actual declarations;
gated, `+0`, `+0.0001`, `+0.001` and `-0.001` pay exactly 0 (|c| < s0). What the gate does NOT do,
measured on the same rollouts and recorded rather than hidden: a constant of the PRIOR sign and
size (`-0.003`) still earns 0.171 against the actual declarations' 0.011 and the oracle's 0.434,
because those declarations are themselves a template (58% are +0.0001 / +0.0005 / +0.001, 8%
have |d| >= s0, sign agreement 26% -- BELOW the 48% a fixed "-" would score). No non-negative
correctness payment can pay such declarations more than a sign-prior constant; what the form
guarantees is that no constant is its maximiser (0.171 vs 0.434) and that the named templates earn
nothing. Whether a trained policy beats the prior is what the declared-share and sign-agreement
readouts measure. `think_edge` (the `cohort_think` formula) stays reported for the process gate;
`T3_THINK_W_EDGE` applies to `cohort_think` only.

The PAID process term is `delta_cal`, not `edge_cal` (`edge_cal` and `cal_edge` are computed and
REPORTED UNPAID beside `think_edge`; `T3_CAL_W_EDGE` keeps its value and is not read by this
variant). The design correction the form follows:

    "Always moving to the biggest neighbour is not a meaningless shortcut. (1) A clinician also
    leans on the largest cohort, because it is the most reliable -- the prompt already asks for a
    group large enough to read reliably. (2) The five-axis information DIFFERENCE between the entry
    cohort and that neighbour is the INPUT CONDITION that determines the delta; converging on the
    edge does not converge on the delta. The delta is what cohort-think has to REASON OUT."

So which edge is taken is not the learning target, and a group whose eight rollouts take the same
edge is not a degenerate group: their eight DELTA PREDICTIONS should differ, and that is where
GRPO's gradient is supposed to come from. The 34.7% zero-variance groups measured while `edge_cal`
was the paid term were caused by the one coordinate that should carry the variance being nearly
unpaid -- the `cohort_think` edge term weighted it 0.05 and paid the measured run's real
declarations a NEGATIVE mean; `edge_cal` paid on 4% of declarations, a mean 0.0006 of the reward.
`delta_cal` makes it a first-class term:

    acc(d, m) = exp(-|d - m| / max(|m|, s0))            # s0 = T3_THINK_EDGE_SCALE
    delta_cal = max(0, acc(d, m) - acc(c*, m))          # c* = cal_delta_const(), the PINNED constant
    reward    = ALPHA * R2 + (1 - Q2) + T3_THINK_W_RANK * rank + T3_CAL_W_DELTA * delta_cal

`c*` is the best CONSTANT declaration on a fixed, policy-independent reference sample of the delta
a declaration has to predict (`delta_reference()`: one `answer_rate - rate(reference_cohort)` per
item over `T3_CAL_DELTA_REF_N` seeded train items -- derived from the ITEM SET, never from a run,
exactly as `rank_reference()` and `cal_l2_scale()` are, hashed into CONSTANTS() as
`t3_cal_delta_ref_sha16`), maximising mean acc over `T3_CAL_DELTA_GRID` plus 0. On the official
items c* = 0.0 EXACTLY (41.7% of the pinned sample measures a delta of exactly 0, see
`T3_THINK_EDGE_SCALE`, and no other constant can hold that atom), so the payment reduces to the
improvement over declaring zero -- the `cohort_think` edge term's intent, now DERIVED rather than
assumed, rectified so it is never negative, and |m|-normalised.

What the centring buys, and why `delta_cal` has no gate. By construction the mean CENTRED accuracy
of any constant on the reference distribution is <= 0, with equality only at c*, so no constant
template is the maximiser of this term. With c* = 0 two clauses of the `edge_cal` gate become
theorems instead of clauses: a wrong-sign declaration has |d - m| > |m| so it pays exactly 0, and
every declaration on an edge that measured exactly 0 pays exactly 0 (acc(c*, 0) = 1 is the maximum
the term can reach). The third clause, |d| >= s0, is NOT implied and is DROPPED: it was there to
stop template-sized numbers, and the centring already does that, while the gate also refused to pay
a CORRECT declaration smaller than s0 -- on the measured run that is most of the ones the policy makes.

The limit, measured and recorded rather than hidden. Exact constant-proofness is IMPOSSIBLE for any
payment that is both non-negative and a function of (d, m) alone: E[max(0, X)] > 0 whenever
X = acc(c, m) - acc(c*, m) varies at all, even though E[X] <= 0. Measured in a replay sweep: on the reference
distribution the best constant of any size (-0.0048) collects a mean 0.090 against 0.381 for an
oracle declaration (d = m), 24% of the oracle; on the measured run's 336 declared-and-measured rollouts the
best constant (-0.0042) collects 0.102 against the oracle's 0.343 and the policy's own declarations'
0.018 -- those declarations are themselves a template (22 distinct values over 336 declarations,
corr(d, m) = -0.17). The `edge_cal` gate did not do better on that: its best constant took 0.171 of an
oracle 0.434 (39%). So what the form guarantees is the CENTRED statement -- the mean centred
accuracy of every constant on the pinned reference is <= 0, exactly 0 at c* = 0 and strictly
negative for all 1,000 others -- plus the ratios above. The claim "a constant earns exactly 0"
would be false under the rectification and is not made.

Why the |d| >= s0 gate is dropped rather than kept, measured on the same rollouts: with the gate
the actual declarations are paid 0.0065, without it 0.0177 (2.7x, and the payment then varies in 17
groups instead of 4), while THE BEST CONSTANT IS IDENTICAL either way (-0.0042 -> 0.1022) -- the
gate costs correct small declarations and buys nothing against templates.

`T3_CAL_W_DELTA` is NOT a guess: it is chosen on a replay sweep of a training run over {0.05, 0.1,
0.2, 0.4, 0.8, 1.6}. No weight satisfies all four pre-registered criteria and two of them are
weight-INDEPENDENT: zero-variance groups sit at 0.331 for every weight (the payment can only break
a tie in a group where some rollout declares, and the start policy declares in 13% of rollouts),
and no constant can be made to earn 0 (above). 0.4 is the smallest weight at which an EXACT
declaration (0.632 * w = 0.253) is worth about the outcome term's own p99 within-group advantage
(0.239) -- first-class, as the correction requires -- while the outcome still governs the argmax by
a wide margin: corr(reward, -|p - y|) 0.847 against corr(reward, payment) 0.162, and the argmax is
still the argmin |p - y| rollout in 91% of groups.

A ROLLOUT THAT NEVER MOVED IS NOT PAID FOR IT. Measured on 2,597 training rollouts of one run: 1,669 (64.3%)
executed a move, **926 (35.7%) called `cw_move` and were REFUSED** ("not a legal move from here"),
and 2 never tried. The 926 had all called `cw_look` first and still passed an id that does not
exist -- a medically sensible chapter with a FABRICATED 16-hex destination (e.g. `bnf.add.1001->...`
against a legal set of `bnf.add.0103 / 0106 / 0202 / ...`). At the judge's temperature 0.6 the same
clone fabricates 8.5% of the time; training runs at 1.0. So `think_moved == 0` never measured
deliberate abstention -- it measured id fabrication -- and the failure was PAID: a refused move
leaves the rollout in the 1000+ entry cohort, whose published rate is often nearer the truth than
any reachable neighbour's (in one walked example the two rejected-move rollouts committed the
entry cohort's rate and took the group's LARGEST advantages, +0.596 each, while the six that
really moved landed on their neighbours' rates and were all punished). Over 100
steps GRPO would be trained to write invalid move ids. So:

    reward = T3_CAL_NOMOVE_FLOOR                                          if it did not move
             ALPHA*R2 + (1-Q2) + W_RANK*rank + W_DELTA*delta_cal + T3_CAL_W_MOVE*moved   otherwise

The sweep chose OPTION B: `T3_CAL_W_MOVE = 0.03` with the FLOOR DISABLED (`None`). Both were swept
on the replay against five criteria, and the floor lost on the evidence: at
every value that made the non-mover advantage non-positive it cost 3-4x more proxy slope
(-0.034 vs -0.013) and 5-50x more cell AUROC (-0.056 vs -0.001) than the payment, because flooring
throws away the outcome gradient of the 35.7% of rollouts that do not move, while the payment keeps
grading them on accuracy and merely removes the bonus they were collecting for failing. At 0.03 the
mean advantage of a no-accepted-move rollout goes +0.0056 -> **-0.0071** (criterion 1), the
zero-variance group share falls 0.328 -> **0.130** (81% of groups are mixed, so the move fact breaks
the ties the delta payment could not), cell AUROC vs the `edge_cal`-paying form is -0.001
(criterion 3), and a rollout that moved and landed FURTHER from the truth than staying still loses
(-0.0176, criterion 5). What it costs: proxy slope -0.0133 [-0.0199, -0.0080] vs the
`edge_cal`-paying form, of which -0.0097 is the delta term's and only -0.0036 is W_MOVE's --
criterion 2 fails, and it fails for every option that satisfies criterion 1, because on that run
the rejected-move rollouts really do commit better points.

`accepted_move()` reads the environment's own acknowledgement (`hit` + `took`), the same evidence
`edge_prediction` uses, and is the single reader of that fact. Nothing else changes: no `vf.stop`,
no taskset, prompt, environment or temperature change (the judge shares that code and stays frozen,
and temperature 1.0 is the only remaining source of within-group diversity). The floor value and
`T3_CAL_W_MOVE` are SWEPT on the replay, not chosen, against five acceptance criteria, including
the arbitrage check that a rollout which moved and landed FURTHER from the truth than staying must
not gain.

THE DECLARATION PARSE differs for this variant only. `_SIGNED_NUM` cannot see a signed number that
ends a sentence, so the declaration rate it reports (20.1% of the measured run's rollouts) is a
property of punctuation, not of the policy: the true rate under `_SIGNED_NUM_V2` is 99.77%. The
declaration statistics quoted above were computed with `_SIGNED_NUM`, i.e. on that biased fifth.
`delta_cal` reads v2; `think_edge`, `edge_cal` and `think_declared` still read v1 so `cohort_think`
stays byte-identical.

`T3_THINK_PROMPT_VARIANTS` names the variants whose prompt IS `internal.cohort_think_question`
(the cold start's own question, not an appended paragraph); it
is the one source of truth internal.py reads, so a run on the new variant trains with the chain.
"""

_THINK_OPEN = re.compile(r"<cohort-think>", re.I)
_SIGNED_NUM = re.compile(r"(?<![\w.])([+-]\s?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)(?![\w.])")
_SIGNED_NUM_V2 = re.compile(r"(?<![\w.])([+-]\s?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)(?![\w])")
r"""The declaration pattern, and the v2 pattern `cohort_think_cal` reads. They differ in ONE character
class and that character cost the study its declaration statistics.

`_SIGNED_NUM`'s trailing lookahead is `(?![\w.])`, so a signed number that ENDS A SENTENCE does not
match: `"-0.004."` finds nothing, `"-0.004 in the rate"` and `"-0.004, so"` match. Measured on
2,597 training rollouts of one run: **100.00% opened `<cohort-think>` before the move, the v1
pattern finds a signed number in 20.10% of them and the v2 pattern in 99.77%**. So `think_declared`
(~0.14-0.27 over that run) never measured whether the policy declares -- it measured whether the
declaration happened to fall anywhere but the end of a sentence -- and every statistic derived from
it was computed on that biased fifth of the data: the "58% are +0.0001/+0.0005/+0.001", the "8%
have |d| >= s0", the "sign agreement 26%", and the `edge_cal` constant-template table.

`_SIGNED_NUM` is KEPT, unchanged, and is still what `cohort_think` and `edge_cal` read, because
those two must stay byte-identical on archived replays.
`edge_prediction` takes the pattern as an argument whose DEFAULT is the old one -- one function, two
patterns, rather than two copies of a fifty-line parser that would drift -- and only
`cohort_think_cal`'s delta payment passes `_SIGNED_NUM_V2`."""


def _to_float(tok: str):
    try:
        v = float(tok.replace(" ", ""))
    except ValueError:
        return None
    return v if math.isfinite(v) and abs(v) <= 1.0 else None


def _walk(trace):
    """(index, role, name, content, tool_calls) for every message, oldest first."""
    for i, m in enumerate(getattr(trace, "messages", None) or []):
        get = (lambda k: m.get(k)) if isinstance(m, dict) else (lambda k: getattr(m, k, None))
        yield i, get("role"), get("name"), get("content"), (get("tool_calls") or [])


def _call_name(call) -> str:
    cg = (lambda k: call.get(k)) if isinstance(call, dict) else (lambda k: getattr(call, k, None))
    return (cg("name") or ((cg("function") or {}).get("name") if isinstance(cg("function"), dict)
                           else getattr(cg("function"), "name", None)) or "")


def edge_prediction(trace, disease: str, horizon: int, pattern=None, truth=None) -> dict:
    """The policy's FIRST edge: what it said the rate would do, and what its own readings measured.

    Returns {"declared", "measured", "src", "dst", "declared_before_move"} with None where absent.
    `pattern` selects the declaration regex and DEFAULTS to `_SIGNED_NUM`, so every other caller
    keeps the v1 parse; `cohort_think_cal` passes `_SIGNED_NUM_V2` (see that constant's docstring).
    Order is the whole point. The declaration is the last signed number (falling back to the last
    number at all) in ASSISTANT text that comes at or after the first `<cohort-think>` and BEFORE
    the first `cw_move` call. A number written after the destination was read is a copy of an
    observation and is not a prediction. The DECLARATION rules do not depend on `truth`.

    `truth` picks which delta the declaration is scored against, and it exists because the internal
    T3 episode runs the chain the other way:

      * `truth is None` (the default, so every caller outside the internal T3 path is unchanged, and
        an archive rescored under its own geometry keeps its number): `measured = rate(dst) - rate(src)`,
        what the move DID, read from the trace's own two readings.
      * `truth` given -- an internal T3 item: `measured = truth - rate(dst)`, the
        correction from the reference class back to the patient. `src` is now the item's own cohort
        and its rate is WITHHELD, so a reading before the move can never exist and the old form would
        be None on every single rollout -- a 0.4-weighted reward term paying identically zero with
        nothing saying so. This is also the quantity `delta_reference` samples, so the payment and
        the constant it is centred on are on ONE distribution.
    """
    out = {"declared": None, "measured": None, "src": None, "dst": None, "declared_before_move": 0.0}
    opened_at = None; move_at = None; text = []
    for i, role, name, content, calls in _walk(trace):
        if role == "assistant":
            c = content if isinstance(content, str) else ""
            if opened_at is None and _THINK_OPEN.search(c):
                opened_at = i
                c = c[_THINK_OPEN.search(c).end():]
            if opened_at is not None and move_at is None:
                text.append(c)
            if move_at is None and any("move" in _call_name(x) for x in calls):
                move_at = i
                # text in THIS message still counts: it precedes the call the message carries
                break
    if opened_at is None or move_at is None:
        return out
    blob = "\n".join(text).replace("\u2212", "-")     # unicode minus reads as a sign
    # SIGNED numbers only. The unsigned fallback manufactured declarations: in a 2-step smoke 98 of
    # 125 "declarations" were `1.0` from "1 year" or `0.0` from "0 diagnoses", and an unsigned "0"
    # in prose plus a zero-delta move paid the whole edge term. The prompt asks for a signed number.
    nums = [v for v in (_to_float(t) for t in (pattern or _SIGNED_NUM).findall(blob)) if v is not None]
    if nums:
        out["declared"] = nums[-1]; out["declared_before_move"] = 1.0
    # readings of the asked cell, tagged by position; find the last before the move and first after
    here = None; before = None; after = None
    for i, role, name, content, calls in _walk(trace):
        if role != "tool":
            continue
        d = _json(content)
        if d is None:
            continue
        if d.get("state_id"):
            here = d["state_id"]
        if d.get("status") == "published" and isinstance(d.get("rate"), (int, float)) \
                and d.get("disease") == disease and str(d.get("horizon")) == str(horizon):
            if i < move_at:
                before = (here, float(d["rate"]))
            elif after is None and i > move_at and here is not None and (before is None or here != before[0]):
                after = (here, float(d["rate"]))
    if truth is not None:
        # The new direction: only the reading AFTER the move is needed, because the reading before
        # it is the answer and is withheld. `src` is still reported for the diagnostics that print it.
        if after is not None:
            out["src"] = before[0] if before is not None else None
            out["dst"] = after[0]
            out["measured"] = float(truth) - after[1]
    elif before is not None and after is not None:
        out["src"], out["dst"] = before[0], after[0]
        out["measured"] = after[1] - before[1]
    return out


def accepted_move(trace) -> tuple:
    """`(moved, code)` for the FIRST move the environment ACCEPTED: 1.0 and the move parameter, or
    (0.0, "") if no move of this rollout ever landed.

    The evidence is the tool result the environment writes back -- `hit` true and a `took` reading
    `"<axis>.<op>.<code>-><state>"` -- which is the same evidence `edge_prediction` uses for the
    destination reading, so "moved" cannot mean one thing to one term and another to the next.

    Why it is a named function. 926 of 2,597 training rollouts of one run (35.7%) CALLED
    `cw_move` and were REFUSED ("not a legal move from here"): the model picks a medically sensible
    chapter and fabricates the whole id, destination hash included, after having been shown the
    legal list. Under the judge prompt at temperature 0.6 the same clone fabricates 8.5% of the
    time; training runs at 1.0. So a rollout that "did not move" is almost never an abstention, and
    the refusal leaves it in the 1000+ entry cohort whose published rate is often closer to the
    truth than any reachable neighbour's -- which the outcome term then PAYS. Measured on that run
    without a move payment: mean advantage of a rejected-move rollout in a mixed group +0.0096, i.e.
    GRPO was being taught to write invalid move ids. `cohort_think_cal` therefore pays `T3_CAL_W_MOVE`
    only for an ACCEPTED move (and can floor a rollout that never moved -- swept, and left disabled);
    see the T3_CAL_* docstring.
    """
    if trace is None:
        return 0.0, ""
    for _i, role, _name, content, _calls in _walk(trace):
        if role != "tool":
            continue
        d = _json(content)
        if d and d.get("hit") and isinstance(d.get("took"), str) and "->" in d["took"]:
            try:
                return 1.0, d["took"].split("->", 1)[0].split(".", 2)[2]
            except Exception:
                return 1.0, ""
    return 0.0, ""


@lru_cache(maxsize=None)
def rank_reference(version: str = "internal") -> tuple:
    """A FIXED sample of (truth, base_rate) pairs from the internal T3 TRAIN items, seeded. It is
    the reference landscape a rollout's prediction is ranked against. Built from ITEMS at first use
    so it follows `CW_ITEMS`; its content hash is recorded by CONSTANTS() so a run says which one."""
    import random
    from .internal import ITEMS as _ITEMS
    rows = []
    with open(f"{_ITEMS}/t3_rate_train.jsonl") as f:
        for line in f:
            d = json.loads(line)
            y = float(d["answer"]["rate"])
            b = base_rate("internal", d["ask"]["disease"], int(d["ask"]["horizon"]))
            rows.append((y, b))
    rng = random.Random(T3_THINK_REF_SEED)
    rng.shuffle(rows)
    return tuple(rows[:T3_THINK_REF_N])


def rank_reference_hash() -> str:
    return hashlib.sha256(json.dumps(rank_reference(), sort_keys=True).encode()).hexdigest()[:16]


def rank_contribution(point: float, truth: float) -> float:
    """This rollout's share of a cross-item ranking: over the reference items whose truth differs
    from this one's, the fraction on which the prediction sits on the SAME SIDE of the reference
    truth as this item's truth does. That is "does p have the percentile the truth has among real
    outcomes" -- the per-rollout form of a cross-item AUROC. Returned CENTRED: a coin flip pays 0,
    p == truth pays the maximum, and it is monotone in |rank(p) - rank(truth)|.

    Compared against reference TRUTHS, not reference base rates: the first draft compared p to
    other items' base rates, which is asymmetric -- on a high-truth item the extreme p = 1.0 beat
    p = truth (0.448 vs 0.407 centred), because 1.0 is above every base rate. A ranking term that
    prefers an extreme to the truth would train the wrong thing.
    """
    hit = tot = 0
    for y_j, _b_j in rank_reference():
        if y_j == truth:
            continue
        tot += 1
        if (point > y_j) == (truth > y_j):
            hit += 1
    return (hit / tot - 0.5) if tot else 0.0


@lru_cache(maxsize=None)
def cal_l2_scale() -> float:
    """S2 for `cohort_think_cal`: `T3_CAL_L2_SCALE_MULT` times the pinned base-rate forecaster's
    MAE over the internal T3 TRAIN items. Read from ITEMS at first use so it follows `CW_ITEMS`;
    recorded by CONSTANTS(). With MULT = 2 the gradient of (p-y)^2 / S2, i.e. 2|p-y| / S2, equals
    the L1 term's gradient (1) exactly at |p-y| = MAE_b: L1-parity at the base-rate forecaster's
    error, so the existing weight T3_THINK_W_RANK keeps its "visible, never dominant" calibration.
    Official items (items_md5 12d8d1c574520e819ea43b1e05a4e85b): MAE_b 0.035877, S2 0.071754."""
    from .internal import ITEMS as _ITEMS
    tot = n = 0
    with open(f"{_ITEMS}/t3_rate_train.jsonl") as f:
        for line in f:
            d = json.loads(line)
            b = base_rate("internal", d["ask"]["disease"], int(d["ask"]["horizon"]))
            tot += abs(b - float(d["answer"]["rate"])); n += 1
    if not n:
        raise ValueError(f"{_ITEMS}/t3_rate_train.jsonl is empty; cal_l2_scale has no items to derive from")
    return T3_CAL_L2_SCALE_MULT * tot / n


@lru_cache(maxsize=None)
def delta_reference(version: str = "internal") -> tuple:
    """A FIXED, POLICY-INDEPENDENT sample of the delta a declaration has to predict.

    Anchored on the reference cohort, not the demographic entry. The episode starts on the item's
    own cohort with its rate withheld and moves toward a wider reference class, so "every legal
    one-hop edge out of the ENTRY cohort, destination rate minus entry rate" would describe a journey
    nobody takes and an entry rate nobody can read. What a declaration must predict is the correction
    from the reference class back to the
    patient: `answer_rate - rate(reference_cohort)`, one value per item, which is exactly what
    `edge_prediction(..., truth=...)` measures on a rollout.

    Still from the ITEM SET and the frozen lattice, never from a run's rollouts: a reference derived
    from a policy would move when the policy moves and the constant it pins would stop being one.
    `T3_CAL_DELTA_REF_N` items drawn with `T3_THINK_REF_SEED`; items whose reference cohort publishes
    nothing for the asked cell are skipped, because there is no delta to declare there either.
    Built at first use (like `rank_reference`), hashed into CONSTANTS().

    One value per item, not one per edge, so the sample is ~N rather than ~34N. That is the
    correct denominator: the quantity is a property of the ITEM under a deterministic reference, not
    of a menu of moves.
    """
    import random
    from .internal import ITEMS as _ITEMS
    from .lattice import _shared
    with open(f"{_ITEMS}/t3_rate_train.jsonl") as f:
        rows = [json.loads(line) for line in f]
    rng = random.Random(T3_THINK_REF_SEED)
    rng.shuffle(rows)
    out = []
    for d in rows[:T3_CAL_DELTA_REF_N]:
        lat = _shared(d["version"])
        dis, hor = d["ask"]["disease"], int(d["ask"]["horizon"])
        tc = d.get("target_cohort") or ""
        if not tc:
            continue
        r0 = _published_rate(lat, reference_cohort(d["version"], tc), dis, hor)
        if r0 is None:
            continue
        out.append(float(d["answer_rate"]) - r0)
    if not out:
        raise ValueError(f"no item with a published reference cohort over {T3_CAL_DELTA_REF_N} of "
                         f"{_ITEMS}/t3_rate_train.jsonl; cal_delta_const has no reference")
    return tuple(out)


def delta_reference_hash() -> str:
    return hashlib.sha256(json.dumps(delta_reference(), sort_keys=True).encode()).hexdigest()[:16]


def delta_accuracy(declared: float, measured: float) -> float:
    """`exp(-|d - m| / max(|m|, s0))` in (0, 1]: how close a declared delta came, normalised by the
    size of the delta it was predicting, so the same RELATIVE error scores the same at every size
    and no payment can be bought by choosing a big-|m| move. Below s0 the tolerance is s0, not |m|,
    so a tiny edge does not make a tiny error fatal."""
    return math.exp(-abs(declared - measured) / max(abs(measured), T3_THINK_EDGE_SCALE))


@lru_cache(maxsize=None)
def cal_delta_const() -> float:
    """c*: the BEST CONSTANT declaration on `delta_reference()`, i.e. argmax over
    `T3_CAL_DELTA_GRID` (plus exactly 0) of the mean `delta_accuracy(c, m)`.

    It is what `cohort_think_cal`'s delta payment is centred on, so that by construction the mean
    centred accuracy of ANY constant is <= 0 -- no template that ignores the item can be the term's
    maximiser, and only a declaration CONDITIONED on the item's five-axis difference earns. Derived,
    lru_cached and hash-bound exactly like `rank_reference()` and `cal_l2_scale()`; recorded as
    `CONSTANTS()["t3_cal_delta_const"]`. On the official items it is 0.0.
    """
    lo, hi, step = T3_CAL_DELTA_GRID
    grid = sorted({0.0} | {round(lo + i * step, 10) for i in range(int(round((hi - lo) / step)) + 1)})
    ms = delta_reference()
    return max(grid, key=lambda c: sum(delta_accuracy(c, m) for m in ms))


"""The `tanh` variant: `+ GAMMA * tanh((median - b) * (y - b) / SCALE**2)`, a BOUNDED covariance term
that pays for a report that departs from the base rate on the truth's side by an amount that grows
with the truth's own departure -- the per-item form of the board's slope, which rises only when
predictions spread in proportion to the truth. SCALE is the population sd of `answer_rate - b` over
the internal T3 TRAIN items (16,770 rows: 0.051209; the external set's is 0.077), so the
term saturates for departures of about one sd. Offline (31,765 external items): flattest shrink-toward-b curve of the proper-score family, best 2-parameter response raises
slope 0.384 -> 0.494 with AUROC held (0.605 -> 0.607), and the response that keeps reading the entry
cohort raises both (0.406 / 0.638). Known blind spot: ~0 in low-base-rate cells."""

NEAR = 0.25
"""T1's partial credit at distance 1: `S = NEAR / (1 + dist)` for a non-exact position.

Inherited from the measured 0.25/0.75 split of the old `locate` rubric. The task the environment
exists for outputs a rate, and a cohort one hop from the target reads literally zero on the asked
cell 69.7% of the time, so being exactly right has to carry the bulk of the score.
"""

MIN_REPORT = 5
"""Values a STATED report must carry, or it is rejected and the ladder falls through.

This is the guard against the worst exploit found in the environment: under the fair
`n*(n-1)` CRPS estimator, `commit([0.0, 1.0])` scores CRPS **exactly 0.0 for every truth in [0, 1]**
-- maximum reward, zero reads, and a zero-gradient attractor once a group finds it. The tool
signature accepts any length and an n=2 commit already occurs in archived data, so this is not
hypothetical. Five values is also what every prompt asks for, and it caps one free extreme sample
(reward-spec 5.4) at 20% of the reported mass.

Cost, measured on all 594 archived commits carrying rates: 552 already state exactly five, and 34
state one to four (10 at m=1, 3 at m=2, 9 at m=3, 12 at m=4). Those 34 fall to the ladder's next
rung rather than being scored, which is a 5.7% format cost. The free lunch is exactly and only at
m = 2 -- the best worst-case CRPS over truths in [0, 1] is 0.00000 at m=2, 0.16667 at m=3 and m=4,
0.20000 at m=5 -- so a lower floor would be defensible on that evidence alone; 5 is kept because it
is the number the prompt asks for and reward-spec 6.1 pins.

Do NOT "fix" the estimator back to `n*n` instead. That would make spreading pay everywhere rather
than only in the tail, which is the bias `crps` documents.
"""

T2_THRESH_LOW, T2_THRESH_HIGH = 0.25, 0.75
"""Where the median of a T2 report is read as "lower" / "higher". Inherited from `DeltaTask`
unchanged, so the old scalar contract is the special case of five identical values."""

T2_PRIOR_CLASS = "indistinguishable"
"""T2's reference decision: ALWAYS HEDGE. Not the majority class -- see below.

The second term of T2's `R` is constant per item and changes no gradient; it exists so that T2 is
referenced to a trivial answer the way T3 is referenced to the base rate and T1 to the entry cohort.

The docstring here used to justify the value as "the train-split majority (0.3823 of 429 items)",
and that justification is wrong twice over. `DeltaTaskset.load` serves only the 125 of those 429
items whose edge lattice and outcome granularity agree, and on the SERVED train set the majority is
`down` (46/125, 0.368), not `indistinguishable` (43/125, 0.344). More importantly the majority class
is not what this constant is for: the trivial T2 answer under this reward is the HEDGE, because the
ladder's floor rung is `[0.5]` and `t2_decision([0.5])` is `indistinguishable`. Referencing to the
hedge is what makes "did nothing" score 0.0 on R; referencing to whichever class happened to be
commonest would not. The value is unchanged; only the reason for it is now the true one.

`test_t2_reference_class_is_the_hedge_the_floor_produces` pins both halves, and the served class
balance MUST still be published beside any T2 number -- a three-way label with an unbalanced prior
flatters any constant answer.

Note also that "a reward of 0.0 means no better than the trivial answer" is exact only on T1, whose
reward is a difference of two scores. On T2 the hedge policy scores +0.672 and on T3 quoting the base
rate scores +0.9488, because both carry the `1 - CRPS` term on top of `R`. What is referenced to zero
on all three tasks is `R`, not the reward.
"""

FLOOR_REPORT = (0.0, 0.25, 0.5, 0.75, 1.0)
"""The uninformative report on [0, 1]. The last rung of the T3 ladder, reached only when the agent
neither stated an answer nor read the asked cell even once."""

BUDGET = 26
"""Tool calls a rollout may spend.

Was hardcoded 12 in five `out_of_budget` stops while the toolset advertised 20, so the stop fired
eight calls early and the resolved config disagreed with the behaviour. 12 is also below the job: the
measured minimum distance from a patient to any released cohort has median 17 operations, and 63.2%
of patients sit 12 or more away, so a 12-call cap made most items unreachable by construction rather
than by the agent's choice. Scores under a 12-call cap are not comparable with scores under 26,
which is why it is pinned HERE and imported everywhere rather than written four times.
"""

GRACE = 3
"""Turns left to answer after navigation is refused; see `CohortWorldToolset._exhausted`."""

PER_AXIS = 6
"""Moves listed per `(axis, op)` group by `legal_moves.summary`; the rest are counted, not shown.

Pinned HERE rather than left as `summary`'s default argument, which is where it used to live. A
default argument is not a contract: it appeared in no run's `config.yaml`, in no
`CONSTANTS()` call and in §2 of `docs/ENVIRONMENT_SPEC.md`, so someone could change 6 to 3 and every score
stayed formally comparable while the achievable policy set had moved -- which §8 calls MAJOR.

Why it changes the policy set and not just the token count: truncation is not uniform over the axes.
Across all 38 archived trace files, 980 of 28,599 served `(axis, op)` groups were truncated, hiding
13,179 legal moves, and 785 of those 980 were `bnf.add` (17.1% of that group against 0.0% for ten of
the fourteen groups). 84% of the truncated listings were at an `n_band` of `1000+`, i.e. the generic
entry cohorts every T1/T3 episode starts on, and the miss rate after a truncated listing is 45.5%
against 30.7% after a complete one. **There is no action that reveals the rest**, so a hidden move is
a move the agent cannot name.

The VALUE is unchanged at 6, so no archived score is invalidated by naming it. 6 and not 12 because
entry cohorts carry a median 32 legal moves: the full listing put prompt tokens at a p99 of 25,924
against a 32,768 window and 1.4% of turns hit the generation cap mid-sentence. 6 and not 0 because
"no truncation" is what it once silently was while the docstring described six.
"""


def CONSTANTS() -> dict:
    """The contract, as a run directory records it in `config.yaml`.

    A run that cannot say which constants produced its numbers is a run whose numbers cannot be
    compared to anything.

    `items` is here because item-set work is only safely outside the code contract if the item files
    are pinned and hashed by it. Hashing `official/`, `prompts/` and the eval TOMLs alone never
    covers the item files, several sibling item builds can exist side by side, and `CW_ITEMS` is an
    environment variable -- so pointing a run at a different item set would leave no trace anywhere in
    the run directory. Naming the root here is what lets `newrun.sh` hash its nine files.

    `semantic_hash` and the two panel roots are here for the same reason the numbers are. A run that
    recorded ALPHA but not which lattice it read was pinning the cheaper half: `dist` and
    `entry_cohort` silently returned a different T1 score when a panel file was absent (measured: 7
    of 12 archived T1 rollouts moved, max |delta| 0.1225, and it moved UPWARD -- a missing input made
    the agent look better). They RAISE now, and the root they raise about is recorded here.
    """
    from .lattice import PANEL, PRIV
    from .internal import ITEMS
    from cohortworld_phenocal.taskset import PHENOCAL, SIDECAR
    return {"env_version": ENV_VERSION, "semantic_hash": semantic_hash(), "items": ITEMS,
            "phenocal": PHENOCAL, "sidecar": SIDECAR,
            "reward_spec_version": REWARD_SPEC_VERSION, "alpha": ALPHA, "near": NEAR,
            "min_report": MIN_REPORT, "t2_thresholds": [T2_THRESH_LOW, T2_THRESH_HIGH],
            "t2_prior_class": T2_PRIOR_CLASS, "floor_report": list(FLOOR_REPORT),
            "budget": BUDGET, "grace": GRACE, "per_axis": PER_AXIS,
            "panel": PANEL, "priv": PRIV,
            "reference_rates": os.path.basename(_reference_path()),
            "t3_default_variant": DEFAULT_T3_VARIANT, "t3_think_w_rank": T3_THINK_W_RANK,
            "t3_think_w_edge": T3_THINK_W_EDGE, "t3_think_edge_scale": T3_THINK_EDGE_SCALE,
            "t3_think_ref_n": T3_THINK_REF_N, "t3_think_ref_seed": T3_THINK_REF_SEED,
            "t3_think_ref_sha16": rank_reference_hash(), "t3_step_beta": T3_STEP_BETA,
            "t3_tanh_gamma": T3_TANH_GAMMA, "t3_tanh_scale": T3_TANH_SCALE,
            "t3_cal_l2_scale": cal_l2_scale(), "t3_cal_l2_scale_mult": T3_CAL_L2_SCALE_MULT,
            "t3_cal_q2_cap": T3_CAL_Q2_CAP, "t3_cal_w_edge": T3_CAL_W_EDGE,
            "t3_cal_w_delta": T3_CAL_W_DELTA, "t3_cal_delta_ref_n": T3_CAL_DELTA_REF_N,
            "t3_cal_delta_grid": list(T3_CAL_DELTA_GRID), "t3_cal_delta_const": cal_delta_const(),
            "t3_cal_delta_ref_sha16": delta_reference_hash(),
            "t3_cal_nomove_floor": T3_CAL_NOMOVE_FLOOR, "t3_cal_w_move": T3_CAL_W_MOVE}


# --------------------------------------------------------------------------- version <-> semantics

_SEMANTIC_MODULES = (
    "cohortworld_env/cohortworld_env/reward.py",
    "cohortworld_env/cohortworld_env/lattice.py",
    "cohortworld_env/cohortworld_env/legal_moves.py",
    "cohortworld_env/cohortworld_env/taskset.py",
    "cohortworld_env/cohortworld_env/toolset.py",
    "cohortworld_env/cohortworld_env/internal.py",
    "cohortworld_phenocal/cohortworld_phenocal/taskset.py",
)
"""Every module whose code can change a reward. `cohortworld_patient` is deliberately absent -- it is
not one of the three frozen tasks (docs/ENVIRONMENT_SPEC.md) and nothing in the frozen surface imports
it -- and so is `gen/`, which builds items and reads runs but is never on the scoring path."""

_UNHASHED = ("ENV_VERSION", "VERSIONED_SEMANTICS")
"""The two names the hash cannot contain, or no version could ever agree with its own code."""


def _semantic_source() -> str:
    """The frozen surface with everything a reader can change freely removed.

    `ast.unparse` of the parsed module drops comments and normalises formatting, and the walk below
    drops every docstring and every bare string statement -- the attribute docstrings under each
    constant included. What survives is the code: the constants' VALUES, every branch, every
    comparison, every literal a reward passes through.

    Bound to the CPython version that unparses it (3.12 here, recorded in every run's `env.txt`). A
    Python upgrade may change the normalised text without changing the semantics; that shows up as
    the "reward changed" branch of the test message, and the fix is to re-record the hash after
    confirming the diff is empty, not to weaken the check.
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return "\n".join(f"### {rel}\n" + _normalise(open(f"{root}/{rel}").read())
                     for rel in _SEMANTIC_MODULES)


def _normalise(src: str) -> str:
    """One module's source with comments, docstrings and formatting taken out. See above."""
    import ast

    def strip(node):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if not isinstance(body, list):
                continue
            keep = []
            for st in body:
                if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) \
                        and isinstance(st.value.value, str):
                    continue                       # a docstring, or a constant's attribute docstring
                if isinstance(st, ast.Assign) and len(st.targets) == 1 \
                        and isinstance(st.targets[0], ast.Name) and st.targets[0].id in _UNHASHED:
                    continue
                keep.append(strip(st))
            setattr(node, field, keep)
        return node

    return ast.unparse(strip(ast.parse(src)))


def semantic_hash() -> str:
    """A content hash of the frozen surface. `VERSIONED_SEMANTICS` says which version it must be."""
    return hashlib.sha256(_semantic_source().encode()).hexdigest()[:16]


VERSIONED_SEMANTICS = {"7.6.3": "7c8817fb92465033"}
"""ENV_VERSION -> the hash of the code that version names.

`semantic_hash()` is computed on the modules in `_SEMANTIC_MODULES` with comments and docstrings
removed, so it changes exactly when reward-relevant code changes. `tests/test_reward.py` fails if the
code no longer hashes to the value registered for `ENV_VERSION`: a change to the reward then needs a
new version and a new entry here, and a version bump with no code change is refused as well.
"""


# Report provenance, recorded per episode as the `report_level` metric. Numeric because
# `Trace.record_metric` casts to float; the names are the ones reward-spec 6.6 requires published
# beside any score.
STATED_COMMIT, STATED_TEXT, AUTO_OBSERVED, FLOOR = 1.0, 2.0, 3.0, 4.0
LEVEL_NAMES = {1.0: "stated_commit", 2.0: "stated_text", 3.0: "auto_observed", 4.0: "floor"}


# --------------------------------------------------------------------------- the scoring rule

def crps(report, truth: float) -> float:
    """Empirical CRPS of a sample report against a scalar truth. LOWER is better; 0.0 is exact.

    `mean|X - y| - 0.5 * mean|X - X'|`, with the second term computed FAIRLY: the double sum is
    divided by n*(n-1), not n*n. Dividing by n^2 leaves the n diagonal zeros in the average and
    shrinks the subtracted term by (n-1)/n -- 20% at the five values these tasks ask for -- which
    pays a forecaster for spreading its samples. That biased estimator is what produced a spurious
    +0.3032 trivial-policy floor.
    """
    xs = [float(x) for x in report]
    if not xs:
        raise ValueError("crps() called with an empty report; the ladder in this module exists so "
                         "that cannot happen -- an empty report means a caller bypassed it")
    n = len(xs)
    term1 = sum(abs(x - truth) for x in xs) / n
    if n == 1:
        return term1
    term2 = sum(abs(a - b) for a in xs for b in xs) / (n * (n - 1))
    return term1 - 0.5 * term2


def median(xs) -> float:
    s = sorted(float(x) for x in xs)
    m = len(s)
    return s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])


# --------------------------------------------------------------------------- reading the trace

# The leading-dot alternation is not cosmetic: without it `.05` parsed as `5.0` and `.7e-2` as
# `0.07`, i.e. a silent 100x scale error on a value the model wrote correctly. Three of 3,421
# archived answer strings hit it, and all three parse RIGHT under this pattern; the rest are
# byte-identical, which is what makes the change safe to apply retroactively.
#
# The trailing `%` group is the SAME 100x class, in the other direction, and it is the one unit
# marker this project honours. `commit` accepts a string precisely so a payload FastMCP used to
# reject reaches the trace, and its own docstring cites `'[0., .6%, .8%, .8%, .8%]'` as one of the
# two it rescues -- which parsed as `[0.0, 0.6, 0.8, 0.8, 0.8]`, i.e. a model that wrote 0.8% scored
# as a confident report of 80%, `well_formed`, `report_level = stated_commit`, no `malformed` flag.
# `well_formed`'s [0, 1] rule catches `[2%, ..., 6%]` but by construction CANNOT catch a percent
# convention already below 1, and `%` is a natural token for a probability, so under RL that channel
# pays. Dividing by 100 here is not a "clever percent heuristic" of the kind `parse_values` removed:
# it does not GUESS a convention from the magnitudes, it reads the unit the model wrote. A number with no `%` after it is
# never rescaled, which is why `[1, 2, 3, 4, 5]` is still rejected whole rather than read as 1-5%.
_NUM_RE = re.compile(r"(-?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(%?)")
_ID_RE = re.compile(r"\b((?:C1|C2|default):[0-9a-f]{16})\b")


def messages(trace):
    """(role, name, content) for every message on the trace, oldest first.

    Walks `trace.messages`, never `trace.tool_messages`. Both derive from `trace.nodes`, but the
    tool-message view is what the `committed` stop used and that stop fired on 6-16 of 120 rollouts
    while 94-111 actually committed -- verifiers commits a tool result as a node only when the NEXT
    model request arrives (`Session.handle_tool` holds it in `prepared_tool_results`), so anything
    reading it inside a rollout is one turn behind. At scoring time the two agree; keeping one
    accessor means the scoring path and the archived-trace rescoring path are literally the same code.
    """
    for m in (getattr(trace, "messages", None) or []):
        get = (lambda k: m.get(k)) if isinstance(m, dict) else (lambda k: getattr(m, k, None))
        yield get("role"), get("name"), get("content")


def _json(content):
    """The dict a tool result carries, or None. Tool results are JSON text on the wire."""
    if isinstance(content, dict):
        return content
    if not isinstance(content, str) or "{" not in content:
        return None
    try:
        return json.loads(content[content.index("{"):content.rindex("}") + 1])
    except Exception:
        return None


def tool_results(trace):
    """(name, dict) for every tool result that parsed, oldest first."""
    for role, name, content in messages(trace):
        if role != "tool":
            continue
        d = _json(content)
        if d is not None:
            yield (name or ""), d


def commit_calls(trace):
    """The arguments of every assistant `cw_commit` tool call, oldest first.

    Needed because the commit's RESULT is usually not on the trace at all -- see `commit_payload`.
    """
    for m in (getattr(trace, "messages", None) or []):
        get = (lambda k: m.get(k)) if isinstance(m, dict) else (lambda k: getattr(m, k, None))
        if get("role") != "assistant":
            continue
        for call in (get("tool_calls") or []):
            cg = (lambda k: call.get(k)) if isinstance(call, dict) else (
                lambda k: getattr(call, k, None))
            if "commit" not in (cg("name") or ""):
                continue
            args = cg("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = None
            yield args if isinstance(args, dict) else {}


def commit_payload(trace) -> dict | None:
    """What the agent committed: the `cw_commit` tool RESULT, or the tool CALL when there is none.

    Read off the environment's own record rather than the reply text: the agent never types a cohort
    id (a cohort is somewhere you walk to), and reading the call removes the whole class of parsing
    failures around a model restating ids while it reasons.

    **The fallback is not defensive coding, it is the normal path.** Since the `committed` stop
    started firing on the ASSISTANT's call (reading the result made it fire a turn late, or
    never), the episode ends before verifiers promotes that call's result to a trace node: v1 holds
    it in `Session.handle_tool`'s `prepared_tool_results` until the NEXT model request, and after a
    stop there is no next request. So a healthy committed rollout has the commit in its message list
    and no commit result anywhere.

    Measured, on the first live smoke after the stop fix (T3 internal, 12 rollouts): 11 stopped `committed`, 0 had a commit result on the trace, and **all 12 were scored
    at `auto_observed` -- every stated report in the run was discarded and scored on the rung below.**
    `report_malformed` stayed 0, because a payload that is not there is not malformed, so nothing in
    the summary said anything was wrong. No archive replay could have found this: every archived run
    predates the stop fix and therefore still has its commit results.

    Reconstructing `cohort_id` as `last_state_id` is faithful rather than a guess: `commit()` returns
    `self.lat.current`, and the last `state_id` any tool result printed IS the lattice's current
    position -- the toolset has no other way to move.
    """
    out = None
    for name, d in tool_results(trace):
        if d.get("committed") is True:
            out = d
    if out is not None:
        return out
    args = None
    for args in commit_calls(trace):
        pass
    if args is None:
        return None
    return {"committed": True, "cohort_id": last_state_id(trace), "rates": args.get("rates")}


def readings(trace) -> list[dict]:
    """Every published `cw_observe` result, tagged with where the agent was standing at the time.

    `cw_observe` does not echo the position, so the position is carried forward from the most recent
    `cw_look` / `cw_move` result. T2's auto report needs two readings at DIFFERENT cohorts, and
    without this it would compare a cohort with itself.
    """
    here, out = None, []
    for name, d in tool_results(trace):
        if d.get("state_id"):
            here = d["state_id"]
        if d.get("status") == "published" and isinstance(d.get("rate"), (int, float)):
            out.append({"state_id": here, "disease": d.get("disease"),
                        "horizon": d.get("horizon"), "rate": float(d["rate"])})
    return out


def last_state_id(trace) -> str:
    """The last cohort id that appeared in any tool result -- where the agent ended up."""
    sid = ""
    for _name, d in tool_results(trace):
        if d.get("state_id"):
            sid = d["state_id"]
        elif d.get("cohort_id"):
            sid = d["cohort_id"]
    return sid


def last_reply(trace) -> str:
    reply = getattr(trace, "last_reply", None)
    if reply:
        return reply
    for role, _name, content in reversed(list(messages(trace))):
        if role == "assistant" and content:
            return content
    return ""


def answer_line(text: str) -> str | None:
    """The `ANSWER:` line the prompt asks for.

    Reading that line rather than the whole reply keeps a worked example from earlier in the model's
    reasoning out of the commitment. Falling back to the whole reply scraped every number in [0, 1]
    out of the chain-of-thought and scored it as a forecast: in one archived run two rollouts that never committed
    still scored 0.959 and 0.794 that way, one off 75 numbers including 1.0, 0.95 and 0.4 -- mean
    absolute error 88x the true rate, and CRPS made it worse rather than catching it, because the
    dispersion term refunds spread. Spraying numbers while thinking paid better than answering.
    """
    for line in reversed((text or "").splitlines()):
        if "ANSWER:" in line.upper():
            return line.split(":", 1)[1]
    return None


# --------------------------------------------------------------------------- the one parse rule

def parse_values(text) -> list[float]:
    """Every number a stated answer contains, in order, WITHOUT rescaling or filtering.

    Deliberately does no range work of its own: judging the values is `well_formed`'s job and there
    is exactly one such judge in the project. The `(1, 100] -> /100` branch that used to live here is
    GONE -- it disagreed with the commit path, which silently dropped the same values, and that
    disagreement is the drift the module docstring describes.
    """
    # A BARE NUMBER IS A VALID THING FOR THE MODEL TO SEND, and before this it crashed the scorer.
    # `cw_commit`'s schema is `anyOf[array of number, string, null]` -- no bare number -- but prime-rl
    # sets VLLM_ENFORCE_STRICT_TOOL_CALLING=0 (inference/server.py:27) so the schema is advertised and
    # not enforced, and `cw_commit(rates=0.05)` arrives as a float. `_NUM_RE.findall(0.05)` then raises
    # `TypeError: expected string or bytes-like object, got 'float'`, which verifiers wraps as
    # `TaskError: task RateTask scoring` and which DISCARDS THE WHOLE TRACE.
    #
    # It was latent until the policy started acting: a run with 93.9% of rollouts making no tool call
    # at all hit it ~never; a later run, with the credit finally reaching the action turn, lost 293
    # traces to it in one step -- 31.9% Error against ~1%. Nothing archived depended on the old behaviour,
    # because the old behaviour was an exception and an exception produces no score.
    #
    # Read, do not reject: the number the agent committed is the claim it made, whatever container it
    # arrived in. Filtering and range-checking stay `well_formed`'s job, which is the one judge.
    if isinstance(text, bool):
        return []                         # True is not a rate; it would otherwise reach the regex
    if isinstance(text, dict):
        # an already-parsed object, not its JSON text. The branch below handles the same shapes after
        # `json.loads`; a dict arriving directly used to fall through to the regex and raise. Found by
        # the "nothing raises on any container a tool call could carry" assertion, not by thinking of it.
        for k in ("samples", "quantiles", "values", "distribution"):
            if isinstance(text.get(k), (list, tuple)):
                return [float(x) for x in text[k] if isinstance(x, (int, float))
                        and not isinstance(x, bool)]
        return []
    if isinstance(text, (int, float)):
        return [float(text)]
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        try:
            return [float(x) for x in text]
        except (TypeError, ValueError):
            return []
    vals: list[float] = []
    try:
        if "{" in text and "}" in text:
            obj = json.loads(text[text.index("{"):text.rindex("}") + 1])
        elif "[" in text and "]" in text:
            obj = json.loads(text[text.index("["):text.rindex("]") + 1])
        else:
            obj = None
        if isinstance(obj, dict):
            for k in ("samples", "quantiles", "values", "distribution"):
                if isinstance(obj.get(k), list):
                    vals = [float(x) for x in obj[k]]
                    break
        elif isinstance(obj, list):
            vals = [float(x) for x in obj]
    except Exception:
        pass
    return vals or [float(n) / 100.0 if pct else float(n)
                    for n, pct in _NUM_RE.findall(text)]


def well_formed(vals) -> bool:
    """One rule, no heuristic: at least `MIN_REPORT` values, every one of them inside [0, 1].

    Returns a single boolean for two causes on purpose -- "the agent stated something this contract
    cannot score" is one fact from the reward's point of view, and `report_malformed` is that fact.
    Which cause it was is recoverable from the trace and from the `m` metric; splitting it into two
    reward-visible signals would be a knob with nothing behind it.

    A report containing ANY value outside [0, 1] is malformed AS A WHOLE and is rejected entirely --
    no dropping, no clamping, no percent rescale. Grounds, from all 1741 archived commits carrying
    rates: 20 (1.15%) were out of range and 18 of those 20 are mixed-scale runaways
    (`[0.005, 0.006, 0.007, 1.0, 2.0]`, `[0.001, 0.0013, 0.002, 0.03, 7.0]`) that a percent heuristic
    gets wrong. Only 2 are a clean percent convention, and the prompt already says "write 2% as 0.02".

    Dropping the offending values -- what the commit path used to do -- is not neutral: it discards
    exactly the values that would have hurt, scoring those 20 at mean CRPS 0.1462 against 0.2633 for
    clamping, which makes a malformed report CHEAPER than an honest wide one.

    The length floor is `MIN_REPORT`; see its docstring for the n=2 exploit it closes.
    """
    return len(vals) >= MIN_REPORT and all(0.0 <= float(v) <= 1.0 for v in vals)


def parse_report(text) -> list[float]:
    """The stated report, or [] if there is not a well-formed one. THE extraction rule, one copy."""
    vals = parse_values(text)
    return vals if well_formed(vals) else []


# --------------------------------------------------------------------------- the always-emitted report

def rate_report(trace, disease: str, horizon: int) -> tuple[list, float, float]:
    """(values, level, malformed) for a T3/PhenoCal rate question. Never returns an empty report.

    The ladder, first rule that yields one:
      1 stated_commit  the last `cw_commit`'s `rates`, if well-formed
      2 stated_text    the `ANSWER:` line of the final assistant message, if well-formed
      3 auto_observed  the last published `cw_observe` of the ASKED (disease, horizon), as a point
                       mass -- a number the environment printed and the agent read
      4 floor          `FLOOR_REPORT`

    Nothing is gained by failing to state a report: for every level the agent could have stated the
    level below it verbatim, so stated weakly dominates auto and auto weakly dominates floor. On the
    archived internal arms that is what the data shows -- B and C beat their own auto level, A is
    level with it. And the ladder's next rung is deliberately the agent's OWN LAST READING rather
    than a well-chosen default: on the 20 malformed archived payloads a good default scores 0.0264,
    i.e. BETTER than honest handling, so a default would make malforming the report a prize.

    No level references a stop condition. Budget exhaustion, generation truncation, context overflow,
    provider error and "the agent simply stopped talking" all resolve down the same ladder. There is
    no `NO_ANSWER` branch and no episode whose score is independent of what the agent did.
    """
    malformed = 0.0
    payload = commit_payload(trace)
    if payload is not None and payload.get("rates") is not None:
        vals = parse_values(payload["rates"])
        if well_formed(vals):
            return vals, STATED_COMMIT, malformed
        malformed = 1.0

    stated = answer_line(last_reply(trace))
    if stated is not None:
        vals = parse_values(stated)
        if well_formed(vals):
            return vals, STATED_TEXT, malformed
        if vals:
            malformed = 1.0

    hits = [r for r in readings(trace)
            if r["disease"] == disease and r["horizon"] == horizon]
    if hits:
        return [hits[-1]["rate"]], AUTO_OBSERVED, malformed
    return list(FLOOR_REPORT), FLOOR, malformed


def locate_report(trace, entry: str) -> tuple[str, float]:
    """(cohort_id, level) for a T1 question. `entry` is where `setup_task` placed the agent.

    There is no stated-text level: an id cannot be typed. `cw_move` only accepts ids the environment
    advertised and `cw_commit` reports where you are standing, so the answer is a position and the
    auto level -- the last cohort any tool result named -- is the same kind of object as the stated
    one. That is also why there is no self-quote attack to gate against.
    """
    payload = commit_payload(trace)
    if payload and payload.get("cohort_id"):
        return payload["cohort_id"], STATED_COMMIT
    sid = last_state_id(trace)
    if sid:
        return sid, AUTO_OBSERVED
    return entry, FLOOR


_T2_Y = {"down": 0.0, "indistinguishable": 0.5, "up": 1.0}


def delta_report(trace, disease: str, horizon: int) -> tuple[list, float, float]:
    """(values, level, malformed) for a T2 direction question.

    Same report channel and same scoring rule as T3 -- a list of values in [0, 1] read as a
    distribution over the ordered effect direction (0 lower, 0.5 indistinguishable, 1 higher). Do not
    add a second report format and do not change the `cw_commit` signature; the scalar contract
    `DeltaTask` used to have is the special case of five identical values.

    The auto level is the measurement the prompt actually prescribes: the sign of the difference
    between the two most recent readings of the asked cell taken at DIFFERENT cohorts. An agent that
    did the experiment and then failed to state a conclusion is scored on the experiment.
    """
    malformed = 0.0
    payload = commit_payload(trace)
    if payload is not None and payload.get("rates") is not None:
        vals = parse_values(payload["rates"])
        if well_formed(vals):
            return vals, STATED_COMMIT, malformed
        malformed = 1.0

    stated = answer_line(last_reply(trace))
    if stated is not None:
        vals = parse_values(stated)
        if well_formed(vals):
            return vals, STATED_TEXT, malformed
        if vals:
            malformed = 1.0

    hits = [r for r in readings(trace) if r["disease"] == disease and r["horizon"] == horizon]
    seen, pair = set(), []
    for r in reversed(hits):                       # newest first, one reading per distinct cohort
        if r["state_id"] not in seen:
            seen.add(r["state_id"])
            pair.append(r)
        if len(pair) == 2:
            break
    if len(pair) == 2:
        later, earlier = pair[0], pair[1]
        d = later["rate"] - earlier["rate"]
        return [1.0 if d > 0 else 0.0 if d < 0 else 0.5], AUTO_OBSERVED, malformed
    return [0.5], FLOOR, malformed


def t2_decision(vals) -> str:
    c = median(vals)
    return "down" if c < T2_THRESH_LOW else "up" if c > T2_THRESH_HIGH else "indistinguishable"


# --------------------------------------------------------------------------- T1 geometry

def _axset(state: dict) -> frozenset:
    """A cohort as the six-element axis SET the prompt describes: age, sex, coarseness, and one
    element per (axis, code, onset rung) entry.

    `coarseness` MUST be in the set. Without it two cohorts with identical axes recorded at different
    granularity read as distance 0 from each other, and `dist == 0` would stop meaning "same cohort".
    """
    ax = state["axes"]
    out = {("age_band", str(ax["age_band"])), ("sex", str(ax["sex"])),
           ("coarseness", str(state["coarseness"]))}
    for k in ("icd", "opcs", "bnf"):
        for entry in ax.get(k) or []:
            code, stage = (list(entry) + [None])[:2]
            out.add((k, str(code), str(stage)))
    return frozenset(out)


def dist(a: str, b: str) -> int:
    """How many recorded features separate two cohorts: `|A xor B|` over the axis sets.

    Replaces the old `_hops` BFS, which was not a distance at all. The lattice's edges are strictly
    DIRECTED -- 466 of 466 sampled same-version edges have no reverse edge -- so `_hops` measured
    reachability, and it was capped at 3 for cost, returning None for 24 of the 34 non-exact commits
    on the one archived T1 run. 71% of near-misses therefore collapsed onto a single 0.0, which is
    the same constant this whole contract exists to remove; the `cap=3` rung was dead on top of that,
    since `0.25 * (1 - 3/3)` is exactly what silence scored. Partial credit was also asymmetric --
    one hop BEFORE the target paid 0.167, one hop PAST it paid 0.0.

    `dist` is always defined, needs no search, and is what the prompt asks for in the first place:
    "the one whose recorded history differs from his own on the fewest features".
    """
    if a == b:
        return 0
    da, db = _released(a), _released(b)
    if da is None or db is None:
        # An id that names no released cohort cannot be measured against one that does. Scoring it as
        # "far" rather than raising keeps one malformed commit from voiding a whole eval.
        #
        # 99 and not 0: `NEAR / (1 + 99)` is 0.0025, a hundredth of the 0.25 a genuine one-hop miss
        # earns. A sentinel of 0 would pay a garbage id the FULL exact-match credit, so the number is
        # load-bearing rather than arbitrary -- `test_an_unresolvable_id_is_scored_as_far_not_near`.
        return 99
    return len(_axset(da) ^ _axset(db))


def _released(sid: str) -> dict | None:
    """The released cohort this id names, None if it names none -- and RAISES if the lattice that
    would answer is not on disk.

    The distinction is the whole point and it was not being made. `lattice._shared` swallows
    `FileNotFoundError` and returns None, which is correct for a ROLLOUT (a partially built lattice
    degrades to fewer resolutions instead of killing the episode) and wrong for a SCORE: `dist` then
    returned its "no such cohort" sentinel for every id in that version and `entry_cohort` returned
    `""`, so T1 was scored against a lattice that was not there. Measured by deleting one
    `CohortWorld_default_*_states.jsonl` and rescoring an archived T1 run: **7 of 12 rollouts changed,
    max |delta| 0.1225, and every change was UPWARD** -- a missing input made the agent look better,
    with no exception, no warning and an unchanged `report_level`. A reward that depends on which
    files happen to exist is not a reward, and nothing in the run directory recorded the panel root
    until `CONSTANTS()` started pinning it.
    """
    from .lattice import COARSER, PANEL, _shared
    ver = sid.split(":", 1)[0] if ":" in sid else ""
    if ver not in COARSER:
        return None                       # not an id any CohortWorld lattice ever issued
    lat = _shared(ver)
    if lat is None:
        raise FileNotFoundError(
            f"scoring {sid!r} needs the {ver} lattice and it is not in {PANEL}. Set CW_PANEL, or "
            f"rebuild it; do NOT score against a missing panel -- it silently pays the agent more")
    return lat.states.get(sid)


def locate_score(sid: str, target: str) -> float:
    """`S` of reward-spec 3.2: 1.0 for the target, `NEAR / (1 + dist)` for anything else."""
    if sid == target:
        return 1.0
    return NEAR / (1.0 + dist(sid, target))


@lru_cache(maxsize=4096)
def entry_cohort(version: str, age_band: str, sex: str) -> str:
    """Where `setup_task` places the agent for this item, recomputed without moving anything.

    T1's reward is `S(final) - S(entry)`, so the entry position is part of the contract and has to be
    derivable at scoring time. Deterministic given (version, age_band, sex), and cached because 120
    items x four scoring hooks would otherwise rescan 7,113 states 480 times.
    """
    from .lattice import PANEL, _shared
    lat = _shared(version)
    if lat is None:
        # Loud for the same reason `_released` is: `S(entry)` is half of T1's reward, and returning
        # "" here scored the subtraction against a cohort that does not exist.
        raise FileNotFoundError(
            f"T1 scoring needs the {version} lattice for its entry cohort and it is not in {PANEL}")
    return lat.entry_for(age_band, sex)


@lru_cache(maxsize=4096)
def reference_cohort(version: str, target: str) -> str:
    """The cohort a T3 episode is meant to WIDEN TO, recomputed without moving anything.

    A T3 episode starts ON the item's own cohort with its rate withheld, so the demographic entry,
    `entry_cohort`, is a position no T3 agent occupies,
    publishing a number no T3 agent can read. Every reward term that meant "the free reading this
    episode gets" has to move with the geometry or it is scoring a different environment. This is
    where it moves to: the destination of one `drop`, the only generalising move there is.

    WHICH destination: the largest published size band, ties to the MOST SPECIFIC cohort, then by id.
    Largest first because the point of widening is a reading worth believing -- the item's own cohort
    is in the 10-49 band for 2,565 of the 2,597 items of one training run -- and most-specific among equals because a
    reference class that keeps more of the patient's history is a closer match. Deterministic given
    (version, target): a reference derived from what a POLICY actually chose would move when the
    policy moves, which is the same objection that keeps `delta_reference` off the rollouts.

    Falls back to `target` itself when there is no code to drop; `_published_rate` then returns that
    cohort's own rate, which is the honest answer to "what can be read from here" for a cohort with
    no history to give up. On the released lattice no coded cohort is in that position (C1 5,601 and
    C2 7,095 coded cohorts, none without a drop), so the branch exists for an item set that places an
    item on a demographic cohort.
    """
    from .lattice import PANEL, _band_lo, _shared
    lat = _shared(version)
    if lat is None:
        raise FileNotFoundError(
            f"T3 scoring needs the {version} lattice for its reference cohort and it is not in "
            f"{PANEL}. Refusing to score against a missing panel; see `_released`.")
    if target not in lat.states:
        return target
    # `_drop_moves`, not `_served_drops`: the shared lattice carries no episode's mask, and what is
    # wanted here is the RULE the agent is offered, not what one episode happened to be served.
    mv = lat._drop_moves(target)
    if not mv:
        return target

    def _rank(m):
        st = lat.states[m["to"]]
        return (_band_lo(st["n_band"]),
                sum(len(st["axes"].get(k) or []) for k in ("icd", "opcs", "bnf")),
                m["to"])
    return max(mv, key=_rank)["to"]


def _published_rate(lat, sid: str, disease: str, horizon: int):
    """The rate `cw_observe` publishes at cohort `sid` for `(disease, horizon)`, or None where it
    publishes nothing (unknown cohort, disease or horizon, or a not-estimable cell). Mirrors
    `Lattice.observe` without moving anything. One source of truth for the two callers that need a
    rate the agent could read: `entry_rate` (the entry cohort) and `delta_reference` (its
    neighbours); a second copy of the not-estimable rule is a rule that drifts."""
    d = lat.states.get(sid) if lat is not None else None
    if d is None or disease not in lat.didx or horizon not in lat.horizons:
        return None
    hi, di = lat.horizons.index(horizon), lat.didx[disease]
    ne = {int(k): v for k, v in (d.get("not_estimable") or {}).items()}
    v = ne.get(hi)
    if v == "all" or (v and di in v):
        return None
    return float(d["rates"][hi][di])


@lru_cache(maxsize=None)
def entry_rate(version: str, age_band: str, sex: str, disease: str, horizon: int,
               target_cohort: str = ""):
    """The rate this episode can read WITHOUT reasoning: the reference the `entry*` variants measure
    the report against. None where nothing is published (unknown disease or horizon, or a
    not-estimable cell).

    TWO GEOMETRIES, and the item says which one it is in:

      * `target_cohort` given -- an internal T3 item. The episode starts ON that cohort and its rate
        is WITHHELD, so the free reading is at `reference_cohort`, one `drop` away. The demographic
        entry's rate would describe a position the agent does not occupy and a number it cannot
        read.
      * `target_cohort` empty -- the external PhenoCal judge, and any item set without the field:
        the demographic entry, `entry_cohort`. The judge starts at the entry, so that is its
        reference, and its archived scores replay byte-identically.

    Keyed on the item's own field rather than on a version flag, so a rescoring of an OLD archive
    (whose task records carry no `target_cohort`) reproduces the number that archive was scored with,
    instead of silently re-anchoring history.
    """
    from .lattice import _shared
    sid = reference_cohort(version, target_cohort) if target_cohort \
        else entry_cohort(version, age_band, sex)
    return _published_rate(_shared(version), sid, disease, horizon)


# --------------------------------------------------------------------------- the reference forecast

def _reference_root() -> str:
    """Where the pinned reference table lives: `official/` of THIS checkout, or `CW_OFFICIAL`.

    One expression, two callers. `_reference_path` and `build_reference` each carried their own copy
    of the `dirname^3(__file__)` walk, which is how a repo grows two conventions for one directory;
    `lattice` and `legal_moves` had meanwhile hardcoded the same directory by name, so in a clone the
    reward's table and the lattice came from different repositories.
    """
    return os.environ.get("CW_OFFICIAL", os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "official"))


def _reference_path() -> str:
    """The one pinned reference-rate table for this spec version.

    A reference that is recomputed per run is a reference that drifts, and a skill score against a
    drifting reference is not a skill score. Built once by `python -m cohortworld_env.reward
    --build-reference`, hashed into its own filename, and read from disk here.
    """
    root = _reference_root()
    hits = sorted(glob.glob(os.path.join(root, f"reference_rates_{REWARD_SPEC_VERSION}_*.json")))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected exactly one reference_rates_{REWARD_SPEC_VERSION}_*.json in {root}, found "
            f"{len(hits)}. Build it with: python -m cohortworld_env.reward --build-reference")
    return hits[0]


_REF: dict = {}


def base_rate(source: str, disease: str, horizon: int) -> float:
    """The reference forecast `b` for one item. Defined for 100% of items, by a pinned cascade.

    `InternalTask.crps_ref_loss` used to return 0.0 when `(disease, horizon)` was missing from the
    train split. Survivable for a metric and fatal for a reward term: **104 of the 304 internal
    unseen items (34.2%) have no train base rate** -- `unseen` is defined by held-out outcome
    chapters, so the key CANNOT exist -- and 43 of those have no entry for the disease at any
    horizon. Worse, a fabricated 0.0 is indistinguishable from a genuinely perfect reference, so
    nothing downstream could filter it, and it halved the denominator of every skill score ever
    quoted: ~40% of items contributed a fabricated zero, inflating |skill| by about 2.4x.

    The cascade, in order: the (disease, horizon) train mean, then that disease at any horizon, then
    that horizon over all diseases, then the train grand mean.
    """
    if not _REF:
        _REF.update(json.load(open(_reference_path()))["sources"])
    t = _REF.get(source)
    if t is None:
        raise KeyError(f"{source!r} is not in {_reference_path()}; rebuild it")
    return float(t["pair"].get(f"{disease}|{horizon}",
                 t["disease"].get(disease,
                 t["horizon"].get(str(horizon), t["grand"]))))


def _fold(rows) -> dict:
    """(disease, horizon, rate) rows -> the four cascade levels."""
    pair, dis, hor, allv = {}, {}, {}, []
    for d, h, r in rows:
        pair.setdefault(f"{d}|{h}", []).append(r)
        dis.setdefault(d, []).append(r)
        hor.setdefault(str(h), []).append(r)
        allv.append(r)
    mean = lambda v: sum(v) / len(v)
    return {"pair": {k: mean(v) for k, v in pair.items()},
            "disease": {k: mean(v) for k, v in dis.items()},
            "horizon": {k: mean(v) for k, v in hor.items()},
            "grand": mean(allv) if allv else 0.0, "n": len(allv)}


def build_reference() -> str:
    """Write the pinned table. Run once per spec version; the hash goes in every run's inputs.txt."""
    from .internal import ITEMS
    from cohortworld_phenocal.taskset import PHENOCAL

    sources = {}
    rows = []
    for line in open(f"{ITEMS}/t3_rate_train.jsonl"):
        d = json.loads(line)
        rows.append((d["disease"], int(d["horizon"]), float(d["answer_rate"])))
    sources["internal"] = _fold(rows)

    for variant in ("PhenoCal", "PhenoCal_C1", "PhenoCal_C2"):
        path = f"{PHENOCAL}/{variant}_train.jsonl"
        if not os.path.exists(path):
            continue
        rows = []
        for line in open(path):
            d = json.loads(line)
            m = d["metadata"]
            rows.append((m["disease_icd10_3char"], int(m["horizon_years"]),
                         float(d["ground_truth"]["rate"])))
        sources[variant] = _fold(rows)

    body = json.dumps({"spec_version": REWARD_SPEC_VERSION, "sources": sources},
                      sort_keys=True, separators=(",", ":"))
    sha = hashlib.sha256(body.encode()).hexdigest()[:8]
    out = os.path.join(_reference_root(), f"reference_rates_{REWARD_SPEC_VERSION}_{sha}.json")
    with open(out, "w") as f:
        f.write(body)
    return out


# --------------------------------------------------------------------------- the three rewards

def rate_reward(trace, *, truth: float, source: str, disease: str, horizon: int,
                variant: str = DEFAULT_T3_VARIANT, **item) -> dict:
    """T3 / PhenoCal: `r = ALPHA * R + (1 - CRPS)`, with `R = |b - y| - |median(F) - y|`.

    `R` depends on the report only through its median, so `dR/d(dispersion) = 0` and the dispersion
    coordinate is elicited exactly as CRPS elicits it, for every alpha -- which is the one property
    the whole design exists to protect (reward-spec 2.2).

    `variant` picks a training-time form from `T3_REWARD_VARIANTS`; the default is this contract
    and the returned dict is unchanged except for one extra metric, `side`. `item` carries what a
    variant needs beyond the contract (`version`, `age_band`, `sex`, for the entry cohort); `base`
    ignores it. Not the PhenoCal `data.variant`, which names the dataset version and is passed
    here as `source`.
    """
    vals, level, malformed = rate_report(trace, disease, horizon)
    b = base_rate(source, disease, horizon)
    point = median(vals)
    loss = crps(vals, truth)
    R = abs(b - truth) - abs(point - truth)
    out = {"reward": ALPHA * R + (1.0 - loss), "crps_loss": loss, "R": R, "point": point,
           "base": b, "report_level": level, "report_malformed": malformed, "m": float(len(vals)),
           "side": 1.0 if (point - b) * (truth - b) > 0 else 0.0}
    form = T3_REWARD_VARIANTS[variant]          # KeyError names the unknown variant
    if form is not None:
        out.update(form(out, truth=truth, disease=disease, horizon=horizon, trace=trace, **item))
    return out


def _t3_entry(out: dict, *, truth: float, disease: str, horizon: int, version: str,
              age_band: str, sex: str, step: bool, target_cohort: str = "", **_) -> dict:
    """`R_ref = |y_entry - y| - |median - y|`: closer to the truth than the ENTRY cohort's own
    published rate, the number the agent reads without moving, instead of closer than the
    population base rate `b`. Where the entry cell publishes nothing, `y_entry` falls back to `b`
    and the term is the contract's `R`. With `step`, `+ T3_STEP_BETA * side` on top.

    INERT UNDER GRPO (found in an offline audit): `R_ref - R = |y_entry - y| - |b - y|`
    depends on the ITEM only, and prime-rl's advantage is `reward - group mean` over the 8 rollouts
    of one item, so this reference cancels exactly (max rollout-to-rollout difference 4.4e-16 on
    31,784 replayed items). `entry` trains identically to `base`; `entry_step` trains as
    `base + T3_STEP_BETA * side`. Kept so that recorded configs still resolve; do not choose it
    expecting a different gradient. `locate_reward`'s docstring states the same principle."""
    ref = entry_rate(version, age_band, sex, disease, horizon, target_cohort)
    if ref is None:
        ref = out["base"]
    R_ref = abs(ref - truth) - abs(out["point"] - truth)
    return {"ref": ref, "R_ref": R_ref,
            "reward": ALPHA * R_ref + (1.0 - out["crps_loss"])
            + (T3_STEP_BETA * out["side"] if step else 0.0)}


def _t3_point(out: dict, *, truth: float, **_) -> dict:
    """`r = ALPHA * R + (1 - |median - y|)`: the contract with the CRPS term replaced by the point
    absolute error. Training-time only. Closes the dispersion channel: under `base`, widening a
    report to [0, p, p, p, 1] at a fixed median gains about 0.4 * MAE (+0.025 on the replayed
    zero-shot traces), which is the whole realised reward gain of an RL checkpoint at step 200 -- a gradient
    direction that neither AUROC nor slope can see. Dispersion is no longer elicited here, which is
    why this is not the scored contract."""
    return {"reward": ALPHA * out["R"] + (1.0 - abs(out["point"] - truth))}


def _t3_tanh(out: dict, *, truth: float, **_) -> dict:
    """`r = ALPHA * R + (1 - CRPS) + T3_TANH_GAMMA * tanh((median - b) * (y - b) / T3_TANH_SCALE**2)`.
    Reads only the median, so `dR/d(dispersion) = 0` still holds and CRPS still elicits the spread.
    Varies ACROSS the rollouts of one item (the median moves), so GRPO sees it -- unlike a reference
    that depends on the item alone."""
    z = (out["point"] - out["base"]) * (truth - out["base"]) / (T3_TANH_SCALE ** 2)
    cov = T3_TANH_GAMMA * math.tanh(z)
    return {"cov": cov, "reward": out["reward"] + cov}


def _t3_point_tanh(out: dict, *, truth: float, **kw) -> dict:
    """`point` and `tanh` together: `ALPHA * R + (1 - |median - y|) + GAMMA * tanh(...)`.

    The two measured effects of the `point` and `tanh` runs, added: `point` was the only variant whose checkpoint beat the
    zero-shot AUROC (+0.038 at step 50, n=1,000, paired lower bound above zero) and it removes the dispersion
    channel that carried the entire realised reward gain measured in `_t3_point`; `tanh` produced the only significant slope gain
    (+0.049 on the both-stated subset at step 100) because its term pays for departing from the base rate in
    proportion to the truth's own departure. Their weaknesses are complementary -- `point` shrinks the report
    (slope down), `tanh` alone buys no ranking -- which is the reason to test the sum."""
    p = _t3_point(out, truth=truth)
    h = _t3_tanh(out, truth=truth, **kw)
    return {"cov": h["cov"], "reward": p["reward"] + h["cov"]}


def _t3_cohort_think(out: dict, *, truth: float, disease: str, horizon: int, version: str,
                     age_band: str, sex: str, trace=None, target_cohort: str = "", **_) -> dict:
    """`point_tanh + W_RANK * rank + W_EDGE * edge`. Nothing here pays for reproducing a template.

    * `rank`: this rollout's centred contribution to a cross-item AUROC against the fixed reference
      set (`rank_contribution`). Reads only the committed median, like both verdict metrics.
    * `edge`: the first move's delta, declared BEFORE the destination is read, scored as the
      improvement over declaring zero (see T3_THINK_EDGE_SCALE). 0 when there is no declaration, no
      move, or no reading of the asked cell on both sides.
    Diagnostics, unpaid: `think_moved` (a cw_move landed), `think_declared`, `think_edge_true`, and
    `think_relevant` (the code the first move added lies in the target cohort's code set -- i.e. the
    move parameter is the part of the patient's own five-axis history the lattice can express).
    `think_big` was dropped at the user's direction: preferring a large reference class is what a
    clinician does, not a shortcut; what must correlate with the question is the move parameter.
    """
    base = _t3_point_tanh(out, truth=truth, disease=disease, horizon=horizon,
                          version=version, age_band=age_band, sex=sex)
    rank = rank_contribution(float(out["point"]), float(truth))
    # `truth` only when the item names its own cohort -- i.e. only under the internal T3 geometry. An
    # archived record carries no `target_cohort`, so rescoring it reproduces its own number.
    ty = truth if target_cohort else None
    ep = edge_prediction(trace, disease, horizon, truth=ty) if trace is not None else \
        {"declared": None, "measured": None, "src": None, "dst": None, "declared_before_move": 0.0}
    edge = 0.0
    if ep["declared"] is not None and ep["measured"] is not None:
        s_ = T3_THINK_EDGE_SCALE
        edge = math.exp(-abs(ep["declared"] - ep["measured"]) / s_) - math.exp(-abs(ep["measured"]) / s_)
    relevant = 0.0
    # `took` reads "<axis>.<op>.<code>-><state>"; the code is the move parameter. One reader, in
    # `accepted_move`, because `cohort_think_cal` pays on the same fact.
    moved, moved_code = accepted_move(trace)
    return {"cov": base["cov"], "think_rank": rank, "think_edge": edge,
            "think_declared": ep["declared_before_move"], "think_moved": moved,
            "think_relevant": relevant, "think_moved_code": moved_code,
            "think_edge_true": ep["measured"] if ep["measured"] is not None else 0.0,
            "reward": base["reward"] + T3_THINK_W_RANK * rank + T3_THINK_W_EDGE * edge}


def _t3_cohort_think_cal(out: dict, *, truth: float, disease: str, horizon: int, version: str,
                         age_band: str, sex: str, trace=None, target_cohort: str = "", **_) -> dict:
    """`ALPHA*R2 + (1 - Q2) + W_RANK*rank + W_DELTA*delta_cal + W_MOVE`, or `T3_CAL_NOMOVE_FLOOR`
    for a rollout no move of which the environment ACCEPTED. See the T3_CAL_* docstring.

    Returns `cohort_think`'s full dict (its `cov`, `think_edge` and the diagnostics are computed by
    the `cohort_think` code and reported UNPAID, so `internal.RateTask`'s metrics and the gate
    diagnostics read the same keys) plus `cal_q2`, `cal_r2`, `cal_edge` (the gated `edge_cal`
    payment, UNPAID and kept as a diagnostic), `cal_delta` (the [0, 1) delta payment before its
    weight) and `cal_declared` (did this rollout declare before the move under `_SIGNED_NUM_V2`? `think_declared` is the
    v1 answer and is 5x lower for a parsing reason, see that constant's docstring); only `reward` is
    redefined.
    """
    ct = _t3_cohort_think(out, truth=truth, disease=disease, horizon=horizon, version=version,
                          age_band=age_band, sex=sex, trace=trace, target_cohort=target_cohort)
    ty = truth if target_cohort else None
    s2 = cal_l2_scale()
    q2 = min((float(out["point"]) - truth) ** 2 / s2, T3_CAL_Q2_CAP)
    r2 = min((float(out["base"]) - truth) ** 2 / s2, T3_CAL_Q2_CAP) - q2
    ep2 = edge_prediction(trace, disease, horizon, pattern=_SIGNED_NUM_V2, truth=ty) \
        if trace is not None else \
        {"declared": None, "measured": None, "declared_before_move": 0.0}
    dc = delta_payment(ep2["declared"], ep2["measured"])
    moved = float(ct["think_moved"])
    paid = (ALPHA * r2 + (1.0 - q2) + T3_THINK_W_RANK * ct["think_rank"]
            + T3_CAL_W_DELTA * dc + T3_CAL_W_MOVE * moved)
    return dict(ct, cal_q2=q2, cal_r2=r2, cal_edge=edge_cal(trace, disease, horizon, truth=ty),
                cal_delta=dc,
                cal_declared=ep2["declared_before_move"],
                reward=T3_CAL_NOMOVE_FLOOR if (not moved and T3_CAL_NOMOVE_FLOOR is not None) else paid)


def edge_cal(trace, disease: str, horizon: int, truth=None) -> float:
    """The gated edge payment in [0, 1] (see the T3_CAL_* docstring): sign-and-size gated,
    |m|-normalised, never negative. 0 without a trace, a declaration before the move, or a reading
    of the asked cell at a different cohort after it."""
    if trace is None:
        return 0.0
    ep = edge_prediction(trace, disease, horizon, truth=truth)
    d, m = ep["declared"], ep["measured"]
    if d is None or m is None or m == 0 or (d > 0) != (m > 0) or abs(d) < T3_THINK_EDGE_SCALE:
        return 0.0
    e1 = math.exp(-1)
    return max(0.0, math.exp(-abs(d - m) / max(abs(m), T3_THINK_EDGE_SCALE)) - e1) / (1.0 - e1)


def delta_payment(declared, measured) -> float:
    """`max(0, acc(d, m) - acc(c*, m))`: the whole of the delta payment, as a pure function of the
    two numbers, so the trace is parsed once by the caller that needs both this and the declaration
    flag. 0 when either is missing."""
    if declared is None or measured is None:
        return 0.0
    return max(0.0, delta_accuracy(declared, measured) - delta_accuracy(cal_delta_const(), measured))


def delta_cal(trace, disease: str, horizon: int, truth=None) -> float:
    """The DELTA payment: `max(0, acc(d, m) - acc(c*, m))`, the accuracy of the declared delta
    ABOVE what the pinned best constant `cal_delta_const()` would have scored on the same edge.

    Never negative, so declaring never costs reward. 0 without a trace, without a declaration made
    BEFORE the first move, without a reading of the asked cell at a DIFFERENT cohort after it, and
    for a number copied from an observation -- all four are `edge_prediction`'s rules, unchanged
    except that this payment reads the declaration with `_SIGNED_NUM_V2` (a sentence-final number counts).
    With c* = 0 it is also 0 for a wrong-sign declaration and on any edge that measured exactly 0,
    without a gate saying so; see the T3_CAL_* docstring for what the centring does and does not
    guarantee. No size gate: a CORRECT declaration smaller than s0 is paid.
    """
    if trace is None:
        return 0.0
    ep = edge_prediction(trace, disease, horizon, pattern=_SIGNED_NUM_V2, truth=truth)
    return delta_payment(ep["declared"], ep["measured"])


T3_REWARD_VARIANTS = {
    DEFAULT_T3_VARIANT: None,
    "entry": lambda out, **kw: _t3_entry(out, step=False, **kw),         # inert under GRPO, see _t3_entry
    "entry_step": lambda out, **kw: _t3_entry(out, step=True, **kw),     # trains as base + side term
    "point": lambda out, **kw: _t3_point(out, **kw),
    "tanh": lambda out, **kw: _t3_tanh(out, **kw),
    "point_tanh": lambda out, **kw: _t3_point_tanh(out, **kw),
    "cohort_think": lambda out, **kw: _t3_cohort_think(out, **kw),
    "cohort_think_cal": lambda out, **kw: _t3_cohort_think_cal(out, **kw),
}
"""name -> the form's extra terms, or None for the contract. `RateTaskConfig.reward` validates
against these keys, so a typo in a run config fails at load, not as a silently-default run.
A variant can only change training if it changes the reward DIFFERENCES between the rollouts of
one item; a term that depends on the item alone is a constant to GRPO's `reward - group mean`."""


def locate_reward(trace, *, target: str, version: str, age_band: str, sex: str) -> dict:
    """T1: `r = S(final) - S(entry)`. No proper-score term, because there is nothing to elicit.

    T1's answer is a POSITION and the target is deterministic given the item -- no outcome random
    variable, so no belief, no dispersion coordinate, and no properness to preserve. Adding a proper
    score here would be scoring a quantity that does not exist.

    Subtracting `S(entry)` is invisible to GRPO (within a group every rollout shares the item, hence
    the same constant) but it makes the contract readable: **a policy that never moves scores exactly
    0.0 on every item, by construction.**
    """
    entry = entry_cohort(version, age_band, sex)
    got, level = locate_report(trace, entry)
    d = dist(got, target)
    return {"reward": locate_score(got, target) - locate_score(entry, target),
            "exact": 1.0 if got == target else 0.0, "dist": float(d),
            "report_level": level, "report_malformed": 0.0}


def delta_reward(trace, *, direction: str, disease: str, horizon: int) -> dict:
    """T2: the same form as T3, against `y in {0, 0.5, 1}`, with `R` on the DECISION.

    T2 is the one task where the degenerate optimum Prop. 9 warns about is real and measurable here:
    expected `1 - CRPS` on the train class prior is 0.838 at accuracy q = 0, dips to 0.788 at chance,
    and does not recover past a policy at q = 0 until q ~= 0.75. A systematically-wrong-but-honest
    policy beats every policy below 75% accuracy under a pure proper score. `R` is a step function of
    the report, so `dR/d(report)` is zero almost everywhere and properness survives as it does on T3.
    """
    y = _T2_Y.get(direction)
    if y is None:
        raise ValueError(f"answer_direction must be one of {sorted(_T2_Y)}; got {direction!r}")
    vals, level, malformed = delta_report(trace, disease, horizon)
    said = t2_decision(vals)
    loss = crps(vals, y)
    R = (1.0 if said == direction else 0.0) - (1.0 if T2_PRIOR_CLASS == direction else 0.0)
    return {"reward": ALPHA * R + (1.0 - loss), "crps_loss": loss, "R": R,
            "correct": 1.0 if said == direction else 0.0, "report_level": level,
            "report_malformed": malformed, "m": float(len(vals))}


if __name__ == "__main__":
    import sys
    if "--build-reference" in sys.argv:
        print(build_reference())
    else:
        print(json.dumps(CONSTANTS(), indent=2))
