"""The rate task as a binary decision: is this stratum's rate ABOVE its (disease, horizon) base rate?

Why binarise at all. The rate target is itself an estimate -- a cohort of 10-49 people gives rates
like 1/12 -- and the base-rate reference carries a relative standard error with a median of 15.7%.
That noise sits in the TARGET, so it does not shrink as items are added, and every magnitude-based
score (CRPS, BSS, squared error) pays for it. A direction call is far less sensitive to it: getting
1/12 when the truth is 2/15 is a large relative error and the same answer to "above or below?".

Why the threshold is the per-(disease, horizon) base rate and not a global one. A global median
splits 50/50 and looks statistically tidy, but the question it asks is mostly "is this disease
common?", which a model can answer from prior knowledge without reading anything about the patient.
Against the per-cell base rate the disease's own prevalence is divided out, so what is left is the
question this benchmark exists for: **does THIS history raise or lower the risk relative to people
with the same disease and horizon?** Measured on PhenoCal_C2 test (31,784 items) that split is
36.1% positive, so the majority-class baseline is 63.9% and must be quoted beside every accuracy.

Two things to report honestly with it:
  * **AUROC uses the predicted rate as a score**, not the thresholded class, so the ranking
    information in the prediction is not thrown away.
  * **12% of items sit within 10% of their threshold**, where the label is decided by the target's
    own sampling noise. Accuracy is therefore also reported on the subset outside that band; a
    method that only beats the baseline inside it has learned nothing.
"""
import json, math, random, statistics as st


def auroc(scores, labels):
    """Rank-based AUROC with ties handled by average rank (Mann-Whitney U)."""
    pairs = sorted(zip(scores, labels))
    ranks, i = {}, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    pos = [ranks[k] for k, (_s, y) in enumerate(pairs) if y == 1]
    npos, nneg = len(pos), len(pairs) - len(pos)
    if npos == 0 or nneg == 0:
        return float("nan")
    return (sum(pos) - npos * (npos + 1) / 2.0) / (npos * nneg)


def evaluate(items, seed=42, B=2000, margin=0.10):
    """items: (predicted_rate, true_rate, threshold). Returns the full binary report."""
    items = [(p, t, f) for p, t, f in items if f is not None and f > 0]
    y = [1 if t > f else 0 for _p, t, f in items]
    yhat = [1 if p > f else 0 for p, _t, f in items]
    s = [p / f for p, _t, f in items]              # score = prediction relative to its own threshold
    n = len(items)
    acc = sum(a == b for a, b in zip(y, yhat)) / n
    base = max(sum(y), n - sum(y)) / n
    tp = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, yhat) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 0)
    tn = n - tp - fp - fn
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    bal = (sens + spec) / 2 if not math.isnan(sens) and not math.isnan(spec) else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    # F1 as 2tp/(2tp+fp+fn), not 2*prec*rec/(prec+rec). The second is 0/0 -- and so nan -- whenever
    # tp==0, which is a real result (the model predicted no positive, or every positive it predicted
    # was wrong) and has an F1 of exactly 0. It is nan only when there is no positive at all, in
    # neither the labels nor the predictions. The bootstrap below already used this form, so the
    # point estimate and its own CI used to disagree: f1 nan beside f1_ci (0.0, 0.0).
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan")
    au = auroc(s, y)

    far = [(p, t, f) for p, t, f in items if abs(t - f) / f >= margin]
    yf = [1 if t > f else 0 for _p, t, f in far]
    yhf = [1 if p > f else 0 for p, _t, f in far]
    acc_far = (sum(a == b for a, b in zip(yf, yhf)) / len(far)) if far else float("nan")
    base_far = (max(sum(yf), len(yf) - sum(yf)) / len(yf)) if far else float("nan")

    def _f1(yy, hh):
        tp_ = sum(1 for a, b in zip(yy, hh) if a == 1 and b == 1)
        fp_ = sum(1 for a, b in zip(yy, hh) if a == 0 and b == 1)
        fn_ = sum(1 for a, b in zip(yy, hh) if a == 1 and b == 0)
        return (2 * tp_ / (2 * tp_ + fp_ + fn_)) if (2 * tp_ + fp_ + fn_) else float("nan")

    rng = random.Random(seed)
    accs, aus, f1s = [], [], []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        bs = [items[k] for k in idx]
        by = [1 if t > f else 0 for _p, t, f in bs]
        bh = [1 if p > f else 0 for p, _t, f in bs]
        accs.append(sum(a == b for a, b in zip(by, bh)) / n)
        aus.append(auroc([p / f for p, _t, f in bs], by))
        f1s.append(_f1(by, bh))
    accs.sort()
    aus = sorted(a for a in aus if not math.isnan(a))
    # F1 gets a CI for the same reason ACC and AUROC do: a leaderboard column without one is a
    # point estimate a reader will difference between rows. nan resamples (no positive predicted
    # AND no positive label) are dropped, exactly as the AUROC line above drops its undefined ones.
    f1s = sorted(v for v in f1s if not math.isnan(v))
    return {"n": n, "positive_share": sum(y) / n, "accuracy": acc, "majority_baseline": base,
            "acc_ci": (accs[int(B * .025)], accs[int(B * .975)]),
            "sensitivity": sens, "specificity": spec, "balanced_accuracy": bal,
            "precision": prec, "f1": f1,
            "f1_ci": ((f1s[int(len(f1s) * .025)], f1s[int(len(f1s) * .975)]) if f1s
                      else (float("nan"), float("nan"))),
            "auroc": au, "auroc_ci": (aus[int(len(aus) * .025)], aus[int(len(aus) * .975)]),
            "n_far": len(far), "accuracy_off_threshold": acc_far, "baseline_off_threshold": base_far}


def show(rep, label):
    print(f"{label}: n={rep['n']:,}  正类 {rep['positive_share']*100:.1f}%")
    print(f"    准确率 {rep['accuracy']*100:5.1f}%  CI [{rep['acc_ci'][0]*100:.1f}, {rep['acc_ci'][1]*100:.1f}]"
          f"   多数类基线 {rep['majority_baseline']*100:.1f}%")
    print(f"    AUROC  {rep['auroc']:.4f}  CI [{rep['auroc_ci'][0]:.4f}, {rep['auroc_ci'][1]:.4f}]"
          f"   平衡准确率 {rep['balanced_accuracy']*100:.1f}%"
          f"  (敏感度 {rep['sensitivity']*100:.0f}% / 特异度 {rep['specificity']*100:.0f}%)")
    print(f"    离阈值>10% 的子集: 准确率 {rep['accuracy_off_threshold']*100:5.1f}%"
          f"   基线 {rep['baseline_off_threshold']*100:.1f}%   n={rep['n_far']:,}")
