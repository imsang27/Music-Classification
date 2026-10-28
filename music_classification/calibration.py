"""Decision-threshold fitting, not posterior-probability calibration."""

import math
from fractions import Fraction


def _validate(truth, scores):
    if (not truth or len(truth) != len(scores) or any(y not in (0, 1) for y in truth)
            or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores)
            or set(truth) != {0, 1}):
        raise ValueError("Both binary classes and aligned finite unit scores required")


def fit_threshold(truth, scores, default=0.5):
    """Maximize balanced accuracy; ties nearest default, then lower threshold."""
    _validate(truth, scores)
    if not math.isfinite(default) or not 0 <= default <= 1:
        raise ValueError("Invalid default threshold")
    unique = sorted(set(scores))
    candidates = {0.0, math.nextafter(unique[-1], math.inf), default}
    candidates.update((a + b) / 2 for a, b in zip(unique, unique[1:]))
    positive, negative = sum(truth), len(truth) - sum(truth)
    def objective(threshold):
        tp = sum(y == 1 and s >= threshold for y, s in zip(truth, scores))
        tn = sum(y == 0 and s < threshold for y, s in zip(truth, scores))
        balanced = (Fraction(tp, positive) + Fraction(tn, negative)) / 2
        return balanced, -abs(threshold - default), -threshold
    return min(candidates, key=lambda t: tuple(-v for v in objective(t)))


def leave_one_out(truth, scores, default=0.5):
    """Each held-out decision uses a threshold fit without that observation."""
    _validate(truth, scores)
    if min(sum(truth), len(truth) - sum(truth)) < 2:
        raise ValueError("At least two examples of each class required for leave-one-out")
    thresholds, decisions = [], []
    for index, score in enumerate(scores):
        threshold = fit_threshold(truth[:index] + truth[index + 1:],
                                  scores[:index] + scores[index + 1:], default)
        thresholds.append(threshold)
        decisions.append(int(score >= threshold))
    return {"thresholds": thresholds, "decisions": decisions}
