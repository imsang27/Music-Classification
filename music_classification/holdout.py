"""Evaluate already fixed decisions; never fit to holdout references."""

import math
import random

from music_classification.benchmark import binary_classification_metrics


def compare_fixed_thresholds(truth, scores, threshold, *, bootstrap_samples=2000, seed=20261025):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid fixed threshold")
    baseline = binary_classification_metrics(truth, scores, threshold=0.5)
    fixed = binary_classification_metrics(truth, scores, threshold=threshold)
    if baseline['balanced_accuracy'] is None:
        raise ValueError("Both reference classes required")
    if bootstrap_samples < 100:
        raise ValueError("Insufficient bootstrap samples")
    if any(not 0 <= s <= 1 for s in scores):
        raise ValueError("Unit scores required")
    # Paired, class-stratified resampling preserves both classes in each sample.
    groups = [[i for i, y in enumerate(truth) if y == label] for label in (0, 1)]
    rng = random.Random(seed)
    deltas = []
    for _ in range(bootstrap_samples):
        delta = 0.0
        for label, group in enumerate(groups):
            drawn = rng.choices(group, k=len(group))
            delta += sum(int((scores[i] >= threshold) == bool(label))
                         - int((scores[i] >= 0.5) == bool(label)) for i in drawn) / len(group) / 2
        deltas.append(delta)
    deltas.sort()
    def percentile(q):
        at = q * (len(deltas) - 1)
        lo = int(at)
        return deltas[lo] + (deltas[min(lo + 1, len(deltas) - 1)] - deltas[lo]) * (at - lo)
    return {'baseline': baseline, 'fixed_calibration_threshold': fixed,
            'balanced_accuracy_delta': fixed['balanced_accuracy'] - baseline['balanced_accuracy'],
            'paired_stratified_bootstrap_delta_95_percentile': [percentile(.025), percentile(.975)],
            'bootstrap_samples': bootstrap_samples, 'seed': seed}


def evaluate_rows(rows, thresholds):
    if (not rows or len({r['track_id'] for r in rows}) != len(rows)
            or any(r['status'] != 'ok' for r in rows)):
        raise ValueError("Complete unique successful predictions required")
    result = {}
    for axis, config in thresholds.items():
        if config['positive'] != axis.removeprefix('mood_') or config['comparison'] != '>=':
            raise ValueError("Unsupported label or decision rule")
        positive = config['positive']
        reviewed = [r for r in rows if axis in r['annotations']]
        if any(r['annotations'][axis] not in {positive, 'not_' + positive, 'non_' + positive} for r in reviewed):
            raise ValueError("Unknown reference label")
        truth = [int(r['annotations'][axis] == positive) for r in reviewed]
        scores = [r['scores'][axis] for r in reviewed]
        result[axis] = {'reviewed_tracks': len(reviewed), 'excluded_missing_consensus': len(rows) - len(reviewed),
                        **compare_fixed_thresholds(truth, scores, config['threshold'])}
    return result
