"""Descriptive diagnosis of fixed predictions; does not select new thresholds."""

import math
import statistics


def score_summary(values):
    if not values or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Finite nonempty unit scores required')
    ordered = sorted(values)
    def quantile(q):
        at = q * (len(ordered) - 1)
        lo = int(at)
        return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (at - lo)
    return {'count': len(values), 'min': ordered[0], 'q25': quantile(.25),
            'median': statistics.median(ordered), 'q75': quantile(.75), 'max': ordered[-1]}


def analyze_axis(rows, axis, threshold):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid fixed threshold')
    if len({r['track_id'] for r in rows}) != len(rows) or any(r['status'] != 'ok' for r in rows):
        raise ValueError('Unique successful predictions required')
    positive = axis.removeprefix('mood_')
    cases = []
    for row in rows:
        if axis not in row['annotations']:
            continue
        label = row['annotations'][axis]
        if label not in {positive, 'non_' + positive, 'not_' + positive}:
            raise ValueError('Unknown native reference label')
        score = row['scores'][axis]
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid score')
        truth = label == positive
        old, new = score >= .5, score >= threshold
        old_correct, new_correct = old == truth, new == truth
        kind = ('persistent_correct' if old_correct else 'rescued') if new_correct else (
            'worsened' if old_correct else 'persistent_error')
        cases.append({'track_id': row['track_id'], 'truth': int(truth), 'score': score,
                      'baseline_decision': int(old), 'fixed_decision': int(new),
                      'baseline_outcome': ('tp' if truth else 'fp') if old else ('fn' if truth else 'tn'),
                      'fixed_outcome': ('tp' if truth else 'fp') if new else ('fn' if truth else 'tn'),
                      'transition': kind})
    positives = [r['score'] for r in cases if r['truth']]
    negatives = [r['score'] for r in cases if not r['truth']]
    if not positives or not negatives:
        raise ValueError('Both reference classes required')
    counts = {kind: sum(c['transition'] == kind for c in cases) for kind in
              ['persistent_correct', 'rescued', 'worsened', 'persistent_error']}
    positive_delta = sum((c['fixed_decision'] - c['baseline_decision']) for c in cases if c['truth']) / len(positives) / 2
    negative_delta = sum((c['baseline_decision'] - c['fixed_decision']) for c in cases if not c['truth']) / len(negatives) / 2
    pairs = [(p, n) for p in positives for n in negatives]
    reversed_pairs = sum(p < n for p, n in pairs)
    tied_pairs = sum(p == n for p, n in pairs)
    return {'reviewed_tracks': len(cases), 'excluded_missing_consensus': len(rows) - len(cases),
            'threshold': threshold, 'positive_scores': score_summary(positives),
            'negative_scores': score_summary(negatives), 'transitions': counts,
            'balanced_accuracy_delta_decomposition': {'positive_recall_contribution': positive_delta,
                                                      'negative_recall_contribution': negative_delta,
                                                      'total': positive_delta + negative_delta},
            'ranking': {'positive_negative_pairs': len(pairs), 'reversed_pairs': reversed_pairs,
                        'tied_pairs': tied_pairs, 'roc_auc': 1 - (reversed_pairs + .5 * tied_pairs) / len(pairs)},
            'changed_score_band': {'lower_inclusive': min(.5, threshold), 'upper_exclusive': max(.5, threshold),
                                   'positive_tracks': sum(c['truth'] and c['baseline_decision'] != c['fixed_decision'] for c in cases),
                                   'negative_tracks': sum(not c['truth'] and c['baseline_decision'] != c['fixed_decision'] for c in cases)},
            'cases': cases}
