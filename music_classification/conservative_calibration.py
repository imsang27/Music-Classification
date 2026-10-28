"""Bounded threshold candidates selected inside nested, artist-disjoint folds.

The current design supports one reviewed track per artist. Repeated artists are
rejected rather than silently split across training and validation.
"""

import math
import random
import statistics
from fractions import Fraction

CAPS = (0.0, 0.05, 0.10, 0.15)
SEEDS = (17, 29, 43)


def validate(rows):
    if (not rows or any(not {'track_id', 'artist_id', 'truth', 'score'} <= set(r) for r in rows)
            or len({r['track_id'] for r in rows}) != len(rows)
            or len({r['artist_id'] for r in rows}) != len(rows)
            or any(r['truth'] not in (0, 1) or not math.isfinite(r['score']) or not 0 <= r['score'] <= 1 for r in rows)
            or {r['truth'] for r in rows} != {0, 1}):
        raise ValueError('Distinct artists/tracks with finite binary observations required')


def balanced(rows, threshold):
    return sum(sum((r['score'] >= threshold) == bool(y) for r in rows if r['truth'] == y)
               / sum(r['truth'] == y for r in rows) for y in (0, 1)) / 2


def fit_bounded(rows, cap):
    validate(rows)
    if cap not in CAPS:
        raise ValueError('Unknown preregistered movement bound')
    low, high = .5 - cap, .5 + cap
    unique = sorted({r['score'] for r in rows})
    candidates = {low, .5, high}
    candidates.update(t for a, b in zip(unique, unique[1:]) if low <= (t := (a + b) / 2) <= high)
    def key(t):
        rates = [Fraction(sum((r['score'] >= t) == bool(y) for r in rows if r['truth'] == y),
                          sum(r['truth'] == y for r in rows)) for y in (0, 1)]
        return (sum(rates) / 2, -abs(t - .5), -t)
    return max(candidates, key=key)


def folds(rows, count, seed):
    validate(rows)
    if count < 2 or min(sum(r['truth'] == y for r in rows) for y in (0, 1)) < count:
        raise ValueError('Too few observations for stratified folds')
    result = [[] for _ in range(count)]
    rng = random.Random(seed)
    for y in (0, 1):
        indices = sorted((i for i, r in enumerate(rows) if r['truth'] == y), key=lambda i: rows[i]['track_id'])
        rng.shuffle(indices)
        for at, index in enumerate(indices):
            result[at % count].append(index)
    return result


def select_candidate(rows, seed):
    inner = folds(rows, 3, seed)
    evidence = []
    for cap in CAPS:
        values = []
        for held in inner:
            train = [r for i, r in enumerate(rows) if i not in held]
            test = [rows[i] for i in held]
            threshold = fit_bounded(train, cap)
            values.append(balanced(test, threshold))
        evidence.append({'cap': cap, 'fold_balanced_accuracy': values, 'mean': statistics.mean(values),
                         'standard_error_heuristic': statistics.stdev(values) / math.sqrt(len(values))})
    best = max(evidence, key=lambda r: (r['mean'], -r['cap']))
    # One-standard-error simplicity rule + a fixed two-point improvement guard.
    cutoff = best['mean'] - best['standard_error_heuristic']
    chosen = min((r for r in evidence if r['mean'] >= cutoff - 1e-12), key=lambda r: r['cap'])
    if chosen['mean'] < evidence[0]['mean'] + .02 - 1e-12:
        chosen = evidence[0]
    return {'cap': chosen['cap'], 'threshold': fit_bounded(rows, chosen['cap']),
            'inner_evidence': evidence, 'inner_validation_ids': [[rows[i]['track_id'] for i in h] for h in inner]}


def nested_validate(rows):
    validate(rows)
    repeats = []
    for seed in SEEDS:
        records = []
        for index, held in enumerate(folds(rows, 4, seed)):
            train = [r for i, r in enumerate(rows) if i not in held]
            chosen = select_candidate(train, seed * 100 + index)
            records.append({'fold': index, 'train_ids': [r['track_id'] for r in train],
                            'validation_ids': [rows[i]['track_id'] for i in held], **chosen,
                            'decisions': [{'track_id': rows[i]['track_id'], 'truth': rows[i]['truth'],
                                           'score': rows[i]['score'], 'baseline': int(rows[i]['score'] >= .5),
                                           'selected': int(rows[i]['score'] >= chosen['threshold'])} for i in held]})
        decisions = [r for fold in records for r in fold['decisions']]
        def metric(key):
            return sum(sum(r[key] == y for r in decisions if r['truth'] == y)
                       / sum(r['truth'] == y for r in decisions) for y in (0, 1)) / 2
        repeats.append({'seed': seed, 'baseline_balanced_accuracy': metric('baseline'),
                        'nested_balanced_accuracy': metric('selected'), 'folds': records})
    return {'repeats': repeats,
            'mean_nested_balanced_accuracy': statistics.mean(r['nested_balanced_accuracy'] for r in repeats),
            'baseline_balanced_accuracy': repeats[0]['baseline_balanced_accuracy'],
            'development_candidate': select_candidate(rows, 20261025),
            'independent_evaluation': False, 'default_override_enabled': False}
