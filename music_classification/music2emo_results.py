"""Validate native Music2Emo results without fabricating labels or probabilities."""

import math


def segment_ranges(total_samples, sample_rate, window_seconds=30):
    """Consecutive30s windows, merging a sub1s tail without dropping samples."""
    if (not isinstance(total_samples, int) or not isinstance(sample_rate, int) or sample_rate <= 0 or total_samples < sample_rate
            or not isinstance(window_seconds, int) or window_seconds < 1):
        raise ValueError('At least one second of audio required')
    size = window_seconds * sample_rate
    ranges = [(start, min(start + size, total_samples)) for start in range(0, total_samples, size)]
    if len(ranges) > 1 and ranges[-1][1] - ranges[-1][0] < sample_rate:
        ranges[-2] = (ranges[-2][0], total_samples)
        ranges.pop()
    return ranges


def validated_scores(result, expected_labels):
    if len(expected_labels) != 56 or len(set(expected_labels)) != 56:
        raise ValueError('Expected 56 unique native mood labels')
    scores = result.get('raw_mood_scores')
    if not isinstance(scores, dict) or set(scores) != set(expected_labels):
        raise ValueError('Missing or changed native mood scores')
    if any(not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
        raise ValueError('Invalid raw sigmoid score')
    if not {'happy', 'sad'} <= set(scores):
        raise ValueError('Native happy/sad outputs unavailable')
    va = [result.get('valence'), result.get('arousal')]
    if any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in va):
        raise ValueError('Invalid valence/arousal')
    return {'native_scores': {k: float(scores[k]) for k in expected_labels},
            'happy_score': float(scores['happy']), 'sad_score': float(scores['sad']),
            'valence': float(va[0]), 'arousal': float(va[1]),
            'va_outside_nominal_1_to_9': any(not 1 <= v <= 9 for v in va),
            'score_semantics': 'uncalibrated native sigmoid; not correctness probability'}
