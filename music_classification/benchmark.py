"""Public-tag benchmark utilities; label spaces remain dataset-native."""

import math
from pathlib import Path, PurePosixPath


def read_consensus_annotations(path):
    """Read only exact three-rater agreement; retain the dataset's native labels."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].split("\t") != ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "ANNOTATIONS"]:
        raise ValueError("Unexpected consensus annotation header")
    records = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.split("\t")
        if len(values) < 6 or values[0] in records:
            raise ValueError("Malformed or duplicate consensus track")
        annotations = {}
        for tag in values[5:]:
            namespace, ratings = tag.split("---", 1)
            ratings = ratings.split(",")
            if (len(ratings) != 3 or len(set(ratings)) != 1 or not ratings[0]
                    or ratings[0] == "unmatched" or namespace in annotations):
                raise ValueError("Annotation is not valid three-rater consensus")
            annotations[namespace] = ratings[0]
        records[values[0]] = {"track_id": values[0], "artist_id": values[1], "album_id": values[2],
                              "path": values[3], "duration": float(values[4]), "annotations": annotations}
    return records


def binary_classification_metrics(truth, scores, *, threshold=0.5):
    """Evaluate an explicit positive/negative reference, with no missing-label imputation."""
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, s = np.asarray(truth), np.asarray(scores, dtype=float)
    if (not len(y) or y.ndim != 1 or s.shape != y.shape or not np.isfinite(s).all()
            or not math.isfinite(threshold) or not np.isin(y, [0, 1]).all()):
        raise ValueError("Aligned binary references and finite scores are required")
    predicted = s >= threshold
    tp = int(((y == 1) & predicted).sum())
    tn = int(((y == 0) & ~predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    positive, negative = tp + fn, tn + fp
    both = positive > 0 and negative > 0
    sensitivity = tp / positive if positive else None
    specificity = tn / negative if negative else None
    return {"tracks": len(y), "positive_tracks": positive, "negative_tracks": negative,
            "threshold": threshold, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp + tn) / len(y),
            "balanced_accuracy": (sensitivity + specificity) / 2 if both else None,
            "positive_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "positive_recall": sensitivity, "negative_recall": specificity,
            "roc_auc": float(roc_auc_score(y, s)) if both else None,
            "average_precision": float(average_precision_score(y, s)) if both else None}


def read_jamendo_split(path):
    """Read variable-width tag columns without silently dropping additional tags."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].split("\t") != ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "TAGS"]:
        raise ValueError("Unexpected MTG-Jamendo header")
    records, seen = [], set()
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.split("\t")
        if len(values) < 6 or values[0] in seen:
            raise ValueError("Malformed or duplicate track")
        relative = PurePosixPath(values[3])
        duration = float(values[4])
        if (relative.is_absolute() or ".." in relative.parts or "\\" in values[3]
                or not math.isfinite(duration) or duration <= 0):
            raise ValueError("Invalid track path or duration")
        if any(not tag.startswith(("genre---", "instrument---", "mood/theme---")) for tag in values[5:]):
            raise ValueError("Unknown tag namespace")
        seen.add(values[0])
        records.append({"track_id": values[0], "artist_id": values[1], "album_id": values[2],
                        "path": values[3], "duration": duration, "tags": values[5:]})
    return records


def tag_ranking_metrics(references, scores, labels):
    """Observed-tag retrieval, not subjective correctness or calibrated accuracy.

    Rows without an observed positive in this label vocabulary are excluded and
    counted. Missing tags are treated as negatives for AP, matching conventional
    tag benchmarks; incomplete annotations limit the interpretation.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score

    if not labels or len(labels) != len(set(labels)) or len(references) != len(scores):
        raise ValueError("Unique labels and aligned references/scores required")
    matrix = np.asarray(scores, dtype=float)
    if matrix.shape != (len(references), len(labels)) or not np.isfinite(matrix).all():
        raise ValueError("Invalid score matrix")
    truth = np.asarray([[label in tags for label in labels] for tags in references], dtype=int)
    included = truth.sum(axis=1) > 0
    truth, matrix = truth[included], matrix[included]
    if not len(truth):
        return {"evaluated_tracks": 0, "excluded_without_target_tags": len(references),
                "hit_at_1": None, "hit_at_3": None, "recall_at_3": None,
                "macro_average_precision": None, "per_label": {}}
    order = np.argsort(-matrix, axis=1, kind="stable")
    ranked = np.take_along_axis(truth, order, axis=1)
    k = min(3, len(labels))
    per_label = {}
    for index, label in enumerate(labels):
        positive = int(truth[:, index].sum())
        valid = 0 < positive < len(truth)
        per_label[label] = {"positive_tracks": positive, "negative_tracks": len(truth) - positive,
                            "average_precision": float(average_precision_score(truth[:, index], matrix[:, index]))
                            if valid else None}
    ap = [row["average_precision"] for row in per_label.values() if row["average_precision"] is not None]
    most_frequent = int(np.argmax(truth.sum(axis=0)))
    return {"evaluated_tracks": len(truth), "excluded_without_target_tags": int((~included).sum()),
            "hit_at_1": float(ranked[:, 0].mean()),
            "hit_at_3": float((ranked[:, :k].sum(axis=1) > 0).mean()),
            "recall_at_3": float((ranked[:, :k].sum(axis=1) / truth.sum(axis=1)).mean()),
            "macro_average_precision": float(np.mean(ap)) if ap else None,
            "labels_with_positive_and_negative_examples": len(ap),
            "constant_most_frequent_label": labels[most_frequent],
            "constant_label_hit_at_1": float(truth[:, most_frequent].mean()), "per_label": per_label}
