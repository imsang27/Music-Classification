"""Reference-based metrics. Unreviewed annotations never become ground truth."""


def evaluate_records(references, predictions):
    """Compare normalized records using stable label IDs in the same taxonomy.

    Records contain track_id and axes. Reference axes have reviewed, primary,
    labels. Prediction axes have primary (ID or None), labels and status.
    Missing or failed predictions count as empty predictions, not omitted tracks.
    """
    def index(records):
        indexed = {}
        for row in records:
            key = row["track_id"]
            if key in indexed:
                raise ValueError(f"Duplicate track_id: {key}")
            indexed[key] = row
        return indexed

    refs, preds = index(references), index(predictions)
    report = {}
    for axis in ("genre", "emotion", "mood"):
        pairs, primary_pairs, available = [], [], 0
        for key, reference in refs.items():
            target = reference.get("axes", {}).get(axis, {})
            if target.get("reviewed") is not True:
                continue
            actual = preds.get(key, {}).get("axes", {}).get(axis, {})
            usable = actual.get("status") == "ok"
            expected = set(target.get("labels", []))
            supplied = set(actual.get("labels", [])) if usable else set()
            pairs.append((expected, supplied))
            available += usable
            if target.get("primary") is not None:
                primary_pairs.append((target["primary"], actual.get("primary") if usable else None))
        if not pairs:
            report[axis] = {"reviewed_tracks": 0, "coverage": None, "micro_f1": None,
                            "macro_f1": None, "sample_jaccard": None,
                            "primary_accuracy": None, "per_label": {}}
            continue
        labels = sorted(set().union(*(truth | guess for truth, guess in pairs)))
        per_label = {}
        for label in labels:
            tp = sum(label in truth and label in guess for truth, guess in pairs)
            fp = sum(label not in truth and label in guess for truth, guess in pairs)
            fn = sum(label in truth and label not in guess for truth, guess in pairs)
            per_label[label] = {"tp": tp, "fp": fp, "fn": fn,
                                "support": tp + fn,
                                "precision": tp / (tp + fp) if tp + fp else 0.0,
                                "recall": tp / (tp + fn) if tp + fn else 0.0,
                                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
        tp = sum(v["tp"] for v in per_label.values())
        fp = sum(v["fp"] for v in per_label.values())
        fn = sum(v["fn"] for v in per_label.values())
        report[axis] = {
            "reviewed_tracks": len(pairs), "coverage": available / len(pairs),
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "macro_f1": sum(v["f1"] for v in per_label.values()) / len(labels) if labels else None,
            "sample_jaccard": sum(len(t & p) / len(t | p) if t | p else 1.0 for t, p in pairs) / len(pairs),
            "primary_accuracy": sum(t == p for t, p in primary_pairs) / len(primary_pairs) if primary_pairs else None,
            "per_label": per_label}
    return report
