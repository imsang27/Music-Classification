"""Append-only feedback with partial labels and explicit dataset split isolation."""

from datetime import datetime, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3
from uuid import uuid4


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class FeedbackStore:
    """SQLite event store. Recording feedback never triggers model training."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY, audio_sha256 TEXT NOT NULL,
                    split TEXT NOT NULL CHECK(split IN ('train', 'development', 'test')),
                    payload TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS feedback (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE NOT NULL,
                    prediction_id TEXT NOT NULL REFERENCES predictions(prediction_id),
                    axis TEXT NOT NULL, label TEXT NOT NULL, action TEXT NOT NULL,
                    corrected_labels TEXT NOT NULL, comment TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL, created_at TEXT NOT NULL);
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    def register_prediction(self, prediction, *, split="train"):
        if split not in {"train", "development", "test"}:
            raise ValueError("Unknown dataset split")
        key = prediction.get("prediction_id")
        sha = prediction.get("input", {}).get("audio_sha256") if prediction.get("input") else None
        model = prediction.get("model") or {}
        analysis = prediction.get("analysis") or {}
        if (not isinstance(key, str) or not key or prediction.get("success") is not True
                or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)
                or not model.get("name") or not model.get("revision")
                or not model.get("prompt_sha256") or not model.get("taxonomy_sha256")
                or not prediction.get("taxonomy_version") or not analysis.get("segments_seconds")):
            raise ValueError("Successful, versioned prediction with audio identity and segments required")
        for axis, value in prediction.get("axes", {}).items():
            if axis not in {"genre", "emotion", "mood"}:
                raise ValueError("Unknown prediction axis")
            if value.get("status") == "ok":
                supported = value.get("supported_labels", [])
                candidates = value.get("candidates", [])
                if (not supported or not candidates
                        or any(c.get("label") not in supported for c in candidates)):
                    raise ValueError("Prediction label vocabulary is incomplete")
        if not any(v.get("status") == "ok" for v in prediction.get("axes", {}).values()):
            raise ValueError("No available prediction axis")
        payload = _json(prediction)
        with self._connect() as db:
            existing = db.execute("SELECT payload, split FROM predictions WHERE prediction_id=?", (key,)).fetchone()
            if existing:
                if existing["payload"] != payload or existing["split"] != split:
                    raise ValueError("Prediction snapshots and splits are immutable")
                return key
            db.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?)",
                       (key, sha, split, payload, datetime.now(timezone.utc).isoformat()))
        return key

    def record_feedback(self, prediction_id, axis, label, action, *, reviewer_id,
                        corrected_labels=None, comment=""):
        if action not in {"accept", "reject", "correct", "skip"}:
            raise ValueError("Action must be accept, reject, correct or skip")
        if not isinstance(reviewer_id, str) or not reviewer_id.strip() or not isinstance(comment, str):
            raise ValueError("Reviewer identifier and text comment required")
        if axis not in {"genre", "emotion", "mood"}:
            raise ValueError("Unknown axis")
        if corrected_labels is None:
            corrected_labels = []
        if (not isinstance(corrected_labels, list)
                or any(not isinstance(v, str) for v in corrected_labels)
                or len(corrected_labels) != len(set(corrected_labels))):
            raise ValueError("Corrected labels must be a list of unique label IDs")
        if (action == "correct" and not corrected_labels) or (action != "correct" and corrected_labels):
            raise ValueError("Only corrections require replacement labels")
        with self._connect() as db:
            row = db.execute("SELECT payload FROM predictions WHERE prediction_id=?", (prediction_id,)).fetchone()
            if row is None:
                raise ValueError("Prediction does not exist")
            prediction = json.loads(row["payload"])
            target = prediction["axes"].get(axis, {})
            proposed = {v["label"] for v in target.get("candidates", [])}
            if target.get("status") != "ok" or label not in proposed:
                raise ValueError("Feedback must target a displayed candidate on an available axis")
            if not set(corrected_labels) <= set(target.get("supported_labels", [])):
                raise ValueError("Correction contains unsupported labels")
            event = {"event_id": str(uuid4()), "prediction_id": prediction_id, "axis": axis,
                     "label": label, "action": action, "corrected_labels": corrected_labels,
                     "comment": comment, "reviewer_id": reviewer_id,
                     "created_at": datetime.now(timezone.utc).isoformat()}
            db.execute("""INSERT INTO feedback
                (event_id, prediction_id, axis, label, action, corrected_labels, comment, reviewer_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                tuple(_json(event[k]) if k == "corrected_labels" else event[k]
                      for k in ("event_id", "prediction_id", "axis", "label", "action", "corrected_labels",
                                "comment", "reviewer_id", "created_at")))
        return event

    def list_feedback(self):
        with self._connect() as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM feedback ORDER BY sequence")]
        for row in rows:
            row["corrected_labels"] = json.loads(row["corrected_labels"])
        return rows

    def export_training_examples(self):
        """Export explicit positive/negative evidence, never complete gold labels.

        Skips/comments are not labels. Latest non-skip judgement per candidate
        wins. Conflicts across judgements are excluded. Reviewers stay separate.
        Any exact audio also registered as development/test is excluded entirely.
        """
        with self._connect() as db:
            db.execute("BEGIN")
            predictions = {r["prediction_id"]: dict(r) for r in db.execute("SELECT * FROM predictions")}
            events = [dict(r) for r in db.execute("SELECT * FROM feedback ORDER BY sequence")]
        for event in events:
            event["corrected_labels"] = json.loads(event["corrected_labels"])
        excluded = {r["audio_sha256"] for r in predictions.values() if r["split"] != "train"}
        latest = {}
        for event in events:
            if event["action"] != "skip":
                key = (event["prediction_id"], event["axis"], event["label"], event["reviewer_id"])
                latest[key] = event
        groups = {}
        for event in latest.values():
            record = predictions[event["prediction_id"]]
            if record["split"] != "train" or record["audio_sha256"] in excluded:
                continue
            prediction = json.loads(record["payload"])
            segments = prediction["analysis"]["segments_seconds"]
            key = (record["audio_sha256"], _json(segments), prediction["taxonomy_version"],
                   prediction["model"]["taxonomy_sha256"], event["axis"], event["reviewer_id"])
            sample = groups.setdefault(key, {"audio_sha256": key[0], "segments_seconds": segments,
                                            "taxonomy_version": key[2], "taxonomy_sha256": key[3],
                                            "axis": key[4], "reviewer_id": key[5],
                                            "positive_labels": set(), "negative_labels": set(),
                                            "feedback_event_ids": [], "prediction_ids": set(),
                                            "annotation_type": "partial_feedback", "split": "train"})
            sample["feedback_event_ids"].append(event["event_id"])
            sample["prediction_ids"].add(event["prediction_id"])
            if event["action"] == "accept":
                sample["positive_labels"].add(event["label"])
            elif event["action"] == "reject":
                sample["negative_labels"].add(event["label"])
            else:
                sample["positive_labels"].update(event["corrected_labels"])
                if event["label"] not in event["corrected_labels"]:
                    sample["negative_labels"].add(event["label"])
        exported = []
        for sample in groups.values():
            conflict = sample["positive_labels"] & sample["negative_labels"]
            sample["conflicting_labels"] = sorted(conflict)
            for key in ("positive_labels", "negative_labels"):
                sample[key] = sorted(sample[key] - conflict)
            sample["prediction_ids"] = sorted(sample["prediction_ids"])
            if sample["positive_labels"] or sample["negative_labels"]:
                exported.append(sample)
        return exported
