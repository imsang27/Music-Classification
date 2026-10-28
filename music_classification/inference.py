"""Wav2Vec2 baseline with explicit axis availability and uncalibrated scores."""

from pathlib import Path
import math


def _axis():
    return {"status": "unavailable", "primary": None, "candidates": [],
            "labels": [], "reason": "No model configured for this axis"}


def classify_audio(audio_path, model=None, processor=None, *, max_duration=30,
                   warning_threshold=0.5, top_k=3):
    """Classify the first audio segment using an already loaded genre model.

    Scores are softmax values, not calibrated correctness probabilities.
    top-k alternatives do not imply that the track has multiple genres.
    The supplied model is put in evaluation mode and kept there.
    """
    result = {"schema_version": "1.0", "status": "error", "success": False,
              "axes": {name: _axis() for name in ("genre", "emotion", "mood")},
              "analysis": None, "model": None, "warnings": []}

    def fail(status, message):
        result.update(status=status, error=message)
        return result

    if (isinstance(max_duration, bool) or not isinstance(max_duration, (int, float))
            or not math.isfinite(max_duration) or max_duration < 1
            or isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
            or not isinstance(warning_threshold, (int, float))
            or not math.isfinite(warning_threshold) or not 0 <= warning_threshold <= 1):
        return fail("error", "Invalid inference options")
    try:
        path = Path(audio_path)
        if not path.is_file() or path.stat().st_size == 0:
            return fail("invalid_audio", "Audio file is missing or empty")
        if path.stat().st_size > 100 * 1024 * 1024:
            return fail("invalid_audio", "Audio file exceeds 100 MB")
    except (TypeError, ValueError, OSError) as exc:
        return fail("invalid_audio", str(exc))
    if model is None or processor is None:
        return fail("unavailable", "Genre model or processor is not configured")
    try:
        import numpy as np
        import librosa
        import torch
    except ImportError as exc:
        return fail("error", f"Missing inference dependency: {exc}")
    try:
        audio, sr = librosa.load(str(path), sr=16000, duration=max_duration)
    except Exception as exc:
        return fail("invalid_audio", f"Audio decoding failed: {exc}")
    if len(audio) == 0 or not np.isfinite(audio).all():
        return fail("invalid_audio", "Audio is empty or contains non-finite samples")
    # Reject digital silence before normalization; this is not a noise/music detector.
    if np.max(np.abs(audio)) <= 1e-7:
        return fail("invalid_audio", "Audio contains only silence")
    analyzed_duration = len(audio) / sr
    if len(audio) < sr:
        return fail("invalid_audio", "At least one second of audio is required")
    try:
        audio = librosa.util.normalize(audio)
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt",
                           padding=True, truncation=True, max_length=int(sr * max_duration))
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        model.eval()
        with torch.inference_mode():
            logits = model(**inputs).logits
            if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] == 0:
                raise ValueError("Expected one non-empty classification vector")
            if not torch.isfinite(logits).all():
                raise ValueError("Model returned non-finite scores")
            scores = torch.softmax(logits[0].float(), dim=-1).cpu()
        id2label = model.config.id2label
        candidates = []
        for index in torch.argsort(scores, descending=True)[:top_k].tolist():
            label = id2label.get(index, id2label.get(str(index)))
            if label is None:
                raise ValueError(f"Missing model label for class {index}")
            candidates.append({"label": str(label), "score": float(scores[index])})
        primary = candidates[0]
        result["axes"]["genre"] = {
            "status": "ok", "primary": primary, "candidates": candidates,
            "labels": [primary["label"]], "score_type": "softmax",
            "calibration": "uncalibrated", "nearest_supported": True,
            "label_space": "model_native", "multilabel": False}
        result["model"] = {"name": str(getattr(model.config, "_name_or_path", "")),
                           "device": str(device), "type": type(model).__name__}
        result["analysis"] = {"sample_rate": sr, "segments_seconds": [[0, analyzed_duration]],
                              "strategy": "first_segment", "max_duration": max_duration}
        result["warnings"] = ["Model scores are not accuracy or calibrated probabilities",
                              "Emotion and mood models are not configured",
                              "Model-native genres have not been mapped to the draft taxonomy",
                              "Noise and speech-only rejection is not implemented"]
        if primary["score"] < warning_threshold:
            result["warnings"].append("Top score below diagnostic warning threshold; candidate retained")
        result.update(status="partial", success=True)
        return result
    except Exception as exc:
        return fail("error", f"Genre inference failed: {exc}")
