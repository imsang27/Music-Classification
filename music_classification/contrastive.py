"""Experimental audio/text classification, independent of web or file workflows."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4


class ExperimentalClapClassifier:
    """Rank supported labels for each axis; no calibrated or multi-label claim.

    Supply already loaded CLAP objects, or use from_local() for an offline
    checkpoint. A classifier instance should not be moved or trained concurrently.
    """

    def __init__(self, model, processor, taxonomy, prompt_config, *, model_id, revision):
        if not model_id or not revision:
            raise ValueError("Model identifier and revision are required")
        if not isinstance(taxonomy.get("version"), str) or not prompt_config.get("version"):
            raise ValueError("Versioned taxonomy and prompts are required")
        self.taxonomy = deepcopy(taxonomy)
        self.prompt_config = deepcopy(prompt_config)
        self.model, self.processor = model, processor
        self.model_id, self.revision = model_id, revision
        self.names = {}
        for axis in ("genre", "emotion", "mood"):
            labels = self.taxonomy.get(axis, [])
            ids = [item["id"] for item in labels]
            prompts = self.prompt_config.get("prompts", {}).get(axis, {})
            if (not ids or len(ids) != len(set(ids)) or set(ids) != set(prompts)
                    or any(not isinstance(v, str) or not v.strip() for v in prompts.values())):
                raise ValueError(f"Taxonomy/prompt mismatch: {axis}")
            self.names[axis] = {item["id"]: item["name"] for item in labels}
        self._text_features = None

    @classmethod
    def from_local(cls, checkpoint, taxonomy, prompt_config, *, device="cpu",
                   model_id="laion/clap-htsat-unfused", revision=None):
        """Load only local files; no implicit model download or device fallback."""
        from transformers import ClapModel, ClapProcessor
        path = Path(checkpoint).resolve()
        if not path.is_dir():
            raise ValueError("Local checkpoint directory does not exist")
        if revision is None:
            if path.parent.name != "snapshots":
                raise ValueError("Supply an exact revision for non-snapshot checkpoints")
            revision = path.name
        return cls(ClapModel.from_pretrained(str(path), local_files_only=True).to(device),
                   ClapProcessor.from_pretrained(str(path), local_files_only=True),
                   taxonomy, prompt_config, model_id=model_id, revision=revision)

    def _prepare_text(self):
        import torch
        from torch.nn.functional import normalize
        device = next(self.model.parameters()).device
        self.model.eval()
        if self._text_features is not None:
            return {axis: values.to(device) for axis, values in self._text_features.items()}
        features = {}
        with torch.inference_mode():
            for axis, prompts in self.prompt_config["prompts"].items():
                if axis not in self.names:
                    continue
                inputs = self.processor(text=["This is " + value + "." for value in prompts.values()],
                                        return_tensors="pt", padding=True)
                values = self.model.get_text_features(**{k: v.to(device) for k, v in inputs.items()})
                if (values.ndim != 2 or values.shape[0] != len(prompts)
                        or not torch.isfinite(values).all() or (values.norm(dim=-1) <= 1e-12).any()):
                    raise ValueError("Invalid text embeddings")
                features[axis] = normalize(values, dim=-1)
        self._text_features = {axis: values.cpu() for axis, values in features.items()}
        return features

    def classify(self, audio_path, *, max_duration=30, offset=0, top_k=3):
        """Analyze consecutive up-to-10-second segments; keep the nearest candidate."""
        result = {"schema_version": "1.0", "prediction_id": str(uuid4()),
                  "status": "error", "success": False, "experimental": True,
                  "taxonomy_version": self.taxonomy["version"], "input": None,
                  "model": {"name": self.model_id, "revision": self.revision,
                            "prompt_version": self.prompt_config["version"],
                            "prompt_sha256": hashlib.sha256(json.dumps(self.prompt_config, sort_keys=True,
                                                                       ensure_ascii=False).encode()).hexdigest(),
                            "taxonomy_sha256": hashlib.sha256(json.dumps(self.taxonomy, sort_keys=True,
                                                                         ensure_ascii=False).encode()).hexdigest()},
                  "analysis": None, "axes": {axis: {"status": "unavailable", "primary": None,
                                                     "candidates": [], "labels": []} for axis in self.names},
                  "warnings": []}

        def fail(status, message):
            result.update(status=status, error=message)
            return result

        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in (max_duration, offset)) or max_duration < 1 or offset < 0
                or isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
            return fail("error", "Invalid inference options")
        try:
            path = Path(audio_path)
            if not path.is_file() or not 0 < path.stat().st_size <= 100 * 1024 * 1024:
                return fail("invalid_audio", "Audio is missing, empty or exceeds 100 MB")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            result["input"] = {"audio_sha256": digest}
        except (OSError, TypeError, ValueError) as exc:
            return fail("invalid_audio", str(exc))
        try:
            import numpy as np
            import librosa
            import torch
            from torch.nn.functional import normalize
        except ImportError as exc:
            return fail("error", f"Missing inference dependency: {exc}")
        try:
            sr = int(self.processor.feature_extractor.sampling_rate)
            if sr < 1:
                raise ValueError("Invalid sampling rate")
        except (AttributeError, TypeError, ValueError) as exc:
            return fail("error", f"Invalid processor configuration: {exc}")
        try:
            audio, _ = librosa.load(str(path), sr=sr, offset=offset, duration=max_duration)
        except Exception as exc:
            return fail("invalid_audio", f"Audio decoding failed: {exc}")
        if len(audio) < sr or not np.isfinite(audio).all() or np.max(np.abs(audio)) <= 1e-7:
            return fail("invalid_audio", "Audio is silent, non-finite or shorter than one second")
        try:
            device = next(self.model.parameters()).device
            text = self._prepare_text()
            # Only full model windows and a final >=1s remainder; shorter tails are recorded as omitted.
            window = int(min(10, self.processor.feature_extractor.max_length_s) * sr)
            segments = [(start, audio[start:start + window]) for start in range(0, len(audio), window)
                        if len(audio[start:start + window]) >= sr]
            audio_features = []
            with torch.inference_mode():
                for _, segment in segments:
                    inputs = self.processor(audios=segment, sampling_rate=sr, return_tensors="pt")
                    features = self.model.get_audio_features(**{k: v.to(device) for k, v in inputs.items()})
                    if (features.ndim != 2 or features.shape[0] != 1 or not torch.isfinite(features).all()
                            or (features.norm(dim=-1) <= 1e-12).any()):
                        raise ValueError("Invalid audio embeddings")
                    audio_features.append(normalize(features, dim=-1))
                embeddings = torch.cat(audio_features)
                axes = {}
                for axis, features in text.items():
                    per_segment = embeddings @ features.T
                    scores = per_segment.mean(dim=0).cpu()
                    ids = list(self.prompt_config["prompts"][axis])
                    order = torch.argsort(scores, descending=True)[:top_k].tolist()
                    candidates = [{"label": ids[i], "name": self.names[axis][ids[i]],
                                   "score": float(scores[i])} for i in order]
                    axes[axis] = {"status": "ok", "primary": candidates[0], "candidates": candidates,
                                  "labels": [candidates[0]["label"]],
                                  "supported_labels": list(self.names[axis]),
                                  "score_type": "mean_cosine_similarity", "calibration": "uncalibrated",
                                  "nearest_supported": True, "label_space": "draft_taxonomy",
                                  "multilabel": False,
                                  "segment_primary_labels": [ids[i] for i in per_segment.argmax(dim=-1).tolist()]}
            result["axes"] = axes
            result["model"]["device"] = str(device)
            result["analysis"] = {"sample_rate": sr, "strategy": "consecutive_segments_mean_similarity",
                                  "segments_seconds": [[offset + start / sr, offset + (start + len(segment)) / sr]
                                                       for start, segment in segments],
                                  "max_duration": max_duration,
                                  "omitted_tail_seconds": (len(audio) - sum(len(a) for _, a in segments)) / sr}
            result["warnings"] = ["Experimental candidates; accuracy has not been established",
                                  "Similarity scores are not probabilities or accuracy",
                                  "Top-k alternatives do not imply multiple simultaneous labels",
                                  "Noise and speech-only rejection is not implemented"]
            result.update(status="ok", success=True)
            return result
        except Exception as exc:
            return fail("error", f"CLAP inference failed: {exc}")
