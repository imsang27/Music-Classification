"""Explicit model/interpreter/cache paths, relative to a JSON settings file."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class BackendSettings:
    installation: Path
    python_executable: Path
    work_directory: Path
    cache_directory: Path
    timeout: float = 600

    def kwargs(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ModelSettings:
    genre_checkpoint: Path | None = None
    genre_device: str = 'cpu'
    music2emo: BackendSettings | None = None
    muq: BackendSettings | None = None

    @classmethod
    def from_file(cls, path):
        path = Path(path).expanduser().resolve()
        data = json.loads(path.read_text(encoding='utf-8'))
        if (not isinstance(data, dict) or isinstance(data.get('version'), bool) or data.get('version') != 1
                or set(data) - {'version', 'genre', 'music2emo', 'muq'}):
            raise ValueError('Expected version1 model settings with known keys')

        def resolve(value):
            if not isinstance(value, str) or not value.strip():
                raise ValueError('Paths must be non-empty strings')
            result = Path(value).expanduser()
            return (path.parent / result).resolve() if not result.is_absolute() else result.resolve()

        def backend(name):
            config = data.get(name)
            if config is None:
                return None
            allowed = {'installation', 'python_executable', 'work_directory', 'cache_directory', 'timeout'}
            if not isinstance(config, dict) or set(config) - allowed or not {'installation', 'python_executable'} <= set(config):
                raise ValueError(f'Invalid {name} path settings')
            timeout = config.get('timeout', 600)
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError('Timeout must be positive finite seconds')
            return BackendSettings(resolve(config['installation']), resolve(config['python_executable']),
                resolve(config.get('work_directory', f'.classification-cache/{name}/requests')),
                resolve(config.get('cache_directory', f'.classification-cache/{name}/cache')), timeout)

        genre = data.get('genre')
        checkpoint, device = None, 'cpu'
        if genre is not None:
            if not isinstance(genre, dict) or set(genre) - {'checkpoint', 'device'} or 'checkpoint' not in genre:
                raise ValueError('Invalid genre settings')
            checkpoint, device = resolve(genre['checkpoint']), genre.get('device', 'cpu')
            if not isinstance(device, str) or not device:
                raise ValueError('Genre device must be explicit text')
        return cls(checkpoint, device, backend('music2emo'), backend('muq'))

    def create_mood_classifier(self):
        from .music2emo import ExperimentalMusic2EmoClassifier
        if self.music2emo is None:
            raise ValueError('Music2Emo paths are not configured')
        return ExperimentalMusic2EmoClassifier(**self.music2emo.kwargs())

    def create_emotion_classifier(self):
        from .muq import ExperimentalMuQClassifier
        if self.muq is None:
            raise ValueError('MuQ paths are not configured')
        return ExperimentalMuQClassifier(**self.muq.kwargs())

    def create_muq_genre_classifier(self):
        from .muq_genre import ExperimentalMuQGenreClassifier
        if self.muq is None:
            raise ValueError('MuQ paths are not configured')
        return ExperimentalMuQGenreClassifier(**self.muq.kwargs())

    def create_classifier(self, *, include_emotion=False, genre_backend='wav2vec2', share_muq_model=True):
        """Load the configured local genre model; optional workers remain lazy."""
        from .combined import ExperimentalCombinedClassifier
        if genre_backend not in ('wav2vec2', 'muq'):
            raise ValueError('genre_backend must be wav2vec2 or muq')
        if not isinstance(share_muq_model, bool):
            raise ValueError('share_muq_model must be a bool')
        mood = self.create_mood_classifier()
        emotion = self.create_emotion_classifier() if include_emotion else None
        if genre_backend == 'muq':
            return ExperimentalCombinedClassifier(None, None, mood_classifier=mood, emotion_classifier=emotion,
                                                  genre_classifier=self.create_muq_genre_classifier(), share_muq_model=share_muq_model)
        model, processor = None, None
        if self.genre_checkpoint is not None:
            if not self.genre_checkpoint.is_dir():
                raise ValueError('Local genre checkpoint is missing')
            import torch
            from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
            model = Wav2Vec2ForSequenceClassification.from_pretrained(str(self.genre_checkpoint), local_files_only=True)
            safe = self.genre_checkpoint / 'model.safetensors'
            if safe.is_file():
                from safetensors.torch import load_file
                state = load_file(str(safe))
            else:
                state = torch.load(self.genre_checkpoint / 'pytorch_model.bin', map_location='cpu', weights_only=True)
            prefix = 'wav2vec2.encoder.pos_conv_embed.conv.'
            for old, new in [('weight_g', 'parametrizations.weight.original0'), ('weight_v', 'parametrizations.weight.original1')]:
                old, new = prefix + old, prefix + new
                if old in state and new in model.state_dict():
                    if new in state:
                        raise ValueError('Ambiguous genre checkpoint keys')
                    state[new] = state.pop(old)
            model.load_state_dict(state, strict=True)
            if any(not torch.equal(model.state_dict()[name], value) for name, value in state.items()):
                raise ValueError('Genre weights differ after loading')
            del state
            model.to(self.genre_device).eval()
            processor = AutoFeatureExtractor.from_pretrained(str(self.genre_checkpoint), local_files_only=True)
        return ExperimentalCombinedClassifier(model, processor, mood_classifier=mood, emotion_classifier=emotion,
                                              share_muq_model=share_muq_model)
