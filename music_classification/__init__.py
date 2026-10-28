"""Independent audio classification API; optional model dependencies load on use."""

from .inference import classify_audio
from .contrastive import ExperimentalClapClassifier
from .feedback import FeedbackStore
from .music2emo import ExperimentalMusic2EmoClassifier
from .combined import ExperimentalCombinedClassifier
from .muq import ExperimentalMuQClassifier
from .muq_genre import ExperimentalMuQGenreClassifier
from .settings import ModelSettings, BackendSettings

__all__ = ["classify_audio", "ExperimentalClapClassifier", "ExperimentalMusic2EmoClassifier",
           "ExperimentalCombinedClassifier", "ExperimentalMuQClassifier", "ExperimentalMuQGenreClassifier", "FeedbackStore", "ModelSettings", "BackendSettings"]
