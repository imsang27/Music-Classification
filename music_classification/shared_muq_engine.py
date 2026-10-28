"""One physical MuQ model, with independent axis text, audio and RNG protocols."""

from .inference_engines import BaseEngine, MuQEngine
from .muq_genre_engine import MuQGenreEngine


class SharedMuQEngine(BaseEngine):
    def __init__(self, base, job, cache):
        super().__init__(base, job, cache)
        self.engines = {}
        for axis, kind in (('genre', MuQGenreEngine), ('emotion', MuQEngine)):
            folder = self.job / axis
            folder.mkdir()
            engine = kind(self.base, folder, self.cache)
            loader = engine._load
            engine._load = lambda engine=engine, loader=loader: self._load_axis(loader)
            self.engines[axis] = engine

    def _load_axis(self, loader):
        import numpy as np
        import torch
        import librosa
        from .portable_runtime import load_muq
        if not self.loaded:
            torch.set_num_threads(4)
            torch.manual_seed(20261025)
            np.random.seed(20261025)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model, self.info = load_muq(self.base, self.job / 'runtime', device)
            self._capture_rng()
            self.loaded = True
        # Reproduce each standalone loader's post-model initialization state.
        # Axis text embeddings and subsequent inference RNG remain independent.
        self._restore_rng()
        loader(shared=(self.model, self.info))

    def predict(self, path, *, axis):
        if axis not in self.engines:
            raise ValueError('Shared MuQ axis must be genre or emotion')
        return self.engines[axis].predict(path)
