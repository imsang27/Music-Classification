"""Compatible adapter pooling; one physical session is counted exactly once."""

from contextlib import contextmanager
from copy import copy

from .muq import ExperimentalMuQClassifier
from .muq_genre import ExperimentalMuQGenreClassifier
from .batching import WorkerSession


def compatible(genre, emotion):
    return (type(genre) is ExperimentalMuQGenreClassifier and type(emotion) is ExperimentalMuQClassifier
            and not hasattr(genre, '_worker_session') and not hasattr(emotion, '_worker_session')
            and all(getattr(genre, field) == getattr(emotion, field)
                    for field in ('installation', 'python', 'work_directory', 'cache_directory')))


class AxisSession:
    def __init__(self, session, axis):
        self.session, self.axis = session, axis

    @property
    def statistics(self):
        return self.session.statistics

    def run(self, argv, **options):
        return self.session.run(argv, axis=self.axis, **options)


@contextmanager
def shared_session(genre, emotion):
    if not compatible(genre, emotion):
        raise ValueError('MuQ adapter installations and worker paths must match')
    session = WorkerSession(genre, 'muq_shared')
    genre, emotion = copy(genre), copy(emotion)
    genre._worker_session = AxisSession(session, 'genre')
    emotion._worker_session = AxisSession(session, 'emotion')
    try:
        yield genre, emotion
    finally:
        session.close()
