"""Private synchronous numerical boundaries; inactive in ordinary Python calls."""
from contextlib import contextmanager
from contextvars import ContextVar

_callback = ContextVar('optimizer_job_boundary', default=None)

@contextmanager
def _boundaries(callback):
    token = _callback.set(callback)
    try:
        yield
    finally:
        _callback.reset(token)

def _checkpoint(family, state, iteration):
    callback = _callback.get()
    if callback is not None:
        callback(family, state, iteration)
