"""Same registered calls for native Python and the service/terminal adapter."""
from __future__ import annotations


def _invoke(service, operation, arguments):
    from optimizer.operations import OperationRequest, OperationService
    from optimizer.operations.contracts import _ServiceError
    if not isinstance(service, OperationService):
        raise TypeError("Python jobs calls require service=OperationService(...).")
    arguments = dict(arguments)
    context = arguments.pop("context", None)
    response = service.execute(OperationRequest(operation, arguments, context=context))
    if not response.ok:
        raise _ServiceError(response.error["code"], response.error["message"])
    return response.result


def configure(*, service=None, **limits):
    """Set JobsConfig limits while no jobs are retained; inspect with status()."""
    return _invoke(service, "jobs.configure", limits)


def submit(operation, *, controls=None, variables=None, options=None, label=None,
           worker_bytes=256*1024*1024, result_bytes=4*1024*1024, context=None, service=None):
    """Freeze a source-backed context and initial state, then queue one solver."""
    return _invoke(service, "jobs.submit", {k:v for k,v in locals().items() if k != "service"})


def status(job=None, *, service=None):
    """Inspect one job or the scheduler, without activating or importing results."""
    return _invoke(service, "jobs.status", {"job":job})


def wait(job, *, timeout=10.0, service=None):
    """Wait up to 30 seconds for a terminal job state; returns the status."""
    return _invoke(service, "jobs.wait", {"job":job,"timeout":timeout})


def result(job, *, service=None):
    """Retain a completed native result as a physical candidate, never activate."""
    return _invoke(service, "jobs.result", {"job":job})


def cancel(job, *, service=None):
    """Request safe-boundary cancellation; unresponsive workers become interrupted."""
    return _invoke(service, "jobs.cancel", {"job":job})


def compare(jobs, *, metric="J", direction="minimize", service=None):
    """Rank compatible completed candidates; OLSS requires qualified verification."""
    return _invoke(service, "jobs.compare", {"jobs":jobs,"metric":metric,"direction":direction})


def release(job, *, service=None):
    """Release a terminal job's runtime inputs/results; retained candidates survive."""
    return _invoke(service, "jobs.release", {"job":job})


def shutdown(*, service=None):
    """Cancel and reap this service's workers; no jobs survive client shutdown."""
    return _invoke(service, "jobs.shutdown", {})
