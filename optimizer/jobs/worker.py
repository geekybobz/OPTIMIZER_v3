"""Private child runner: numeric codec, frozen source and boundary cancellation."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

from optimizer.checkpoints.codec import _Codec, _json
from optimizer.operations import OperationRequest, OperationService
from optimizer.serialization import to_plain
from optimizer.jobs.boundary import _boundaries


def _write(directory, filename, value, limit):
    codec = _Codec(max_bytes=limit)
    raw = _json(codec.encode(value))
    if len(raw) + sum(map(len, codec.blobs.values())) > limit:
        raise MemoryError('Serialized boundary/result exceeds result_bytes.')
    blobs = directory / 'output_blobs'; blobs.mkdir(exist_ok=True)
    for digest, data in codec.blobs.items():
        (blobs / digest).write_bytes(data)
    temporary = directory / (filename + '.prepare')
    temporary.write_bytes(raw); temporary.replace(directory / filename)


class _Cancelled(BaseException):
    pass


class _BoundarySystem:
    """Wrap project calls, leaving physics and numerical return values unchanged."""
    _calls = {"evaluate", "gradient", "value_and_gradient", "residuals", "jacobian",
              "residuals_and_jacobian", "verify", "cost_gradient", "hvp", "hessian", "smooth_gradient", "prox"}

    def __init__(self, system, cancelled):
        self._system, self._cancelled = system, cancelled

    def __getattr__(self, name):
        value = getattr(self._system, name)
        if name not in self._calls or not callable(value):
            return value
        def call(*args, **kwargs):
            if self._cancelled(): raise _Cancelled()
            result = value(*args, **kwargs)
            if self._cancelled(): raise _Cancelled()
            return result
        return call


def main():
    directory = Path(sys.argv[1])
    limit = int(sys.argv[2])
    lost_parent = threading.Event()
    def watch_parent():
        try: sys.stdin.buffer.read(1)
        finally:
            lost_parent.set()
            time.sleep(float(sys.argv[3]))
            # No PID guessing: a worker exits itself if its owner disappeared.
            shutil.rmtree(directory, ignore_errors=True)
            os._exit(70)
    threading.Thread(target=watch_parent, daemon=True).start()
    cancelled = lambda: lost_parent.is_set() or (directory / "cancel").exists()
    outcome = {}
    try:
        manifest = json.loads((directory / "input.json").read_bytes())
        inputs = _Codec(lambda digest: (directory / "input_blobs" / digest).read_bytes(),
                        max_bytes=int(sys.argv[4])).decode(manifest)
        recipe = inputs["recipe"]
        source = directory / "source" / inputs["source_path"]
        sys.path.insert(0, str(source.parent))
        service = OperationService()
        loaded = service.execute(OperationRequest("session.load", {"name":"worker","factory":str(source),
            "factory_name":recipe["factory_name"], "parameters":recipe.get("arguments", {})}))
        if not loaded.ok: raise RuntimeError(loaded.error)
        for updates in recipe.get("secondary_updates", []):
            response = service.execute(OperationRequest("session.update", {"updates":updates}))
            if not response.ok: raise RuntimeError(response.error)
        context = service.session._context("worker")
        if to_plain(context.system.system_spec()) != inputs["system_spec"]:
            raise RuntimeError("Reconstructed system differs from the frozen declaration.")
        factory = context.factory
        context.system = _BoundarySystem(context.system, cancelled)
        context.factory = lambda: _BoundarySystem(factory(), cancelled)
        argument = "controls" if inputs["family"] == "olgs" else "initial"
        arguments = {**inputs["options"], argument:inputs["initial"]}
        if cancelled(): raise _Cancelled()
        def boundary(family, state, iteration):
            if cancelled(): raise _Cancelled()
            if not (directory / 'pause').exists(): return
            from optimizer.session import _physical
            snapshot = {'physical':_physical(state), 'native_state':state, 'iteration':iteration,
                        'family':family, 'resume_level':'physical_only', 'boundary':'iteration'}
            _write(directory, 'paused.json', snapshot, limit)
            while (directory / 'pause').exists() and not (directory / 'resume').exists():
                if cancelled(): raise _Cancelled()
                time.sleep(.05)
            (directory / 'paused.json').unlink(missing_ok=True)
            shutil.rmtree(directory / 'output_blobs',ignore_errors=True)
            (directory / 'pause').unlink(missing_ok=True)
            (directory / 'resume').unlink(missing_ok=True)
        with _boundaries(boundary):
            response = service.execute(OperationRequest(inputs["operation"], arguments, activate=False))
        if not response.ok:
            outcome = {"status":"failed", "error":response.error, "result":None}
        else:
            result = response.result
            from optimizer.core.stop_reasons import STOP_REASONS
            stop = STOP_REASONS.get(result.stop_reason)
            failed = stop is not None and stop.outcome == "failed"
            outcome = {"status":"failed" if failed else "succeeded", "result":result,
                       "error":{"code":"numerical_failure","message":result.stop_reason} if failed else None}
    except _Cancelled:
        outcome = {"status":"cancelled", "result":None,
                   "error":None, "cancellation_boundary":"project_call"}
    except BaseException as exc:
        outcome = {"status":"interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "result":None,
                   "error":{"code":"worker_failed","message":f"{type(exc).__name__}: {str(exc)[:1024]}"}}
    try:
        _write(directory, 'output.json', outcome, limit)
    except BaseException as exc:
        # A fixed-size error can still fit even when the numerical result cannot.
        raw = _json(_Codec().encode({"status":"failed", "result":None,
                    "error":{"code":"result_quota","message":str(exc)[:512]}}))
        (directory / "output.json").write_bytes(raw)
    return 0


if __name__ == "__main__": raise SystemExit(main())
