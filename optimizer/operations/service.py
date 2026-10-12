"""Validated dispatch to native tools, isolated candidates and atomic publication."""

from __future__ import annotations

import dataclasses
import inspect
import threading
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

import numpy as np

from optimizer.controls import Controls
from optimizer.core.engine import run_chunk
from optimizer.core.stop_reasons import STOP_REASONS
from optimizer.result import OptimizerResult
from optimizer.state import RunState
from optimizer.system_olss import ShootingResult, ShootingVariables
from optimizer.operations.contracts import ObjectRef, OperationRequest, OperationResponse, _ServiceError
from optimizer.operations.registry import _Registry
from optimizer.session import Session, _compatible, _owned, _physical


def _key(value: Any) -> Any:
    """Hashable identity of effective method configuration, including schedules."""
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, Mapping):
        return tuple(sorted((k, _key(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_key(v) for v in value)
    if dataclasses.is_dataclass(value):
        return (type(value), tuple((f.name, _key(getattr(value, f.name))) for f in dataclasses.fields(value)))
    if callable(value):
        return ("ephemeral_callable", id(value))
    return value


def _compact(value: Any, depth: int = 0) -> Any:
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (Controls, ShootingVariables)):
        return {"type": type(value).__name__, "size": value.spec.size}
    if value is None or isinstance(value, (str, bool, int, float, complex, np.generic)):
        return value
    if depth >= 3:
        return {"type": type(value).__name__}
    if isinstance(value, Mapping):
        items = {str(k): _compact(v, depth + 1) for k, v in list(value.items())[:24]}
        if len(value) > 24:
            return {"type": "mapping", "items": items, "omitted": len(value) - 24}
        return items
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "length": len(value)}
    return {"type": type(value).__name__}


def _summary(result: Any) -> dict[str, Any]:
    summary = {"type": type(result).__name__}
    if isinstance(result, (OptimizerResult, ShootingResult)):
        summary.update({k: _compact(getattr(result, k)) for k in (
            "metrics", "stop_reason", "iterations")})
        if isinstance(result, ShootingResult):
            summary.update({k: getattr(result, k) for k in ("qualified", "verified", "solver_converged")})
            summary["error"] = result.error
    elif isinstance(result, Mapping):
        summary["values"] = _compact(result)
    elif isinstance(result, np.ndarray):
        summary.update(_compact(result))
    return summary


class OperationService:
    """One foreground service shared by direct Python and future CLI/MCP.

    No history writer or disk storage is installed. Execution serializes at the
    service boundary; a session revision also catches reentrant/outside writes.
    """

    def __init__(self, session: Session | None = None) -> None:
        self.session = Session() if session is None else session
        if not isinstance(self.session, Session):
            raise TypeError("session must be Session.")
        self._registry = _Registry()
        self._writer = threading.RLock()

    def list(self) -> list[dict[str, Any]]:
        return [self._registry.describe(name) for name in self._registry.items]

    def info(self, operation: str) -> dict[str, Any]:
        return self._registry.describe(operation)

    def register_reference(self, name: str, value: Any) -> ObjectRef:
        """Register Python-owned callbacks/data; durability remains ephemeral."""
        from optimizer.operations.contracts import _name
        _name(name, "name")
        with self.session._lock:
            return self.session._publish(value, revision=self.session._revision, bind=name)

    def _resolve(self, value: Any) -> Any:
        if isinstance(value, ObjectRef):
            return self.session.get(value)
        if isinstance(value, Mapping):
            return {k: self._resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._resolve(v) for v in value)
        return _owned(value)

    def _session_call(self, request: OperationRequest) -> tuple[Any, tuple[ObjectRef, ...]]:
        methods = {"session.load": self.session.load, "session.attach": self.session.attach,
                   "session.select": self.session.select, "session.use": self.session.use,
                   "session.update": self.session.update, "session.release": self.session.release,
                   "session.bind": self.session.bind_reference,
                   "session.get": self.session.get, "session.state": self.session.snapshot}
        method = methods.get(request.operation)
        if method is None:
            raise _ServiceError("unknown_operation", f"Unknown session operation {request.operation!r}.")
        if request.activate is not None:
            raise _ServiceError("invalid_arguments", "Session operations declare their state effect; activate is not accepted.")
        args = dict(request.arguments)
        if request.context is not None:
            if request.operation not in {"session.use", "session.update"} or "context" in args:
                raise _ServiceError("invalid_arguments", "context is accepted once, only for use/update.")
            args["context"] = request.context
        if request.bind is not None:
            if request.operation != "session.use" or "bind" in args:
                raise _ServiceError("invalid_arguments", "bind is accepted once, only for session.use.")
            args["bind"] = request.bind
        try:
            inspect.signature(method).bind(**args)
        except TypeError as exc:
            raise _ServiceError("invalid_arguments", str(exc)) from exc
        # Preserve refs for use (configuration provenance) and get, and don't
        # deepcopy trusted system objects passed to attach.
        if request.operation not in {"session.use", "session.get", "session.attach", "session.bind"}:
            args = self._resolve(args)
        result = method(**args)
        return result, (result,) if isinstance(result, ObjectRef) else ()

    def _execute_tool(self, request: OperationRequest) -> tuple[Any, tuple[ObjectRef, ...], dict[str, Any], str]:
        item, adapter = self._registry.resolve(request.operation)
        if adapter.classification not in {"tool", "constructor"}:
            raise _ServiceError("adapter_unavailable", f"{item.id}: {adapter.reason}")
        activate = adapter.effect == "solve" if request.activate is None else request.activate
        if activate and adapter.effect not in {"solve", "candidate"}:
            raise _ServiceError("invalid_arguments", "Inspection/constructor results do not implicitly become numerical state; use session.use.")
        obj = item.public_object()
        signature = inspect.signature(obj)
        with self.session._lock:
            revision = self.session._revision
            execution_before = self.session.snapshot()
            args = self._resolve(dict(request.arguments))
        if "system" in args:
            raise _ServiceError("invalid_arguments", "Select a named context instead of supplying a live system.")
        if any(name in args for name in ("evaluator", "trace", "observers")):
            raise _ServiceError("invalid_arguments", "Live evaluator/trace/observer injection requires a lifecycle adapter.")
        if item.id == "utils.compare_methods":
            methods = args.get("methods", ())
            nested = args.get("options", {}) or {}
            for method in (() if isinstance(methods, Mapping) else methods):
                if not isinstance(method, str):
                    # Explicit registered Python callbacks are trusted extensions.
                    continue
                native_item, native_adapter = self._registry.resolve(method)
                if native_item.group != "optimizers" or native_adapter.classification != "tool":
                    raise _ServiceError("invalid_arguments", f"Not an optimizer: {method!r}.")
                native_signature = inspect.signature(native_item.public_object())
                valid = set(native_signature.parameters) | set(inspect.signature(run_chunk).parameters)
                unknown = set(nested.get(method, {})) - valid
                if unknown:
                    raise _ServiceError("invalid_arguments", f"Unknown {method} options: {sorted(unknown)}.")
        # Explicit methods may forward engine options. Only declared run_chunk
        # parameters are accepted, never arbitrary optimizer **kwargs.
        allowed = {n for n, p in signature.parameters.items() if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
        engine_options = set(inspect.signature(run_chunk).parameters) - {"system", "controls", "step", "optimizer_name"}
        if any(p.kind == p.VAR_KEYWORD and p.name == "engine_kwargs" for p in signature.parameters.values()):
            allowed |= engine_options
        if item.id == "guesses.perturb_guess":
            kind = args.get("kind", "random_fourier")
            from optimizer import guesses
            targets = {"random_fourier": guesses.random_fourier_guess, "random_smooth": guesses.random_smooth_guess,
                       "random": guesses.random_guess, "fourier": guesses.fourier_guess}
            if kind in targets:
                allowed |= set(inspect.signature(targets[kind]).parameters) - {"target"}
        if item.id == "campaigns.random_smooth_guesses":
            from optimizer.guesses import random_smooth_guess
            allowed |= set(inspect.signature(random_smooth_guess).parameters) - {"target"}
        project_options = item.id in {"base.plot_data", "system_olgs.optional_plot_data"}
        if set(args) - allowed and not project_options:
            raise _ServiceError("invalid_arguments", f"Unknown arguments: {sorted(set(args) - allowed)}.")

        with self.session._lock:
            self.session._check_revision(revision)
            protected = set(self.session._bindings.values()) | {
                c.head for c in self.session._contexts.values() if c.head}
            if len(protected) >= self.session.max_objects:
                raise _ServiceError("object_quota", "Release a named object before executing another retained tool.")
            context_needed = any(name not in args for name, _ in adapter.inject) or activate or request.context is not None
            context = self.session._context(request.context) if context_needed else None
            head = None
            if context is not None and adapter.family and context.family not in adapter.family:
                raise _ServiceError("wrong_family", f"{item.id} accepts {adapter.family}, not {context.family}.")
            if context is not None:
                # Read tools use reconstruction too when available. Attached
                # read-only systems retain their project cache semantics.
                system = context.isolated() if context.factory else context.system
                if adapter.effect in {"solve", "candidate"} and context.factory is None:
                    raise _ServiceError("unsupported_isolation", "State-changing tools require a reconstruction/clone recipe.")
                head = self.session.get(context.head) if context.head else None
                for name, source in adapter.inject:
                    if name not in args:
                        if source == "system":
                            args[name] = system
                        elif source == "layout":
                            args[name] = _owned(context.layout)
                        elif source == "physical" and head is not None:
                            args[name] = _owned(_physical(head))
                for hook in adapter.hooks:
                    if not callable(getattr(system, hook, None)):
                        raise _ServiceError("capability_missing", f"Missing system hook {hook!r}.")
                if item.id in {"shooting.solve_shooting_multistart", "shooting.refine_shooting_seeds"}:
                    # Managed concurrency is later; existing foreground orchestration is serial.
                    parallel = args.get("parallel")
                    if parallel is not None and getattr(parallel, "backend", "serial") != "serial":
                        raise _ServiceError("capability_missing", "Managed parallel scheduling is Task 6; use serial here.")
                    from optimizer.utils.parallel import ParallelConfig
                    args["parallel"] = parallel or ParallelConfig(backend="serial")
                    args.setdefault("system_factory", context.factory)
                if item.id in {"shooting.solve_shooting_multistart", "shooting.refine_shooting_seeds", "shooting.continue_parameter"}:
                    from optimizer.shooting import solve_shooting
                    nested = args.get("solver_options", {}) or {}
                    unknown = set(nested) - set(inspect.signature(solve_shooting).parameters)
                    forbidden = set(nested) & {"system", "initial", "evaluator", "previous_result"}
                    if unknown or forbidden:
                        raise _ServiceError("invalid_arguments", f"Invalid nested solver options: {sorted(unknown | forbidden)}.")
                for name in ("controls", "variables", "initial", "anchor", "center", "gradient", "previous", "current"):
                    if isinstance(args.get(name), (Controls, ShootingVariables)):
                        _compatible(context.layout, args[name])
                for name in ("state", "warmstart"):
                    if args.get(name) is not None and _physical(args[name]) is not None:
                        _compatible(context.layout, args[name])
                for name in ("starts", "guesses"):
                    if name in args:
                        for value in args[name] if isinstance(args[name], (tuple, list)) else [args[name]]:
                            _compatible(context.layout, value)
                if item.id == "shooting.compare_shooting_replay":
                    for argument, family in (("shooting_system", "olss"), ("gradient_system", "olgs")):
                        named = args.get(argument)
                        if not isinstance(named, str):
                            raise _ServiceError("invalid_arguments", f"{argument} must be an explicit context name.")
                        other = self.session._context(named)
                        if other.family != family:
                            raise _ServiceError("wrong_family", f"{argument} requires {family}.")
                        args[argument] = other.isolated()
            if item.id == "base.transact":
                transaction = args.get("request")
                transaction_op = transaction.get("operation") if isinstance(transaction, Mapping) else getattr(transaction, "operation", None)
                if transaction_op == "with_secondary":
                    raise _ServiceError("invalid_arguments", "Use session.update for configuration changes.")

            try:
                bound = signature.bind(**args)
                bound.apply_defaults()
            except TypeError as exc:
                raise _ServiceError("invalid_arguments", str(exc)) from exc

            # Effective configuration excludes only run budgets and input/state.
            # Any change to variant, schedule or numerical settings resets memory.
            config = {k: v for k, v in bound.arguments.items() if k not in {
                "system", "controls", "initial", "state", "warmstart", "previous_result", "maxiter", "engine_kwargs"}}
            engine_defaults = {k: p.default for k, p in inspect.signature(run_chunk).parameters.items()
                               if p.default is not inspect.Parameter.empty and k not in {
                                   "system", "controls", "state", "warmstart", "maxiter", "chunk", "stage"}}
            if item.group == "optimizers" and item.name != "cma_es":
                config.update({k: args.get(k, v) for k, v in engine_defaults.items() if k not in config})
            method_key = (item.id, _key(config)) if adapter.effect == "solve" else None
            explicit_start = any(k in request.arguments for k in ("controls", "initial", "state", "warmstart", "previous_result"))
            continuation = "physical_restart" if head is not None and adapter.effect == "solve" else "new_start"
            if adapter.effect == "solve" and context is not None and not explicit_start and context.method_key == method_key:
                if adapter.continuation == "method_state" and (isinstance(head, RunState) or isinstance(head, OptimizerResult) and head.state):
                    args.pop("controls", None)
                    args["state"] = _owned(head if isinstance(head, RunState) else head.state)
                    continuation = "method_state"
                elif adapter.continuation == "variables_and_history" and isinstance(head, ShootingResult):
                    args["previous_result"] = _owned(head)
                    continuation = "variables_and_history"
            elif head is not None and adapter.effect == "solve":
                continuation = "physical_restart"
            # Explicit RunState/WarmStart is only safe with session provenance;
            # accept refs to a same-config result.state, never detached raw state.
            for field in ("state", "warmstart", "previous_result"):
                if request.arguments.get(field) is not None:
                    original = request.arguments[field]
                    if not isinstance(original, ObjectRef):
                        raise _ServiceError("unsupported_continuation", f"{field} requires a compatible retained object reference.")
                    stored = self.session._objects[self.session._resolve_id(original)]
                    if context is None or stored.context != context.name or stored.configuration != context.configuration or stored.method_key != method_key:
                        raise _ServiceError("unsupported_continuation", "Method/configuration state is incompatible; provide physical controls/variables instead.")
                    if field == "state":
                        continuation = "method_state"
                    elif field == "previous_result":
                        continuation = "variables_and_history"

        try:
            result = obj(**args)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            code = "capability_missing" if isinstance(exc, NotImplementedError) else "computation_failed"
            raise _ServiceError(code, f"{type(exc).__name__}: {exc}") from exc
        if adapter.effect == "solve":
            if not isinstance(result, (OptimizerResult, ShootingResult)):
                raise _ServiceError("invalid_result", "Solver must return a native numerical result.")
            stop = STOP_REASONS.get(result.stop_reason)
            if stop is not None and stop.outcome == "failed":
                activate = False
            if isinstance(result, OptimizerResult) and result.state is None:
                activate = False
        physical = _physical(result)
        if adapter.effect in {"solve", "candidate"} and physical is not None and context is not None:
            _compatible(context.layout, physical)
        if activate and physical is None:
            # Tuple transfer returns control + transfer report. Complex campaigns
            # remain candidates whose chosen component is activated explicitly.
            if isinstance(result, tuple) and result and isinstance(result[0], Controls):
                raise _ServiceError("invalid_state", "Activate the transfer's control with session.use; its tuple is a report.")
            raise _ServiceError("invalid_state", "Select a numerical member from this candidate report before activation.")
        response_value = _owned(result)  # prepare before committing
        with self.session._lock:
            ref = self.session._publish(result, revision=revision, context=context, bind=request.bind,
                                        activate=activate, method_key=method_key)
        effective = dict(bound.arguments)
        effective.update(args)
        if "controls" not in args and "state" in args:
            effective.pop("controls", None)
        metadata = {"summary": _summary(result), "continuation": continuation,
                    "activated": activate, "durability": "ephemeral",
                    "execution_state_before": execution_before,
                    "effective_arguments": _compact({k: v for k, v in effective.items() if k != "system"})}
        return response_value, (ref,), metadata, item.id

    def execute(self, request: OperationRequest | Mapping[str, Any]) -> OperationResponse:
        request_id, operation = uuid4().hex, None
        with self._writer:
            before = self.session.snapshot()
            try:
                if isinstance(request, Mapping):
                    request_id = request.get("request_id") or request_id
                    operation = request.get("operation")
                    request = OperationRequest(**request)
                elif isinstance(request, OperationRequest):
                    request_id, operation = request.request_id, request.operation
                    request = OperationRequest(**vars(request))
                else:
                    raise TypeError("Expected OperationRequest or mapping.")
            except Exception as exc:
                return OperationResponse(request_id if isinstance(request_id, str) else uuid4().hex,
                    operation if isinstance(operation, str) else None, "error", state_before=before,
                    state_after=self.session.snapshot(), error={"code": "invalid_request", "type": type(exc).__name__, "message": str(exc)})
            try:
                if request.operation.startswith("session."):
                    result, refs = self._session_call(request)
                    metadata = {"summary": _summary(result)}
                else:
                    result, refs, metadata, operation = self._execute_tool(request)
                    before = metadata.pop("execution_state_before")
                return OperationResponse(request_id, operation, "ok", result, refs, before,
                                         self.session.snapshot(), metadata=metadata)
            except Exception as exc:
                return OperationResponse(request_id, operation, "error", state_before=before,
                    state_after=self.session.snapshot(), error={"code": exc.code if isinstance(exc, _ServiceError) else "invalid_arguments",
                    "type": type(exc).__name__, "message": str(exc)})
