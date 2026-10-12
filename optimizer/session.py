"""Owned foreground contexts and bounded object versions for operation dispatch."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from optimizer.controls import Controls, _validate_control_spec_compatibility
from optimizer.families import system_family
from optimizer.result import OptimizerResult
from optimizer.state import RunState
from optimizer.system_olgs import get_system_spec, require_system
from optimizer.system_olgs.validation import validate_control_spec
from optimizer.system_olss import require_shooting_system, validate_shooting_spec
from optimizer.system_olss import ShootingResult
from optimizer.system_olss.variables import ShootingVariables, _validate_shooting_spec_compatibility
from optimizer.operations.contracts import ObjectRef, _ServiceError, _name
from optimizer.serialization import to_plain


def _owned(value: Any) -> Any:
    """Copy supported data graphs, never use this to clone a physics system."""
    try:
        return copy.deepcopy(value)
    except Exception as exc:
        raise _ServiceError("unsupported_object", f"Cannot retain {type(value).__name__}: {exc}") from exc


def _size(value: Any, seen: set[int] | None = None) -> int:
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    total = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return total + (value.nbytes if value.base is not None else 0) + (
            sum(_size(item, seen) for item in value.flat) if value.dtype.hasobject else 0)
    if isinstance(value, Mapping):
        return total + sum(_size(k, seen) + _size(v, seen) for k, v in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return total + sum(_size(v, seen) for v in value)
    if hasattr(value, "__dict__") and not callable(value):
        return total + _size(vars(value), seen)
    return total


def _physical(value: Any) -> Controls | ShootingVariables | None:
    if isinstance(value, (Controls, ShootingVariables)):
        return value
    if isinstance(value, RunState):
        return value.controls
    for field_name in ("controls", "variables"):
        physical = getattr(value, field_name, None)
        if isinstance(physical, (Controls, ShootingVariables)):
            return physical
    return None


def _validate_system(system: Any) -> tuple[str, Any]:
    try:
        family = system_family(system)
        if family == "olgs":
            require_system(system)
            get_system_spec(system)
            return family, validate_control_spec(system)
        require_shooting_system(system)
        return family, validate_shooting_spec(system)
    except Exception as exc:
        raise _ServiceError("invalid_system", str(exc)) from exc


def _compatible(layout: Any, value: Any) -> None:
    physical = _physical(value)
    if physical is None:
        raise _ServiceError("invalid_state", "Activation requires controls, variables, or a numerical result/state.")
    try:
        if isinstance(physical, Controls):
            _validate_control_spec_compatibility(layout, physical.spec)
            array = physical.as_matrix(copy=False)
            expected = layout.shape
        else:
            _validate_shooting_spec_compatibility(layout, physical.spec)
            array = physical.values
            expected = (layout.size,)
        if array.shape != expected or not np.all(np.isfinite(array)):
            raise ValueError("Numerical state must have the declared shape and finite values.")
    except (ValueError, TypeError, AttributeError) as exc:
        raise _ServiceError("incompatible_layout", str(exc)) from exc


@dataclass
class _Context:
    name: str
    system: Any
    family: str
    layout: Any
    factory: Callable[[], Any] | None
    recipe: dict[str, Any]
    head: str | None = None
    method_key: Any = None
    configuration: int = 0

    def isolated(self) -> Any:
        if self.factory is None:
            raise _ServiceError("unsupported_isolation", "Attach a reconstruction factory or clone before running a state-changing tool.")
        candidate = self.factory()
        if candidate is self.system:
            raise _ServiceError("unsupported_isolation", "The factory/clone must return an independent system.")
        family, layout = _validate_system(candidate)
        if family != self.family:
            raise _ServiceError("invalid_system", "Factory changed system family.")
        if json.dumps(to_plain(candidate.system_spec()), sort_keys=True) != json.dumps(to_plain(self.system.system_spec()), sort_keys=True):
            raise _ServiceError("invalid_system", "Factory/clone changed the declared configuration; load a new context.")
        if self.family == "olgs":
            _validate_control_spec_compatibility(self.layout, layout)
        else:
            _validate_shooting_spec_compatibility(self.layout, layout)
        return candidate


@dataclass
class _Stored:
    value: Any
    bytes: int
    context: str | None
    configuration: int | None
    method_key: Any = None


class Session:
    """Independent named systems, selected context and bounded retained objects.

    Reads return owned data. One revision guards all atomic commits, including
    reentrant callbacks. Byte accounting concerns retained data, not process RAM.
    """

    def __init__(self, *, max_bytes: int = 64 * 1024 * 1024, max_objects: int = 256) -> None:
        for name, value in (("max_bytes", max_bytes), ("max_objects", max_objects)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.max_bytes, self.max_objects = max_bytes, max_objects
        self._contexts: dict[str, _Context] = {}
        self._objects: dict[str, _Stored] = {}
        self._bindings: dict[str, str] = {}
        self._selected: str | None = None
        self._revision = 0
        self._lock = threading.RLock()

    def _retained_bytes(self) -> int:
        # Physics runtimes/native libraries/temporary arrays are outside this
        # retained-data accounting; never present it as process RSS.
        return _size(self._bindings) + sum(o.bytes for o in self._objects.values()) + sum(
            _size(c.recipe) + _size(c.layout) for c in self._contexts.values())

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"revision": self._revision, "selected": self._selected,
                    "contexts": {name: {"family": c.family, "head": c.head,
                        "configuration": c.configuration, "recipe": _owned(c.recipe)}
                        for name, c in self._contexts.items()},
                    "bindings": dict(self._bindings), "objects": len(self._objects),
                    "retained_bytes": self._retained_bytes()}

    def _context(self, name: str | None) -> _Context:
        name = self._selected if name is None else name
        if name not in self._contexts:
            raise _ServiceError("missing_context", f"No loaded context {name!r}.")
        return self._contexts[name]

    def _resolve_id(self, reference: str | ObjectRef) -> str:
        name = reference.object_id if isinstance(reference, ObjectRef) else reference
        object_id = self._bindings.get(name, name)
        if object_id not in self._objects:
            raise _ServiceError("missing_object", f"No retained object {name!r}.")
        return object_id

    def get(self, reference: str | ObjectRef) -> Any:
        with self._lock:
            value = self._objects[self._resolve_id(reference)].value
            if isinstance(reference, ObjectRef):
                for part in reference.path:
                    if isinstance(value, (tuple, list, np.ndarray)) and type(part) is int:
                        try:
                            value = value[part]
                        except IndexError as exc:
                            raise _ServiceError("invalid_reference", str(exc)) from exc
                    elif isinstance(value, Controls) and part in value.spec.keys:
                        value = value.as_matrix(copy=False)[value.spec.keys.index(part)]
                    elif isinstance(value, Mapping):
                        try:
                            value = value[part]
                        except KeyError as exc:
                            raise _ServiceError("invalid_reference", str(exc)) from exc
                    elif dataclasses.is_dataclass(value) and part in {f.name for f in dataclasses.fields(value)}:
                        value = getattr(value, part)
                    else:
                        raise _ServiceError("invalid_reference", f"{part!r} is not a declared data field.")
            return _owned(value)

    def active(self, context: str | None = None) -> Any:
        with self._lock:
            head = self._context(context).head
            if head is None:
                raise _ServiceError("missing_state", "Set an initial control/variable with session.use first.")
            return self.get(head)

    def bind_reference(self, name: str, reference: ObjectRef) -> ObjectRef:
        """Protect an object version or retain a field with its source provenance."""
        _name(name, "name")
        if not isinstance(reference, ObjectRef):
            raise TypeError("reference must be ObjectRef.")
        with self._lock:
            oid = self._resolve_id(reference)
            source = self._objects[oid]
            if not reference.path:
                bindings = {**self._bindings, name: oid}
                extra = _size(bindings) - _size(self._bindings)
                if len(bindings) > self.max_objects or self._retained_bytes() + extra > self.max_bytes:
                    raise _ServiceError("object_quota", "Binding quota exceeded; release named objects.")
                self._bindings = bindings
                self._revision += 1
                return ObjectRef(oid)
            # The field is owned separately, but keeps the old configuration even
            # if its context has since been updated. Never invent resume authority.
            value = self.get(reference)
            result = self._publish(value, revision=self._revision, bind=name,
                                   method_key=source.method_key)
            stored = self._objects[result.object_id]
            stored.context, stored.configuration = source.context, source.configuration
            return result

    def _check_revision(self, revision: int) -> None:
        if self._revision != revision:
            raise _ServiceError("revision_conflict", "Session changed during execution; the candidate was not committed.")

    def _publish(self, value: Any, *, revision: int, context: _Context | None = None,
                 bind: str | None = None, activate: bool = False, method_key: Any = None,
                 extra_bytes: int = 0) -> ObjectRef:
        self._check_revision(revision)
        owned = _owned(value)
        size = _size(owned) + _size(method_key)
        oid = uuid4().hex
        bindings = {**self._bindings, bind: oid} if bind else self._bindings
        if len(bindings) > self.max_objects:
            raise _ServiceError("object_quota", "Binding count exceeds the configured object limit.")
        extra_bytes += _size(bindings) - _size(self._bindings)
        protected = set(self._bindings.values()) | {c.head for c in self._contexts.values() if c.head}
        # Old bindings remain protected during replacement; release is explicit.
        evict: list[str] = []
        count, total = len(self._objects) + 1, self._retained_bytes() + size + extra_bytes
        for old_oid, stored in self._objects.items():
            if count <= self.max_objects and total <= self.max_bytes:
                break
            if old_oid not in protected:
                evict.append(old_oid)
                count -= 1
                total -= stored.bytes
        if count > self.max_objects or total > self.max_bytes:
            raise _ServiceError("object_quota", "Retained object quota exceeded; release named objects or raise the limit.")
        if activate:
            if context is None:
                raise _ServiceError("missing_context", "Activation requires a context.")
            _compatible(context.layout, owned)
        self._check_revision(revision)
        for old in evict:
            del self._objects[old]
        self._objects[oid] = _Stored(owned, size, context.name if context else None,
                                    context.configuration if context else None, method_key)
        if bind:
            self._bindings[bind] = oid
        if activate:
            context.head, context.method_key = oid, method_key
        self._revision += 1
        return ObjectRef(oid)

    def load(self, name: str, factory: Callable[..., Any] | str | Path, *,
             parameters: Mapping[str, Any] | None = None, factory_name: str = "system",
             select: bool = True) -> dict[str, Any]:
        """Load an explicit callable or trusted project file; never import a search tree."""
        _name(name, "name")
        if type(select) is not bool:
            raise TypeError("select must be bool.")
        with self._lock:
            revision = self._revision
            if name in self._contexts:
                raise _ServiceError("context_exists", f"Context {name!r} already exists; choose a new name.")
            if len(self._contexts) >= self.max_objects:
                raise _ServiceError("object_quota", "Context count exceeds the configured object limit.")
            source: dict[str, Any] = {}
            if isinstance(factory, (str, Path)):
                path = Path(factory).expanduser().resolve(strict=True)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                module_name = f"_optimizer_project_{uuid4().hex}"
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    raise ValueError("Factory path must identify a Python source file.")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                try:
                    spec.loader.exec_module(module)
                    factory = getattr(module, factory_name)
                except BaseException:
                    sys.modules.pop(module_name, None)
                    raise
                source = {"path": str(path), "sha256": digest, "factory_name": factory_name}
            if not callable(factory):
                raise TypeError("factory must be callable.")
            parameter_values = _owned(dict(parameters or {}))
            factory_signature = inspect.signature(factory)
            bound = factory_signature.bind(**parameter_values)
            bound.apply_defaults()
            effective = _owned(dict(bound.arguments))

            def construct() -> Any:
                if source and hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
                    raise _ServiceError("source_changed", "Factory source changed; load a new named context.")
                args = inspect.BoundArguments(factory_signature, _owned(effective))
                return factory(*args.args, **args.kwargs)

            system = construct()
            family, layout = _validate_system(system)
            recipe = {**source, "callable": f"{getattr(factory, '__module__', type(factory).__module__)}:{getattr(factory, '__qualname__', type(factory).__qualname__)}",
                      "parameters": effective, "durability": "source_reference" if source else "ephemeral"}
            if self._retained_bytes() + _size(recipe) + _size(layout) > self.max_bytes:
                raise _ServiceError("object_quota", "Retained context recipe/layout exceeds the byte limit.")
            self._check_revision(revision)
            self._contexts[name] = _Context(name, system, family, layout, construct, recipe)
            if select:
                self._selected = name
            self._revision += 1
            return {"name": name, "family": family, "recipe": _owned(recipe)}

    def attach(self, name: str, system: Any, *, clone: Callable[[], Any] | None = None,
               select: bool = True) -> dict[str, Any]:
        """Attach Python-only systems; mutation requires an independent clone recipe."""
        _name(name, "name")
        if clone is not None and not callable(clone):
            raise TypeError("clone must be callable or None.")
        if type(select) is not bool:
            raise TypeError("select must be bool.")
        with self._lock:
            revision = self._revision
            if name in self._contexts:
                raise _ServiceError("context_exists", f"Context {name!r} already exists.")
            if len(self._contexts) >= self.max_objects:
                raise _ServiceError("object_quota", "Context count exceeds the configured object limit.")
            family, layout = _validate_system(system)
            if self._retained_bytes() + _size(layout) > self.max_bytes:
                raise _ServiceError("object_quota", "Retained context layout exceeds the byte limit.")
            self._check_revision(revision)
            self._contexts[name] = _Context(name, system, family, layout, clone, {"durability": "ephemeral"})
            if select:
                self._selected = name
            self._revision += 1
            return {"name": name, "family": family}

    def select(self, name: str) -> dict[str, Any]:
        with self._lock:
            context = self._context(name)
            self._selected = context.name
            self._revision += 1
            return {"selected": context.name}

    def use(self, value: Any, *, context: str | None = None, bind: str | None = None) -> ObjectRef:
        _name(bind, "bind", optional=True)
        with self._lock:
            c = self._context(context)
            key = None
            if isinstance(value, ObjectRef):
                stored = self._objects[self._resolve_id(value)]
                value = self.get(value)
                if isinstance(value, (RunState, OptimizerResult, ShootingResult)) and stored.context == c.name and stored.configuration == c.configuration:
                    key = stored.method_key
            if isinstance(value, (RunState, OptimizerResult)) and key is None:
                raise _ServiceError("unsupported_continuation", "Full method state requires a result reference from the same configuration; use its controls for a new start.")
            return self._publish(value, revision=self._revision, context=c, bind=bind, activate=True, method_key=key)

    def release(self, name: str) -> dict[str, Any]:
        with self._lock:
            if name not in self._bindings:
                raise _ServiceError("missing_object", f"No binding {name!r}.")
            oid = self._bindings.pop(name)
            if oid not in self._bindings.values() and all(c.head != oid for c in self._contexts.values()):
                del self._objects[oid]
            self._revision += 1
            return {"released": name}

    def update(self, *, context: str | None = None, updates: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            c = self._context(context)
            revision = self._revision
            system = c.isolated()
            candidate = system.with_secondary(**_owned(dict(updates)))
            family, layout = _validate_system(candidate)
            if family != c.family:
                raise _ServiceError("invalid_system", "Secondary update changed family.")
            try:
                if family == "olgs":
                    _validate_control_spec_compatibility(c.layout, layout)
                else:
                    _validate_shooting_spec_compatibility(c.layout, layout)
            except (ValueError, TypeError) as exc:
                raise _ServiceError("incompatible_layout", "Load a new context for a changed layout.") from exc
            if c.head:
                _compatible(layout, self._objects[c.head].value)
            old_factory = c.factory
            payload = _owned(dict(updates))
            recipe = {**c.recipe, "secondary_updates": [*c.recipe.get("secondary_updates", []), payload]}
            delta = _size(recipe) + _size(layout) - _size(c.recipe) - _size(c.layout)
            if self._retained_bytes() + delta > self.max_bytes:
                raise _ServiceError("object_quota", "Updated context recipe exceeds the byte limit.")
            def construct() -> Any:
                return old_factory().with_secondary(**_owned(payload))
            # Retain physical input, discard metrics and private method memory.
            physical = _physical(self._objects[c.head].value) if c.head else None
            self._check_revision(revision)
            if physical is not None:
                head = self._publish(physical, revision=revision, context=c, activate=True, extra_bytes=delta)
                c.head = head.object_id
            else:
                self._revision += 1
            c.system, c.layout, c.factory = candidate, layout, construct
            c.configuration += 1
            c.method_key = None
            c.recipe = recipe
            if c.head:
                self._objects[c.head].configuration = c.configuration
            return {"name": c.name, "configuration": c.configuration}
