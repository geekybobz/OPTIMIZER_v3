"""Restricted AST interpreter translating typed calls into service requests.

There is deliberately no eval/exec and no second numerical tool registry.
"""

from __future__ import annotations

import ast
import inspect
import math
from dataclasses import replace
from typing import Callable
from uuid import uuid4

from optimizer.cli.discovery import _discover
from optimizer.cli.display import _describe
from optimizer.cli.input import _frame
from optimizer.operations import ObjectRef, OperationRequest, OperationResponse, OperationService
from optimizer.operations.contracts import _ServiceError

_MAX_NODES, _MAX_DEPTH, _MAX_CALLS = 8192, 32, 128
_MUTATING_SESSION = {"session.load", "session.attach", "session.select", "session.update", "session.release"}
_SESSION_METHODS = {"session.load": "load", "session.attach": "attach", "session.select": "select",
                    "session.use": "use", "session.update": "update", "session.release": "release",
                    "session.get": "get", "session.state": "snapshot", "session.bind": "bind_reference"}


def _public(name: str) -> bool:
    return name.isidentifier() and not name.startswith("_")


def _operation(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id != "opt" or not parts or not all(_public(p) for p in parts):
        raise _ServiceError("unsupported_syntax", "Calls must use a public registered opt.operation(...).")
    return ".".join(reversed(parts))


def _literal_number(node: ast.AST) -> int | float | complex:
    if isinstance(node, ast.Constant) and type(node.value) in {int, float, complex}:
        value = node.value
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _literal_number(node.operand)
        value = -value if isinstance(node.op, ast.USub) else value
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        # Only Python's real +/- imaginary literal form, never generic arithmetic.
        left, right = _literal_number(node.left), _literal_number(node.right)
        if type(left) not in {int, float} or type(right) is not complex or right.real != 0:
            raise _ServiceError("unsupported_syntax", "Only literal real +/- imaginary numbers support binary operators.")
        value = left + right if isinstance(node.op, ast.Add) else left - right
    else:
        raise _ServiceError("unsupported_syntax", "Signs apply only to numerical literals.")
    if isinstance(value, complex):
        finite = math.isfinite(value.real) and math.isfinite(value.imag)
    else:
        finite = type(value) is int or math.isfinite(value)
    if not finite:
        raise _ServiceError("invalid_literal", "Numerical literals must be finite.")
    return value


def _validate(node: ast.AST, depth: int = 0) -> None:
    if depth > _MAX_DEPTH:
        raise _ServiceError("input_limit", f"Expression nesting exceeds {_MAX_DEPTH}.")
    if isinstance(node, ast.Constant):
        if node.value is not None and type(node.value) not in {str, bool, int, float, complex}:
            raise _ServiceError("unsupported_syntax", "Unsupported literal type.")
        if type(node.value) in {int, float, complex}:
            _literal_number(node)
    elif isinstance(node, (ast.UnaryOp, ast.BinOp)):
        _literal_number(node)
    elif isinstance(node, ast.Name):
        if not _public(node.id) or node.id == "opt":
            raise _ServiceError("unsupported_syntax", "Use a public retained name; opt is reserved.")
    elif isinstance(node, (ast.List, ast.Tuple)):
        for value in node.elts:
            _validate(value, depth + 1)
    elif isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if key is None:
                raise _ServiceError("unsupported_syntax", "Dictionary unpacking is not supported.")
            _validate(key, depth + 1)
            _validate(value, depth + 1)
    elif isinstance(node, ast.Call):
        _operation(node.func)
        for value in node.args:
            _validate(value, depth + 1)
        seen = set()
        for keyword in node.keywords:
            if keyword.arg is None or not _public(keyword.arg) or keyword.arg in seen:
                raise _ServiceError("unsupported_syntax", "Keywords must be unique public names; unpacking is not supported.")
            seen.add(keyword.arg)
            _validate(keyword.value, depth + 1)
    elif isinstance(node, ast.Attribute):
        if not _public(node.attr):
            raise _ServiceError("unsupported_syntax", "Private/dunder data access is forbidden.")
        _validate(node.value, depth + 1)
    elif isinstance(node, ast.Subscript):
        _validate(node.value, depth + 1)
        if not isinstance(node.slice, ast.Constant) or not (
                type(node.slice.value) is int and node.slice.value >= 0 or
                isinstance(node.slice.value, str) and node.slice.value and not node.slice.value.startswith("_")):
            raise _ServiceError("unsupported_syntax", "Indices must be nonnegative integers or public literal keys.")
    else:
        raise _ServiceError("unsupported_syntax", f"{type(node).__name__} is outside the command grammar.")


def _parse(source: str) -> list[tuple[ast.AST, str | None, str]]:
    commands = []
    count = calls = 0
    for raw in _frame(source):
        try:
            tree = ast.parse(raw.strip(), mode="exec")
        except (SyntaxError, ValueError, RecursionError) as exc:
            raise _ServiceError("syntax_error", str(exc)) from exc
        nodes = list(ast.walk(tree))
        count += len(nodes)
        calls += sum(isinstance(n, ast.Call) for n in nodes)
        if count > _MAX_NODES or calls > _MAX_CALLS:
            raise _ServiceError("input_limit", "Submission exceeds the node/call limit.")
        if len(tree.body) != 1:
            raise _ServiceError("missing_semicolon", "Each statement requires its own ';'.")
        statement, binding = tree.body[0], None
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                raise _ServiceError("unsupported_syntax", "Assignment requires one simple public name.")
            binding = statement.targets[0].id
            if not _public(binding) or binding == "opt":
                raise _ServiceError("unsupported_syntax", "Assignment cannot replace opt or a private name.")
        elif not isinstance(statement, ast.Expr):
            raise _ServiceError("unsupported_syntax", "Only calls, assignments and data inspection are supported.")
        expression = statement.value
        _validate(expression)
        if binding and isinstance(expression, ast.Call) and (
                _operation(expression.func) in _MUTATING_SESSION | {"session.exit", "session.bind"}):
            raise _ServiceError("unsupported_syntax", "Run this session mutation without assignment, then inspect session.state().")
        commands.append((expression, binding, raw))
    return commands


class _Halt(Exception):
    pass


class _Terminal:
    """Foreground coordinator with an optional explicit future-history observer.

    The observer receives raw text, canonical request and terminal response. It
    is an informational hook, not a durable commit protocol or installed writer.
    """

    def __init__(self, service: OperationService | None = None, *,
                 emit: Callable[[OperationResponse], None] | None = None,
                 observer: Callable[[dict], None] | None = None) -> None:
        self.service = service if service is not None else OperationService()
        self.emit = emit or (lambda response: None)
        self.observer = observer
        self.closed = False
        self._raw = ""
        self._interrupted = False
        self._request = None
        self._operation_name = None

    def _output(self, response: OperationResponse, request: OperationRequest | None = None) -> None:
        self.emit(response)
        if self.observer is not None:
            self.observer({"raw_command": self._raw, "request": request,
                           "response": response, "interrupted": self._interrupted})

    def _send(self, request: OperationRequest) -> ObjectRef | object:
        self._request = request
        response = self.service.execute(request)
        if response.ok:
            response = replace(response, metadata={**response.metadata,
                "summary": _describe(response.result), **({"binding": request.bind} if request.bind else {})})
        if isinstance(response.metadata.get("summary"), dict) and response.metadata["summary"].get("stop_reason") == "interrupted":
            self._interrupted = True
        self._output(response, request)
        if not response.ok or self._interrupted:
            raise _Halt()
        return response.object_refs[0] if response.object_refs else response.result

    def _help(self, operation: str | None = None) -> dict:
        if operation is not None:
            return self._info(operation)
        return {"syntax": "name = opt.namespace.operation(...); name.public_field;",
                "commands": ["opt.help(operation=None)", "opt.list(group=None, classification=None)",
                    "opt.info(operation)", "opt.inspect(value)", "opt.session.discover(root='.', depth=3)",
                    *[f"opt.{name}{inspect.signature(getattr(self.service.session, method))}"
                      for name, method in _SESSION_METHODS.items()],
                    *[f"opt.{name}{inspect.signature(method)}" for name,method in self.service._persistence.methods.items()], "opt.session.exit()"],
                "lifetime": "RAM unless project.init/open selects a durable experiment; project.status reports storage.",
                "deferred": ["CLI/MCP command history and legacy logger retirement (Task 5)",
                             "managed parallel jobs/control transforms (Task 6)"]}

    def _info(self, operation: str) -> dict:
        if not isinstance(operation, str):
            raise _ServiceError("invalid_arguments", "operation must be a name string.")
        name = operation.removeprefix("opt.")
        if name in _SESSION_METHODS:
            return {"id": name, "signature": str(inspect.signature(getattr(self.service.session, _SESSION_METHODS[name]))),
                    "classification": "session_operation"}
        if name in self._conveniences():
            return {"id": name, "signature": str(inspect.signature(self._conveniences()[name])),
                    "classification": "terminal_convenience"}
        return self.service.info(name)

    def _list(self, group: str | None = None, classification: str | None = None) -> dict:
        for name, value in (("group", group), ("classification", classification)):
            if value is not None and not isinstance(value, str):
                raise _ServiceError("invalid_arguments", f"{name} must be a name string or None.")
        entries = [entry for entry in self.service.list() if (group is None or entry["id"].split(".", 1)[0] == group)
                   and (classification is None or entry["classification"] == classification)]
        entries = list({entry["id"]: entry for entry in entries}.values())
        return {"operations": [{k: entry[k] for k in ("id", "classification", "effect")} for entry in entries],
                "count": len(entries), "detail": "opt.info('operation') shows its signature and availability reason."}

    def _inspect(self, value):
        return self.service.session.get(value) if isinstance(value, ObjectRef) else value

    def _exit(self):
        persistent=self.service._persistence.store is not None
        if self.service._persistence.store is not None:
            self._send(OperationRequest('project.close'))
        self.closed = True
        return {"session": "closed", "durability": "persistent" if persistent else "ephemeral"}

    def _conveniences(self) -> dict:
        return {"help": self._help, "info": self._info, "list": self._list,
                "inspect": self._inspect, "session.discover": _discover, "session.exit": self._exit}

    def _value(self, node: ast.AST, binding: str | None = None):
        if isinstance(node, ast.Call):
            return self._call(node, binding)
        if isinstance(node, ast.Name):
            with self.service.session._lock:
                return ObjectRef(self.service.session._resolve_id(node.id))
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            ref = self._value(node.value)
            if not isinstance(ref, ObjectRef):
                raise _ServiceError("invalid_reference", "Field/index inspection requires a retained object.")
            part = node.attr if isinstance(node, ast.Attribute) else node.slice.value
            return ObjectRef(ref.object_id, ref.path + (part,))
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.UnaryOp, ast.BinOp)):
            return _literal_number(node)
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [self._value(v) for v in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        return {self._value(k): self._value(v) for k, v in zip(node.keys, node.values)}

    def _call(self, node: ast.Call, binding: str | None):
        operation = _operation(node.func)
        args = [self._value(value) for value in node.args]
        kwargs = {keyword.arg: self._value(keyword.value) for keyword in node.keywords}
        self._operation_name, self._request = operation, None
        if operation in self.service._persistence.methods:
            method=self.service._persistence.methods[operation]
            bound=inspect.signature(method).bind(*args, **kwargs)
            return self._send(OperationRequest(operation, dict(bound.arguments), bind=binding))
        conveniences = self._conveniences()
        if operation in conveniences:
            method = conveniences[operation]
            before = self.service.session.snapshot()
            inspect.signature(method).bind(*args, **kwargs)
            if operation not in {"inspect"}:
                args, kwargs = self.service._resolve(args), self.service._resolve(kwargs)
            value = method(*args, **kwargs)
            refs = ()
            if binding:
                source = args[0] if args else kwargs.get("value")
                ref = (self._send(OperationRequest('session.bind', {'name':binding,'reference':source}))
                       if operation == "inspect" and isinstance(source, ObjectRef)
                       else self.service.register_reference(binding, value))
                refs = (ref,)
            summary = _describe(value)
            # Discovery/catalog output is already bounded, and must show all
            # candidates/tools instead of truncating the list to sixteen entries.
            if operation in {"list", "session.discover", "help", "info"}:
                summary = value
            response = OperationResponse(uuid4().hex, operation, "ok", value, refs, before,
                self.service.session.snapshot(), metadata={"summary": summary, **({"binding": binding} if binding else {})})
            self._output(response, OperationRequest(operation, {**kwargs, "positional": args}, bind=binding))
            return refs[0] if refs else value
        if operation in _SESSION_METHODS:
            method = getattr(self.service.session, _SESSION_METHODS[operation])
            bound = inspect.signature(method).bind(*args, **kwargs)
            if binding and operation == "session.use":
                if "bind" in bound.arguments:
                    raise _ServiceError("invalid_arguments", "Use assignment or bind, not both.")
                bound.arguments["bind"] = binding
            result = self._send(OperationRequest(operation, dict(bound.arguments)))
            if binding and operation in {"session.get", "session.state"}:
                # A get alias preserves source provenance, including method state.
                if operation == "session.get":
                    source = bound.arguments["reference"]
                    source = source if isinstance(source, ObjectRef) else ObjectRef(source)
                    result = self._send(OperationRequest("session.bind", {"name": binding, "reference": source}))
                else:
                    result = self.service.register_reference(binding, result)
            return result
        item, adapter = self.service._registry.resolve(operation)
        if adapter.classification not in {"tool", "constructor"}:
            raise _ServiceError("adapter_unavailable", f"{item.id}: {adapter.reason}")
        options = {name: kwargs.pop(name) for name in ("context", "activate", "bind") if name in kwargs}
        if binding:
            if "bind" in options:
                raise _ServiceError("invalid_arguments", "Use assignment or bind, not both.")
            options["bind"] = binding
        signature = inspect.signature(item.public_object())
        # The selected context supplies system. All other native positions stay
        # intact; native **kwargs are still validated by the service adapter.
        parameters = [p for p in signature.parameters.values() if p.name != "system"]
        bound = signature.replace(parameters=parameters).bind_partial(*args, **kwargs)
        arguments = {}
        for name, value in bound.arguments.items():
            if signature.parameters[name].kind == inspect.Parameter.VAR_KEYWORD:
                arguments.update(value)
            elif signature.parameters[name].kind == inspect.Parameter.VAR_POSITIONAL:
                raise _ServiceError("invalid_arguments", "Variadic positional adapters are not supported.")
            else:
                arguments[name] = value
        return self._send(OperationRequest(item.id, arguments, **options))

    def submit(self, source: str) -> int:
        """Execute a bounded submission; 0 success, 2 input/dispatch, 130 interrupt."""
        self._raw, self._interrupted = source, False
        self._request = None
        self._operation_name = None
        before = self.service.session.snapshot()
        try:
            commands = _parse(source)  # Validate every statement before any operation.
            for expression, binding, raw in commands:
                if self.closed:
                    break
                self._raw = raw
                self._request, self._operation_name = None, None
                with self.service._writer:
                    before = self.service.session.snapshot()
                    value = self._value(expression, binding)
                    if not isinstance(expression, ast.Call):
                        if binding:
                            if isinstance(value, ObjectRef):
                                value = self._send(OperationRequest("session.bind", {"name": binding, "reference": value}))
                            else:
                                value = self.service.register_reference(binding, self.service._resolve(value))
                        actual = self.service.session.get(value) if isinstance(value, ObjectRef) else value
                        response = OperationResponse(uuid4().hex, "inspect", "ok", actual,
                            (value,) if isinstance(value, ObjectRef) else (), before, self.service.session.snapshot(),
                            metadata={"summary": _describe(actual), **({"binding": binding} if binding else {})})
                        self._output(response, OperationRequest("inspect", {"value": value}, bind=binding))
            return 0
        except _Halt:
            return 130 if self._interrupted else 2
        except KeyboardInterrupt:
            self._interrupted = True
            after = self.service.session.snapshot()
            response = OperationResponse(uuid4().hex, None, "error", state_before=before, state_after=after,
                error={"code": "interrupted", "type": "KeyboardInterrupt", "message": "Command interrupted; inspect the last committed head."})
            self._output(response, self._request)
            return 130
        except Exception as exc:
            state = self.service.session.snapshot()
            response = OperationResponse(uuid4().hex, self._request.operation if self._request else self._operation_name, "error", state_before=before, state_after=state,
                error={"code": exc.code if isinstance(exc, _ServiceError) else "invalid_arguments",
                       "type": type(exc).__name__, "message": str(exc)})
            self._output(response, self._request)
            return 2

    def completions(self) -> list[str]:
        entries = self.service.list()
        names = ["opt." + entry["id"] for entry in entries if entry["classification"] in {"tool", "constructor", "lifecycle"}]
        names += ["opt." + item.name for item in self.service._registry.items.values() if item.root
                  and self.service._registry.adapters[item.id].classification in {"tool", "constructor"}]
        names += ["opt." + name for name in (*_SESSION_METHODS, *self._conveniences())]
        names += list(self.service.session.snapshot()["bindings"])
        return sorted(set(names))
