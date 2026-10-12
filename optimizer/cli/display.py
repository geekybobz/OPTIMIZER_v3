"""Bounded data-only summaries; never evaluate physics for display."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from optimizer.controls import Controls
from optimizer.operations.contracts import OperationResponse
from optimizer.operations.service import _summary
from optimizer.serialization import to_plain
from optimizer.system_olss import ShootingVariables


def _describe(value: Any, depth: int = 0) -> Any:
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return _describe(value.item(), depth)
    if value is None or isinstance(value, (str, bool, int, float, complex)):
        return value if not isinstance(value, str) or len(value) <= 512 else value[:512] + "…"
    if isinstance(value, Controls):
        return {"type": "Controls", "keys": list(value.spec.keys), "shape": list(value.spec.shape),
                "dtype": str(value.spec.dtype)}
    if isinstance(value, ShootingVariables):
        return {"type": "ShootingVariables", "names": list(value.spec.names), "size": value.spec.size}
    if depth >= 3:
        return {"type": type(value).__name__}
    if isinstance(value, Mapping):
        items = {str(k): _describe(v, depth + 1) for k, v in list(value.items())[:24]}
        return items if len(value) <= 24 else {"items": items, "omitted": len(value) - 24}
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "length": len(value),
                "items": [_describe(v, depth + 1) for v in value[:16]], "omitted": max(0, len(value) - 16)}
    summary = _summary(value)
    if "metrics" not in summary and dataclasses.is_dataclass(value):
        summary["fields"] = {field.name: _describe(getattr(value, field.name), depth + 1)
                             for field in dataclasses.fields(value)[:12] if not field.name.startswith("_")}
    return summary


def _render(response: OperationResponse, *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(response.to_dict(), allow_nan=False, separators=(",", ":"))
    if not response.ok:
        return f"error [{response.error['code']}]: {response.error['message']}"
    summary = response.metadata.get("summary", {})
    binding = response.metadata.get("binding")
    names = f"{binding} = " if binding else ""
    refs = ",".join(ref.object_id[:8] for ref in response.object_refs)
    head_before = {n: c.get("head") for n, c in response.state_before.get("contexts", {}).items()}
    head_after = {n: c.get("head") for n, c in response.state_after.get("contexts", {}).items()}
    facts = json.dumps(to_plain(summary), ensure_ascii=False, allow_nan=False)
    return f"{names}{response.operation}: {facts}" + (
        f" [ref={refs}]" if refs else "") + (
        f" [context={response.state_after.get('selected')}, head changed]" if head_after != head_before else "")


# ---------------------------------------------------------------------------
# Rich panels for a terminal. Piped output keeps `_render`; --json is unchanged.
# ---------------------------------------------------------------------------

_HINTS = {
    "wrong_family": "Select a context of the matching family: opt.session.select(\"name\");",
    "capability_missing": "The selected system does not provide a hook this tool needs.",
    "invalid_arguments": ":help <name> shows the accepted arguments.",
    "missing_object": "opt.session.state(); lists the retained names.",
    "missing_state": "Set a start first, for example opt.session.use(u0);",
    "unknown_operation": ":help lists the commands.",
    "unsupported_syntax": "Use name = opt.namespace.operation(...); or name.field;",
    "missing_semicolon": "End each command with ';'.",
}
_NUMERIC = ("optimizers.", "shooting.", "utils.", "guesses.", "base.", "schedules.", "controls.")
_FACTS = ("name", "family", "selected", "configuration", "durability", "generation", "mode",
          "kind", "resume_level", "restored", "project", "data", "session")


def _short(operation: str | None) -> str:
    operation = operation or "result"
    return operation.rsplit(".", 1)[-1] if operation.startswith(_NUMERIC) else operation


def _title(response: OperationResponse) -> str:
    binding = response.metadata.get("binding")
    return f"{binding} = {_short(response.operation)}" if binding else _short(response.operation)


def _changed(response: OperationResponse) -> list[str]:
    before = {name: ctx.get("head") for name, ctx in (response.state_before or {}).get("contexts", {}).items()}
    after = {name: ctx.get("head") for name, ctx in (response.state_after or {}).get("contexts", {}).items()}
    return [name for name, head in after.items() if head != before.get(name)]


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return repr(value)
    if value is None or isinstance(value, (int, str)):
        return str(value)
    try:
        text = json.dumps(to_plain(value), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= 96 else text[:95] + "…"


def _grid(rows: list[tuple[str, Any]]):
    from rich.table import Table
    from rich.text import Text

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim")
    grid.add_column()
    for key, value in rows:
        grid.add_row(str(key), value if isinstance(value, Text) else _fmt(value))
    return grid


def _flag(value: Any):
    from rich.text import Text

    return Text("yes", style="green") if value is True else Text("no", style="red") if value is False else Text(_fmt(value))


def _metric_rows(metrics: Mapping[str, Any], first: tuple[str, ...]) -> list[tuple[str, Any]]:
    order = [key for key in first if key in metrics] + [key for key in metrics if key not in first]
    return [(key, metrics[key]) for key in order]


def _list_text(summary: Mapping[str, Any]) -> str:
    items = summary.get("items") or []
    kinds = sorted({item.get("type", type(item).__name__) for item in items if isinstance(item, Mapping)})
    names = sorted({name for item in items if isinstance(item, Mapping) for name in item.get("names", [])})
    detail = f" ({', '.join(names)})" if names else ""
    return f"{summary.get('type', 'list')} of {summary.get('length', len(items))} {', '.join(kinds) or 'items'}{detail}"


def _compact(response: OperationResponse, summary: Any, notes: list[str]):
    from rich.text import Text

    line = Text.assemble(("✓ ", "green"), (_title(response), "bold"))
    facts: list[str] = []
    if isinstance(summary, Mapping):
        kind = summary.get("type")
        if kind == "Controls":
            shape = "×".join(str(n) for n in summary.get("shape", []))
            facts.append(f"Controls {', '.join(summary.get('keys', []))} · {shape} · {summary.get('dtype')}")
        elif kind == "ShootingVariables":
            facts.append(f"ShootingVariables {', '.join(summary.get('names', []))} · size {summary.get('size')}")
        elif kind in ("list", "tuple"):
            facts.append(_list_text(summary))
        elif kind != "ObjectRef":
            recipe = summary.get("recipe") if isinstance(summary.get("recipe"), Mapping) else None
            facts += [f"{key} {_fmt(summary[key])}" for key in _FACTS if key in summary and not isinstance(summary[key], (Mapping, list))]
            if recipe and recipe.get("path"):
                facts.append(f"{str(recipe['path']).rsplit('/', 1)[-1]}:{recipe.get('factory_name')}")
    for fact in facts + notes:
        line.append("  ·  ", style="dim")
        line.append(fact)
    return line


_HELP_SECTIONS = (("session", "Session"), ("project", "Project"), ("checkpoints", "Checkpoints"),
                  ("storage", "Storage"))


def _help_overview(summary: Mapping[str, Any]):
    """Group command names instead of printing every raw signature."""
    from rich.console import Group
    from rich.text import Text

    names = [str(command).split("(", 1)[0].removeprefix("opt.") for command in summary.get("commands", [])]
    grouped: dict[str, list[str]] = {}
    for name in names:
        head, _, rest = name.partition(".")
        grouped.setdefault(head if rest else "", []).append(rest or head)
    rows: list[tuple[str, Any]] = [
        ("Terminal", ":help  ·  :help <name>  ·  :exit  (Ctrl-D also works)"),
        ("Discovery", "  ·  ".join(grouped.get("", []))),
    ]
    rows += [(title, "  ·  ".join(grouped[key])) for key, title in _HELP_SECTIONS if grouped.get(key)]
    history = [str(entry).partition(".")[2] or str(entry) for entry in summary.get("history", [])]
    if history:
        rows.append(("History", "  ·  ".join(history)))
    if summary.get("groups"):
        rows.append(("Tools", "  ·  ".join(summary["groups"])))
        rows.append(("", 'opt.list(group="optimizers"); lists one group'))
    if summary.get("lifetime"):
        rows.append(("Lifetime", summary["lifetime"]))
    if summary.get("deferred"):
        rows.append(("Deferred", "; ".join(summary["deferred"])))
    intro = Text(f"Commands end with ';'.  {summary.get('syntax', '')}", style="dim")
    outro = Text(":help <name> shows one command, for example :help adam or :help session.load", style="dim")
    return Group(intro, _grid(rows), outro)


def _help_one(summary: Mapping[str, Any]):
    """The facts a user needs about one operation, usage first."""
    rows: list[tuple[str, Any]] = []
    if summary.get("usage"):
        rows.append(("usage", summary["usage"]))
    elif summary.get("signature"):
        rows.append(("signature", summary["signature"]))
    for key in ("summary", "family", "variants", "returns", "requires", "classification"):
        if summary.get(key) not in (None, [], ""):
            rows.append((key, summary[key]))
    effect = "  ·  ".join(str(summary[key]) for key in ("effect", "continuation") if summary.get(key))
    if effect:
        rows.append(("effect", effect))
    if summary.get("reason"):
        rows.append(("unavailable", summary["reason"]))
    return _grid(rows)


def _rich_renderable(response: OperationResponse, elapsed: float | None = None):
    """Build one rich renderable for a response; data only, no evaluation."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    if not response.ok:
        error = response.error or {}
        code = str(error.get("code", "error"))
        body = Text(str(error.get("message", "")))
        hint = _HINTS.get(code)
        if hint:
            body.append("\n" + hint, style="dim")
        return Panel(body, title=f"[bold red]✗ {code}[/]", title_align="left", border_style="red", expand=False)

    summary = response.metadata.get("summary")
    notes = [f"{', '.join(_changed(response))} updated"] if _changed(response) else []
    if elapsed is not None and elapsed >= 0.05:
        notes.append(f"{elapsed:.2f} s")
    subtitle = "  ·  ".join(notes) or None
    kind = summary.get("type") if isinstance(summary, Mapping) else None

    def panel(body, border):
        return Panel(body, title=f"[bold]{_title(response)}[/]", title_align="left",
                     subtitle=subtitle, subtitle_align="right", border_style=border, expand=False)

    if kind == "OptimizerResult":
        metrics = summary.get("metrics") or {}
        rows = [("iterations", summary.get("iterations")), ("stop", summary.get("stop_reason"))]
        return panel(_grid(rows + _metric_rows(metrics, ("J",))), "green")
    if kind == "ShootingResult":
        rows = [("qualified", _flag(summary.get("qualified"))), ("verified", _flag(summary.get("verified"))),
                ("solver converged", _flag(summary.get("solver_converged"))),
                ("stop", summary.get("stop_reason")), ("iterations", summary.get("iterations"))]
        if summary.get("error"):
            rows.append(("error", summary["error"]))
        rows += _metric_rows(summary.get("metrics") or {}, ("residual_norm",))
        return panel(_grid(rows), "green" if summary.get("qualified") else "yellow")
    if kind == "MultiStartResult":
        fields = summary.get("fields") or {}
        metric = fields.get("selection_metric") or "metric"
        counts = Text.assemble(
            ("results ", "dim"), str((fields.get("results") or {}).get("length", "?")),
            ("   qualified ", "dim"), str((fields.get("qualified") or {}).get("length", "?")),
            ("   unique ", "dim"), str((fields.get("unique_qualified") or {}).get("length", "?")),
            ("   ranked by ", "dim"), str(metric))
        table = Table(box=None, padding=(0, 2), show_edge=False)
        table.add_column("ranked[i]", justify="right", style="dim")
        table.add_column(str(metric), justify="right")
        table.add_column("residual", justify="right")
        table.add_column("qualified")
        for rank, item in enumerate(((fields.get("ranked") or {}).get("items") or [])[:5]):
            metrics = item.get("metrics") or {}
            table.add_row(str(rank), _fmt(metrics.get(metric)), _fmt(metrics.get("residual_norm")),
                          _flag(item.get("qualified")))
        from rich.console import Group

        return panel(Group(counts, table), "green")
    if response.operation == "help" and isinstance(summary, Mapping) and "commands" in summary:
        return panel(_help_overview(summary), "cyan")
    if response.operation == "help" and isinstance(summary, Mapping) and "id" in summary:
        return panel(_help_one(summary), "cyan")
    if response.operation == "list" and isinstance(summary, Mapping) and "operations" in summary:
        table = Table(box=None, padding=(0, 2), show_edge=False)
        for column in ("id", "classification", "effect"):
            table.add_column(column)
        for entry in summary.get("operations", []):
            table.add_row(*(str(entry.get(column, "")) for column in ("id", "classification", "effect")))
        return panel(table, "cyan")
    if (kind in ("Controls", "ShootingVariables", "ObjectRef", "list", "tuple")
            or str(response.operation or "").startswith(("session.", "project.", "checkpoints.", "storage."))
            and response.operation not in ("session.state", "session.discover")):
        return _compact(response, summary, notes)
    if isinstance(summary, Mapping) and isinstance(summary.get("fields"), Mapping):
        # Dataclass summaries (Evaluation, RunState, ...): expand one level.
        rows: list[tuple[str, Any]] = []
        for key, value in summary["fields"].items():
            if key == "metrics" and isinstance(value, Mapping):
                rows += _metric_rows(value, ("J",))
            elif isinstance(value, Mapping) and value.get("type") == "Controls":
                shape = "×".join(str(n) for n in value.get("shape", []))
                rows.append((key, f"Controls {', '.join(value.get('keys', []))} · {shape}"))
            else:
                rows.append((key, value))
        return panel(_grid(rows), "cyan")
    if isinstance(summary, Mapping):
        return panel(_grid([(key, value) for key, value in summary.items()]), "cyan")
    label = Text.assemble((f"{_title(response)}  ", "bold")) if response.metadata.get("binding") else Text()
    return Text.assemble(("→ ", "dim"), label, _fmt(summary))


class _Display:
    """Print responses as JSON lines, rich panels on a terminal, or compact text."""

    def __init__(self, *, json_output: bool = False, stream=None, rich: bool | None = None) -> None:
        import sys

        self.json_output = json_output
        self.stream = stream if stream is not None else sys.stdout
        self.timings: dict[str, float] = {}
        self.console = None
        if not json_output:
            if rich is None:
                try:
                    rich = bool(self.stream.isatty())
                except Exception:  # noqa: BLE001
                    rich = False
            if rich:
                try:
                    from rich.console import Console

                    self.console = Console(file=self.stream, highlight=False)
                except ImportError:
                    self.console = None

    def emit(self, response: OperationResponse) -> None:
        if self.console is None:
            print(_render(response, json_output=self.json_output), file=self.stream, flush=True)
            return
        self.console.print(_rich_renderable(response, self.timings.get(response.request_id)))
