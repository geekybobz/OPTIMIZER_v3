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
