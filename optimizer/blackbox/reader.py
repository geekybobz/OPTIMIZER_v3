"""Readers for blackbox run folders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def manifest_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / "blackbox.json"


def ledger_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / "ledger.jsonl"


def read_manifest(run_dir: Path | str) -> dict[str, Any]:
    """Read ``blackbox.json``."""

    path = manifest_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"blackbox manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def iter_records(run_dir: Path | str) -> Iterable[dict[str, Any]]:
    """Yield ledger records from ``ledger.jsonl``."""

    path = ledger_path(run_dir)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def read_records(run_dir: Path | str, *, kind: str | None = None) -> list[dict[str, Any]]:
    """Return all ledger records, optionally filtered by kind."""

    records = list(iter_records(run_dir))
    if kind is not None:
        records = [record for record in records if record.get("kind") == kind]
    return records


def latest_record(run_dir: Path | str, *, kind: str | None = None) -> dict[str, Any] | None:
    """Return the newest record, optionally filtered by kind."""

    records = read_records(run_dir, kind=kind)
    return records[-1] if records else None


def get_path(payload: dict[str, Any], path: str, default: Any = None) -> Any:
    """Read a dotted path from a nested dictionary."""

    current: Any = payload
    for part in str(path).split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def series(run_dir: Path | str, keys: Iterable[str]) -> dict[str, list[Any]]:
    """Return time series for dotted paths from iteration records."""

    key_list = [str(key) for key in keys]
    out = {key: [] for key in key_list}
    out["iteration"] = []
    for record in read_records(run_dir, kind="iteration"):
        out["iteration"].append(record.get("i", record.get("iteration")))
        for key in key_list:
            out[key].append(get_path(record, key))
    return out
