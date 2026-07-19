"""Reset and pruning helpers for blackbox run folders."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterable

from optimizer.blackbox.reader import read_manifest
from optimizer.blackbox.records import json_safe, utc_now


def _append_admin_record(run_dir: Path, kind: str, payload: dict[str, Any]) -> None:
    ledger = run_dir / "ledger.jsonl"
    seq = 0
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seq += 1
    record = {
        "v": 1,
        "seq": seq + 1,
        "kind": kind,
        "run_id": None,
        "t_utc": utc_now(),
        **json_safe(payload),
    }
    with ledger.open("a", encoding="utf-8") as handle:
        import json

        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def reset(run_dir: Path | str, *, section: str | None = None) -> dict[str, Any]:
    """Clear a blackbox run fully or by section.

    ``section=None`` clears the full run folder contents.  Supported partial sections
    are ``"arrays"``, ``"ledger"``, and ``"final"``.
    """

    root = Path(run_dir)
    if not root.exists():
        return {"kind": "reset", "section": section, "removed": [], "missing": True}

    removed: list[str] = []
    if section is None:
        for child in root.iterdir():
            removed.append(str(child.relative_to(root)))
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        return {"kind": "reset", "section": None, "removed": removed, "missing": False}

    if section == "arrays":
        arrays = root / "arrays"
        if arrays.exists():
            for child in arrays.iterdir():
                removed.append(str(child.relative_to(root)))
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        _append_admin_record(root, "reset", {"section": "arrays", "removed": removed})
        return {"kind": "reset", "section": "arrays", "removed": removed, "missing": False}

    if section == "ledger":
        ledger = root / "ledger.jsonl"
        if ledger.exists():
            removed.append(str(ledger.relative_to(root)))
            ledger.unlink()
        return {"kind": "reset", "section": "ledger", "removed": removed, "missing": False}

    if section == "final":
        manifest = root / "blackbox.json"
        if manifest.exists():
            data = read_manifest(root)
            data["final"] = {}
            data["status"] = "running"
            manifest.write_text(__import__("json").dumps(json_safe(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _append_admin_record(root, "reset", {"section": "final", "removed": []})
        return {"kind": "reset", "section": "final", "removed": [], "missing": False}

    raise ValueError("section must be None, 'arrays', 'ledger', or 'final'.")


def prune(
    run_dir: Path | str,
    *,
    keep_labels: Iterable[str] = ("initial", "best", "final", "repair", "failure", "threshold"),
) -> dict[str, Any]:
    """Prune array artifacts that are not referenced by important labels."""

    root = Path(run_dir)
    arrays = root / "arrays"
    if not arrays.exists():
        return {"kind": "prune", "removed": [], "kept": [], "missing": True}

    keep_terms = tuple(str(label) for label in keep_labels)
    kept: list[str] = []
    removed: list[str] = []
    for child in arrays.iterdir():
        rel = str(child.relative_to(root))
        if any(term in child.name for term in keep_terms):
            kept.append(rel)
            continue
        removed.append(rel)
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    _append_admin_record(root, "prune", {"removed": removed, "kept": kept, "keep_labels": list(keep_terms)})
    return {"kind": "prune", "removed": removed, "kept": kept, "missing": False}
