"""Bounded source discovery without importing any candidate."""

from __future__ import annotations

import ast
import os
from pathlib import Path

from optimizer.operations.contracts import _ServiceError


def _discover(root: str = ".", depth: int = 3) -> dict:
    if not isinstance(root, str) or type(depth) is not int or not 0 <= depth <= 12:
        raise _ServiceError("invalid_arguments", "root must be a path string; depth must be an integer from 0 to 12.")
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise _ServiceError("invalid_arguments", f"Not a directory: {path}")
    candidates, scanned, truncated = [], 0, False
    skipped = {"__pycache__", "node_modules", "build", "dist", "venv", "env"}
    for directory, dirs, files in os.walk(path, followlinks=False):
        level = len(Path(directory).relative_to(path).parts)
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in skipped
                         and not (Path(directory) / d).is_symlink() and level < depth)
        for name in sorted(files):
            if not name.endswith(".py") or name.startswith("."):
                continue
            scanned += 1
            if scanned > 512:
                truncated = True
                break
            file = Path(directory) / name
            try:
                if file.is_symlink() or file.stat().st_size > 262144:
                    continue
                tree = ast.parse(file.read_text(encoding="utf-8"))
                factories = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                             and node.name in {"system", "System"}]
                if factories:
                    candidates.append({"path": str(file), "factories": factories})
            except (OSError, UnicodeError, SyntaxError, RecursionError):
                continue
        if truncated:
            break
    return {"root": str(path), "depth": depth, "candidates": candidates,
            "scanned": min(scanned, 512), "truncated": truncated,
            "selection": "Load an explicit path and factory_name; no candidate is imported automatically."}
