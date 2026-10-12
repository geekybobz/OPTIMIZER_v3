#!/usr/bin/env python3
"""Print a runtime-derived context snapshot for this OPTIMIZER V4 checkout.

The library owns optimizer and system-contract self-description. This
small command only combines those public payloads with workspace and Git state.
It does not parse implementation source, evaluate a system, or modify Git state.

    conda run --no-capture-output -n Optimizer python optimizer_context.py
    conda run --no-capture-output -n Optimizer python optimizer_context.py --json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


SCHEMA_VERSION = "optimizer.context-command/v1"
ROOT = Path(__file__).resolve().parent


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _import_checkout_optimizer() -> tuple[Any, Path]:
    """Import exactly the package next to this command, not an installed copy."""

    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    import optimizer as opt

    module_file = Path(getattr(opt, "__file__", "")).resolve()
    expected = (ROOT / "optimizer" / "__init__.py").resolve()
    if module_file != expected:
        raise RuntimeError(
            "optimizer_context must use this checkout's optimizer package; "
            f"expected {expected}, imported {module_file}."
        )
    return opt, module_file


def _git(*args: str) -> tuple[int, str]:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode, completed.stdout.rstrip()


def _git_snapshot() -> dict[str, Any]:
    branch_code, branch = _git("branch", "--show-current")
    head_code, head = _git("rev-parse", "--short", "HEAD")
    status_code, raw_status = _git("status", "--short")
    paths: list[dict[str, str]] = []
    by_state: dict[str, int] = {}

    if status_code == 0:
        for line in raw_status.splitlines():
            if len(line) < 4:
                continue
            state = line[:2]
            paths.append({"state": state, "path": line[3:]})
            by_state[state] = by_state.get(state, 0) + 1

    return {
        "available": branch_code == 0 and head_code == 0,
        "branch": branch if branch_code == 0 else None,
        "head": head if head_code == 0 else None,
        "changes": {
            "total": len(paths),
            "by_state": dict(sorted(by_state.items())),
            "paths": paths,
        },
    }


def _workspace_inventory() -> dict[str, Any]:
    """List a few stable handoff anchors without reading their contents."""

    documentation = [
        ROOT / "README.md",
        ROOT / "INDEX.md",
        ROOT / "optimizer" / "mcp" / "MCP.md",
        ROOT / "optimizer" / "system_olgs" / "CONTRACT.md",
        ROOT / "optimizer" / "core" / "CORE_ENGINE.md",
    ]
    tests = sorted((ROOT / "tests").glob("test_*.py"))
    return {
        "documentation": [
            {
                "path": str(path.relative_to(ROOT)),
                "exists": path.is_file(),
            }
            for path in documentation
        ],
        "tests": {
            "count": len(tests),
            "files": [str(path.relative_to(ROOT)) for path in tests],
        },
    }


def build_snapshot() -> dict[str, Any]:
    """Build a read-only snapshot from the current public runtime surface."""

    opt, module_file = _import_checkout_optimizer()
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "workspace": {
            "root": str(ROOT),
            "optimizer_module": str(module_file),
            "python": sys.version.split()[0],
        },
        "git": _git_snapshot(),
        "context": opt.context_info(),
        "inventory": _workspace_inventory(),
        "recommended_commands": [
            "conda run --no-capture-output -n Optimizer python optimizer_context.py",
            "conda run --no-capture-output -n Optimizer python optimizer_context.py --json",
            "conda run --no-capture-output -n Optimizer python -B -m unittest discover -s tests",
        ],
    }


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _render_human(snapshot: dict[str, Any], *, verbose: bool) -> str:
    workspace = snapshot["workspace"]
    git = snapshot["git"]
    context = snapshot["context"]
    optimizer = context["optimizer"]
    contract = context["system_contract"]
    inventory = snapshot["inventory"]
    methods = optimizer["methods"]
    groups = optimizer["catalog"]["groups"]
    group_summary = ", ".join(
        f"{name} ({data['count']})" for name, data in groups.items()
    )

    lines = [
        "OPTIMIZER runtime context",
        f"root: {workspace['root']}",
        f"package: {workspace['optimizer_module']}",
        f"generated: {snapshot['generated_utc']}",
        "",
        "Git",
        f"  branch/head: {git['branch'] or 'unavailable'} / {git['head'] or 'unavailable'}",
        f"  worktree changes: {git['changes']['total']}",
        "",
        "Registered optimizer surface",
        f"  methods: {len(methods)}",
        f"  catalog groups: {group_summary}",
        f"  run_chunk parameters: {', '.join(optimizer['core']['run_chunk_parameters'])}",
        "",
        "System contract",
        f"  required hooks: {', '.join(contract['required_hooks'])}",
        f"  optional static hooks: {', '.join(contract['optional_static_hooks'])}",
        "",
        "Workspace anchors",
        f"  tests: {inventory['tests']['count']}",
        f"  available documents: {sum(1 for item in inventory['documentation'] if item['exists'])}",
        "",
        "Useful commands",
        *[f"  {command}" for command in snapshot["recommended_commands"]],
    ]

    if verbose:
        lines.extend(["", "Registered methods"])
        for name, method in methods.items():
            lines.append(f"  {name}: {method['status']} ({method['module']})")
        lines.extend(["", "Changed paths"])
        lines.extend(f"  {item['state']} {item['path']}" for item in git["changes"]["paths"])

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the runtime OPTIMIZER API and system context snapshot."
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="include registered methods and changed paths",
    )
    args = parser.parse_args(argv)

    try:
        snapshot = build_snapshot()
    except RuntimeError as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
    else:
        print(_render_human(snapshot, verbose=args.verbose), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
