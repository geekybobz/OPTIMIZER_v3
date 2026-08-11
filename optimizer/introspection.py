"""Runtime self-description for the public optimizer library.

The payloads in this module describe registered public APIs and signatures. They
never evaluate a system, request a gradient, or construct controls. A caller may
optionally supply a system to include its declared static metadata.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
from typing import Any, Mapping

from optimizer import catalog
from optimizer.core.engine import run_chunk
from optimizer.system_olgs import contract_info, describe_system
from optimizer.tracker import info as tracker_info


SCHEMA_VERSION = "optimizer.context/v1"


def _method_payload(methods: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize registered facade methods without inspecting source files."""

    payload: dict[str, dict[str, Any]] = {}
    for name, method in sorted(methods.items()):
        to_dict = getattr(method, "to_dict", None)
        if callable(to_dict):
            payload[str(name)] = dict(to_dict())
        elif isinstance(method, Mapping):
            payload[str(name)] = dict(method)
        else:
            payload[str(name)] = {"name": str(name), "status": "unknown"}
    return payload


def context_info(
    methods: Mapping[str, Any],
    *,
    system: Any | None = None,
) -> dict[str, Any]:
    """Return a current, read-only description of the registered library surface.

    When system is supplied, only its optional describe() and tracking_spec()
    hooks are consulted through describe_system. Numerical hooks such as
    evaluate() and gradient() are never called.
    """

    run_chunk_parameters = list(inspect.signature(run_chunk).parameters)
    package_file = Path(__file__).resolve().with_name("__init__.py")

    return {
        "schema_version": SCHEMA_VERSION,
        "optimizer": {
            "runtime": {
                "python": sys.version.split()[0],
                "package_file": str(package_file),
            },
            "public_calls": [
                "list",
                "info",
                "search",
                "path",
                "context_info",
                "system_info",
            ],
            "methods": _method_payload(methods),
            "catalog": catalog.list(),
            "core": {
                "run_chunk_parameters": run_chunk_parameters,
                "has_live_tracking_argument": any(
                    name in {"tracking", "tracker", "journal", "observer", "observers"}
                    for name in run_chunk_parameters
                ),
            },
        },
        "system_contract": contract_info(),
        "tracker": tracker_info(),
        "system": None if system is None else describe_system(system),
    }
