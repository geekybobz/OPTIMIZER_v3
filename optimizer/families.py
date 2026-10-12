"""One rule for telling OLGS and OLSS systems apart.

A system declares its family in ``system_spec()["family"]``. OLSS systems
always declare ``"olss"``; an OLGS specification built by the library declares
``"olgs"``, and a missing family also means OLGS so that hand-written OLGS
specifications keep working.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

FAMILY_NAMES = ("olgs", "olss")


def system_family(system: Any) -> str:
    """Return ``"olgs"`` or ``"olss"`` from the system's static specification.

    Raises ``TypeError`` when the object has no callable ``system_spec()`` and
    ``ValueError`` for a declared family the library does not know.
    """

    hook = getattr(system, "system_spec", None)
    if not callable(hook):
        raise TypeError("A system must provide system_spec() to declare its family.")
    spec = hook()
    family = spec.get("family") if isinstance(spec, Mapping) else None
    if family is None or family == "olgs":
        return "olgs"
    if family == "olss":
        return "olss"
    raise ValueError(f"Unsupported system family {family!r}; expected one of {FAMILY_NAMES}.")
