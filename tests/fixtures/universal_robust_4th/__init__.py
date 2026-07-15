"""Temporary universal-robust-4th fixture package for facade tests.

This fixture package exists so public API tests exercise a system shaped like the
downstream fourth-order robust-control model instead of another abstract toy problem.
It is deliberately temporary and can be removed once real downstream integration
tests are connected.
"""

from tests.fixtures.universal_robust_4th.system import (
    TemporaryUniversalFourthOrderSystem,
    UniversalFourthOrderParams,
)

__all__ = ["TemporaryUniversalFourthOrderSystem", "UniversalFourthOrderParams"]
