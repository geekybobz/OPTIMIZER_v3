"""Diagnostics, repair, and training-aid utilities.

Why this package exists
-----------------------
Optimizers move controls using gradients.  Serious control work also needs tools that
answer different questions: is the analytical gradient correct, are hard residuals
small, is the residual Jacobian ill-conditioned, can a candidate be repaired, and is
the pulse too rough.  Those tools should not live in ``optimizer/optimizers`` because
they are not ordinary descent algorithms.

How it fits the architecture
----------------------------
- ``diagnostics`` summarizes metrics, controls, residuals, and local geometry.
- ``derivatives`` verifies analytical derivatives and provides finite-difference
  fallbacks when a system does not expose optional hooks.
- ``repairs`` moves controls specifically to reduce residual violations.
- ``geometry`` provides Jacobian/nullspace/projection helpers.
- ``spectrum`` measures pulse roughness and frequency content.

Reviewer invariants
-------------------
- utilities use the same ``Controls`` and system contract as optimizers.
- derivative fallbacks are explicit and reported.
- repair/projection utilities do not pretend to be Adam/momentum-style optimizers.
"""

from optimizer.utils.derivatives import (
    finite_difference_gradient,
    finite_difference_jacobian,
    verify_gradient,
    verify_jacobian,
)
from optimizer.utils.diagnostics import diagnostic_report, geometry_probe, metric_report
from optimizer.utils.geometry import nullspace_basis, project_gradient
from optimizer.utils.repairs import RepairResult, repair_newton
from optimizer.utils.spectrum import control_spectrum, smoothness_report
from optimizer.catalog import attach_namespace_helpers


attach_namespace_helpers(globals(), "utils")

__all__ = [
    "RepairResult",
    "control_spectrum",
    "diagnostic_report",
    "finite_difference_gradient",
    "finite_difference_jacobian",
    "geometry_probe",
    "info",
    "list",
    "metric_report",
    "nullspace_basis",
    "project_gradient",
    "repair_newton",
    "smoothness_report",
    "verify_gradient",
    "verify_jacobian",
]
