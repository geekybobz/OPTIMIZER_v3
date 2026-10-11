"""Numerical methods for Open Loop Shooting Systems."""

from optimizer.shooting.campaigns import (
    perturb_shooting_seeds,
    random_shooting_seeds,
    refine_shooting_seeds,
    solve_shooting_multistart,
)
from optimizer.shooting.continuation import continue_parameter, continue_shooting
from optimizer.shooting.derivatives import (
    finite_difference_shooting_jacobian,
    verify_shooting_jacobian,
)
from optimizer.shooting.evaluator import ShootingEvaluator
from optimizer.shooting.solvers import solve_direct_shooting, solve_shooting
from optimizer.shooting.replay import compare_shooting_replay
from optimizer.catalog import attach_namespace_helpers


attach_namespace_helpers(globals(), "shooting")


__all__ = [
    "ShootingEvaluator",
    "continue_shooting",
    "continue_parameter",
    "compare_shooting_replay",
    "finite_difference_shooting_jacobian",
    "solve_direct_shooting",
    "solve_shooting",
    "solve_shooting_multistart",
    "random_shooting_seeds",
    "perturb_shooting_seeds",
    "refine_shooting_seeds",
    "verify_shooting_jacobian",
    "info",
    "list",
    "search",
]
