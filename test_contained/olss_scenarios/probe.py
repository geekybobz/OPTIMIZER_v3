"""Read-only compatibility probe for OLSS systems and numerical methods."""

from __future__ import annotations

import json
from typing import Any

import optimizer as opt

from test_contained.olss_scenarios.systems import (
    ArticleFourthOrderTwentySystem,
    ArticleGateTwelveSystem,
    ArticleSixthOrderPlanarSystem,
    BratuSystem,
    CartDirectPolynomialSystem,
    RobustTransferSecondOrderSystem,
    UnstableMultipleShootingSystem,
)


def _method(available: bool, applicable: bool, reason: str) -> dict[str, Any]:
    return {
        "available": bool(available),
        "applicable": bool(applicable),
        "reason": str(reason),
    }


def probe_method_compatibility(system: Any) -> dict[str, Any]:
    """Report public method availability separately from system applicability."""

    info = opt.system_info(system)
    spec = info["system_spec"]["value"]
    hooks = info["probe"]
    n = int(spec["variables"]["size"])
    m = int(spec["residuals"]["size"])
    required = bool(hooks["required_ok"])
    selection = spec.get("selection")
    public = {
        "root": callable(getattr(opt, "solve_shooting", None)),
        "multistart": callable(getattr(opt, "solve_shooting_multistart", None)),
        "pseudo_arclength": callable(getattr(opt, "continue_shooting", None)),
        "direct": callable(getattr(opt, "solve_direct_shooting", None)),
        "jacobian_check": callable(getattr(opt, "verify_shooting_jacobian", None)),
    }
    methods = {
        "root": _method(public["root"] and required, required, "requires the OLSS contract"),
        "multistart": _method(
            public["multistart"] and required,
            required,
            "parallel execution additionally requires a caller-supplied system factory",
        ),
        "ranked_multistart": _method(
            public["multistart"] and required,
            required and selection is not None,
            "ranking requires a declared selection metric",
        ),
        "pseudo_arclength": _method(
            public["pseudo_arclength"] and required,
            required and m == n - 1,
            "requires a one-dimensional root branch (m = n - 1)",
        ),
        "direct": _method(
            public["direct"] and required,
            required and selection is not None and bool(hooks["has_cost_gradient"]),
            "requires a selection metric and cost_gradient",
        ),
        "jacobian_check": _method(
            public["jacobian_check"] and required,
            required,
            "compares the declared Jacobian with explicit finite differences",
        ),
        "bounded_root": _method(
            public["root"] and required,
            required and bool(hooks["has_shooting_bounds"]),
            "root clipping is active when shooting_bounds is provided",
        ),
        "control_recovery": _method(
            bool(hooks["has_recover_controls"]),
            bool(hooks["has_recover_controls"]),
            "project-owned optional reconstruction hook",
        ),
    }
    return {
        "system_id": spec["identity"]["id"],
        "contract_valid": required,
        "dimensions": {
            "variables": n,
            "residuals": m,
            "shape": "square" if n == m else ("overdetermined" if m > n else "underdetermined"),
        },
        "jacobian_source": spec["derivatives"]["jacobian"]["source"],
        "hooks": hooks,
        "methods": methods,
    }


def article_scenario_matrix() -> dict[str, dict[str, Any]]:
    """Return representative article-shaped and numerical-stress scenarios."""

    scenarios = {
        "second_order_8x9": RobustTransferSecondOrderSystem(primary={"steps": 80}),
        "gate_12x12": ArticleGateTwelveSystem(),
        "fourth_order_20x21": ArticleFourthOrderTwentySystem(),
        "sixth_order_planar_4x4": ArticleSixthOrderPlanarSystem(),
        "direct_3x2": CartDirectPolynomialSystem(),
        "fold_2x1": BratuSystem(primary={"steps": 80}),
        "multiple_shooting_39x39": UnstableMultipleShootingSystem(),
    }
    return {name: probe_method_compatibility(system) for name, system in scenarios.items()}


def main() -> None:
    print(json.dumps(article_scenario_matrix(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
