"""OLGS system package.

This package replaces the old ``optimizer.system`` module.  It keeps the existing
validation/probing helpers available while adding the OLGS class, primary/secondary
parameter helpers, structured lifecycle result containers, and optional derivative
fallback tools.
"""

from optimizer.system_olgs.contract import (
    Metrics,
    OLGSystem,
    OptimizerSystem,
    SystemProbe,
    get_secondary_update_hook,
    probe_system,
    require_system,
)
from optimizer.system_olgs.derivatives import (
    finite_difference_hessian,
    finite_difference_hvp,
    finite_difference_jacobian,
    get_hessian,
    get_hvp,
    get_jacobian,
)
from optimizer.system_olgs.olgs import OLGS, with_secondary
from optimizer.system_olgs.params import (
    PrimaryParams,
    SecondaryParams,
    coerce_dataclass_params,
    normalize_channels,
    params_to_dict,
    replace_params,
)
from optimizer.system_olgs.results import (
    BackwardResult,
    ForwardResult,
    GradientResult,
    SimulationResult,
    SystemEvaluation,
)
from optimizer.system_olgs.validation import (
    evaluate_system,
    gradient_system,
    optional_hessian,
    optional_hvp,
    optional_jacobian,
    optional_residuals,
    validate_control_spec,
    validate_controls_for_system,
    validate_metrics,
)


__all__ = [
    "BackwardResult",
    "ForwardResult",
    "GradientResult",
    "Metrics",
    "OLGS",
    "OLGSystem",
    "OptimizerSystem",
    "PrimaryParams",
    "SecondaryParams",
    "SimulationResult",
    "SystemEvaluation",
    "SystemProbe",
    "coerce_dataclass_params",
    "evaluate_system",
    "finite_difference_hessian",
    "finite_difference_hvp",
    "finite_difference_jacobian",
    "get_hessian",
    "get_hvp",
    "get_jacobian",
    "get_secondary_update_hook",
    "gradient_system",
    "normalize_channels",
    "optional_hessian",
    "optional_hvp",
    "optional_jacobian",
    "optional_residuals",
    "params_to_dict",
    "probe_system",
    "replace_params",
    "require_system",
    "validate_control_spec",
    "validate_controls_for_system",
    "validate_metrics",
    "with_secondary",
]
