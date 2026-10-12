"""OLGS system package.

This package replaces the old ``optimizer.system`` module.  It keeps the existing
validation/probing helpers available while adding the OLGS class, structured
lifecycle result containers, and optional derivative fallback tools.
"""

from optimizer.system_olgs.contract import (
    ControlBounds,
    Metrics,
    OLGSystem,
    OptimizerSystem,
    SystemProbe,
    SupportsBounds,
    SupportsControlTransfer,
    SupportsEqualityConstraints,
    SupportsHVP,
    SupportsPlotData,
    SupportsProximalObjective,
    SupportsValueAndGradient,
    TRACKING_SPEC_SCHEMA,
    TrackingSpec,
    ValueGradient,
    contract_info,
    describe_system,
    get_secondary_update_hook,
    get_system_spec,
    get_tracking_spec,
    probe_system,
    require_system,
)
from optimizer.system_olgs.specs import (
    SYSTEM_SPEC_SCHEMA,
    basic_system_spec,
    metric_spec,
    validate_declared_metrics,
    validate_system_spec,
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
    optional_control_bounds,
    optional_control_transfer,
    optional_hessian,
    optional_hvp,
    optional_jacobian,
    optional_plot_data,
    optional_prox,
    optional_residuals,
    optional_smooth_gradient,
    validate_control_spec,
    validate_controls_for_system,
    validate_metrics,
    value_and_gradient_system,
)
from optimizer.catalog import exports


__all__ = exports("system_olgs")
