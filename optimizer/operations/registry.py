"""Explicit dispatch policy layered over the existing public catalog."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from optimizer import catalog
from optimizer.operations.contracts import _ServiceError


@dataclass(frozen=True)
class _Adapter:
    classification: str
    effect: str = "inspect"
    inject: tuple[tuple[str, str], ...] = ()
    hooks: tuple[str, ...] = ()
    family: tuple[str, ...] = ()
    reason: str | None = None
    continuation: str = "unsupported"


_OLGS_READ = {
    "get_system_spec", "describe_system", "probe_system",
    "validate_control_spec", "validate_controls_for_system", "evaluate_system", "gradient_system",
    "value_and_gradient_system", "optional_residuals", "optional_jacobian", "optional_hvp",
    "optional_hessian", "optional_smooth_gradient", "optional_prox", "optional_plot_data",
    "optional_control_bounds", "finite_difference_jacobian", "finite_difference_hessian",
    "finite_difference_hvp", "get_jacobian", "get_hvp", "get_hessian",
}
_OLSS_READ = {
    "get_shooting_system_spec", "validate_shooting_spec", "describe_shooting_system", "probe_shooting_system",
    "validate_variables_for_shooting_system", "residuals_system", "jacobian_system",
    "residuals_and_jacobian_system", "evaluate_shooting_system", "verify_shooting_system",
    "roots_equivalent_system", "cost_gradient_system", "shooting_bounds_system",
}
_UTIL_READ = {
    "metric_report", "diagnostic_report", "geometry_probe", "finite_difference_gradient",
    "verify_gradient", "finite_difference_jacobian", "verify_jacobian", "nullspace_basis",
    "project_gradient", "control_spectrum", "smoothness_report", "diagnose_run",
    "curvature_probe", "compare_methods", "shooting_geometry_probe",
}
_PURE_BASE = {"control_spec", "controls", "zeros", "constant", "warmstart", "to_plain", "methods",
              "constant_schedule", "adaptive_step_schedule", "exponential_decay_schedule",
              "linear_warmup_schedule", "cosine_decay_schedule"}
_PURE_SYSTEM = {"basic_system_spec", "metric_spec", "validate_system_spec", "validate_declared_metrics",
                "validate_metrics", "contract_info", "basic_shooting_system_spec", "shooting_metric_spec",
                "validate_shooting_system_spec", "shooting_contract_info"}
_VISUAL_DATA = {"coerce_plot_data", "validate_bloch_data", "validate_controls_data"}
_HOOKS = {
    "optional_residuals": ("residuals",), "optional_jacobian": ("jacobian",),
    "optional_hvp": ("hvp",), "optional_hessian": ("hessian",),
    "optional_smooth_gradient": ("smooth_gradient",), "optional_prox": ("prox",),
    "optional_control_transfer": ("transfer_controls",), "optional_plot_data": ("plot_data",),
    "optional_control_bounds": ("control_bounds",), "cost_gradient_system": ("cost_gradient",),
}


def _injections(obj: Any) -> tuple[tuple[str, str], ...]:
    # Policy opts into this fixed vocabulary. Reflection only checks its presence.
    params = inspect.signature(obj).parameters
    names = {"system": "system", "controls": "physical", "variables": "physical",
             "initial": "physical", "spec": "layout", "target": "layout"}
    return tuple((name, source) for name, source in names.items() if name in params)


def _policy(item: catalog.CatalogItem) -> _Adapter:
    obj = item.public_object()
    group, name = item.group, item.name
    family = item.family
    if item.kind in {"constant", "registry", "object", "type_alias"}:
        return _Adapter("inspectable_value", reason="Discovery only; this entry is not executable.")
    if group == "optimizers":
        hooks = {"proximal_gradient": ("smooth_gradient", "prox"), "projected_gradient": ("residuals",)}.get(name, ())
        return _Adapter("tool", "solve", _injections(obj), hooks=hooks, family=family,
                        continuation="controls_only" if name == "cma_es" else "method_state")
    if group == "guesses":
        return _Adapter("tool", "candidate", _injections(obj), family=family)
    if group == "utils" and item.kind == "control_transform":
        # Pure transforms validate their own source/grid. A requested context
        # constrains only an explicit target, never the source array.
        if name == "recover_shooting_controls":
            return _Adapter("tool", "inspect", _injections(obj), hooks=("recover_controls",), family=("olss",))
        return _Adapter("tool", "inspect", (("target", "layout"),) if name in {"resample_controls", "assemble_channels"} else ())
    if group == "utils" and name in _UTIL_READ | {"repair_newton"}:
        return _Adapter("tool", "candidate" if name == "repair_newton" else "inspect",
                        _injections(obj), family=("olss",) if name == "shooting_geometry_probe" else family)
    if group == "system_olgs" and name in _OLGS_READ | {"optional_control_transfer"}:
        return _Adapter("tool", "candidate" if name == "optional_control_transfer" else "inspect",
                        _injections(obj), _HOOKS.get(name, ()), family)
    if group == "system_olss" and name in _OLSS_READ:
        return _Adapter("tool", inject=_injections(obj), hooks=_HOOKS.get(name, ()), family=family)
    if group == "shooting" and name in {
        "solve_shooting", "solve_direct_shooting", "solve_shooting_multistart", "refine_shooting_seeds",
        "continue_shooting", "continue_parameter", "verify_shooting_jacobian",
        "finite_difference_shooting_jacobian", "random_shooting_seeds", "perturb_shooting_seeds",
        "compare_shooting_replay",
    }:
        effect = "solve" if name in {"solve_shooting", "solve_direct_shooting"} else "candidate" if name in {
            "solve_shooting_multistart", "refine_shooting_seeds", "continue_shooting", "continue_parameter",
            "random_shooting_seeds", "perturb_shooting_seeds"} else "inspect"
        return _Adapter("tool", effect, _injections(obj), family=family,
                        continuation="variables_and_history" if name == "solve_shooting" else "variables_only")
    if group == "base" and name in {"evaluate", "gradient", "value_and_gradient", "plot_data", "system_info", "system_family", "transact"}:
        return _Adapter("tool", inject=_injections(obj), family=family)
    if group == "base" and name in _PURE_BASE or group in {"system_olgs", "system_olss"} and name in _PURE_SYSTEM:
        return _Adapter("tool", inject=() if name == "control_spec" else tuple(
            pair for pair in _injections(obj) if pair[1] == "layout"))
    if group == "visualization" and name in _VISUAL_DATA:
        return _Adapter("tool")
    if group == "core" and name in {"run_chunk", "metric_guard", "default_accept"}:
        return _Adapter("tool", "solve" if name == "run_chunk" else "inspect",
                        _injections(obj), family=family, continuation="controls_only")
    if group == "campaigns" and name == "random_smooth_guesses":
        return _Adapter("tool", "candidate", _injections(obj), family=family)
    if group == "schedules" or item.kind in {"data_object", "result_object", "guard"}:
        return _Adapter("constructor")
    if name in {"with_secondary", "require_system", "require_shooting_system", "get_secondary_update_hook"}:
        return _Adapter("adapter_unavailable", reason="Use session.load/attach/update; live systems/hooks are not retained data objects.")
    if group == "visualization":
        return _Adapter("extension", reason="Live figure/axes/context-manager lifecycle needs a renderer adapter; plot-data tools are available.")
    if group == "campaigns" or name == "continuation":
        return _Adapter("extension", reason="Existing workflow retains live systems or writes legacy campaigns; workflow/history migration precedes this adapter.")
    return _Adapter("adapter_unavailable", reason="Live runtime/facade/protocol is not an executable retained tool; direct Python remains available.")


class _Registry:
    def __init__(self) -> None:
        self.items = {item.id: item for group in catalog.groups() for item in catalog.group(group).items}
        self.adapters = {key: _policy(item) for key, item in self.items.items()}

    def resolve(self, name: str) -> tuple[catalog.CatalogItem, _Adapter]:
        try:
            item = catalog._resolve_item(name)
            # The legacy ncg wrapper is itself declared; its execution policy uses
            # the canonical method signature, not unrestricted wrapper **kwargs.
            if item.id == "optimizers.ncg":
                item = self.items["optimizers.nonlinear_cg"]
            return item, self.adapters[item.id]
        except KeyError as exc:
            raise _ServiceError("unknown_operation", str(exc)) from exc

    def describe(self, name: str) -> dict[str, Any]:
        item, adapter = self.resolve(name)
        return {**item.short_dict(), "classification": adapter.classification,
                "effect": adapter.effect, "activation_default": adapter.effect == "solve",
                "continuation": adapter.continuation, "reason": adapter.reason,
                "signature": item.signature(), "context_inputs": dict(adapter.inject)}
