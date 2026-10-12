"""Public exports and the runtime catalog stay in step.

Every exported name is either described in the runtime catalog or listed below
as a known gap. The gaps are mostly contract helpers, protocols, and engine
event types; the planned operation registry is meant to cover them. A new
export without a catalog entry fails this test instead of drifting silently.
"""

from __future__ import annotations

import importlib
import unittest

import optimizer as opt


KNOWN_UNCATALOGUED = {
    "system_olss": {
        "OpenLoopShootingSystem", "SHOOTING_SYSTEM_SPEC_SCHEMA", "ShootingMetrics", "ShootingOptions",
        "ShootingSystemProbe", "cost_gradient_system", "describe_shooting_system", "evaluate_shooting_system",
        "get_shooting_system_spec", "jacobian_system", "probe_shooting_system", "require_shooting_system",
        "residuals_and_jacobian_system", "residuals_system", "roots_equivalent_system", "shooting_bounds_system",
        "shooting_contract_info", "validate_shooting_spec", "validate_shooting_system_spec",
        "validate_variables_for_shooting_system", "verify_shooting_system",
    },
    "system_olgs": {
        "BackwardResult", "ControlBounds", "ForwardResult", "GradientResult", "Metrics", "OLGS", "OLGSystem",
        "OptimizerSystem", "SYSTEM_SPEC_SCHEMA", "SimulationResult", "SupportsBounds", "SupportsControlTransfer",
        "SupportsEqualityConstraints", "SupportsHVP", "SupportsPlotData", "SupportsProximalObjective",
        "SupportsValueAndGradient", "SystemEvaluation", "SystemProbe", "TRACKING_SPEC_SCHEMA", "TrackingSpec",
        "ValueGradient", "basic_system_spec", "contract_info", "describe_system", "evaluate_system",
        "finite_difference_hessian", "finite_difference_hvp", "get_hessian", "get_hvp", "get_jacobian",
        "get_secondary_update_hook", "get_tracking_spec", "gradient_system", "metric_spec",
        "optional_control_transfer", "optional_hessian", "optional_hvp", "optional_jacobian",
        "optional_plot_data", "optional_prox", "optional_residuals", "optional_smooth_gradient", "probe_system",
        "validate_control_spec", "validate_controls_for_system", "validate_declared_metrics", "validate_metrics",
        "validate_system_spec", "value_and_gradient_system",
    },
    "core": {
        "CheckpointEvent", "ChunkEvent", "EvaluationOutcome", "GradientOutcome", "IterationEvent", "MetricGuard",
        "ObserverAdapter", "RunObserver", "RunStartEvent", "SnapshotFacts", "StopDecision", "StopTracker",
        "TerminalEvent", "ValueGradientOutcome", "default_accept",
    },
    "campaigns": {"CandidateRef", "ModeResult", "ValidationResult", "random_smooth_guesses"},
}
NAMESPACES = ("shooting", "system_olss", "system_olgs", "utils", "optimizers", "core", "schedules", "guesses", "campaigns")
HELPERS = {"info", "list", "search", "path"}


class CatalogCoverageTests(unittest.TestCase):
    def test_every_export_is_catalogued_or_a_known_gap(self):
        catalogued = {entry["name"] for entries in opt.list()["items"].values() for entry in entries}
        for namespace in NAMESPACES:
            module = importlib.import_module(f"optimizer.{namespace}")
            exported = set(getattr(module, "__all__", ())) - HELPERS
            missing = exported - catalogued - KNOWN_UNCATALOGUED.get(namespace, set())
            with self.subTest(namespace=namespace):
                self.assertEqual(missing, set(), f"exported but not in the catalog: {sorted(missing)}")

    def test_known_gaps_are_still_exported(self):
        for namespace, names in KNOWN_UNCATALOGUED.items():
            exported = set(getattr(importlib.import_module(f"optimizer.{namespace}"), "__all__", ()))
            with self.subTest(namespace=namespace):
                self.assertEqual(names - exported, set(), "remove stale names from the known-gap list")


if __name__ == "__main__":
    unittest.main()
