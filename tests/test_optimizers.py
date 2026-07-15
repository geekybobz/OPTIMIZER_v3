"""Tests for the first implemented optimizer phase.

These tests intentionally use the temporary fourth-order robust-control fixture
instead of a scalar toy function.  The goal is to verify the optimizer APIs against
the same control shape, metric names, analytical-gradient contract, and curriculum
style expected by downstream systems.
"""

import unittest

import numpy as np

import optimizer as opt
from fixtures.universal_robust_4th.system import TemporaryUniversalFourthOrderSystem


def fourth_order_case():
    """Return a small but nontrivial fourth-order fixture for optimizer tests."""

    return TemporaryUniversalFourthOrderSystem(
        N=17,
        lambda2=0.25,
        lambda4=0.05,
        energy_weight=1.0e-3,
    )


class OptimizerPhaseTests(unittest.TestCase):
    def test_line_search_backtracking_decreases_fourth_order_objective(self):
        system = fourth_order_case()
        controls = opt.zeros(system.control_spec(), name="zero")
        initial = opt.evaluate(system, controls)

        result = opt.line_search(
            system,
            controls,
            variant="backtracking",
            step_size=1.0,
            grow=1.0,
            maxiter=4,
        )

        self.assertEqual(result.optimizer, "line_search")
        self.assertLess(result.J, initial.J)
        self.assertGreater(result.state.optimizer_state["accept_count"], 0)
        self.assertIn("last_accepted_step_size", result.state.optimizer_state)

    def test_line_search_armijo_variant_records_attempts(self):
        system = fourth_order_case()
        controls = opt.zeros(system.control_spec(), name="zero")

        result = opt.line_search(
            system,
            controls,
            variant="armijo",
            step_size=1.0,
            maxiter=2,
            max_backtracks=6,
        )

        self.assertLess(result.J, opt.evaluate(system, controls).J)
        record = result.trace.iteration_records[0]
        attempts = record.technical["proposal"]["attempts"]
        self.assertGreaterEqual(len(attempts), 1)

    def test_momentum_variants_move_controls_and_store_velocity(self):
        system = fourth_order_case()
        controls = opt.zeros(system.control_spec(), name="zero")
        initial_j = opt.evaluate(system, controls).J

        for variant in ("heavy_ball", "nesterov", "clipped"):
            with self.subTest(variant=variant):
                result = opt.momentum(
                    system,
                    controls,
                    variant=variant,
                    step_size=0.05,
                    max_step_norm=0.25 if variant == "clipped" else None,
                    maxiter=3,
                )
                self.assertEqual(result.optimizer, "momentum")
                self.assertLess(result.J, initial_j)
                velocity = result.state.optimizer_state["velocity"]
                self.assertEqual(velocity.shape, (system.control_spec().size,))
                self.assertTrue(np.all(np.isfinite(velocity)))

    def test_momentum_restart_resets_velocity_on_rejected_step(self):
        system = fourth_order_case()
        controls = opt.zeros(system.control_spec(), name="zero")

        result = opt.momentum(
            system,
            controls,
            variant="restart",
            step_size=100.0,
            maxiter=1,
        )

        self.assertFalse(result.trace.iteration_records[0].accepted)
        self.assertEqual(result.state.optimizer_state["reject_count"], 1)
        np.testing.assert_allclose(result.state.optimizer_state["velocity"], 0.0)

    def test_adam_variants_decrease_fourth_order_objective(self):
        system = fourth_order_case()
        controls = opt.zeros(system.control_spec(), name="zero")
        initial_j = opt.evaluate(system, controls).J

        for variant in ("adam", "amsgrad", "adamw", "radam", "adabelief"):
            with self.subTest(variant=variant):
                result = opt.adam(
                    system,
                    controls,
                    variant=variant,
                    step_size=0.05,
                    weight_decay=0.0,
                    maxiter=4,
                )
                self.assertEqual(result.optimizer, "adam")
                self.assertLess(result.J, initial_j)
                self.assertEqual(result.state.optimizer_state["variant"], variant)
                self.assertGreaterEqual(result.state.optimizer_state["t"], 1)

    def test_adam_warmstart_transfers_moment_state(self):
        system = fourth_order_case()
        controls = opt.zeros(system.control_spec(), name="zero")

        first = opt.adam(
            system,
            controls,
            variant="amsgrad",
            step_size=0.05,
            maxiter=2,
        )
        warm = first.warmstart(target_optimizer="adam")
        second = opt.adam(
            system,
            variant="amsgrad",
            warmstart=warm,
            maxiter=2,
        )

        self.assertGreater(second.state.optimizer_state["t"], first.state.optimizer_state["t"])
        self.assertLess(second.J, first.J)
        self.assertEqual(second.state.optimizer_state["variant"], "amsgrad")


if __name__ == "__main__":
    unittest.main()
