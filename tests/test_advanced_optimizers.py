"""Tests for advanced optimizer methods."""

import unittest

import numpy as np

import optimizer as opt
from fixtures.quadratic_system import QuadraticVectorSystem


def advanced_system():
    """Return a small vector fixture with nonzero gradient at zero controls."""

    return QuadraticVectorSystem(
        N=17,
        residual_weight=0.25,
        energy_weight=1.0e-3,
    )


class AdvancedOptimizerTests(unittest.TestCase):
    def test_adagrad_and_rmsprop_decrease_fixture_objective(self):
        system = advanced_system()
        controls = opt.constant_guess(system, value=0.0)
        initial = opt.evaluate(system, controls).J

        adagrad = opt.adagrad(system, controls, step_size=0.05, maxiter=4)
        rmsprop = opt.rmsprop(system, controls, step_size=0.02, decay=0.8, maxiter=4)

        self.assertLess(adagrad.J, initial)
        self.assertLess(rmsprop.J, initial)
        self.assertIn("accumulator", adagrad.state.optimizer_state)
        self.assertIn("accumulator", rmsprop.state.optimizer_state)

    def test_lbfgs_decreases_fixture_objective_and_tracks_history(self):
        system = advanced_system()
        controls = opt.gaussian_guess(system, amplitude=0.2, width=0.2)
        initial = opt.evaluate(system, controls).J

        result = opt.lbfgs(
            system,
            controls,
            step_size=0.2,
            history_size=4,
            maxiter=4,
            max_step_norm=0.5,
        )

        self.assertLess(result.J, initial)
        self.assertLessEqual(len(result.state.optimizer_state["s_history"]), 4)
        self.assertEqual(result.optimizer, "lbfgs")

    def test_nonlinear_cg_variants_decrease_fixture_objective(self):
        system = advanced_system()
        controls = opt.constant_guess(system, value=0.0)
        initial = opt.evaluate(system, controls).J

        for variant in ("fletcher_reeves", "polak_ribiere_plus", "hestenes_stiefel"):
            with self.subTest(variant=variant):
                result = opt.nonlinear_cg(
                    system,
                    controls,
                    variant=variant,
                    step_size=0.05,
                    maxiter=3,
                    max_step_norm=0.5,
                )
                self.assertLess(result.J, initial)
                self.assertEqual(result.optimizer, "nonlinear_cg")

    def test_cma_es_population_search_returns_improved_standard_result(self):
        system = advanced_system()
        controls = opt.constant_guess(system, value=0.0)
        initial = opt.evaluate(system, controls).J

        result = opt.cma_es(
            system,
            controls,
            variant="diagonal",
            population_size=24,
            elite_fraction=0.35,
            sigma=0.15,
            maxiter=6,
            seed=7,
        )

        self.assertEqual(result.optimizer, "cma_es")
        self.assertLess(result.J, initial)
        self.assertIn("population_size", result.state.optimizer_state)
        self.assertGreaterEqual(len(result.trace.iteration_records), 1)

    def test_public_advanced_optimizer_status(self):
        methods = opt.methods()

        for name in ("adagrad", "rmsprop", "lbfgs", "nonlinear_cg", "ncg", "cma_es"):
            with self.subTest(name=name):
                self.assertEqual(methods[name].status, "implemented")


if __name__ == "__main__":
    unittest.main()
