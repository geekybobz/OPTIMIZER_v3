"""Tests for initial guess generators.

The guess phase matters because optimization outcomes depend strongly on the starting
controls.  These tests use a named-channel vector fixture so every guess is validated
against the same endpoint-grid control layout used by later optimizer and repair
tests.
"""

import unittest

import numpy as np

import optimizer as opt
from fixtures.quadratic_system import QuadraticVectorSystem


def system():
    return QuadraticVectorSystem(N=33)


class GuessGeneratorTests(unittest.TestCase):
    def test_zero_constant_and_ramp_guesses_match_system_spec(self):
        sys = system()
        zero = opt.zero_guess(sys)
        const = opt.constant_guess(sys, value={"ux": 0.2, "uy": 0.0, "uz": -0.1})
        ramp = opt.ramp_guess(sys, start=0.0, stop={"ux": 0.3, "uy": 0.0, "uz": 0.0})

        self.assertEqual(zero.shape, sys.control_spec().shape)
        np.testing.assert_allclose(zero.as_matrix(), 0.0)
        self.assertAlmostEqual(float(np.max(const.channel("ux"))), 0.2)
        self.assertAlmostEqual(float(np.min(const.channel("uz"))), -0.1)
        self.assertAlmostEqual(ramp.channel("ux")[0], 0.0)
        self.assertAlmostEqual(ramp.channel("ux")[-1], 0.3)

    def test_harmonic_guesses_control_amplitude_channels_and_endpoints(self):
        sys = system()
        sine = opt.sine_guess(
            sys,
            amplitude={"ux": 0.3, "uy": 0.0, "uz": 0.0},
            frequency=2,
            endpoint="zero",
        )
        gaussian = opt.gaussian_guess(sys.control_spec(), amplitude=0.4, width=0.12, envelope="hann")
        sinc = opt.sinc_guess(sys, amplitude=0.25, width=5.0)

        self.assertLessEqual(sine.max_abs(), 0.3 + 1.0e-12)
        self.assertAlmostEqual(sine.channel("ux")[0], 0.0)
        self.assertAlmostEqual(sine.channel("ux")[-1], 0.0)
        np.testing.assert_allclose(sine.channel("uy"), 0.0)
        self.assertLessEqual(gaussian.max_abs(), 0.4 + 1.0e-12)
        self.assertLessEqual(sinc.max_abs(), 0.25 + 1.0e-12)

    def test_fourier_guess_supports_modes_phase_and_amplitude(self):
        sys = system()
        guess = opt.fourier_guess(
            sys,
            amplitude=0.2,
            modes=4,
            phases=np.zeros(4),
            decay="1/k2",
            endpoint="zero",
        )

        self.assertEqual(guess.meta["guess"], "fourier")
        self.assertEqual(guess.meta["modes"], 4)
        self.assertLessEqual(guess.max_abs(), 0.2 + 1.0e-12)
        self.assertAlmostEqual(guess.channel("ux")[0], 0.0)
        self.assertAlmostEqual(guess.channel("ux")[-1], 0.0)

    def test_random_guesses_are_reproducible_and_smooth_option_reduces_roughness(self):
        sys = system()
        first = opt.random_fourier_guess(sys, amplitude=(0.05, 0.2), modes=5, seed=10)
        second = opt.random_fourier_guess(sys, amplitude=(0.05, 0.2), modes=5, seed=10)
        raw = opt.random_guess(sys, amplitude=0.2, seed=4, distribution="normal")
        smooth = opt.random_smooth_guess(sys, amplitude=0.2, seed=4, correlation=0.2)

        np.testing.assert_allclose(first.as_matrix(), second.as_matrix())
        self.assertLessEqual(first.max_abs(), 0.2 + 1.0e-12)
        raw_rough = opt.smoothness_report(raw)["global_first_difference_norm"]
        smooth_rough = opt.smoothness_report(smooth)["global_first_difference_norm"]
        self.assertLess(smooth_rough, raw_rough)

    def test_scale_mix_and_perturb_guess_support_restarts(self):
        sys = system()
        base = opt.gaussian_guess(sys, amplitude=0.2)
        perturb = opt.perturb_guess(base, amplitude=0.01, kind="random_fourier", seed=2, modes=3)
        scaled = opt.scale_guess(base, amplitude=0.1)
        mixed = opt.mix_guess([base, perturb], weights=[0.8, 0.2])

        self.assertEqual(perturb.shape, base.shape)
        self.assertGreater(np.linalg.norm(perturb.as_matrix() - base.as_matrix()), 0.0)
        self.assertLessEqual(scaled.max_abs(), 0.1 + 1.0e-12)
        np.testing.assert_allclose(
            mixed.as_matrix(),
            0.8 * base.as_matrix() + 0.2 * perturb.as_matrix(),
        )

    def test_context_guess_methods_and_public_status(self):
        sys = system()
        ctx = opt.context(sys)

        guess = ctx.random_smooth_guess(amplitude=0.1, seed=1)

        self.assertEqual(guess.shape, sys.control_spec().shape)
        self.assertEqual(opt.methods()["fourier_guess"].status, "implemented")
        self.assertEqual(opt.methods()["random_fourier_guess"].status, "implemented")


if __name__ == "__main__":
    unittest.main()
