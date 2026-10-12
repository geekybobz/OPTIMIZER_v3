"""Functional tests for the six public campaign modes."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import optimizer as opt
from optimizer.system_olgs import ControlBounds
from optimizer.utils.parallel import ParallelConfig
from tests.fixtures.quadratic_system import QuadraticVectorSystem, system


def _quadratic_factory() -> QuadraticVectorSystem:
    """Top-level factory: safe to use for isolated worker instances."""

    return system()


def _bad_factory() -> object:
    """Deliberately invalid worker product for campaign failure closure tests."""

    return object()


def _quadratic_resolver(reference: opt.campaigns.SystemReference) -> QuadraticVectorSystem:
    """Rebuild a fixture only from the persisted, system-owned descriptor."""

    parameters = reference.system_spec["parameters"]
    primary = parameters["primary"]
    secondary = parameters["secondary"]
    return QuadraticVectorSystem(N=primary["N"], tau=primary["tau"], **secondary)


class CountingQuadraticSystem(QuadraticVectorSystem):
    def __init__(self) -> None:
        super().__init__()
        self.evaluate_calls = 0
        self.gradient_calls = 0

    def evaluate(self, controls):
        self.evaluate_calls += 1
        return super().evaluate(controls)

    def gradient(self, controls):
        self.gradient_calls += 1
        return super().gradient(controls)


class BoundedQuadraticWithoutTransfer(QuadraticVectorSystem):
    """Declares bounds but intentionally offers no transfer-controls API."""

    transfer_controls = None

    def control_bounds(self) -> ControlBounds:
        spec = self.control_spec()
        return ControlBounds(
            lower=opt.Controls.constant(spec, -1.0, name="lower"),
            upper=opt.Controls.constant(spec, 1.0, name="upper"),
        )


class CampaignTests(unittest.TestCase):
    def _guesses(self, target: QuadraticVectorSystem, count: int = 3):
        return [
            opt.guesses.random_smooth_guess(target, amplitude=0.2, seed=index)
            for index in range(count)
        ]

    def test_all_six_modes_retain_owned_numerical_candidates(self) -> None:
        base = system()
        short_adam = opt.campaigns.MethodSpec("adam", {"maxiter": 3, "step_size": 0.1})
        short_lbfgs = opt.campaigns.MethodSpec("lbfgs", {"maxiter": 2, "step_size": 0.1})
        with tempfile.TemporaryDirectory() as directory:
            campaign = opt.campaigns.create(base, Path(directory) / "campaign", system_factory=_quadratic_factory)
            scout = campaign.scout(
                self._guesses(base),
                method=short_adam,
                parallel=ParallelConfig(backend="thread", workers=2),
            )
            portfolio = campaign.portfolio(
                scout.top(2), methods=[short_adam, short_lbfgs], parallel=ParallelConfig(backend="thread", workers=2)
            )
            refined = campaign.refine(portfolio.best, method=short_adam)
            challenged = campaign.challenge(
                refined.best,
                method=short_adam,
                variants=4,
                fourier_terms=2,
                parallel=ParallelConfig(backend="thread", workers=2),
            )
            validation = campaign.validate(challenged.best)
            continued = campaign.continue_parameter(
                challenged.best,
                updates={"energy_weight": 2.0e-3},
                method=short_adam,
            )
            report = campaign.report()

            self.assertEqual(len(scout.candidates), 3)
            self.assertEqual(len(portfolio.candidates), 4)
            self.assertEqual(len(refined.candidates), 1)
            self.assertEqual(len(challenged.candidates), 4)
            self.assertEqual(len(continued.candidates), 1)
            self.assertEqual(report["stage_count"], 13)
            self.assertEqual(validation.counters["evaluation"], 1)
            self.assertEqual(validation.counters["gradient"], 1)
            self.assertFalse(campaign.root.exists())
            self.assertEqual(report["selected"]["candidate_id"], continued.best.candidate_id)

    def test_report_is_passive_and_parallel_requires_an_isolated_factory(self) -> None:
        base = CountingQuadraticSystem()
        method = opt.campaigns.MethodSpec("adam", {"maxiter": 2, "step_size": 0.1})
        with tempfile.TemporaryDirectory() as directory:
            campaign = opt.campaigns.create(base, Path(directory) / "campaign")
            result = campaign.scout(self._guesses(base, 2), method=method)
            before = (base.evaluate_calls, base.gradient_calls)
            report = campaign.report()
            self.assertEqual((base.evaluate_calls, base.gradient_calls), before)
            self.assertEqual(report["best_J"], result.best.J)

        with tempfile.TemporaryDirectory() as directory:
            campaign = opt.campaigns.create(base, Path(directory) / "parallel_campaign")
            with self.assertRaisesRegex(ValueError, "system_factory"):
                campaign.scout(
                    self._guesses(base, 2),
                    method=method,
                    parallel=ParallelConfig(backend="thread", workers=2),
                )

    def test_public_info_names_exactly_the_supported_modes(self) -> None:
        payload = opt.campaigns.info()
        self.assertEqual(
            payload["modes"],
            ["scout", "portfolio", "refine", "challenge", "validate", "continue"],
        )
        self.assertTrue(payload["invariants"]["report_makes_no_numerical_calls"])

    def test_guess_helper_requires_declared_amplitude_and_makes_no_numerical_calls(self) -> None:
        base = CountingQuadraticSystem()
        guesses = opt.campaigns.random_smooth_guesses(base, 3, amplitude=0.2, seed=4)
        self.assertEqual(len(guesses), 3)
        self.assertEqual((base.evaluate_calls, base.gradient_calls), (0, 0))
        with self.assertRaises(TypeError):
            opt.campaigns.random_smooth_guesses(base, 2)  # type: ignore[call-arg]

    def test_continuation_validation_uses_the_candidate_system_not_campaign_base(self) -> None:
        base = system(energy_weight=1.0e-3)
        method = opt.campaigns.MethodSpec("adam", {"maxiter": 2, "step_size": 0.1})
        with tempfile.TemporaryDirectory() as directory:
            campaign = opt.campaigns.create(base, Path(directory) / "campaign")
            scout = campaign.scout(self._guesses(base, 1), method=method)
            continued = campaign.continue_parameter(
                scout.best,
                updates={"energy_weight": 1.0},
                method=method,
            )
            validated = campaign.validate(continued.best, gradient=False)
            expected_system = base.with_secondary(energy_weight=1.0)
            expected = expected_system.evaluate(continued.best.result.controls)["J"]
            original = base.evaluate(continued.best.result.controls)["J"]

            self.assertEqual(validated.system_fingerprint, continued.best.system.fingerprint)
            self.assertAlmostEqual(validated.metrics["J"], expected)
            self.assertNotAlmostEqual(validated.metrics["J"], original)

    def test_evidence_saved_before_specs_declared_a_family_still_matches(self) -> None:
        base = system()
        spec = {key: value for key, value in opt.get_system_spec(base).items() if key != "family"}
        canonical = json.dumps(opt.to_plain(spec), sort_keys=True, separators=(",", ":")).encode("utf-8")
        legacy = opt.campaigns.SystemReference.from_dict(
            {
                "system_id": spec["identity"]["id"],
                "fingerprint": hashlib.sha256(canonical).hexdigest(),
                "system_spec": spec,
            }
        )

        legacy.assert_matches(base)
        current = opt.campaigns.SystemReference.from_system(base)
        self.assertEqual(current.system_spec["family"], "olgs")
        self.assertEqual(current.fingerprint, legacy.fingerprint)
        opt.campaigns.SystemReference.from_dict(current.to_dict())

    def test_load_candidate_is_owned_and_validation_uses_its_system(self):
        base = system()
        campaign = opt.campaigns.create(base)
        method = opt.campaigns.MethodSpec('adam', {'maxiter': 2})
        produced = campaign.scout(self._guesses(base, 1), method=method).best
        restored = campaign.load_candidate(produced.candidate_id)
        self.assertIsNot(restored.result.controls, produced.result.controls)
        self.assertEqual(campaign.validate(restored, gradient=False).metrics, produced.result.metrics)
        self.assertFalse(hasattr(opt.campaigns, 'open'))

    def test_worker_failure_closes_campaign_and_constrained_challenge_requires_system_transfer(self) -> None:
        base = system()
        method = opt.campaigns.MethodSpec("adam", {"maxiter": 1, "step_size": 0.1})
        with tempfile.TemporaryDirectory() as directory:
            campaign = opt.campaigns.create(base, Path(directory) / "failed", system_factory=_bad_factory)
            with self.assertRaises(TypeError):
                campaign.scout(
                    self._guesses(base, 2),
                    method=method,
                    parallel=ParallelConfig(backend="thread", workers=2),
                )
            self.assertEqual(campaign.report()["status"], "failed")
            self.assertFalse(campaign.root.exists())

        bounded = BoundedQuadraticWithoutTransfer()
        with tempfile.TemporaryDirectory() as directory:
            campaign = opt.campaigns.create(bounded, Path(directory) / "bounded")
            source = campaign.scout(self._guesses(bounded, 1), method=method).best
            with self.assertRaisesRegex(ValueError, "transfer_controls"):
                campaign.challenge(source, method=method, variants=2, perturbation=0.1)

    def test_selection_spec_uses_declared_direction_not_the_J_convention(self) -> None:
        declared = opt.campaigns.SelectionSpec(metric_id="fidelity", label="Fidelity", direction="maximize")
        self.assertLess(declared.key(0.97, "b"), declared.key(0.91, "a"))


if __name__ == "__main__":
    unittest.main()
