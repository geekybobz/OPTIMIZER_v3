"""One serialization policy and one family rule for every result type.

``opt.to_plain`` converts results, reports, and transaction responses to strict
JSON: non-finite numbers become the strings ``"nan"``, ``"inf"``, and
``"-inf"``, and an object without a known form raises instead of falling back
to ``repr()``. ``opt.system_family`` reads the family from ``system_spec()``.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import unittest
from pathlib import PurePosixPath
from typing import Any

import numpy as np

import optimizer as opt
from fixtures.quadratic_system import QuadraticVectorSystem
from fixtures.shooting_systems import CubicRootShootingSystem


def strict_json(payload: Any) -> str:
    return json.dumps(payload, allow_nan=False)


@dataclasses.dataclass
class _Pair:
    left: float
    right: Any


class _NoCodec:
    pass


class _FailingStartSystem(CubicRootShootingSystem):
    """Residuals raise beyond x = 5, so a seed there cannot start."""

    def residuals(self, variables, *, options=None):
        if float(variables.block("x")) > 5.0:
            raise FloatingPointError("overflow")
        return super().residuals(variables, options=options)


def _variables(system, *values: float) -> opt.ShootingVariables:
    return opt.ShootingVariables(system.shooting_spec(), np.array(values, dtype=float))


class ToPlainTests(unittest.TestCase):
    def test_non_finite_numbers_become_strings(self):
        self.assertEqual(
            opt.to_plain([math.nan, math.inf, -math.inf, np.float32("nan"), 1.5]),
            ["nan", "inf", "-inf", "nan", 1.5],
        )
        self.assertEqual(opt.to_plain(np.array([1.0, np.inf])), [1.0, "inf"])
        self.assertEqual(opt.to_plain(complex(math.nan, 2.0)), {"real": "nan", "imag": 2.0})
        self.assertEqual(
            opt.to_plain(np.array([1 + 2j, 3 - 1j])), {"real": [1.0, 3.0], "imag": [2.0, -1.0]}
        )

    def test_numpy_scalars_become_python_values(self):
        payload = opt.to_plain({"flag": np.bool_(True), "count": np.int64(3), "value": np.float64(0.5)})
        self.assertEqual(payload, {"flag": True, "count": 3, "value": 0.5})
        self.assertIs(type(payload["flag"]), bool)
        self.assertIs(type(payload["count"]), int)

    def test_library_objects_use_their_known_forms(self):
        system = QuadraticVectorSystem()
        controls = system.reference_controls(amplitude=0.5)
        self.assertIn("matrix", opt.to_plain(controls))
        self.assertEqual(opt.to_plain(system)["system_type"], "QuadraticVectorSystem")
        self.assertEqual(opt.to_plain(_Pair(math.inf, {3, 1, 2})), {"left": "inf", "right": [1, 2, 3]})
        self.assertEqual(opt.to_plain(PurePosixPath("runs/a.json")), "runs/a.json")
        self.assertEqual(opt.to_plain(strict_json), f"{__name__}:strict_json")
        self.assertEqual(
            opt.to_plain(functools.partial(strict_json)), f"{__name__}:strict_json (partial)"
        )

    def test_an_object_without_a_known_form_raises(self):
        with self.assertRaisesRegex(TypeError, "to_dict"):
            opt.to_plain({"value": _NoCodec()})


class StrictJsonResultTests(unittest.TestCase):
    """Phase 1 acceptance: every result type is strict JSON, also with non-finite values."""

    def test_olgs_results(self):
        system = QuadraticVectorSystem()
        controls = system.reference_controls(amplitude=0.5)
        result = opt.adam(system, controls, maxiter=3)
        strict_json(result.to_dict(include_state=True, include_trace=True))
        strict_json(opt.diagnose_run(result))
        strict_json(opt.diagnostic_report(system, controls))

        evaluation = opt.Evaluation(controls, {"J": 1.0, "leakage": math.nan})
        self.assertEqual(json.loads(strict_json(evaluation.to_dict()))["metrics"]["leakage"], "nan")

        repair = opt.RepairResult(
            controls=controls,
            residuals=np.array([math.inf]),
            residual_norm=math.inf,
            residual_max_abs=math.inf,
            converged=False,
            iterations=1,
            method="lm",
            residual_name="hard",
            jacobian_source="analytical",
            history=[{"residual_norm": np.float64(math.inf)}],
        )
        payload = json.loads(strict_json(repair.to_dict(include_controls=True)))
        self.assertEqual(payload["residual_norm"], "inf")
        self.assertEqual(payload["history"], [{"residual_norm": "inf"}])

    def test_a_failed_olss_seed_is_strict_json(self):
        system = _FailingStartSystem()
        campaign = opt.solve_shooting_multistart(system, [_variables(system, 9.0), _variables(system, 0.9)])
        failed = campaign.results[0]

        self.assertEqual(failed.stop_reason, "initial_evaluation_failed")
        payload = json.loads(strict_json(failed.to_dict()))
        self.assertEqual(payload["residuals"], ["nan"])
        self.assertEqual(payload["residual_norm"], "nan")
        strict_json(campaign.to_dict())
        strict_json(opt.diagnose_run(failed))

    def test_olss_evaluation_and_verification_with_non_finite_residuals(self):
        system = CubicRootShootingSystem()
        variables = _variables(system, 0.5)
        evaluation = opt.ShootingEvaluation(variables, np.array([math.nan]), {"cost": math.inf})
        verification = opt.ShootingVerification(np.array([math.inf]), {"cost": math.nan})

        self.assertEqual(json.loads(strict_json(evaluation.to_dict()))["residual_norm"], "nan")
        self.assertEqual(json.loads(strict_json(verification.to_dict()))["residuals"], ["inf"])


class SystemFamilyTests(unittest.TestCase):
    def test_families_come_from_the_static_specification(self):
        self.assertEqual(opt.system_family(QuadraticVectorSystem()), "olgs")
        self.assertEqual(opt.system_family(CubicRootShootingSystem()), "olss")

    def test_a_missing_family_means_olgs(self):
        class Legacy:
            def system_spec(self):
                return {"identity": {"id": "legacy"}}

        self.assertEqual(opt.system_family(Legacy()), "olgs")

    def test_unknown_and_missing_declarations_are_errors(self):
        class Unknown:
            def system_spec(self):
                return {"family": "pde"}

        with self.assertRaisesRegex(ValueError, "pde"):
            opt.system_family(Unknown())
        with self.assertRaises(TypeError):
            opt.system_family(object())

    def test_olgs_specifications_declare_their_family(self):
        spec = QuadraticVectorSystem().system_spec()
        self.assertEqual(spec["family"], "olgs")
        self.assertEqual(opt.validate_system_spec(spec)["family"], "olgs")
        with self.assertRaisesRegex(ValueError, "family"):
            opt.validate_system_spec({**spec, "family": "olss"})


if __name__ == "__main__":
    unittest.main()
