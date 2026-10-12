"""Transaction boundary: payload decoding, contract validation, and optional hooks."""

from __future__ import annotations

import json
import unittest

import numpy as np

import optimizer as opt
from optimizer import ControlBounds, Controls
from fixtures.quadratic_system import QuadraticVectorSystem
from fixtures.shooting_systems import DirectCostShootingSystem, RectangularShootingSystem


def _round_trip(request: opt.SystemRequest) -> dict:
    """Send a request through JSON, as a terminal or MCP client would."""

    return json.loads(json.dumps(request.to_dict()))


class CombinedQuadratic(QuadraticVectorSystem):
    def value_and_gradient(self, controls: Controls):
        return self.evaluate(controls), self.gradient(controls)

    def control_bounds(self):
        spec = self.control_spec()
        return ControlBounds(Controls.constant(spec, -2.0), Controls.constant(spec, 2.0))


class BrokenQuadratic(QuadraticVectorSystem):
    def evaluate(self, controls: Controls):
        metrics = dict(super().evaluate(controls))
        metrics["J"] = float("nan")
        return metrics


class OlgsTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system = QuadraticVectorSystem()
        self.controls = self.system.reference_controls(amplitude=0.3)

    def test_json_round_trip_matches_the_direct_call(self):
        for operation in ("evaluate", "gradient"):
            request = opt.SystemRequest(operation, {"controls": self.controls})
            response = opt.transact(self.system, _round_trip(request))
            self.assertTrue(response.ok, response.error)
            direct = opt.transact(self.system, request)
            self.assertEqual(response.to_dict()["result"], direct.to_dict()["result"])
            json.dumps(response.to_dict(), allow_nan=False)

    def test_channel_mapping_flat_and_matrix_payloads_decode(self):
        matrix = self.controls.as_matrix()
        payloads = (
            {key: matrix[index].tolist() for index, key in enumerate(self.controls.spec.keys)},
            matrix.reshape(-1).tolist(),
            matrix.tolist(),
            {"flat": matrix.reshape(-1).tolist()},
        )
        expected = opt.transact(self.system, opt.SystemRequest("evaluate", {"controls": self.controls}))
        for payload in payloads:
            response = opt.transact(self.system, {"operation": "evaluate", "payload": {"controls": payload}})
            self.assertTrue(response.ok, response.error)
            self.assertEqual(response.result["J"], expected.result["J"])

    def test_a_layout_mismatch_is_an_invalid_payload(self):
        response = opt.transact(
            self.system, {"operation": "evaluate", "payload": {"controls": [1.0, 2.0, 3.0]}}
        )
        self.assertFalse(response.ok)
        self.assertEqual(response.error["type"], "invalid_payload")

    def test_contract_violations_are_reported_not_returned(self):
        response = opt.transact(
            BrokenQuadratic(), opt.SystemRequest("evaluate", {"controls": self.controls})
        )
        self.assertFalse(response.ok)
        self.assertEqual(response.error["type"], "ValueError")

    def test_optional_hooks_are_available_when_declared(self):
        system = CombinedQuadratic()
        combined = opt.transact(system, opt.SystemRequest("value_and_gradient", {"controls": self.controls}))
        bounds = opt.transact(system, opt.SystemRequest("control_bounds"))
        residuals = opt.transact(system, opt.SystemRequest("residuals", {"controls": self.controls}))

        self.assertTrue(combined.ok, combined.error)
        self.assertIn("gradient", combined.to_dict()["result"])
        self.assertTrue(bounds.ok, bounds.error)
        self.assertTrue(residuals.ok, residuals.error)
        for response in (combined, bounds, residuals):
            json.dumps(response.to_dict(), allow_nan=False)

    def test_a_missing_capability_fails_before_computation(self):
        response = opt.transact(self.system, opt.SystemRequest("hvp", {"controls": self.controls, "vector": self.controls}))
        self.assertFalse(response.ok)
        self.assertEqual(response.error["type"], "capability_missing")

    def test_unsupported_and_malformed_requests_are_structured(self):
        unknown = opt.transact(self.system, opt.SystemRequest("step"))
        static = opt.transact(self.system, opt.SystemRequest("system_spec", {"extra": 1}))
        missing = opt.transact(self.system, opt.SystemRequest("evaluate"))
        mismatch = opt.transact(self.system, opt.SystemRequest("evaluate", {"controls": self.controls}, family="olss"))

        self.assertEqual(unknown.error["type"], "unsupported_operation")
        self.assertIn("value_and_gradient", unknown.error["message"])
        self.assertEqual(static.error["type"], "invalid_payload")
        self.assertEqual(missing.error["type"], "invalid_payload")
        self.assertEqual(mismatch.error["type"], "family_mismatch")


class OlssTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system = RectangularShootingSystem()
        self.variables = opt.ShootingVariables.from_flat(self.system.shooting_spec(), [4.0, 3.0])

    def test_json_round_trip_for_every_numerical_call(self):
        for operation in ("residuals", "jacobian", "residuals_and_jacobian", "evaluate", "verify"):
            request = opt.SystemRequest(operation, {"variables": self.variables}, options={"profile": "check"})
            response = opt.transact(self.system, _round_trip(request))
            self.assertTrue(response.ok, (operation, response.error))
            json.dumps(response.to_dict(), allow_nan=False)

    def test_block_mapping_and_flat_payloads_decode(self):
        expected = opt.transact(self.system, opt.SystemRequest("residuals", {"variables": self.variables}))
        for payload in (self.variables.unpack(), [4.0, 3.0]):
            payload = {key: np.asarray(value).tolist() for key, value in payload.items()} if isinstance(payload, dict) else payload
            response = opt.transact(self.system, {"operation": "residuals", "payload": {"variables": payload}})
            self.assertTrue(response.ok, response.error)
            np.testing.assert_allclose(response.result, expected.result)

    def test_optional_cost_gradient_and_its_absence(self):
        direct = DirectCostShootingSystem()
        variables = opt.ShootingVariables.zeros(direct.shooting_spec())
        present = opt.transact(direct, opt.SystemRequest("cost_gradient", {"variables": variables}))
        absent = opt.transact(self.system, opt.SystemRequest("cost_gradient", {"variables": self.variables}))

        self.assertTrue(present.ok, present.error)
        self.assertEqual(absent.error["type"], "capability_missing")


if __name__ == "__main__":
    unittest.main()
