"""Transaction boundary: payload decoding, contract validation, and optional hooks."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from dataclasses import replace

import numpy as np

import optimizer as opt
from optimizer import ControlBounds, Controls
from optimizer.system_olss import residuals_system
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


class CountingQuadratic(QuadraticVectorSystem):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def evaluate(self, controls):
        self.calls += 1
        return super().evaluate(controls)


class ComplexSystem:
    def __init__(self, *, dtype=complex):
        self.dtype = dtype
        self.calls = 0

    def control_spec(self):
        return opt.ControlSpec(("drive",), 2, dtype=self.dtype, dt=0.5)

    def system_spec(self):
        return opt.basic_system_spec("complex-fixture", control_spec=self.control_spec())

    def evaluate(self, controls):
        self.calls += 1
        return {"J": float(np.sum(np.abs(controls.as_matrix()) ** 2))}

    def gradient(self, controls):
        return Controls.from_matrix(self.control_spec(), 2 * controls.as_matrix())

    def with_secondary(self, **updates):
        if updates:
            raise KeyError("No secondary updates.")
        return ComplexSystem(dtype=self.dtype)


class RequestBoundaryTests(unittest.TestCase):
    def test_malformed_envelopes_are_correlated_strict_json_failures(self):
        system = CountingQuadratic()
        bad = [None, [], 1, {}, {"operation": None}, {"operation": 1},
               {"operation": " "}, {"operation": "evaluate", "payload": []},
               {"operation": "evaluate", "options": []},
               {"operation": "evaluate", "family": "other"},
               {"operation": "evaluate", "request_id": 1},
               {"operation": "evaluate", "request_id": " "},
               {"operation": "evaluate", "payload": {1: 2}},
               {"operation": "evaluate", "options": {1: 2}},
               {"operation": "evaluate", "unknown": 1}]
        for request in bad:
            with self.subTest(request=request):
                response = opt.transact(system, request)
                self.assertFalse(response.ok)
                self.assertEqual(response.error["type"], "invalid_request")
                self.assertIsNone(response.family)
                self.assertTrue(response.request_id)
                json.dumps(response.to_dict(), allow_nan=False)
        correlated = opt.transact(system, {"request_id": "caller-id", "operation": "evaluate", "typo": 1})
        self.assertEqual(correlated.request_id, "caller-id")
        self.assertEqual(correlated.operation, "evaluate")
        self.assertEqual(system.calls, 0)

    def test_direct_request_construction_keeps_python_exceptions(self):
        for kwargs in ({"operation": None}, {"operation": "evaluate", "payload": []},
                       {"operation": "evaluate", "request_id": ""}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                opt.SystemRequest(**kwargs)

    def test_mutated_request_mappings_are_revalidated(self):
        request = opt.SystemRequest("system_spec")
        request.payload[1] = "bad"
        self.assertEqual(opt.transact(CountingQuadratic(), request).error["type"], "invalid_request")

    def test_family_discovery_failures_are_inside_the_boundary(self):
        class Invalid:
            def system_spec(self):
                return []

        class Raises:
            def system_spec(self):
                raise RuntimeError("bad spec")

        class Unknown:
            def system_spec(self):
                return {"family": "unknown"}

        class Incomplete(CountingQuadratic):
            def system_spec(self):
                return {"family": "olgs"}

        for system in (object(), Invalid(), Raises(), Unknown(), Incomplete()):
            with self.subTest(system=type(system).__name__):
                response = opt.transact(system, {"operation": "system_spec", "request_id": "spec-id"})
                self.assertFalse(response.ok)
                self.assertEqual(response.error["type"], "invalid_system")
                self.assertEqual(response.request_id, "spec-id")
                self.assertIsNone(response.family)
                json.dumps(response.to_dict(), allow_nan=False)

        incomplete = Incomplete()
        response = opt.transact(incomplete, {"operation": "gradient", "payload": {"controls": incomplete.reference_controls()}})
        self.assertEqual(response.error["type"], "invalid_system")
        self.assertEqual(incomplete.calls, 0)

    def test_payload_typos_and_disallowed_options_precede_numerical_work(self):
        system = CountingQuadratic()
        controls = system.reference_controls()
        for request in (
            {"operation": "evaluate", "payload": {"controls": controls, "typo": 1}},
            {"operation": "evaluate", "payload": {}, "options": {"accuracy": 1}},
            {"operation": "evaluate", "payload": {}},
            {"operation": "control_bounds", "payload": {"controls": controls}},
        ):
            with self.subTest(request=request):
                self.assertEqual(opt.transact(system, request).error["type"], "invalid_payload")
        self.assertEqual(system.calls, 0)

    def test_missing_secondary_hook_is_a_capability_failure(self):
        class NoSecondary:
            def system_spec(self):
                return CountingQuadratic().system_spec()

        response = opt.transact(NoSecondary(), {"operation": "with_secondary", "payload": {"weight": 1}})
        self.assertEqual(response.error["type"], "capability_missing")

    def test_valid_evaluation_reads_static_spec_and_runs_numerics_once(self):
        class CountSpecs(CountingQuadratic):
            spec_calls = 0

            def system_spec(self):
                self.spec_calls += 1
                return super().system_spec()

        system = CountSpecs()
        controls = system.reference_controls()
        response = opt.transact(system, {"operation": "evaluate", "payload": {"controls": controls}})
        self.assertTrue(response.ok, response.error)
        self.assertEqual(system.spec_calls, 1)
        self.assertEqual(system.calls, 1)


class InputCompatibilityTests(unittest.TestCase):
    def test_serialized_controls_check_every_declared_layout_field(self):
        system = CountingQuadratic()
        original = opt.to_plain(system.reference_controls())
        changes = {"keys": ["uz", "uy", "ux"], "control_dim": 16, "dtype": "float32",
                   "dt": 123.0, "shape": [1, 51], "n_controls": 2}
        for field, value in changes.items():
            for representation in ("matrix", "flat"):
                payload = deepcopy(original)
                payload["spec"][field] = value
                if representation == "flat":
                    payload["flat"] = np.asarray(payload.pop("matrix")).reshape(-1).tolist()
                with self.subTest(field=field, representation=representation):
                    response = opt.transact(system, {"operation": "evaluate", "payload": {"controls": payload}})
                    self.assertEqual(response.error["type"], "invalid_payload")
        self.assertEqual(system.calls, 0)

    def test_incomplete_specs_and_ambiguous_representations_fail(self):
        system = CountingQuadratic()
        base = opt.to_plain(system.reference_controls())
        payloads = [dict(base, spec={"keys": list(system.control_spec().keys)}),
                    dict(base, spec=None), dict(base, flat=[0]), dict(base, typo=1)]
        for payload in payloads:
            self.assertEqual(opt.transact(system, {"operation": "evaluate", "payload": {"controls": payload}}).error["type"], "invalid_payload")
        self.assertEqual(system.calls, 0)

    def test_live_controls_and_json_have_the_same_policy(self):
        system = CountingQuadratic()
        controls = system.reference_controls()
        for spec in (replace(controls.spec, dt=0.2), replace(controls.spec, dtype="float32")):
            changed = Controls.from_matrix(spec, controls.as_matrix())
            for payload in (changed, opt.to_plain(changed)):
                response = opt.transact(system, {"operation": "evaluate", "payload": {"controls": payload}})
                self.assertEqual(response.error["type"], "invalid_payload")
            with self.assertRaises(ValueError):
                opt.validate_controls_for_system(system, changed)
        self.assertEqual(system.calls, 0)

    def test_nonfinite_controls_are_rejected_even_after_live_mutation(self):
        system = CountingQuadratic()
        controls = system.reference_controls()
        controls.u[0, 0] = np.nan
        for payload in (controls, opt.to_plain(controls)):
            response = opt.transact(system, {"operation": "evaluate", "payload": {"controls": payload}})
            self.assertEqual(response.error["type"], "invalid_payload")
        with self.assertRaises(ValueError):
            opt.validate_controls_for_system(system, controls)
        self.assertEqual(system.calls, 0)

    def test_complex_controls_round_trip_for_all_input_forms(self):
        system = ComplexSystem()
        controls = Controls.from_matrix(system.control_spec(), [[1 + 2j, 3 + 4j]])
        canonical = opt.to_plain(controls)
        payloads = [controls, canonical,
                    {"flat": opt.to_plain(controls.flatten()), "spec": system.control_spec().to_dict()},
                    {"matrix": opt.to_plain(controls.as_matrix())},
                    {"drive": opt.to_plain(controls.as_matrix()[0])}]
        for payload in payloads:
            for operation in ("evaluate", "gradient"):
                response = opt.transact(system, {"operation": operation, "payload": {"controls": payload}})
                self.assertTrue(response.ok, response.error)
                json.dumps(response.to_dict(), allow_nan=False)
                if operation == "evaluate":
                    self.assertEqual(response.result["J"], 30.0)
                else:
                    np.testing.assert_array_equal(response.result.as_matrix(), 2 * controls.as_matrix())
        np.testing.assert_array_equal(controls.as_matrix(), [[1 + 2j, 3 + 4j]])

    def test_complex_inputs_do_not_broadcast_or_lose_imaginary_values(self):
        for dtype, matrix in ((float, [[1 + 2j, 3 + 4j]]),
                              (complex, {"real": [[1, 2]], "imag": [1, 2]})):
            system = ComplexSystem(dtype=dtype)
            response = opt.transact(system, {"operation": "evaluate", "payload": {"controls": {"matrix": opt.to_plain(matrix)}}})
            self.assertEqual(response.error["type"], "invalid_payload")
            self.assertEqual(system.calls, 0)

    def test_decoder_owns_values_before_calling_a_mutating_system(self):
        class Mutating(CountingQuadratic):
            def evaluate(self, controls):
                result = super().evaluate(controls)
                controls.u[:] = 0.0
                return result

        system = Mutating()
        controls = system.reference_controls()
        original = controls.as_matrix()
        for payload in (controls, original):
            response = opt.transact(system, {"operation": "evaluate", "payload": {"controls": payload}})
            self.assertTrue(response.ok, response.error)
            np.testing.assert_array_equal(controls.as_matrix(), original)
        self.assertEqual(system.calls, 2)

    def test_descriptive_metadata_is_not_numerical_identity(self):
        system = CountingQuadratic()
        controls = system.reference_controls()
        controls.spec = replace(controls.spec, meta={"label": "new label"})
        for payload in (controls, opt.to_plain(controls)):
            self.assertTrue(opt.transact(system, {"operation": "evaluate", "payload": {"controls": payload}}).ok)

    def test_shooting_layout_semantics_are_preserved_for_live_and_json(self):
        system = RectangularShootingSystem()
        original = opt.ShootingVariables.from_flat(system.shooting_spec(), [4, 3])
        variants = [replace(original.spec, dtype="float32")]
        for update in ({"shape": (1,)}, {"unit": "rad"}, {"scale": 2.0}, {"role": "costate"}):
            variants.append(replace(original.spec, blocks=(replace(original.spec.blocks[0], **update), original.spec.blocks[1])))
        variants.append(replace(original.spec, blocks=tuple(reversed(original.spec.blocks))))
        for spec in variants:
            variables = opt.ShootingVariables.from_flat(spec, [4, 3])
            for payload in (variables, variables.to_dict()):
                response = opt.transact(system, {"operation": "residuals", "payload": {"variables": payload}})
                self.assertEqual(response.error["type"], "invalid_payload")
            with self.assertRaises(ValueError):
                residuals_system(system, variables)
        self.assertEqual(system.combined_calls, 0)

    def test_shooting_labels_can_change_without_changing_coordinates(self):
        system = RectangularShootingSystem()
        spec = replace(system.shooting_spec(), metadata={"label": "new"}, blocks=tuple(
            replace(block, meaning="new wording", metadata={"label": "new"}) for block in system.shooting_spec().blocks
        ))
        variables = opt.ShootingVariables.from_flat(spec, [4, 3])
        for payload in (variables, variables.to_dict()):
            response = opt.transact(system, {"operation": "residuals", "payload": {"variables": payload}})
            self.assertTrue(response.ok, response.error)
            np.testing.assert_array_equal(response.result, [3, 5, 8])

    def test_shooting_specs_and_values_cannot_hide_invalid_fields(self):
        system = RectangularShootingSystem()
        original = opt.ShootingVariables.from_flat(system.shooting_spec(), [4, 3]).to_dict()
        payloads = []
        for path, value in (("size", 3), ("dtype", "float32")):
            payload = deepcopy(original)
            payload["spec"][path] = value
            payloads.append(payload)
        payload = deepcopy(original)
        payload["spec"]["blocks"][0]["size"] = 2
        payloads.append(payload)
        payloads.extend([dict(original, spec={"size": 2}), dict(original, values=["nan", 3]),
                         dict(original, values=[1 + 2j, 3]), dict(original, typo=1)])
        for payload in payloads:
            response = opt.transact(system, {"operation": "residuals", "payload": {"variables": payload}})
            self.assertEqual(response.error["type"], "invalid_payload")

    def test_shooting_static_declaration_cannot_disagree_at_equal_size(self):
        class WrongDeclaration(RectangularShootingSystem):
            def system_spec(self):
                spec = super().system_spec()
                spec["variables"]["blocks"][0]["unit"] = "wrong unit"
                return spec

        system = WrongDeclaration()
        response = opt.transact(system, {"operation": "shooting_spec"})
        self.assertFalse(response.ok)
        self.assertEqual(response.error["type"], "ValueError")
        self.assertEqual(system.combined_calls, 0)

    def test_non_mapping_family_spec_does_not_default_to_olgs(self):
        class Invalid:
            def system_spec(self):
                return []

        with self.assertRaises(TypeError):
            opt.system_family(Invalid())


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
