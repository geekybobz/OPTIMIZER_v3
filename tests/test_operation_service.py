"""Foreground workflows and ownership regressions, using neutral systems only."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import numpy as np

import optimizer as opt
from fixtures.quadratic_system import QuadraticVectorSystem
from fixtures.shooting_systems import RectangularShootingSystem


class OperationServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = opt.OperationService()
        self.load()

    def call(self, operation, arguments=None, **options):
        response = self.service.execute(opt.OperationRequest(operation, arguments or {}, **options))
        self.assertTrue(response.ok, response.error)
        return response

    def load(self, name="main", factory=QuadraticVectorSystem, **parameters):
        if factory is QuadraticVectorSystem:
            parameters.setdefault("N", 5)
        return self.call("session.load", {"name": name, "factory": factory, "parameters": parameters})

    def initial(self):
        return self.call("guesses.constant_guess", bind="u0", activate=True)

    def adam(self, **options):
        args = {"maxiter": 2, "step_size": 0.02, "gradient_tolerance": None}
        args.update(options.pop("arguments", {}))
        return self.call("adam", args, **options)

    def fail(self, operation, arguments=None, code=None, **options):
        before = self.service.session.snapshot()
        response = self.service.execute({"operation": operation, "arguments": arguments or {}, **options})
        self.assertFalse(response.ok)
        if code:
            self.assertEqual(response.error["code"], code, response.error)
        self.assertEqual(self.service.session.snapshot(), before)
        return response

    def test_request_validation_and_correlated_failures(self):
        for request in [None, [], {}, {"operation": "adam", "extra": 1},
                        {"operation": "adam", "arguments": {1: 2}},
                        {"operation": "adam", "activate": 1}]:
            with self.subTest(request=request):
                response = self.service.execute(request)
                self.assertEqual(response.error["code"], "invalid_request")
                json.dumps(response.to_dict(), allow_nan=False)
        response = self.service.execute({"operation": "", "request_id": "mine"})
        self.assertEqual(response.request_id, "mine")

    def test_mutated_request_mapping_revalidated(self):
        request = opt.OperationRequest("adam")
        request.arguments[1] = "bad"
        self.assertEqual(self.service.execute(request).error["code"], "invalid_request")

    def test_catalog_complete_classification_alias_and_constructor(self):
        entries = self.service.list()
        expected = {i.id for g in opt.catalog.groups() for i in opt.catalog.group(g).items}
        self.assertEqual({i["id"] for i in entries}, expected)
        self.assertEqual(self.service.info("ncg")["id"], "optimizers.nonlinear_cg")
        self.assertEqual(self.service.info("optimizers.ncg")["id"], "optimizers.nonlinear_cg")
        self.assertTrue(all(i["reason"] for i in entries if i["classification"] not in {"tool", "constructor"}))
        response = self.call("schedules.ConstantSchedule", {"value": 0.02}, bind="schedule")
        self.assertIsInstance(response.result, opt.ConstantSchedule)
        self.fail("tracker.RunTracker", code="adapter_unavailable")
        self.fail("does_not_exist", code="unknown_operation")

    def test_argument_typos_rejected_before_computation(self):
        self.initial()
        for arguments in ({"maxiterr": 2}, {"statee": None}, {"history_writer": True}):
            self.fail("adam", arguments, "invalid_arguments")
        self.fail("session.select", {"name": "main", "unknown": 2}, "invalid_arguments")

    def test_missing_context_state_and_family(self):
        self.fail("adam", code="invalid_arguments")  # native controls required despite default
        empty = opt.OperationService()
        self.assertEqual(empty.execute({"operation": "adam"}).error["code"], "missing_context")
        self.load("shot", RectangularShootingSystem)
        self.fail("adam", code="wrong_family")

    def test_native_adam_equivalence_and_chunk_continuation(self):
        initial = self.initial().result
        first = self.adam(bind="first")
        second = self.adam(arguments={"maxiter": 3}, bind="second")
        native = opt.adam(QuadraticVectorSystem(N=5), initial, maxiter=5, step_size=0.02, gradient_tolerance=None)
        np.testing.assert_allclose(second.result.controls.as_matrix(), native.controls.as_matrix())
        np.testing.assert_allclose(second.result.state.optimizer_state["m"], native.state.optimizer_state["m"])
        self.assertEqual(second.result.state.global_iteration, 5)
        self.assertEqual(second.metadata["continuation"], "method_state")
        self.assertEqual(first.result.stop_reason, "maxiter")
        self.assertTrue(first.metadata["activated"])

    def test_candidate_inspection_then_explicit_activation(self):
        self.initial()
        active = self.adam(bind="active").object_refs[0]
        trial = self.call("lbfgs", {"maxiter": 1}, bind="trial", activate=False)
        self.assertEqual(self.service.session.snapshot()["contexts"]["main"]["head"], active.object_id)
        self.call("utils.diagnose_run", {"result": trial.object_refs[0]})
        self.call("session.use", {"value": trial.object_refs[0]})
        np.testing.assert_allclose(self.service.session.active().controls.as_matrix(), trial.result.controls.as_matrix())

    def test_method_switch_config_change_and_explicit_start_reset(self):
        u = self.initial().result
        self.adam()
        changed = self.adam(arguments={"variant": "amsgrad"})
        self.assertEqual(changed.result.state.optimizer_state["t"], 2)
        self.assertEqual(changed.metadata["continuation"], "physical_restart")
        explicit = self.adam(arguments={"controls": u, "variant": "amsgrad"})
        self.assertEqual(explicit.result.state.optimizer_state["t"], 2)
        self.assertEqual(explicit.result.state.global_iteration, 2)
        switched = self.call("momentum", {"maxiter": 1, "gradient_tolerance": None})
        self.assertNotIn("m", switched.result.state.optimizer_state)

    def test_same_effective_engine_default_preserves_continuation(self):
        self.initial()
        self.adam()
        result = self.adam(arguments={"use_cache": True})
        self.assertEqual(result.metadata["continuation"], "method_state")

    def test_explicit_state_reference_requires_full_configuration(self):
        self.initial()
        result = self.adam(bind="r")
        reference = opt.ObjectRef(result.object_refs[0].object_id, ("state",))
        continued = self.adam(arguments={"state": reference})
        self.assertEqual(continued.result.state.global_iteration, 4)
        self.fail("adam", {"state": reference, "step_size": 0.03}, "unsupported_continuation")
        self.fail("adam", {"state": result.result.state}, "unsupported_continuation")

    def test_layout_checked_before_call(self):
        self.initial()
        wrong = opt.constant_guess(QuadraticVectorSystem(N=7))
        self.fail("adam", {"controls": wrong}, "incompatible_layout")
        spec = opt.ControlSpec(("ux", "uy", "uz"), 5, dt=0.5)
        wrong = opt.Controls(spec, np.zeros(spec.shape))
        self.fail("session.use", {"value": wrong}, "incompatible_layout")

    def test_result_input_and_store_buffers_owned(self):
        initial = self.initial()
        initial.result.set_channel("ux", np.full(5, 7.0))
        self.assertTrue(np.all(self.service.session.active().as_matrix() == 0))
        result = self.adam(bind="r")
        result.result.state.optimizer_state["m"][:] = 900
        result.result.controls.set_channel("ux", np.full(5, 900.0))
        stored = self.service.session.get("r")
        self.assertFalse(np.any(stored.controls.as_matrix() == 900))
        stored.state.optimizer_state["m"][:] = 500
        self.assertFalse(np.any(self.service.session.get("r").state.optimizer_state["m"] == 500))

    def test_named_versions_and_public_field_refs(self):
        first = self.initial()
        second = self.call("guesses.constant_guess", {"value": 0.5}, bind="u0")
        self.assertNotEqual(first.object_refs[0], second.object_refs[0])
        np.testing.assert_allclose(self.service.session.get(first.object_refs[0]).as_matrix(), 0)
        result = self.adam(bind="r")
        controls = self.call("session.get", {"reference": opt.ObjectRef(result.object_refs[0].object_id, ("controls",))})
        self.assertIsInstance(controls.result, opt.Controls)
        self.fail("session.get", {"reference": opt.ObjectRef(result.object_refs[0].object_id, ("warmstart",))}, "invalid_reference")

    def test_context_switch_preserves_independent_heads(self):
        self.initial()
        self.adam()
        head = self.service.session.snapshot()["contexts"]["main"]["head"]
        self.load("other", objective_weight=3)
        self.call("guesses.constant_guess", {"value": 0.3}, activate=True)
        other = self.service.session.snapshot()["contexts"]["other"]["head"]
        self.call("session.select", {"name": "main"})
        self.assertEqual(self.service.session.snapshot()["contexts"]["main"]["head"], head)
        self.assertEqual(self.service.session.snapshot()["contexts"]["other"]["head"], other)

    def test_explicit_context_does_not_change_selection(self):
        self.initial()
        self.load("other")
        result = self.adam(context="main")
        self.assertEqual(result.state_after["selected"], "other")
        self.assertIsNone(result.state_after["contexts"]["other"]["head"])

    def test_invalid_system_load_and_duplicate_preserve_session(self):
        self.fail("session.load", {"name": "bad", "factory": object}, "invalid_system")
        self.fail("session.load", {"name": "main", "factory": QuadraticVectorSystem}, "context_exists")
        self.fail("session.load", {"name": "bad", "factory": QuadraticVectorSystem, "parameters": {"unknown": 2}}, "invalid_arguments")

    def test_secondary_update_invalidates_metrics_and_method_memory(self):
        self.initial()
        old = self.adam(bind="old")
        self.call("session.update", {"updates": {"objective_weight": 2}})
        self.assertIsInstance(self.service.session.active(), opt.Controls)
        resumed = self.adam()
        self.assertEqual(resumed.result.state.global_iteration, 2)
        self.fail("session.use", {"value": old.object_refs[0]}, "unsupported_continuation")
        self.fail("session.update", {"updates": {"unknown": 2}}, "invalid_arguments")

    def test_attached_system_requires_isolation_for_solver(self):
        self.call("session.attach", {"name": "attached", "system": QuadraticVectorSystem(N=5)})
        self.call("session.use", {"value": opt.constant_guess(QuadraticVectorSystem(N=5))})
        self.call("evaluate")
        self.fail("adam", {"maxiter": 1}, "unsupported_isolation")
        self.call("session.attach", {"name": "cloned", "system": QuadraticVectorSystem(N=5), "clone": lambda: QuadraticVectorSystem(N=5)})
        self.initial()
        self.adam()

    def test_isolation_rejects_same_object_clone(self):
        system = QuadraticVectorSystem(N=5)
        self.call("session.attach", {"name": "singleton", "system": system, "clone": lambda: system})
        self.fail("guesses.constant_guess", code="unsupported_isolation")

    def test_factory_parameters_are_captured(self):
        parameters = {"N": 5}
        self.call("session.load", {"name": "captured", "factory": QuadraticVectorSystem, "parameters": parameters})
        parameters["N"] = 9
        result = self.call("guesses.constant_guess")
        self.assertEqual(result.result.spec.control_dim, 5)

    def test_source_file_factory_digest_and_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "system.py"
            path.write_text("from fixtures.quadratic_system import QuadraticVectorSystem\ndef system(N=5):\n    return QuadraticVectorSystem(N=N)\n")
            loaded = self.call("session.load", {"name": "file", "factory": str(path)})
            self.assertEqual(loaded.result["recipe"]["parameters"], {"N": 5})
            self.assertEqual(len(loaded.result["recipe"]["sha256"]), 64)
            path.write_text(path.read_text() + "# changed\n")
            self.fail("guesses.constant_guess", code="source_changed")

    def test_shooting_solver_history_and_native_equivalence(self):
        self.load("shot", RectangularShootingSystem)
        system = RectangularShootingSystem()
        variables = opt.ShootingVariables(system.shooting_spec(), np.zeros(2))
        self.call("session.use", {"value": variables})
        first = self.call("solve_shooting", {"maxiter": 1, "damping": 0.2, "tolerance": 0}, bind="s1")
        second = self.call("solve_shooting", {"maxiter": 1, "damping": 0.2, "tolerance": 0})
        native = opt.solve_shooting(system, variables, maxiter=1, damping=0.2, tolerance=0)
        np.testing.assert_allclose(first.result.variables.values, native.variables.values)
        self.assertEqual(second.metadata["continuation"], "variables_and_history")
        self.assertGreater(len(second.result.history), len(first.result.history))
        self.assertIn("qualified", second.to_dict()["metadata"]["summary"])

    def test_shooting_seed_and_foreground_multistart(self):
        self.load("shot", RectangularShootingSystem)
        seeds = self.call("random_shooting_seeds", {"count": 2, "seed": 1}, bind="seeds")
        result = self.call("solve_shooting_multistart", {"starts": seeds.object_refs[0], "solver_options": {"maxiter": 2}})
        self.assertEqual(len(result.result.results), 2)
        self.assertFalse(result.metadata["activated"])
        first_seed = opt.ObjectRef(seeds.object_refs[0].object_id, (0,))
        self.call("session.use", {"value": first_seed})
        self.assertIsInstance(self.service.session.active(), opt.ShootingVariables)

    def test_quota_rejects_atomically_and_release_allows_new_result(self):
        self.service = opt.OperationService(opt.Session(max_objects=2))
        self.load()
        self.initial()
        self.adam(bind="r")
        self.fail("adam", {"maxiter": 1}, "object_quota")
        self.call("session.release", {"name": "u0"})
        self.adam()
        self.assertLessEqual(self.service.session.snapshot()["objects"], 2)

    def test_byte_quota_and_constructor_without_context(self):
        self.service = opt.OperationService(opt.Session(max_bytes=200))
        self.fail("base.controls", {"spec": opt.ControlSpec(("a",), 100), "values": np.zeros((1, 100))}, "object_quota")
        service = opt.OperationService()
        result = service.execute({"operation": "schedules.ConstantSchedule", "arguments": {"value": 0.02}})
        self.assertTrue(result.ok, result.error)

    def test_reentrant_callback_cannot_overwrite_newer_revision(self):
        self.initial()
        before_head = self.service.session.snapshot()["contexts"]["main"]["head"]
        def accept(*args):
            self.service.session.select("main")
            return opt.AcceptanceDecision(True, "accepted")
        response = self.service.execute({"operation": "adam", "arguments": {"maxiter": 1, "accept": accept}})
        self.assertEqual(response.error["code"], "revision_conflict")
        self.assertEqual(self.service.session.snapshot()["contexts"]["main"]["head"], before_head)

    def test_two_services_share_revision_guard(self):
        self.initial()
        other = opt.OperationService(self.service.session)
        native = opt.optimizers.adam
        barrier = threading.Barrier(2)
        @wraps(native)
        def synchronized(*args, **kwargs):
            barrier.wait(timeout=5)
            return native(*args, **kwargs)
        responses = []
        def run(service):
            responses.append(service.execute({"operation": "adam", "arguments": {"maxiter": 1}}))
        with patch.object(opt.optimizers, "adam", synchronized):
            threads = [threading.Thread(target=run, args=(s,)) for s in (self.service, other)]
            for thread in threads: thread.start()
            for thread in threads: thread.join(timeout=10)
        self.assertEqual(sum(r.ok for r in responses), 1)
        self.assertEqual([r.error["code"] for r in responses if not r.ok], ["revision_conflict"])

    def test_read_result_compact_json_and_no_extra_numerical_calls(self):
        calls = []
        class Counted(QuadraticVectorSystem):
            def evaluate(self, controls):
                calls.append("evaluate")
                return super().evaluate(controls)
        self.load("counted", Counted, N=5)
        self.initial()
        result = self.call("evaluate")
        self.assertEqual(calls, ["evaluate"])
        encoded = json.dumps(result.to_dict(), allow_nan=False)
        self.assertLess(len(encoded), 4000)
        self.assertNotIn('"matrix"', encoded)
        self.assertEqual(calls, ["evaluate"])

    def test_native_failure_does_not_activate_bad_result(self):
        class Broken(QuadraticVectorSystem):
            def gradient(self, controls):
                controls.u[:] = 9
                raise RuntimeError("failed gradient")
        self.load("broken", Broken, N=5)
        self.initial()
        before = self.service.session.snapshot()["contexts"]["broken"]["head"]
        result = self.call("adam", {"maxiter": 1})
        self.assertFalse(result.metadata["activated"])
        self.assertEqual(self.service.session.snapshot()["contexts"]["broken"]["head"], before)
        np.testing.assert_allclose(self.service.session.active().as_matrix(), 0)

    def test_missing_capability_and_python_no_history_files(self):
        self.initial()
        self.fail("optional_hvp", {"vector": np.ones(15)}, "capability_missing")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.adam()
                self.call("evaluate")
                self.assertEqual(list(Path(directory).iterdir()), [])
            finally:
                os.chdir(previous)

    def test_registered_custom_step_and_controls_only_restart(self):
        self.initial()
        def step(context):
            return opt.StepProposal(controls=opt.Controls.from_matrix(
                context.state.controls.spec, context.state.controls.as_matrix() * 0.5))
        reference = self.service.register_reference("step", step)
        result = self.call("core.run_chunk", {"step": reference, "optimizer_name": "custom", "maxiter": 1})
        self.assertEqual(result.result.optimizer, "custom")
        self.assertTrue(result.metadata["activated"])

    def test_nested_options_validated_and_parallel_rejected(self):
        self.initial()
        self.fail("utils.compare_methods", {"methods": ["adam"], "options": {"adam": {"maxiterr": 1}}}, "invalid_arguments")
        self.load("shot", RectangularShootingSystem)
        seeds = self.call("random_shooting_seeds", {"count": 2}).object_refs[0]
        self.fail("solve_shooting_multistart", {"starts": seeds, "solver_options": {"maxiterr": 1}}, "invalid_arguments")
        self.fail("solve_shooting_multistart", {"starts": seeds, "parallel": opt.ParallelConfig(backend="thread", workers=2)}, "capability_missing")

    def test_derivative_report_and_transform_native_equivalence(self):
        initial = self.initial()
        report = self.call("utils.verify_gradient", {"directions": 2})
        self.assertTrue(report.result["passed"])
        transformed = self.call("guesses.scale_guess", {"controls": initial.object_refs[0], "amplitude": 0.2, "offset": 0.1})
        direct = opt.scale_guess(initial.result, amplitude=0.2, offset=0.1)
        np.testing.assert_allclose(transformed.result.as_matrix(), direct.as_matrix())
        head = self.service.session.snapshot()["contexts"]["main"]["head"]
        self.assertEqual(head, initial.object_refs[0].object_id)

    def test_context_count_and_recipe_byte_limits(self):
        self.service = opt.OperationService(opt.Session(max_objects=1))
        self.load()
        self.fail("session.load", {"name": "too_many", "factory": QuadraticVectorSystem}, "object_quota")
        self.service = opt.OperationService(opt.Session(max_bytes=500))
        self.fail("session.load", {"name": "too_large", "factory": QuadraticVectorSystem}, "object_quota")

    def test_factory_changed_configuration_rejected(self):
        state = [1]
        def factory():
            system = QuadraticVectorSystem(N=5, objective_weight=state[0])
            state[0] += 1
            return system
        self.load("unstable", factory)
        self.fail("guesses.constant_guess", code="invalid_system")

    def test_effective_mutable_factory_defaults_frozen(self):
        settings = {"N": 5}
        def factory(config=settings):
            return QuadraticVectorSystem(**config)
        self.load("defaults", factory)
        settings["N"] = 7
        result = self.call("guesses.constant_guess")
        self.assertEqual(result.result.spec.control_dim, 5)

    def test_raw_shooting_transaction_and_candidate_ref(self):
        self.load("shot", RectangularShootingSystem)
        variables = opt.ShootingVariables(RectangularShootingSystem().shooting_spec(), np.zeros(2))
        self.call("session.use", {"value": variables})
        response = self.call("transact", {"request": opt.SystemRequest("residuals", {"variables": variables})})
        self.assertTrue(response.result.ok)
        np.testing.assert_allclose(response.result.result, [-1, 2, 1])

    def test_secondary_layout_change_fails_without_head(self):
        class LayoutChanging(QuadraticVectorSystem):
            def with_secondary(self, **updates):
                return LayoutChanging(N=7)
        self.load("layout", LayoutChanging, N=5)
        self.fail("session.update", {"updates": {}}, "incompatible_layout")

    def test_large_factory_inputs_are_summarized_in_transport(self):
        def factory(table):
            return QuadraticVectorSystem(N=5)
        self.load("large_input", factory, table=np.zeros(10000))
        response = self.call("session.state")
        text = json.dumps(response.to_dict(), allow_nan=False)
        self.assertLess(len(text), 4000)
        table = response.to_dict()["state_after"]["contexts"]["large_input"]["recipe"]["parameters"]["table"]
        self.assertEqual(table["shape"], [10000])


if __name__ == "__main__":
    unittest.main()
