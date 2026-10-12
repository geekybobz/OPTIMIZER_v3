"""Command grammar, provenance and neutral numerical workflow regressions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import optimizer as opt

from fixtures.quadratic_system import QuadraticVectorSystem
from fixtures.shooting_systems import RectangularShootingSystem
from optimizer.cli.discovery import _discover
from optimizer.cli.input import _frame
from optimizer.cli.language import _Terminal
from optimizer.operations.contracts import _ServiceError


class LanguageTests(unittest.TestCase):
    def setUp(self):
        self.responses, self.events = [], []
        self.service = opt.OperationService()
        self.service.session.load("main", QuadraticVectorSystem, parameters={"N": 9})
        self.terminal = _Terminal(self.service, emit=self.responses.append, observer=self.events.append)

    def run_commands(self, source, status=0):
        self.assertEqual(self.terminal.submit(source), status, self.responses[-1].error if self.responses else None)
        return self.responses[-1] if self.responses else None

    def initial(self):
        self.run_commands("u = opt.guesses.constant_guess(value=0.1); opt.session.use(u);")

    def test_native_olgs_parity_and_positional_binding(self):
        self.initial()
        self.run_commands("r = opt.adam(u, maxiter=3, step_size=0.01, gradient_tolerance=None);")
        actual = self.service.session.get("r")
        native = opt.adam(QuadraticVectorSystem(N=9), self.service.session.get("u"),
                          maxiter=3, step_size=0.01, gradient_tolerance=None)
        np.testing.assert_array_equal(actual.controls.as_matrix(), native.controls.as_matrix())
        self.assertEqual(actual.metrics, native.metrics)
        self.assertEqual(self.events[-1]["request"].operation, "optimizers.adam")
        self.assertIn("r = opt.adam", self.events[-1]["raw_command"])

    def test_nested_schedule_and_continuation(self):
        self.initial()
        self.run_commands("r = opt.adam(maxiter=2, step_size=opt.schedules.ConstantSchedule(value=0.01), gradient_tolerance=None);")
        self.run_commands("r2 = opt.adam(maxiter=2, step_size=opt.schedules.ConstantSchedule(value=0.01), gradient_tolerance=None);")
        self.assertEqual(self.service.session.get("r2").state.global_iteration, 4)

    def test_root_alias_protects_old_version_after_rebinding(self):
        self.initial()
        self.run_commands("r = opt.adam(maxiter=2); saved = r;")
        first = self.service.session.snapshot()["bindings"]["saved"]
        self.run_commands("r = opt.adam(maxiter=2); opt.session.use(saved);")
        self.assertNotEqual(first, self.service.session.snapshot()["bindings"]["r"])
        self.assertEqual(self.service.session.get("saved").state.global_iteration, 2)
        self.assertEqual(self.service.session.active().state.global_iteration, 2)

    def test_field_alias_retains_compatible_method_state(self):
        self.initial()
        self.run_commands("r = opt.adam(maxiter=2); s = r.state; r = opt.adam(maxiter=2);")
        self.run_commands("resumed = opt.adam(state=s, maxiter=2);")
        self.assertEqual(self.service.session.get("resumed").state.global_iteration, 4)
        np.testing.assert_array_equal(self.service.session.get("resumed").controls.as_matrix(),
                                      self.service.session.get("r").controls.as_matrix())

    def test_alias_configuration_stays_old_after_update(self):
        self.initial()
        self.run_commands("r = opt.adam(maxiter=2); opt.session.update(updates={}); s = r.state;")
        before = self.service.session.snapshot()
        self.run_commands("opt.adam(state=s, maxiter=2);", 2)
        self.assertEqual(self.responses[-1].error["code"], "unsupported_continuation")
        self.assertEqual(before, self.service.session.snapshot())

    def test_method_change_rejects_state_and_accepts_controls(self):
        self.initial()
        self.run_commands("r = opt.adam(maxiter=2); s = r.state; u1 = r.controls;")
        self.run_commands("opt.momentum(state=s, maxiter=1);", 2)
        self.assertEqual(self.responses[-1].error["code"], "unsupported_continuation")
        self.run_commands("trial = opt.momentum(controls=u1, maxiter=1, activate=False);")
        self.assertEqual(self.service.session.get("trial").state.global_iteration, 1)

    def test_activate_state_field_then_implicit_continuation(self):
        self.initial()
        self.run_commands("r=opt.adam(maxiter=2); opt.session.use(r.state); r2=opt.adam(maxiter=2);")
        self.assertEqual(self.service.session.get("r2").state.global_iteration, 4)
        self.assertEqual(self.responses[-1].metadata["continuation"], "method_state")
        self.run_commands("opt.session.use(r.controls); r3=opt.adam(maxiter=1);")
        self.assertEqual(self.service.session.get("r3").state.global_iteration, 1)

    def test_candidate_and_explicit_activation(self):
        self.initial()
        self.run_commands("r = opt.adam(maxiter=2);")
        head = self.service.session.snapshot()["contexts"]["main"]["head"]
        self.run_commands("trial = opt.lbfgs(maxiter=2, activate=False); trial.metrics;")
        self.assertEqual(self.service.session.snapshot()["contexts"]["main"]["head"], head)
        self.run_commands("opt.session.use(trial);")
        self.assertNotEqual(self.service.session.snapshot()["contexts"]["main"]["head"], head)

    def test_entire_batch_syntax_is_validated_first(self):
        before = self.service.session.snapshot()
        for bad in ["import os;", "x = lambda: 1;", "x=[v for v in [1]];", "opt=1;", "r.__class__;",
                    "a.b=1;", "a=b=1;", "opt.adam(**{});", "opt.adam(*[]);", "x=1+2;", "x=1e999;",
                    "x=opt.session.load('other', 'system.py');", "r[0:2];", "r[-1];", "r.warmstart();"]:
            with self.subTest(bad=bad):
                self.run_commands("u = opt.guesses.constant_guess(); " + bad, 2)
                self.assertEqual(before, self.service.session.snapshot())

    def test_runtime_failure_stops_remaining_batch_only(self):
        self.run_commands("x=1; opt.no_such_method(); y=2;", 2)
        self.assertEqual(self.service.session.get("x"), 1)
        self.assertNotIn("y", self.service.session.snapshot()["bindings"])

    def test_semicolons_comments_and_multiline_calls(self):
        self.run_commands('label="one;two"; # ; ignored\nu = opt.guesses.constant_guess(\n value=0.1\n); label;')
        self.assertEqual(self.service.session.get("label"), "one;two")
        self.assertIsNone(_frame('opt.guesses.constant_guess(\n', incomplete_ok=True))
        self.assertIsNone(_frame('x=1', incomplete_ok=True))
        self.assertIsNotNone(_frame('x=";"; # ;\n', incomplete_ok=True))

    def test_literal_data_and_complex_numbers(self):
        self.run_commands('x = {"a": [None, True, -2, +3.5, 2-3j], "b": (1, 2)}; x["a"][4];')
        self.assertEqual(self.responses[-1].result, 2-3j)

    def test_field_mapping_array_channel_access(self):
        self.initial()
        self.run_commands('u["ux"][0];')
        self.assertEqual(self.responses[-1].result, 0.1)
        self.run_commands('u.as_matrix;', 2)
        self.assertEqual(self.responses[-1].error["code"], "invalid_reference")
        self.run_commands('u["ux"][99];', 2)
        self.assertEqual(self.responses[-1].error["code"], "invalid_reference")

    def test_retention_and_alias_count_are_bounded(self):
        service = opt.OperationService(opt.Session(max_objects=2))
        terminal = _Terminal(service, emit=self.responses.append)
        self.assertEqual(terminal.submit("a=1; b=a;"), 0)
        before = service.session.snapshot()
        self.assertEqual(terminal.submit("c=a;"), 2)
        self.assertEqual(service.session.snapshot(), before)
        self.assertEqual(terminal.submit("opt.session.release('b'); c=a;"), 0)

    def test_input_limits_are_preflight(self):
        before = self.service.session.snapshot()
        for source in ['x="' + 'a' * 65536 + '";', "x=" + "[" * 40 + "1" + "]" * 40 + ";",
                       "opt.help();" * 129, "x=[" + "1," * 8192 + "];" ]:
            self.run_commands(source, 2)
            self.assertEqual(self.responses[-1].error["code"], "input_limit")
            self.assertEqual(self.service.session.snapshot(), before)

    def test_wrong_keywords_positions_and_unavailable_tools(self):
        self.initial()
        before = self.service.session.snapshot()
        for source in ["opt.adam(maxiterr=2);", "opt.adam(u,u);", "opt.adam(system=u);",
                       "opt.adam(maxiter=1, bind='r', activate='yes');", "opt.tracker.save();"]:
            self.run_commands(source, 2)
            self.assertEqual(self.service.session.snapshot(), before)
        self.run_commands("opt.OptimizerResult();", 2)
        self.run_commands("opt.info('tracker.create');", 2)  # Unknown name gives a useful catalog error.

    def test_help_info_and_completion_are_catalog_derived(self):
        self.run_commands("opt.help(); opt.info('adam'); opt.list(group='optimizers');")
        self.assertEqual(self.responses[-2].result["id"], "optimizers.adam")
        self.assertGreater(self.responses[-1].result["count"], 5)
        self.assertIn("opt.optimizers.adam", self.terminal.completions())
        self.assertNotIn("opt.tracker.create", self.terminal.completions())

    def test_shooting_neutral_workflow_and_native_parity(self):
        self.service.session.load("shoot", RectangularShootingSystem)
        self.run_commands('spec = opt.system_olss.validate_shooting_spec(); '
                          'v = opt.ShootingVariables(spec=spec, values=[0,0]); '
                          'opt.session.use(v); r=opt.solve_shooting(maxiter=3); r.variables.values;')
        actual = self.service.session.get("r")
        native = opt.solve_shooting(RectangularShootingSystem(), self.service.session.get("v"), maxiter=3)
        np.testing.assert_array_equal(actual.variables.values, native.variables.values)
        self.assertEqual(actual.qualified, native.qualified)
        self.assertIn("verified", self.responses[-2].metadata["summary"])
        before = self.service.session.snapshot()
        self.run_commands("opt.adam(maxiter=1);", 2)
        self.assertEqual(self.service.session.snapshot(), before)
        self.run_commands("opt.session.select('main'); u=opt.guesses.constant_guess(); opt.session.use(u);")
        self.assertEqual(self.service.session.snapshot()["selected"], "main")

    def test_inspect_and_get_alias_preserve_state_provenance(self):
        self.initial()
        self.run_commands("r=opt.adam(maxiter=1); s=opt.inspect(r.state); t=opt.session.get(r.state);")
        self.run_commands("r2=opt.adam(state=s, maxiter=1); r3=opt.adam(state=t, maxiter=1);")
        self.assertEqual(self.service.session.get("r3").state.global_iteration, 2)

    def test_keyboard_interrupt_preserves_head_and_exposes_hook(self):
        self.initial()
        before = self.service.session.snapshot()
        with patch.object(self.service, "execute", side_effect=KeyboardInterrupt):
            self.run_commands("opt.adam(maxiter=2);", 130)
        self.assertEqual(self.service.session.snapshot(), before)
        self.assertTrue(self.events[-1]["interrupted"])
        self.assertEqual(self.events[-1]["request"].operation, "optimizers.adam")

    def test_registered_callback_and_native_interruption(self):
        self.initial()
        def step(context):
            return opt.StepProposal(controls=opt.Controls.from_matrix(
                context.state.controls.spec, context.state.controls.as_matrix() * 0.5))
        self.service.register_reference("step", step)
        self.run_commands("r=opt.core.run_chunk(step=step, optimizer_name='custom', maxiter=1);")
        np.testing.assert_array_equal(self.service.session.get("r").controls.as_matrix(), 0.05)
        before = self.service.session.snapshot()
        def interrupted_step(context):
            raise KeyboardInterrupt
        self.service.register_reference("halt_step", interrupted_step)
        self.run_commands("r2=opt.core.run_chunk(step=halt_step, optimizer_name='custom', maxiter=1); x=1;", 130)
        self.assertTrue(self.events[-1]["interrupted"])
        self.assertEqual(self.service.session.snapshot()["contexts"], before["contexts"])
        self.assertNotIn("r2", self.service.session.snapshot()["bindings"])
        self.assertNotIn("x", self.service.session.snapshot()["bindings"])

    def test_discovery_is_bounded_and_never_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "system.py").write_text("raise RuntimeError('must not import')\ndef system(): pass\n")
            (root / "nested").mkdir()
            (root / "nested" / "system.py").write_text("class System: pass\n")
            (root / ".hidden").mkdir()
            (root / ".hidden" / "system.py").write_text("def system(): pass\n")
            self.assertEqual(len(_discover(directory, 0)["candidates"]), 1)
            self.assertEqual(len(_discover(directory, 1)["candidates"]), 2)
            self.run_commands(f'opt.session.discover(root={directory!r}, depth=1);')
            self.assertEqual(len(self.responses[-1].metadata["summary"]["candidates"]), 2)
            with self.assertRaises(_ServiceError):
                _discover(directory, True)

    def test_exit_stops_batch_without_running_later_commands(self):
        self.run_commands("opt.session.exit(); x=1;")
        self.assertTrue(self.terminal.closed)
        self.assertNotIn("x", self.service.session.snapshot()["bindings"])
