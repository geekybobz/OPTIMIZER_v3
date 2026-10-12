"""Every stop reason the library assigns is in the shared registry."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import optimizer as opt
from optimizer.core.stop_reasons import FAMILIES, OUTCOMES, STOP_REASONS, StopReason


ROOT = Path(__file__).resolve().parents[1] / "optimizer"
_STOP_TARGETS = {"stop_reason", "requested_stop"}


def _string_constants(node: ast.AST) -> set[str]:
    """Literals that can become the assigned value: not tests, call arguments, or keys."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.JoinedStr) and node.values:
        head = node.values[0]
        return {head.value} if isinstance(head, ast.Constant) and isinstance(head.value, str) else set()
    if isinstance(node, ast.IfExp):
        return _string_constants(node.body) | _string_constants(node.orelse)
    if isinstance(node, ast.BoolOp):
        return set().union(*(_string_constants(value) for value in node.values))
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(_string_constants(item) for item in node.elts)) if node.elts else set()
    return set()


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Attribute):
        return {target.attr}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(item) for item in target.elts))
    return set()


def assigned_stop_reasons() -> dict[str, set[str]]:
    """Collect string literals assigned as stop reasons, by source file."""

    found: dict[str, set[str]] = {}
    for path in sorted(ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        literals: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(_target_names(t) & _STOP_TARGETS for t in node.targets):
                value = node.value
                if isinstance(value, ast.Tuple) and isinstance(node.targets[0], ast.Tuple):
                    names = [_target_names(t) for t in node.targets[0].elts]
                    value = ast.Tuple(
                        elts=[v for n, v in zip(names, value.elts) if n & _STOP_TARGETS], ctx=ast.Load()
                    )
                literals |= _string_constants(value)
            elif isinstance(node, ast.keyword) and node.arg == "stop_reason":
                literals |= _string_constants(node.value)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "stop_now"
                and node.args
            ):
                literals |= _string_constants(node.args[0])
        reasons = {value.split(":", 1)[0].rstrip(":") for value in literals if value and value.replace("_", "").replace(":", "").isalpha() and value.islower()}
        if reasons:
            found[str(path.relative_to(ROOT))] = reasons
    return found


class StopReasonRegistryTests(unittest.TestCase):
    def test_every_assigned_stop_reason_is_registered(self):
        missing = {
            path: sorted(reasons - set(STOP_REASONS))
            for path, reasons in assigned_stop_reasons().items()
            if reasons - set(STOP_REASONS)
        }
        self.assertEqual(missing, {})

    def test_the_scan_sees_both_families(self):
        seen = set().union(*assigned_stop_reasons().values())
        for expected in ("maxiter", "nonfinite_trial", "root_tolerance", "jacobian_failed", "converged", "completed"):
            self.assertIn(expected, seen)

    def test_entries_are_well_formed_and_exposed(self):
        self.assertIs(opt.core.STOP_REASONS, STOP_REASONS)
        for name, entry in STOP_REASONS.items():
            self.assertEqual(entry.name, name)
            self.assertIn(entry.outcome, OUTCOMES)
            self.assertTrue(set(entry.families) <= set(FAMILIES))
            self.assertTrue(entry.meaning.endswith("."))
        self.assertEqual(StopReason.lookup("stage_gate_failed:warmup").name, "stage_gate_failed")
        self.assertIsNone(StopReason.lookup("a_custom_step_reason"))
        with self.assertRaises(TypeError):
            STOP_REASONS["new"] = STOP_REASONS["maxiter"]  # read-only

    def test_diagnosis_reports_the_outcome_class(self):
        from fixtures.quadratic_system import QuadraticVectorSystem

        system = QuadraticVectorSystem()
        result = opt.adam(system, system.reference_controls(amplitude=0.5), maxiter=3)
        diagnosis = opt.diagnose_run(result)

        self.assertEqual(diagnosis["stop_reason"], "maxiter")
        self.assertEqual(diagnosis["stop_outcome"], "stopped")
        self.assertTrue(diagnosis["stop_continuable"])


if __name__ == "__main__":
    unittest.main()
