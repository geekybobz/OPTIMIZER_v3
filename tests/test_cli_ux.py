"""Terminal UX: colon commands, rich panels and the display-only progress line."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import optimizer as opt
from fixtures.quadratic_system import QuadraticVectorSystem
from fixtures.shooting_systems import CubicRootShootingSystem
from optimizer.cli.display import _Display, _render, _rich_renderable
from optimizer.cli.language import _Terminal
from optimizer.cli.progress import _LiveLine, _live_line_factory
from optimizer.core.progress import _REPORTER, _report, _reporting

_ROOT = Path(__file__).resolve().parents[1]
_LOAD = 'opt.session.load(name="main", factory="tests/fixtures/quadratic_system.py", parameters={"N":9});'


class _TtyBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def _text(renderable) -> str:
    from rich.console import Console

    console = Console(record=True, width=120, color_system=None, force_terminal=True, file=io.StringIO())
    console.print(renderable)
    return console.export_text()


class ColonCommandTests(unittest.TestCase):
    def setUp(self):
        self.responses = []
        self.service = opt.OperationService()
        self.service.session.load("main", QuadraticVectorSystem, parameters={"N": 9})
        self.terminal = _Terminal(self.service, emit=self.responses.append)

    def test_help_overview_and_one_command(self):
        self.assertEqual(self.terminal.frontend(":help"), 0)
        self.assertEqual(self.responses[-1].operation, "help")
        self.assertIn(":exit", self.responses[-1].metadata["summary"]["frontend"])
        self.assertEqual(self.terminal.frontend("  :help adam  "), 0)
        self.assertEqual(self.responses[-1].metadata["summary"]["id"], "optimizers.adam")

    def test_exit_closes_the_terminal(self):
        self.assertEqual(self.terminal.frontend(":exit"), 0)
        self.assertTrue(self.terminal.closed)
        self.assertEqual(self.responses[-1].metadata["summary"]["session"], "closed")

    def test_unknown_colon_command_lists_the_valid_ones(self):
        before = self.service.session.snapshot()
        self.assertEqual(self.terminal.frontend(":quit"), 2)
        error = self.responses[-1].error
        self.assertEqual(error["code"], "unknown_command")
        self.assertIn(":help <name>", error["message"])
        self.assertEqual(self.terminal.frontend(":help adam extra"), 2)
        self.assertEqual(self.service.session.snapshot(), before)

    def test_bare_words_get_a_hint_and_are_not_submitted(self):
        before = self.service.session.snapshot()
        for text, hint in (("exit", ":exit"), ("exit;", ":exit"), ("quit()", ":exit"),
                           ("help", ":help"), (" help ; ", ":help")):
            with self.subTest(text=text):
                self.assertEqual(self.terminal.frontend(text), 2)
                self.assertEqual(self.responses[-1].error["code"], "unknown_command")
                self.assertIn(hint, self.responses[-1].error["message"])
        self.assertEqual(self.service.session.snapshot(), before)
        self.assertFalse(self.terminal.closed)

    def test_ordinary_statements_are_not_frontend_commands(self):
        for text in ("x = opt.help();", "opt.session.state();", "help_me;", "exits;"):
            with self.subTest(text=text):
                self.assertIsNone(self.terminal.frontend(text))


class ColonCommandProcessTests(unittest.TestCase):
    def command(self, *args, input=None):
        return subprocess.run([sys.executable, "-B", "-m", "optimizer.cli", *args], cwd=_ROOT,
                              input=input, capture_output=True, text=True, timeout=30)

    def test_one_shot_colon_commands_and_hint(self):
        self.assertEqual(self.command("--command", ":help").returncode, 0)
        self.assertEqual(self.command("--command", ":exit").returncode, 0)
        result = self.command("--json", "--command", ":help adam")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])["operation"], "help")
        hint = self.command("--command", "exit;")
        self.assertEqual(hint.returncode, 2)
        self.assertIn(":exit", hint.stdout)

    def test_piped_session_accepts_colon_commands(self):
        result = self.command(input=_LOAD + "\n:help adam\n:exit\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("session", result.stdout.splitlines()[-1])


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.responses = []
        self.service = opt.OperationService()
        self.service.session.load("main", QuadraticVectorSystem, parameters={"N": 9})
        self.service.session.load("shoot", CubicRootShootingSystem, select=False)
        self.terminal = _Terminal(self.service, emit=self.responses.append)

    def last(self, source, status=0):
        self.assertEqual(self.terminal.submit(source), status)
        return self.responses[-1]

    def test_gradient_result_panel(self):
        self.last("u0 = opt.guesses.constant_guess(value=0.1); opt.session.use(u0);")
        text = _text(_rich_renderable(self.last("r = opt.adam(maxiter=3);"), elapsed=0.5))
        for marker in ("r = adam", "iterations", "stop", "J", "main updated", "0.50 s"):
            self.assertIn(marker, text)
        evaluation = _text(_rich_renderable(self.last("opt.evaluate();")))
        for marker in ("evaluate", "controls", "Controls ux, uy, uz", "J", "fidelity"):
            self.assertIn(marker, evaluation)

    def test_shooting_and_multistart_panels(self):
        self.last('opt.session.select("shoot"); seeds = opt.shooting.random_shooting_seeds(count=3, seed=1);'
                  " opt.session.use(seeds[0]);")
        root = _text(_rich_renderable(self.last("root = opt.shooting.solve_shooting();")))
        for marker in ("root = solve_shooting", "qualified", "verified", "solver converged", "residual_norm"):
            self.assertIn(marker, root)
        many = _text(_rich_renderable(self.last("ms = opt.shooting.solve_shooting_multistart(starts=seeds);")))
        for marker in ("ms = solve_shooting_multistart", "ranked by", "rank", "residual", "qualified"):
            self.assertIn(marker, many)

    def test_error_panel_has_code_and_hint(self):
        text = _text(_rich_renderable(self.last("opt.shooting.solve_shooting();", status=2)))
        self.assertIn("✗ wrong_family", text)
        self.assertIn("opt.session.select", text)

    def test_compact_lines_and_help_panel(self):
        guess = _text(_rich_renderable(self.last("u0 = opt.guesses.constant_guess(value=0.1);")))
        self.assertIn("✓ u0 = constant_guess", guess)
        self.assertIn("Controls ux, uy, uz", guess)
        self.assertEqual(self.terminal.frontend(":help"), 0)
        help_text = _text(_rich_renderable(self.responses[-1]))
        for marker in (":help <name>", ":exit", "Session", "load", "History", "Tools"):
            self.assertIn(marker, help_text)
        self.assertEqual(self.terminal.frontend(":help adam"), 0)
        one = _text(_rich_renderable(self.responses[-1]))
        for marker in ("usage", "opt.adam(controls=<active>", "maxiter=100", "variants"):
            self.assertIn(marker, one)

    def test_display_modes(self):
        response = self.last("x = opt.session.state();")
        plain = io.StringIO()
        _Display(stream=plain, rich=False).emit(response)
        self.assertEqual(plain.getvalue().strip(), _render(response))
        machine = io.StringIO()
        _Display(stream=machine, json_output=True, rich=True).emit(response)
        self.assertEqual(machine.getvalue().strip(), _render(response, json_output=True))
        with mock.patch.dict(sys.modules, {"rich.console": None}):
            fallback = _Display(stream=io.StringIO(), rich=True)
        self.assertIsNone(fallback.console)
        rich_stream = _TtyBuffer()
        display = _Display(stream=rich_stream)
        self.assertIsNotNone(display.console)
        display.emit(response)
        self.assertNotEqual(rich_stream.getvalue().strip(), _render(response))


class ReporterTests(unittest.TestCase):
    def test_report_is_a_no_op_without_a_reporter(self):
        self.assertIsNone(_REPORTER.get())
        _report(kind="iteration", value=1.0)
        events = []
        with _reporting(events.append):
            _report(kind="iteration", value=2.0)
            with _reporting(None):
                _report(kind="iteration", value=3.0)
        self.assertEqual(events, [{"kind": "iteration", "value": 2.0}])
        self.assertIsNone(_REPORTER.get())

    def test_engine_reports_each_iteration_without_changing_results(self):
        system = QuadraticVectorSystem(N=9)
        start = opt.constant_guess(system, value=0.1)
        plain = opt.optimizers.adam(system, start, maxiter=6)
        events = []
        with _reporting(events.append):
            seen = opt.optimizers.adam(system, start, maxiter=6)
        self.assertEqual(len(events), seen.iterations)
        self.assertEqual([event["iteration"] for event in events], list(range(1, seen.iterations + 1)))
        self.assertEqual({(event["kind"], event["method"], event["total"], event["metric"]) for event in events},
                         {("iteration", "adam", 6, "J")})
        self.assertEqual(seen.metrics, plain.metrics)

        def broken(event):
            raise RuntimeError("display failed")

        with _reporting(broken):
            unaffected = opt.optimizers.adam(system, start, maxiter=6)
        self.assertEqual(unaffected.metrics, plain.metrics)

    def test_shooting_and_multistart_report_progress(self):
        system = CubicRootShootingSystem()
        seeds = opt.shooting.random_shooting_seeds(system.shooting_spec(), 3, seed=1)
        events = []
        with _reporting(events.append):
            result = opt.shooting.solve_shooting_multistart(system, seeds)
        items = [event for event in events if event["kind"] == "item"]
        self.assertEqual([event["index"] for event in items], [1, 2, 3])
        self.assertEqual(items[-1]["qualified"], len(result.qualified))
        iterations = [event for event in events if event["kind"] == "iteration"]
        self.assertTrue(iterations)
        self.assertEqual({event["method"] for event in iterations}, {"solve_shooting"})


class LiveLineTests(unittest.TestCase):
    def test_text_shows_bar_counts_and_metric(self):
        line = _LiveLine("optimizers.adam", _TtyBuffer())
        line.update({"kind": "iteration", "iteration": 5, "total": 10, "metric": "J", "value": 0.123456})
        text = line.text(2.0)
        for marker in ("adam", "5/10", "J=0.1235", "2.0s", "~2.0s left", "█"):
            self.assertIn(marker, text)
        line.update({"kind": "item", "index": 2, "total": 4, "unit": "seed", "qualified": 1})
        nested = line.text(1.0)
        for marker in ("2/4 seeds", "qualified 1", "iter 5"):
            self.assertIn(marker, nested)

    def test_unknown_total_uses_a_spinner(self):
        line = _LiveLine("work", _TtyBuffer())
        self.assertRegex(line.text(0.7), r"work  \S  0\.7s")

    def test_draws_on_a_terminal_and_clears_itself(self):
        stream = _TtyBuffer()
        with _LiveLine("optimizers.adam", stream, delay=0.0, interval=0.01) as line:
            line.update({"kind": "iteration", "iteration": 1, "total": 2, "metric": "J", "value": 1.0})
            time.sleep(0.1)
        output = stream.getvalue()
        self.assertIn("adam", output)
        self.assertTrue(output.endswith("\r\x1b[2K"))

    def test_quiet_for_short_calls_and_non_terminals(self):
        stream = _TtyBuffer()
        with _LiveLine("fast", stream, delay=5.0, interval=0.01):
            time.sleep(0.05)
        self.assertEqual(stream.getvalue(), "")
        self.assertIsNone(_live_line_factory(io.StringIO()))
        self.assertIsNotNone(_live_line_factory(_TtyBuffer()))

    def test_terminal_installs_the_reporter_per_call(self):
        labels, events = [], []

        class Recorder:
            def __init__(self, label):
                labels.append(label)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def update(self, event):
                events.append(event)

        service = opt.OperationService()
        service.session.load("main", QuadraticVectorSystem, parameters={"N": 9})
        terminal = _Terminal(service, emit=lambda response: None, progress=Recorder)
        self.assertEqual(terminal.submit("u = opt.guesses.constant_guess(value=0.1); opt.session.use(u);"
                                         " r = opt.adam(maxiter=4);"), 0)
        self.assertIn("optimizers.adam", labels)
        self.assertEqual(len([event for event in events if event["method"] == "adam"]), 4)
        self.assertTrue(all(value >= 0.0 for value in terminal.timings.values()))
        self.assertIsNone(_REPORTER.get())


if __name__ == "__main__":
    unittest.main()
