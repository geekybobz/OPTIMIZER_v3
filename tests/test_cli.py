"""Exercise real checkout-module processes, JSON framing and terminal lifetime."""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest
from pathlib import Path

from optimizer.cli.display import _render
from optimizer.operations import OperationResponse

_ROOT = Path(__file__).resolve().parents[1]
_LOAD = 'opt.session.load(name="main", factory="tests/fixtures/quadratic_system.py", parameters={"N":9});'


class CliProcessTests(unittest.TestCase):
    def command(self, *args, input=None):
        result = subprocess.run([sys.executable, "-B", "-m", "optimizer.cli", *args],
                                cwd=_ROOT, input=input, capture_output=True, text=True, timeout=20)
        return result

    def records(self, result):
        return [json.loads(line) for line in result.stdout.splitlines()]

    def test_console_entry_declaration(self):
        data = tomllib.loads((_ROOT / "pyproject.toml").read_text())
        self.assertEqual(data["project"]["scripts"]["optimizer"], "optimizer.cli.main:main")

    def test_one_shot_json_native_result_and_compact_arrays(self):
        result = self.command("--json", "--command", _LOAD +
                              'u=opt.guesses.constant_guess(value=0.1); opt.session.use(u); '
                              'r=opt.adam(maxiter=2); r.controls; r.metrics;')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = self.records(result)
        self.assertEqual(records[-2]["metadata"]["summary"]["shape"], [3, 9])
        self.assertNotIn("result", records[-2])
        self.assertGreater(records[-1]["metadata"]["summary"]["J"], 0)
        self.assertEqual(records[3]["metadata"]["summary"]["stop_reason"], "maxiter")

    def test_script_mode_and_no_history_files(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "workflow.opt"
            script.write_text('x={"label":"a;b"};\nx["label"];\nopt.session.exit();\n')
            result = self.command("--script", str(script), "--json")
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(self.records(result)[1]["metadata"]["summary"], "a;b")
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["workflow.opt"])

    def test_pipe_multiline_eof_and_explicit_exit(self):
        for text in ["x=(\n 1, 2\n);\nx[1];\n", "x=2;\nopt.session.exit();\nx=3;\n"]:
            result = self.command("--json", input=text)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(len(self.records(result)), 2)

    def test_eof_incomplete_input_is_error(self):
        result = self.command("--json", input="x=1\n")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.records(result)[0]["error"]["code"], "missing_semicolon")

    def test_syntax_error_prevents_loading_first_command(self):
        result = self.command("--json", "--command", _LOAD + "import os;")
        self.assertEqual(result.returncode, 2)
        records = self.records(result)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["state_after"]["contexts"], {})

    def test_runtime_error_retains_prior_committed_head(self):
        result = self.command("--json", "--command", _LOAD +
                              'u=opt.guesses.constant_guess(); opt.session.use(u); opt.adam(maxiterr=2); x=1;')
        self.assertEqual(result.returncode, 2)
        records = self.records(result)
        self.assertEqual(records[-1]["status"], "error")
        self.assertEqual(records[-1]["state_before"], records[-1]["state_after"])
        self.assertIsNotNone(records[-1]["state_after"]["contexts"]["main"]["head"])
        self.assertNotIn("x", records[-1]["state_after"]["bindings"])

    def test_bad_script_missing_large_and_invalid_utf8(self):
        result = self.command("--json", "--script", "/does/not/exist.opt")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.records(result)[0]["error"]["code"], "script_input")
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "bad.opt"
            for content in [b"x" * 65537, b"\xff"]:
                script.write_bytes(content)
                result = self.command("--json", "--script", str(script))
                self.assertEqual(result.returncode, 2)
                self.assertEqual(len(self.records(result)), 1)

    def test_cli_help_and_mutually_exclusive_flags(self):
        self.assertEqual(self.command("--help").returncode, 0)
        self.assertEqual(self.command("--command", "x=1;", "--script", "x.opt").returncode, 2)

    def test_renderer_distinguishes_shooting_flags(self):
        response = OperationResponse("test", "shooting.solve_shooting", "ok",
            metadata={"summary": {"type": "ShootingResult", "qualified": False,
                                  "verified": False, "solver_converged": True}})
        output = _render(response)
        self.assertIn('"qualified": false', output)
        self.assertIn('"solver_converged": true', output)

    @unittest.skipUnless(os.name == "posix", "Real TTY interruption needs POSIX pty.")
    def test_tty_interrupt_clears_incomplete_input_and_can_continue(self):
        import pty
        master, slave = pty.openpty()
        process = subprocess.Popen([sys.executable, "-B", "-m", "optimizer.cli"], cwd=_ROOT,
                                   stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
        os.close(slave)
        output = b""
        def wait_for(marker):
            nonlocal output
            end = time.monotonic() + 10
            while marker not in output and time.monotonic() < end:
                if select.select([master], [], [], 0.1)[0]:
                    output += os.read(master, 65536)
            self.assertIn(marker, output)
        try:
            wait_for(b"opt> ")
            os.write(master, b"x = (\n")
            wait_for(b"...> ")
            process.send_signal(signal.SIGINT)
            wait_for(b"Input cleared.")
            os.write(master, b"x=2;\nx;\nopt.session.exit();\n")
            # A terminal shows the rich compact line instead of raw JSON.
            wait_for(b"session closed")
            self.assertEqual(process.wait(timeout=10), 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            os.close(master)
