"""The OLGS scaffold carries the forward -> backward -> gradient pattern and its cache."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import optimizer as opt
from optimizer import Controls
from optimizer.system_olgs.scaffold import scaffold_system


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OlgsTemplateTests(unittest.TestCase):
    def test_scaffold_declares_forward_backward_and_cache_hooks(self):
        with tempfile.TemporaryDirectory() as directory:
            system_path, test_path = scaffold_system(Path(directory) / "demo", class_name="Demo")
            source = system_path.read_text(encoding="utf-8")
            for name in ("def forward_prop", "def back_prop", "def _propagate_forward", "def cache_reset"):
                self.assertIn(name, source)
            probe = opt.probe_system(_load(system_path, "demo_system").Demo())
            self.assertTrue(probe.required_ok)
            self.assertTrue(probe.has_forward_prop and probe.has_back_prop)

            completed = subprocess.run(
                [sys.executable, "-B", "-m", "unittest", "test_system"],
                cwd=system_path.parent,
                env={"PYTHONPATH": str(ROOT), "PATH": ""},
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_filled_template_runs_one_forward_pass_per_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            system_path, _ = scaffold_system(Path(directory) / "quad", class_name="Quad")
            module = _load(system_path, "quad_system")

            class Quadratic(module.Quad):
                """J = 1/2 |u - 1|^2, written through the template's hooks."""

                forward_passes = 0

                def system_spec(self):
                    return opt.basic_system_spec(
                        "quad",
                        control_spec=self.control_spec(),
                        metrics={"J": opt.metric_spec("Objective", role="objective", required=True)},
                    )

                def _propagate_forward(self, controls: Controls):
                    type(self).forward_passes += 1
                    return {"residual": controls.as_matrix() - 1.0}

                def back_prop(self, controls: Controls):
                    return {"adjoint": self.forward_prop(controls)["residual"]}

                def evaluate(self, controls: Controls):
                    residual = self.forward_prop(controls)["residual"]
                    return {"J": 0.5 * float(np.sum(residual**2))}

                def gradient(self, controls: Controls):
                    return Controls.from_matrix(controls.spec, self.back_prop(controls)["adjoint"])

            system = Quadratic()
            start = Controls.zeros(system.control_spec())
            result = opt.adam(system, start, step_size=0.05, maxiter=20)

            self.assertEqual(result.iterations, 20)
            self.assertEqual(Quadratic.forward_passes, 21)  # the start plus one per accepted trial


if __name__ == "__main__":
    unittest.main()
