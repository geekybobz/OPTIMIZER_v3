"""Contract, generic plot-data, Bloch renderer, and scaffold integration tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import matplotlib
except ModuleNotFoundError:
    matplotlib = None
else:
    matplotlib.use("Agg")

import optimizer as opt
from optimizer.system_olgs.scaffold import scaffold_system


class GenericQubitSystem:
    def __init__(self) -> None:
        self.info_calls = 0
        self.evaluate_calls = 0
        self.spec = opt.ControlSpec(("u",), 3, dt=0.5)

    def system_spec(self):
        self.info_calls += 1
        return opt.basic_system_spec(
            "generic-qubit-test",
            system_type="qubit",
            control_spec=self.spec,
            metrics={"quality": {"label": "Quality", "role": "quality"}},
            visualizations={
                "bloch": {"provider": "plot_data", "status": "implemented", "automatic": False}
            },
        )

    def control_spec(self):
        return self.spec

    def evaluate(self, controls):
        self.evaluate_calls += 1
        return {"J": controls.norm() ** 2, "quality": 1.0 / (1.0 + controls.norm() ** 2)}

    def gradient(self, controls):
        return 2.0 * controls

    def with_secondary(self, **updates):
        del updates
        return type(self)()

    def plot_data(self, controls, *, kind, **options):
        del controls, options
        if kind != "bloch":
            raise KeyError(kind)
        parameter = np.linspace(0.0, 1.0, 5)
        first = np.column_stack((np.sin(parameter), np.zeros(5), np.cos(parameter)))
        second = np.column_stack((np.zeros(5), np.sin(parameter), np.cos(parameter)))
        return opt.PlotData(
            kind="bloch",
            layers=(
                opt.PlotLayer("layer_a", {"points": first, "parameter": parameter}),
                opt.PlotLayer("layer_b", {"points": second, "parameter": parameter}, style={"color_by": "constant"}),
            ),
            annotations={"point_a": {"point": first[0], "label": "A"}},
        )


class UndeclaredMetricSystem(GenericQubitSystem):
    def evaluate(self, controls):
        payload = super().evaluate(controls)
        payload["undeclared"] = 2.0
        return payload


class InvalidSystemSpec(GenericQubitSystem):
    def system_spec(self):
        payload = super().system_spec()
        payload["derivatives"]["gradient"]["source"] = "finite_difference"
        return payload


class SystemSpecVisualizationTests(unittest.TestCase):
    def test_system_info_is_static_and_complete(self):
        system = GenericQubitSystem()
        info = opt.system_info(system)

        self.assertEqual(info["system_spec"]["value"]["identity"]["id"], "generic-qubit-test")
        self.assertTrue(info["probe"]["required_ok"])
        self.assertTrue(info["capabilities"]["plot_data"]["available"])
        self.assertEqual(system.evaluate_calls, 0)

    def test_evaluator_enforces_declared_metrics_and_counts_explicit_plot_data(self):
        system = GenericQubitSystem()
        controls = opt.Controls.zeros(system.control_spec())
        evaluator = opt.SystemEvaluator(system)

        evaluation = evaluator.evaluate(controls)
        data = evaluator.plot_data(controls, kind="bloch")

        self.assertEqual(evaluation.J, 0.0)
        self.assertEqual([layer.name for layer in data.layers], ["layer_a", "layer_b"])
        self.assertEqual(evaluator.counters()["plot_data"], 1)
        with self.assertRaisesRegex(KeyError, "undeclared"):
            opt.evaluate(UndeclaredMetricSystem(), controls)

    def test_analytical_gradient_provenance_is_mandatory(self):
        system = GenericQubitSystem()
        payload = system.system_spec()
        payload["derivatives"]["gradient"]["source"] = "finite_difference"
        with self.assertRaisesRegex(ValueError, "analytical"):
            opt.validate_system_spec(payload)

        info = opt.system_info(InvalidSystemSpec())
        self.assertFalse(info["system_spec"]["valid"])
        self.assertFalse(info["capabilities"]["base_olgs"]["complete"])
        with self.assertRaisesRegex(ValueError, "analytical"):
            opt.require_system(InvalidSystemSpec())

    @unittest.skipUnless(matplotlib is not None, "requires the scientific matplotlib extra")
    def test_bloch_renderer_accepts_generic_named_layers(self):
        system = GenericQubitSystem()
        controls = opt.Controls.zeros(system.control_spec())
        data = opt.plot_data(system, controls, kind="bloch")

        with opt.visualization.style_context():
            figure, sphere, projections = opt.visualization.plot_bloch(data, projections=True)
        self.assertIsNotNone(sphere)
        self.assertIsNotNone(projections)
        matplotlib.pyplot.close(figure)

    def test_scaffold_is_capability_selected_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            system_path = Path(directory) / "example" / "system.py"
            created = scaffold_system(
                system_path,
                class_name="ExampleSystem",
                profile="qubit",
                capabilities=("constraints", "second-order"),
            )
            self.assertEqual(len(created), 2)
            source = system_path.read_text(encoding="utf-8")
            self.assertIn("def system_spec", source)
            self.assertIn("# def plot_data", source)
            self.assertIn("# def jacobian", source)

            spec = importlib.util.spec_from_file_location("generated_system", system_path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            generated = module.ExampleSystem()
            self.assertTrue(opt.probe_system(generated).required_ok)
            self.assertFalse(opt.probe_system(generated).has_plot_data)

            with self.assertRaises(FileExistsError):
                scaffold_system(system_path, class_name="ExampleSystem")


if __name__ == "__main__":
    unittest.main()
