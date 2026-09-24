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

    @unittest.skipUnless(matplotlib is not None, "requires the scientific matplotlib extra")
    def test_parameter_colored_layer_is_rasterized_by_default(self):
        from optimizer.visualization.bloch import _draw_layer

        parameter = np.linspace(0.0, 1.0, 5)
        points = np.column_stack((np.sin(parameter), np.zeros(5), np.cos(parameter)))
        layer = opt.PlotLayer("trajectory", {"points": points, "parameter": parameter})

        figure = matplotlib.pyplot.figure()
        axis = figure.add_subplot(projection="3d")
        artist = _draw_layer(axis, layer, 0)

        self.assertTrue(artist.get_rasterized())
        matplotlib.pyplot.close(figure)

    @unittest.skipUnless(matplotlib is not None, "requires the scientific matplotlib extra")
    def test_parameter_colored_layer_can_opt_out_of_rasterization(self):
        from optimizer.visualization.bloch import _draw_layer

        parameter = np.linspace(0.0, 1.0, 5)
        points = np.column_stack((np.sin(parameter), np.zeros(5), np.cos(parameter)))
        layer = opt.PlotLayer(
            "trajectory", {"points": points, "parameter": parameter}, style={"rasterized": False}
        )

        figure = matplotlib.pyplot.figure()
        axis = figure.add_subplot(projection="3d")
        artist = _draw_layer(axis, layer, 0)

        self.assertFalse(artist.get_rasterized())
        matplotlib.pyplot.close(figure)

    @unittest.skipUnless(matplotlib is not None, "requires the scientific matplotlib extra")
    def test_plot_bloch_panel_returns_two_tuple(self):
        system = GenericQubitSystem()
        controls = opt.Controls.zeros(system.control_spec())
        data = opt.plot_data(system, controls, kind="bloch")

        with opt.visualization.style_context():
            figure, sphere = opt.visualization.plot_bloch_panel(data)
        self.assertIsNotNone(sphere)
        matplotlib.pyplot.close(figure)

    @unittest.skipUnless(matplotlib is not None, "requires the scientific matplotlib extra")
    def test_plot_projection_panel_runs_and_returns_single_axis(self):
        system = GenericQubitSystem()
        controls = opt.Controls.zeros(system.control_spec())
        data = opt.plot_data(system, controls, kind="bloch")

        with opt.visualization.style_context():
            figure, axis = opt.visualization.plot_projection_panel(data)
        self.assertIsNotNone(axis)
        matplotlib.pyplot.close(figure)

    @unittest.skipUnless(matplotlib is not None, "requires the scientific matplotlib extra")
    def test_plot_controls_panel_uses_control_palette_and_skips_bloch_validation(self):
        from optimizer.visualization.panels import CONTROL_COLORS

        parameter = np.linspace(0.0, 1.0, 5)
        # deliberately outside the Bloch unit ball: validate_bloch_data would reject this
        amplitudes = np.column_stack((np.full(5, 2.0), np.zeros(5), np.full(5, -3.0)))
        controls_data = opt.PlotData(
            kind="controls",
            layers=(opt.PlotLayer("u", {"points": amplitudes, "parameter": parameter}),),
        )

        with opt.visualization.style_context():
            figure, axis = opt.visualization.plot_controls_panel(controls_data)
        self.assertIsNotNone(axis)
        self.assertEqual(axis.get_lines()[0].get_color(), CONTROL_COLORS[0])
        matplotlib.pyplot.close(figure)

    def test_plot_controls_panel_rejects_bloch_kind_data(self):
        bloch_shaped = opt.PlotData(
            kind="bloch",
            layers=(opt.PlotLayer("r", {"points": np.zeros((3, 3))}),),
        )
        with self.assertRaisesRegex(ValueError, "kind 'controls'"):
            opt.visualization.validate_controls_data(bloch_shaped)

    @unittest.skipUnless(matplotlib is not None, "requires the scientific matplotlib extra")
    def test_save_panel_writes_pdf_and_png(self):
        system = GenericQubitSystem()
        controls = opt.Controls.zeros(system.control_spec())
        data = opt.plot_data(system, controls, kind="bloch")

        with tempfile.TemporaryDirectory() as directory:
            with opt.visualization.style_context():
                figure, _sphere = opt.visualization.plot_bloch_panel(data)
            written = opt.visualization.save_panel(figure, Path(directory) / "panel")
            self.assertEqual({p.suffix for p in written}, {".pdf", ".png"})
            for path in written:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)

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
