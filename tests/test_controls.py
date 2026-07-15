import unittest

import numpy as np

from optimizer.controls import ControlSpec, Controls


class ControlSpecTests(unittest.TestCase):
    def test_spec_properties(self):
        spec = ControlSpec(keys=("ux", "uy", "uz"), control_dim=5, dt=0.1)

        self.assertEqual(spec.keys, ("ux", "uy", "uz"))
        self.assertEqual(spec.n_controls, 3)
        self.assertEqual(spec.shape, (3, 5))
        self.assertEqual(spec.size, 15)
        self.assertEqual(spec.channel_index("uy"), 1)
        self.assertEqual(spec.to_dict()["shape"], [3, 5])

    def test_spec_rejects_bad_layouts(self):
        with self.assertRaises(ValueError):
            ControlSpec(keys=(), control_dim=5)
        with self.assertRaises(ValueError):
            ControlSpec(keys=("u", "u"), control_dim=5)
        with self.assertRaises(ValueError):
            ControlSpec(keys=("u",), control_dim=0)
        with self.assertRaises(ValueError):
            ControlSpec(keys=("u",), control_dim=5, dt=0.0)
        with self.assertRaises(KeyError):
            ControlSpec(keys=("u",), control_dim=5).channel_index("missing")


class ControlsTests(unittest.TestCase):
    def test_from_dict_preserves_spec_order(self):
        spec = ControlSpec(keys=("uA", "uB"), control_dim=3)
        controls = Controls.from_dict(
            spec,
            {
                "uB": np.array([4.0, 5.0, 6.0]),
                "uA": np.array([1.0, 2.0, 3.0]),
            },
            name="ordered",
        )

        np.testing.assert_allclose(
            controls.as_matrix(),
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        )
        self.assertEqual(controls.name, "ordered")

    def test_zeros_constant_and_channel_access(self):
        spec = ControlSpec(keys=("ux", "uy", "uz"), control_dim=4)

        zeros = Controls.zeros(spec)
        np.testing.assert_allclose(zeros.as_matrix(), np.zeros((3, 4)))

        const = Controls.constant(spec, {"ux": 1.0, "uy": 2.0, "uz": 3.0})
        np.testing.assert_allclose(const.channel("uy"), np.full(4, 2.0))

        const.set_channel("uy", np.arange(4))
        np.testing.assert_allclose(const.channel("uy"), np.array([0.0, 1.0, 2.0, 3.0]))

    def test_flatten_roundtrip(self):
        spec = ControlSpec(keys=("u1", "u2", "u3", "u4", "u5", "u6"), control_dim=2)
        flat = np.arange(12, dtype=float)

        controls = Controls.from_flat(spec, flat)

        self.assertEqual(controls.shape, (6, 2))
        np.testing.assert_allclose(controls.flatten(), flat)

    def test_copy_is_independent(self):
        spec = ControlSpec(keys=("u",), control_dim=3)
        controls = Controls.from_dict(spec, {"u": [1.0, 2.0, 3.0]})

        cloned = controls.copy(name="clone")
        cloned.set_channel("u", [9.0, 9.0, 9.0])

        np.testing.assert_allclose(controls.channel("u"), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(cloned.channel("u"), [9.0, 9.0, 9.0])
        self.assertEqual(cloned.name, "clone")

    def test_vectorized_arithmetic(self):
        spec = ControlSpec(keys=("uA", "uB"), control_dim=2)
        a = Controls.from_matrix(spec, [[1.0, 2.0], [3.0, 4.0]])
        b = Controls.from_matrix(spec, [[10.0, 20.0], [30.0, 40.0]])

        np.testing.assert_allclose((a + b).as_matrix(), [[11.0, 22.0], [33.0, 44.0]])
        np.testing.assert_allclose((b - a).as_matrix(), [[9.0, 18.0], [27.0, 36.0]])
        np.testing.assert_allclose((2.0 * a).as_matrix(), [[2.0, 4.0], [6.0, 8.0]])
        np.testing.assert_allclose((-a).as_matrix(), [[-1.0, -2.0], [-3.0, -4.0]])

    def test_norms(self):
        spec = ControlSpec(keys=("uA", "uB"), control_dim=2)
        controls = Controls.from_matrix(spec, [[3.0, 4.0], [0.0, -2.0]])

        self.assertAlmostEqual(controls.norm(), np.sqrt(29.0))
        self.assertAlmostEqual(controls.channel_norms()["uA"], 5.0)
        self.assertAlmostEqual(controls.max_abs(), 4.0)

    def test_rejects_invalid_controls(self):
        spec = ControlSpec(keys=("uA", "uB"), control_dim=2)

        with self.assertRaises(ValueError):
            Controls.from_matrix(spec, [[1.0, 2.0, 3.0]])
        with self.assertRaises(ValueError):
            Controls.from_matrix(spec, [[1.0, np.nan], [2.0, 3.0]])
        with self.assertRaises(KeyError):
            Controls.from_dict(spec, {"uA": [1.0, 2.0]})
        with self.assertRaises(ValueError):
            Controls.from_flat(spec, [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            Controls.zeros(spec) + np.zeros((1, 2))


if __name__ == "__main__":
    unittest.main()

