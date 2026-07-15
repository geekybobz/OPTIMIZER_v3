import unittest

import numpy as np

from optimizer.controls import ControlSpec, Controls
from optimizer.logs.checkpoint import Checkpoint
from optimizer.logs.trace import Trace
from optimizer.state import RunState


def sample_controls(name="controls", value=1.0):
    spec = ControlSpec(keys=("u",), control_dim=3)
    return Controls.from_dict(spec, {"u": np.full(3, value)}, name=name)


class TraceRecordTests(unittest.TestCase):
    def test_iteration_and_chunk_records_are_exported(self):
        trace = Trace("run-1")

        trace.record_iteration(
            optimizer="adam",
            iteration=1,
            global_iteration=1,
            metrics={"J": 2.0},
            system_params={"lambda2": 1.0},
            technical={"grad_norm": 0.5},
            stage="fid",
            accepted=True,
            reason="accepted",
        )
        trace.record_chunk(
            optimizer="adam",
            chunk=1,
            start_iteration=0,
            end_iteration=5,
            start_metrics={"J": 3.0},
            end_metrics={"J": 2.0},
            accepted=True,
        )

        payload = trace.to_dict()
        self.assertEqual(payload["iterations"][0]["metrics"]["J"], 2.0)
        self.assertEqual(payload["iterations"][0]["technical"]["grad_norm"], 0.5)
        self.assertEqual(payload["chunks"][0]["end_iteration"], 5)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_copies_controls_and_state(self):
        controls = sample_controls(value=1.0)
        state = RunState.initial(controls, metrics={"J": 1.0}, optimizer_name="adam")

        checkpoint = Checkpoint.create(label="stage_start", controls=controls, state=state)
        controls.set_channel("u", [9.0, 9.0, 9.0])
        state.update_current(controls, {"J": 9.0})

        restored_controls, restored_state = checkpoint.restore()

        np.testing.assert_allclose(restored_controls.channel("u"), [1.0, 1.0, 1.0])
        self.assertIsNotNone(restored_state)
        self.assertEqual(restored_state.metrics["J"], 1.0)

    def test_checkpoint_to_dict_contains_metadata(self):
        checkpoint = Checkpoint.create(
            label="accepted",
            controls=sample_controls(value=2.0),
            metrics={"J": 4.0},
            system_params={"lambda4": 10.0},
            optimizer_state={"m": np.array([1.0])},
            iteration=7,
            stage="fourth_order",
        )

        payload = checkpoint.to_dict()
        self.assertEqual(payload["label"], "accepted")
        self.assertEqual(payload["metrics"]["J"], 4.0)
        self.assertEqual(payload["system_params"]["lambda4"], 10.0)
        self.assertEqual(payload["optimizer_state_keys"], ["m"])
        self.assertEqual(payload["iteration"], 7)


class TraceCheckpointTests(unittest.TestCase):
    def test_trace_restores_latest_checkpoint_for_label(self):
        trace = Trace("run-2")
        first = trace.checkpoint("stage_start", sample_controls(value=1.0))
        second = trace.checkpoint("stage_start", sample_controls(value=2.0))

        restored_controls, restored_state = trace.restore("stage_start")

        self.assertNotEqual(first.id, second.id)
        self.assertIsNone(restored_state)
        np.testing.assert_allclose(restored_controls.channel("u"), [2.0, 2.0, 2.0])
        self.assertEqual(trace.event_records[-1].event, "restore")

    def test_trace_restores_by_checkpoint_id(self):
        trace = Trace("run-3")
        first = trace.checkpoint("stage_start", sample_controls(value=1.0))
        trace.checkpoint("stage_start", sample_controls(value=2.0))

        restored_controls, _ = trace.restore(first.id)

        np.testing.assert_allclose(restored_controls.channel("u"), [1.0, 1.0, 1.0])

    def test_checkpoint_updates_state_checkpoint_ids(self):
        trace = Trace("run-4")
        controls = sample_controls(value=3.0)
        state = RunState.initial(controls, metrics={"J": 3.0})

        checkpoint = trace.checkpoint("latest", controls, state)

        self.assertEqual(state.checkpoint_ids["latest"], checkpoint.id)
        self.assertEqual(trace.latest_checkpoint_id("latest"), checkpoint.id)

    def test_missing_checkpoint_label_raises(self):
        trace = Trace("run-5")

        with self.assertRaises(KeyError):
            trace.restore("missing")


if __name__ == "__main__":
    unittest.main()

