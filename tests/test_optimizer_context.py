"""Tests for the read-only optimizer context inspector."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "optimizer_context.py"


class OptimizerContextTests(unittest.TestCase):
    @unittest.skipUnless(SCRIPT.exists(), "requires the source checkout layout")
    def test_human_snapshot_has_no_removed_tracker_dependency(self):
        completed = subprocess.run([sys.executable, "-B", str(SCRIPT)],
            cwd=ROOT.parent, text=True, capture_output=True, check=True)
        self.assertIn("System contract", completed.stdout)
        self.assertNotIn("Tracker", completed.stdout)

    @unittest.skipUnless(SCRIPT.exists(), "requires the source checkout layout")
    def test_json_snapshot_is_runtime_derived_and_uses_contained_package(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--json"],
            cwd=ROOT.parent,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        snapshot = json.loads(completed.stdout)

        self.assertEqual(snapshot["schema_version"], "optimizer.context-command/v1")
        self.assertEqual(snapshot["workspace"]["root"], str(ROOT))
        self.assertEqual(
            snapshot["workspace"]["optimizer_module"],
            str(ROOT / "optimizer" / "__init__.py"),
        )
        self.assertEqual(snapshot["context"]["schema_version"], "optimizer.context/v1")
        self.assertIn("run_chunk_parameters", snapshot["context"]["optimizer"]["core"])
        self.assertNotIn("tracker", snapshot["context"])
        self.assertIn("recommended_commands", snapshot)


if __name__ == "__main__":
    unittest.main()
