"""Freshness check for the generated public API reference."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ApiReferenceTests(unittest.TestCase):
    def test_generated_api_reference_is_current(self):
        completed = subprocess.run(
            [sys.executable, "-B", "tools/generate_api_reference.py", "--check"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
