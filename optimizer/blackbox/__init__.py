"""Numeric blackbox logging for optimizer runs.

The blackbox module stores a compact technical ledger:

``blackbox.json``
    Run identity, policy, status, summary, and artifact index.

``ledger.jsonl``
    Append-only numeric records for iterations, chunks, thresholds, repairs,
    checkpoints, and window analysis.

``arrays/*.npz``
    Selected full arrays only when the retention policy says they are useful.
"""

from __future__ import annotations

from optimizer.blackbox.analysis import analyze_path as analyze
from optimizer.blackbox.analysis import diagnostics
from optimizer.blackbox.artifacts import load_artifact, save_array, save_controls
from optimizer.blackbox.policy import BlackBoxPolicy
from optimizer.blackbox.reader import latest_record, read_manifest, read_records, series
from optimizer.blackbox.reset import prune, reset
from optimizer.blackbox.run import BlackBoxRun, ensure_run, start


__all__ = [
    "BlackBoxPolicy",
    "BlackBoxRun",
    "analyze",
    "diagnostics",
    "ensure_run",
    "latest_record",
    "load_artifact",
    "prune",
    "read_manifest",
    "read_records",
    "reset",
    "save_array",
    "save_controls",
    "series",
    "start",
]
