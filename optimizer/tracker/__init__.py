"""Durable, passive, replayable records for optimization runs."""

from optimizer.tracker.artifacts import CheckpointData, CheckpointReference, load_checkpoint
from optimizer.tracker.journal import RunJournal
from optimizer.tracker.model import (
    AnalysisRecord,
    InterventionRecord,
    IterationRecord,
    LiveSnapshot,
    RunReceipt,
    SCHEMA_VERSION,
    ScalarSnapshot,
    SourceReference,
    TerminalRecord,
)
from optimizer.tracker.reader import (
    TimelineReadResult,
    read_live,
    read_receipt,
    read_terminal,
    read_timeline,
)

def info() -> dict[str, object]:
    """Describe tracker storage without opening a run or system.

    This is a static capability statement. It distinguishes the durable passive
    writer already implemented here from later core integration, campaign
    orchestration, and the optimizer_log user interface.
    """

    return {
        "schema_version": "optimizer.tracker-info/v1",
        "run_schema_version": SCHEMA_VERSION,
        "storage": {
            "receipt": "run_receipt.json",
            "timeline": "timeline.jsonl",
            "live_snapshot": "live.json",
            "terminal_index": "terminal.json",
            "checkpoints": "checkpoints/<checkpoint_id>.npz",
        },
        "capabilities": {
            "passive_recording": True,
            "append_only_timeline": True,
            "atomic_live_snapshot": True,
            "tail_recovery": True,
            "explicit_checkpoint_artifacts": True,
            "read_only_reader": True,
            "core_bridge": False,
            "campaign_model": False,
            "portal_cli": False,
            "run_discovery": False,
            "writer_locking": False,
            "retention_api": False,
        },
        "public_api": sorted(__all__),
    }


__all__ = [
    "AnalysisRecord",
    "CheckpointData",
    "CheckpointReference",
    "InterventionRecord",
    "IterationRecord",
    "LiveSnapshot",
    "RunJournal",
    "RunReceipt",
    "ScalarSnapshot",
    "SourceReference",
    "TerminalRecord",
    "TimelineReadResult",
    "info",
    "load_checkpoint",
    "read_live",
    "read_receipt",
    "read_terminal",
    "read_timeline",
]
