"""Durable, passive, replayable records for optimization runs."""

from optimizer.tracker.artifacts import CheckpointData, CheckpointReference, load_checkpoint
from optimizer.tracker.bridge import CheckpointPolicy, JournalObserver
from optimizer.tracker.journal import RunJournal
from optimizer.tracker.model import (
    AnalysisRecord,
    ChunkRecord,
    InterventionRecord,
    IterationRecord,
    LiveSnapshot,
    MetricDescriptor,
    RunReceipt,
    SCHEMA_VERSION,
    ScalarSnapshot,
    SourceReference,
    TerminalRecord,
)
from optimizer.tracker.reader import (
    TimelineReadResult,
    follow_timeline,
    inspect_writer,
    metric_coverage,
    project_timeline,
    read_live,
    read_receipt,
    read_terminal,
    read_timeline,
    read_writer,
    validate_event,
    validate_receipt,
    validate_timeline,
)
from optimizer.catalog import exports

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
            "writer_ownership": "writer.json + .writer.lock",
            "checkpoints": "checkpoints/<checkpoint_id>.npz",
            "standalone_reader": "optimizer_log.reader",
        },
        "capabilities": {
            "passive_recording": True,
            "append_only_timeline": True,
            "atomic_live_snapshot": True,
            "tail_recovery": True,
            "incremental_follow": True,
            "schema_validation": True,
            "generic_projection": True,
            "explicit_checkpoint_artifacts": True,
            "read_only_reader": True,
            "standalone_reader": True,
            "core_bridge": True,
            "campaign_model": "external_optimizer.campaigns",
            "portal_cli": False,
            "run_discovery": False,
            "writer_locking": True,
            "writer_ownership_record": True,
            "retention_api": True,
        },
        "public_api": sorted(__all__),
    }


__all__ = [*exports("tracker"), "info"]
