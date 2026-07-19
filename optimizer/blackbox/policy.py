"""Retention and sampling policy for blackbox runs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping


@dataclass(frozen=True)
class BlackBoxPolicy:
    """Controls what the blackbox stores automatically.

    Scalar ledger rows are cheap and default to every iteration.  Full arrays are
    saved only for important states such as initial, best, final, repair, failure, or
    threshold crossings.
    """

    mode: str = "standard"
    scalar_every: int = 1
    analysis_every: int = 10
    checkpoint_every: int = 25
    manifest_every: int = 1
    save_initial: bool = True
    save_best: bool = True
    save_final: bool = True
    save_repairs: bool = True
    save_failures: bool = True
    save_threshold_arrays: bool = True
    save_latest_checkpoints: bool = False
    gradient_rel_delta_threshold: float = 1.5
    metric_regression_rel_threshold: float = 0.05
    reject_rate_window_threshold: float = 0.7
    stall_slope_abs: float = 1.0e-12
    step_collapse_ratio: float = 1.0e-3
    control_max_abs_threshold: float | None = None

    def __post_init__(self) -> None:
        if int(self.scalar_every) < 1:
            raise ValueError("scalar_every must be >= 1.")
        if int(self.analysis_every) < 0:
            raise ValueError("analysis_every must be >= 0.")
        if int(self.checkpoint_every) < 1:
            raise ValueError("checkpoint_every must be >= 1.")
        if int(self.manifest_every) < 1:
            raise ValueError("manifest_every must be >= 1.")
        if self.mode not in {"minimal", "standard", "full"}:
            raise ValueError("mode must be 'minimal', 'standard', or 'full'.")

    @classmethod
    def from_value(cls, value: "BlackBoxPolicy | str | Mapping[str, Any] | None" = None) -> "BlackBoxPolicy":
        """Create a policy from a policy object, mode name, mapping, or ``None``."""

        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if isinstance(value, str):
            return cls.for_mode(value)
        if isinstance(value, Mapping):
            mode = str(value.get("mode", "standard"))
            base = cls.for_mode(mode)
            allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
            updates = {str(key): item for key, item in value.items() if str(key) in allowed}
            return replace(base, **updates)
        raise TypeError("policy must be a BlackBoxPolicy, mode string, mapping, or None.")

    @classmethod
    def for_mode(cls, mode: str) -> "BlackBoxPolicy":
        """Return a named retention policy."""

        mode = str(mode)
        if mode == "minimal":
            return cls(
                mode="minimal",
                scalar_every=5,
                analysis_every=25,
                checkpoint_every=100,
                save_repairs=True,
                save_failures=True,
                save_threshold_arrays=False,
            )
        if mode == "standard":
            return cls(mode="standard")
        if mode == "full":
            return cls(
                mode="full",
                scalar_every=1,
                analysis_every=5,
                checkpoint_every=1,
                save_threshold_arrays=True,
                save_latest_checkpoints=True,
            )
        raise ValueError("mode must be 'minimal', 'standard', or 'full'.")

    def should_record_iteration(self, iteration: int) -> bool:
        return int(iteration) % int(self.scalar_every) == 0

    def should_analyze(self, iteration: int) -> bool:
        return self.analysis_every > 0 and int(iteration) > 0 and int(iteration) % int(self.analysis_every) == 0

    def should_save_checkpoint(self, label: str, iteration: int | None, *, force: bool = False) -> bool:
        """Return whether a checkpoint label should save a full controls artifact."""

        if force:
            return True
        label = str(label)
        if label == "initial":
            return bool(self.save_initial)
        if label == "chunk_start":
            return False
        if label.startswith("best"):
            return bool(self.save_best)
        if label == "final":
            return bool(self.save_final)
        if label == "latest" and not self.save_latest_checkpoints:
            return False
        if iteration is None:
            return False
        return int(iteration) % int(self.checkpoint_every) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "scalar_every": int(self.scalar_every),
            "analysis_every": int(self.analysis_every),
            "checkpoint_every": int(self.checkpoint_every),
            "manifest_every": int(self.manifest_every),
            "save_initial": bool(self.save_initial),
            "save_best": bool(self.save_best),
            "save_final": bool(self.save_final),
            "save_repairs": bool(self.save_repairs),
            "save_failures": bool(self.save_failures),
            "save_threshold_arrays": bool(self.save_threshold_arrays),
            "save_latest_checkpoints": bool(self.save_latest_checkpoints),
            "gradient_rel_delta_threshold": float(self.gradient_rel_delta_threshold),
            "metric_regression_rel_threshold": float(self.metric_regression_rel_threshold),
            "reject_rate_window_threshold": float(self.reject_rate_window_threshold),
            "stall_slope_abs": float(self.stall_slope_abs),
            "step_collapse_ratio": float(self.step_collapse_ratio),
            "control_max_abs_threshold": self.control_max_abs_threshold,
        }
