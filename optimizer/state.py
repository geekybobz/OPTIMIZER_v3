"""Run and warmstart state containers.

Why this file exists
--------------------
Optimization is not just a final set of controls.  During a run the library needs to
carry the current controls, current metrics, step-size estimates, best-so-far values,
and optimizer-specific memory such as Adam moments or momentum buffers.  When one
optimizer hands off to another, only a safe subset of that state should transfer.

This module defines those containers before the engine exists:

    RunState        mutable state owned by an active optimization run
    WarmStartState  safe handoff state created from a run or result

How it fits the architecture
----------------------------
- the future engine will update ``RunState`` every iteration and chunk.
- optimizers will store method-specific data in ``optimizer_state``.
- ``OptimizerResult`` will expose the final ``RunState`` for inspection.
- warmstart logic will use ``WarmStartState`` to move between optimizers safely.
- logs/checkpoints will later serialize these objects.

What this file deliberately does not do
---------------------------------------
It does not run optimization, evaluate systems, choose steps, write files, or decide
physical acceptance rules.  It is only the in-memory shape of a run.

Reviewer invariants
-------------------
- ``RunState.controls`` is always the current control object.
- ``RunState.metrics`` describes those current controls.
- best-so-far state is updated only when explicitly requested.
- warmstart always transfers controls, metrics, step estimate, and context.
- optimizer-specific state transfers only when the source and target optimizers are
  compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from optimizer.controls import Controls


def _copy_metrics(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a shallow dict copy for metric payloads."""

    return dict(metrics or {})


def _copy_mapping(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a shallow dict copy for context/state payloads."""

    return dict(payload or {})


@dataclass
class RunState:
    """Mutable state for one active optimization run."""

    controls: Controls
    metrics: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    global_iteration: int = 0
    step_size: float | None = None
    optimizer_name: str | None = None
    optimizer_state: dict[str, Any] = field(default_factory=dict)
    best_controls: Controls | None = None
    best_metrics: dict[str, Any] | None = None
    stop_reason: str | None = None
    trace_id: str | None = None
    checkpoint_ids: dict[str, str] = field(default_factory=dict)
    system_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def initial(
        cls,
        controls: Controls,
        *,
        metrics: Mapping[str, Any] | None = None,
        optimizer_name: str | None = None,
        step_size: float | None = None,
        system_params: Mapping[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> "RunState":
        """Create a clean run state at iteration zero."""

        metric_dict = _copy_metrics(metrics)
        return cls(
            controls=controls,
            metrics=metric_dict,
            step_size=step_size,
            optimizer_name=optimizer_name,
            best_controls=controls.copy(name="best") if metric_dict else None,
            best_metrics=dict(metric_dict) if metric_dict else None,
            trace_id=trace_id,
            system_params=_copy_mapping(system_params),
        )

    def update_current(
        self,
        controls: Controls,
        metrics: Mapping[str, Any],
        *,
        step_size: float | None = None,
        iteration_increment: int = 1,
    ) -> None:
        """Update the current run position after an accepted iteration."""

        self.controls = controls
        self.metrics = _copy_metrics(metrics)
        self.iteration += int(iteration_increment)
        self.global_iteration += int(iteration_increment)
        if step_size is not None:
            self.step_size = float(step_size)

    def mark_best(
        self,
        *,
        controls: Controls | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        """Explicitly mark the current or provided state as best-so-far."""

        source_controls = self.controls if controls is None else controls
        source_metrics = self.metrics if metrics is None else metrics
        self.best_controls = source_controls.copy(name="best")
        self.best_metrics = _copy_metrics(source_metrics)

    def update_best_by_metric(
        self,
        *,
        metric: str = "J",
        mode: str = "min",
    ) -> bool:
        """Update best-so-far when the current metric improves.

        Returns ``True`` when a new best was recorded.
        """

        if metric not in self.metrics:
            raise KeyError(f"Current metrics do not include {metric!r}.")
        current_value = float(self.metrics[metric])
        if self.best_metrics is None or metric not in self.best_metrics:
            self.mark_best()
            return True
        best_value = float(self.best_metrics[metric])
        if mode == "min":
            improved = current_value < best_value
        elif mode == "max":
            improved = current_value > best_value
        else:
            raise ValueError("mode must be 'min' or 'max'.")
        if improved:
            self.mark_best()
        return improved

    def to_warmstart(self, *, target_optimizer: str | None = None) -> "WarmStartState":
        """Create warmstart state from this run state."""

        return WarmStartState.from_run_state(self, target_optimizer=target_optimizer)


@dataclass(frozen=True)
class WarmStartState:
    """Safe handoff state for starting or continuing an optimizer."""

    controls: Controls
    metrics: dict[str, Any] = field(default_factory=dict)
    step_size: float | None = None
    optimizer_state: dict[str, Any] | None = None
    source_optimizer: str | None = None
    target_optimizer: str | None = None
    trace_id: str | None = None
    checkpoint_ids: dict[str, str] = field(default_factory=dict)
    system_params: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def optimizer_state_compatible(
        source_optimizer: str | None,
        target_optimizer: str | None,
    ) -> bool:
        """Return whether optimizer-private state can be reused."""

        if source_optimizer is None or target_optimizer is None:
            return False
        return source_optimizer == target_optimizer

    @classmethod
    def from_run_state(
        cls,
        state: RunState,
        *,
        target_optimizer: str | None = None,
    ) -> "WarmStartState":
        """Build warmstart state from an active run state."""

        compatible = cls.optimizer_state_compatible(state.optimizer_name, target_optimizer)
        optimizer_state = _copy_mapping(state.optimizer_state) if compatible else None
        return cls(
            controls=state.controls.copy(name="warmstart"),
            metrics=_copy_metrics(state.metrics),
            step_size=state.step_size,
            optimizer_state=optimizer_state,
            source_optimizer=state.optimizer_name,
            target_optimizer=target_optimizer,
            trace_id=state.trace_id,
            checkpoint_ids=dict(state.checkpoint_ids),
            system_params=dict(state.system_params),
        )

    @classmethod
    def from_result(
        cls,
        result: Any,
        *,
        target_optimizer: str | None = None,
    ) -> "WarmStartState":
        """Build warmstart state from an ``OptimizerResult``-like object."""

        state = getattr(result, "state", None)
        if isinstance(state, RunState):
            return cls.from_run_state(state, target_optimizer=target_optimizer)
        return cls(
            controls=result.controls.copy(name="warmstart"),
            metrics=_copy_metrics(getattr(result, "metrics", {})),
            step_size=None,
            optimizer_state=None,
            source_optimizer=getattr(result, "optimizer", None),
            target_optimizer=target_optimizer,
            trace_id=getattr(result, "trace_id", None),
            checkpoint_ids=_copy_mapping(getattr(result, "checkpoint_ids", {})),
            system_params=_copy_mapping(getattr(result, "system_params", {})),
        )

