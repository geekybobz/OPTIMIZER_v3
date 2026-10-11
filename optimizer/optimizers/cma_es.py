"""Compact CMA-ES style derivative-free optimizer.

Why this file exists
--------------------
Gradient methods are not always enough for hard landscapes or poor initial guesses.
CMA-ES-style search samples a population around a current mean, keeps elite
candidates, and adapts the sampling distribution.  This is useful for rough global
search, especially when later guess modules parameterize pulses with a small Fourier
or smooth basis.

How it fits the architecture
----------------------------
- this optimizer still uses ``Controls`` and system ``evaluate``.
- it does not require ``system.gradient`` during iterations.
- it returns the standard ``OptimizerResult`` with a ``RunState``.
- trace records and checkpoints are emitted in the same style as engine-driven
  optimizers where practical.

What this file deliberately does not do
---------------------------------------
It is not a full research-grade CMA-ES implementation with evolution paths and full
covariance matrix adaptation.  The initial library version provides two practical
variants:

``isotropic``
    one global sampling radius.

``diagonal``
    one sampling radius per flattened control coordinate.

Reviewer invariants
-------------------
- population candidates are finite controls matching the system spec.
- accepted generations improve the selected metric.
- best-so-far controls advance only through accepted generations.
- random seed controls reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping
from uuid import uuid4

import numpy as np

from optimizer.controls import Controls
from optimizer.core.evaluate import SystemEvaluator
from optimizer.core.observer import (
    CheckpointEvent,
    ChunkEvent,
    IterationEvent,
    ObserverDispatcher,
    RunObserver,
    RunStartEvent,
    SnapshotFacts,
    TerminalEvent,
    classify_terminal_status,
    emit_initial_failure,
    normalize_observers,
)
from optimizer.logs.trace import Trace
from optimizer.optimizers._common import controls_from_flat_like, require_finite, require_variant
from optimizer.optimizers._common import coerce_warmstart
from optimizer.result import Evaluation, OptimizerResult
from optimizer.state import RunState, WarmStartState
from optimizer.core.stopping import StopTracker, StoppingConfig
from optimizer.system_olgs import get_system_spec, validate_controls_for_system


VALID_VARIANTS = ("diagonal", "isotropic")


@dataclass(frozen=True)
class CMAESConfig:
    """Validated options for the compact CMA-ES implementation."""

    variant: str = "diagonal"
    population_size: int | None = None
    elite_fraction: float = 0.5
    sigma: float = 0.1
    min_sigma: float = 1.0e-8
    covariance_lr: float = 0.2
    seed: int | None = None
    accept_metric: str = "J"
    accept_mode: str = "min"

    def __post_init__(self) -> None:
        require_variant(self.variant, VALID_VARIANTS, family="cma_es")
        require_finite("sigma", self.sigma, positive=True)
        require_finite("min_sigma", self.min_sigma, positive=True)
        require_finite("covariance_lr", self.covariance_lr, positive=True)
        if not 0.0 < float(self.elite_fraction) <= 1.0:
            raise ValueError("elite_fraction must satisfy 0 < elite_fraction <= 1.")
        if not 0.0 < float(self.covariance_lr) <= 1.0:
            raise ValueError("covariance_lr must satisfy 0 < covariance_lr <= 1.")
        if self.population_size is not None and int(self.population_size) < 2:
            raise ValueError("population_size must be >= 2.")
        if self.accept_mode not in {"min", "max"}:
            raise ValueError("accept_mode must be 'min' or 'max'.")

    def effective_population_size(self, dimension: int) -> int:
        """Return default or explicit population size."""

        if self.population_size is not None:
            return int(self.population_size)
        return int(max(8, 4 + np.floor(3.0 * np.log(max(2, int(dimension))))))

    def elite_count(self, population_size: int) -> int:
        """Return number of elites retained per generation."""

        return int(max(1, np.ceil(float(self.elite_fraction) * int(population_size))))


def _metric(metrics: Mapping[str, Any], key: str) -> float:
    """Return a finite scalar metric for ranking candidates."""

    if key not in metrics:
        raise KeyError(f"metrics do not include {key!r}.")
    value = np.asarray(metrics[key])
    if value.shape != ():
        raise ValueError(f"metric {key!r} must be scalar for CMA-ES ranking.")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"metric {key!r} must be finite.")
    return out


def _improves(candidate: float, current: float, *, mode: str, tolerance: float) -> bool:
    """Return whether candidate metric improves current metric."""

    if mode == "min":
        return candidate <= current + float(tolerance)
    return candidate >= current - float(tolerance)


def _rank_indices(values: np.ndarray, *, mode: str) -> np.ndarray:
    """Return candidate indices sorted best to worst."""

    order = np.argsort(values)
    return order if mode == "min" else order[::-1]


def _weighted_elite_mean(elites: np.ndarray) -> np.ndarray:
    """Return log-weighted mean of elite candidate vectors."""

    count = elites.shape[0]
    if count == 1:
        return elites[0].copy()
    ranks = np.arange(1, count + 1, dtype=float)
    weights = np.log(count + 0.5) - np.log(ranks)
    weights = np.maximum(weights, 0.0)
    if float(np.sum(weights)) <= np.finfo(float).tiny:
        weights = np.ones(count, dtype=float)
    weights = weights / np.sum(weights)
    return weights @ elites


def _snapshot(evaluator: SystemEvaluator, state: RunState, metrics: Mapping[str, Any]) -> SnapshotFacts:
    return SnapshotFacts.create(
        metrics=metrics,
        metric_sources={str(name): "system.evaluate" for name in metrics},
        counters={
            "evaluation": evaluator.evaluation_count,
            "gradient": evaluator.gradient_count,
            "evaluation_attempt": evaluator.evaluation_attempt_count,
            "gradient_attempt": evaluator.gradient_attempt_count,
            "iteration": state.iteration,
            "global_iteration": state.global_iteration,
        },
    )


def _system_params(system: Any, explicit: Mapping[str, Any] | None) -> dict[str, Any]:
    if explicit is not None:
        return dict(explicit)
    params = getattr(system, "params", None)
    if isinstance(params, Mapping):
        return dict(params)
    if params is not None and hasattr(params, "__dict__"):
        return dict(vars(params))
    return {}


def cma_es(
    system: Any,
    controls: Controls | None = None,
    *,
    variant: str = "diagonal",
    maxiter: int = 100,
    population_size: int | None = None,
    elite_fraction: float = 0.5,
    sigma: float = 0.1,
    min_sigma: float = 1.0e-8,
    covariance_lr: float = 0.2,
    seed: int | None = None,
    accept_metric: str | None = None,
    accept_mode: str | None = None,
    accept_tolerance: float = 0.0,
    state: RunState | None = None,
    warmstart: Any | None = None,
    stopping: StoppingConfig | None = None,
    target_value: float | None = None,
    target_metric: str | None = None,
    target_mode: str | None = None,
    stall_patience: int | None = None,
    stall_tolerance: float = 0.0,
    stall_metric: str | None = None,
    stall_mode: str | None = None,
    trace: Trace | None = None,
    create_trace: bool = True,
    observers: Iterable[RunObserver] | None = None,
    api_context: Mapping[str, Any] | None = None,
    system_params: Mapping[str, Any] | None = None,
    stage: str | None = None,
    chunk: int = 0,
    use_cache: bool = True,
    checkpoint_start: bool = True,
    checkpoint_best: bool = True,
    checkpoint_final: bool = True,
) -> OptimizerResult:
    """Run compact CMA-ES style derivative-free search."""

    objective = get_system_spec(system)["objective"]
    resolved_metric = str(objective["metric_id"]) if accept_metric is None else str(accept_metric)
    resolved_mode = "min" if objective["direction"] == "minimize" else "max"
    if accept_mode is not None:
        resolved_mode = str(accept_mode)
    config = CMAESConfig(
        variant=require_variant(variant, VALID_VARIANTS, family="cma_es"),
        population_size=population_size,
        elite_fraction=elite_fraction,
        sigma=sigma,
        min_sigma=min_sigma,
        covariance_lr=covariance_lr,
        seed=seed,
        accept_metric=resolved_metric,
        accept_mode=resolved_mode,
    )
    if int(maxiter) < 0:
        raise ValueError("maxiter must be >= 0.")
    require_finite("accept_tolerance", accept_tolerance, nonnegative=True)

    previous_state = state
    warm = coerce_warmstart(warmstart, target_optimizer="cma_es")
    if previous_state is not None:
        controls = previous_state.controls.copy(name=previous_state.controls.name)
    elif warm is not None:
        controls = warm.controls.copy(name=warm.controls.name)
    if controls is None:
        raise ValueError("cma_es requires controls, state, or warmstart.")

    evaluator = SystemEvaluator(system, use_cache=use_cache)
    validate_controls_for_system(evaluator.system, controls)
    active_trace = trace if trace is not None else (Trace(run_id=uuid4().hex) if create_trace else None)
    active_observers = list(normalize_observers(observers))
    if active_trace is not None and all(observer is not active_trace for observer in active_observers):
        active_observers.append(active_trace)
    sinks = ObserverDispatcher(active_observers)
    params = _system_params(evaluator.system, system_params)
    resolved_api_context = {
        "cma_es": {
            "variant": config.variant,
            "maxiter": int(maxiter),
            "population_size": population_size,
            "elite_fraction": float(elite_fraction),
            "sigma": float(sigma),
            "min_sigma": float(min_sigma),
            "covariance_lr": float(covariance_lr),
            "seed": seed,
            "accept_metric": config.accept_metric,
            "accept_mode": config.accept_mode,
            "accept_tolerance": float(accept_tolerance),
            "stage": stage,
            "chunk": int(chunk),
            "use_cache": bool(use_cache),
            "trace_supplied": trace is not None,
            "observer_count": len(sinks.active),
            "checkpoint_policy": {
                "start": bool(checkpoint_start),
                "best": bool(checkpoint_best),
                "final": bool(checkpoint_final),
            },
        },
        "caller": dict(api_context or {}),
    }
    initial_outcome = evaluator.try_evaluate(controls)
    if not initial_outcome.ok or initial_outcome.evaluation is None:
        failed = SnapshotFacts.create(
            counters={
                "evaluation": evaluator.evaluation_count,
                "gradient": 0,
                "evaluation_attempt": evaluator.evaluation_attempt_count,
                "gradient_attempt": 0,
                "iteration": 0,
                "global_iteration": 0,
            }
        )
        start_event = RunStartEvent(
            optimizer="cma_es",
            phase="initialization" if stage is None else str(stage),
            initial=failed,
            chunk=int(chunk),
            stage=stage,
            system_params=params,
            api_context={**resolved_api_context, "initial_error": initial_outcome.error},
        )
        terminal_event = TerminalEvent(
            optimizer="cma_es",
            status="failed",
            reason="initial_evaluation_failed",
            final=failed,
            best=None,
            iteration=0,
            global_iteration=0,
            stage=stage,
            chunk=int(chunk),
            system_params=params,
        )
        emit_initial_failure(sinks, start_event, terminal_event)
        raise ValueError(f"initial controls failed evaluation: {initial_outcome.error}")
    initial = initial_outcome.evaluation
    state = RunState.initial(
        controls.copy(name=controls.name),
        metrics=initial.metrics,
        optimizer_name="cma_es",
        step_size=float(config.sigma),
        system_params=params,
        trace_id=None if active_trace is None else active_trace.run_id,
    )
    if previous_state is not None:
        state.global_iteration = previous_state.global_iteration
        state.checkpoint_ids = dict(previous_state.checkpoint_ids)
        if previous_state.optimizer_name == "cma_es":
            state.optimizer_state = dict(previous_state.optimizer_state)
    elif warm is not None and warm.optimizer_state is not None:
        state.optimizer_state = dict(warm.optimizer_state)

    start_event = RunStartEvent(
        optimizer="cma_es",
        phase="initialization" if stage is None else str(stage),
        initial=_snapshot(evaluator, state, initial.metrics),
        chunk=int(chunk),
        stage=stage,
        system_params=params,
        api_context=resolved_api_context,
    )
    sinks.emit("on_start", start_event)

    rng = np.random.default_rng(seed)
    dimension = controls.spec.size
    population = config.effective_population_size(dimension)
    elite_count = config.elite_count(population)
    mean = np.asarray(
        state.optimizer_state.get("mean", controls.flatten(copy=True)), dtype=float
    ).reshape(-1)
    diagonal_sigma = np.asarray(
        state.optimizer_state.get("diagonal_sigma", np.full(dimension, float(config.sigma))),
        dtype=float,
    ).reshape(-1)
    if mean.shape != (dimension,) or diagonal_sigma.shape != (dimension,):
        raise ValueError("CMA-ES warmstart state has the wrong dimension.")
    current_value = _metric(initial.metrics, config.accept_metric)
    accepted_any = False
    usable_generation_any = False
    failed_generation_count = 0
    first_usable_generation: int | None = None
    resolved_target_metric = config.accept_metric if target_metric is None else str(target_metric)
    resolved_target_mode = ("le" if config.accept_mode == "min" else "ge") if target_mode is None else str(target_mode)
    resolved_stall_metric = config.accept_metric if stall_metric is None else str(stall_metric)
    resolved_stall_mode = config.accept_mode if stall_mode is None else str(stall_mode)
    if stopping is None:
        stop_config = StoppingConfig(
            maxiter=maxiter,
            target_value=target_value,
            target_metric=resolved_target_metric,
            target_mode=resolved_target_mode,
            stall_patience=stall_patience,
            stall_tolerance=stall_tolerance,
            stall_metric=resolved_stall_metric,
            stall_mode=resolved_stall_mode,
        )
    else:
        stop_config = replace(
            stopping,
            target_metric=resolved_target_metric if stopping.target_metric is None else stopping.target_metric,
            target_mode=resolved_target_mode if stopping.target_mode is None else stopping.target_mode,
            stall_metric=resolved_stall_metric if stopping.stall_metric is None else stopping.stall_metric,
            stall_mode=resolved_stall_mode if stopping.stall_mode is None else stopping.stall_mode,
        )
    resolved_api_context["cma_es"]["effective_maxiter"] = int(stop_config.maxiter)
    resolved_api_context["cma_es"]["stopping"] = {
        "target_value": stop_config.target_value,
        "target_metric": stop_config.target_metric,
        "target_mode": stop_config.target_mode,
        "stall_patience": stop_config.stall_patience,
        "stall_tolerance": stop_config.stall_tolerance,
        "stall_metric": stop_config.stall_metric,
        "stall_mode": stop_config.stall_mode,
        "check_finite": stop_config.check_finite,
    }
    tracker = StopTracker(stop_config, initial_metrics=state.metrics)
    initial_stop = tracker.check_initial_metrics(state.metrics)
    requested_stop: str | None = initial_stop.reason if initial_stop.stop else None

    def emit_checkpoint(
        label: str,
        reason: str,
        *,
        checkpoint_controls: Controls | None = None,
        checkpoint_metrics: Mapping[str, Any] | None = None,
    ) -> None:
        event = CheckpointEvent(
            label=label,
            reason=reason,
            controls=(state.controls if checkpoint_controls is None else checkpoint_controls).copy(
                name=(state.controls if checkpoint_controls is None else checkpoint_controls).name
            ),
            snapshot=_snapshot(
                evaluator,
                state,
                state.metrics if checkpoint_metrics is None else checkpoint_metrics,
            ),
            optimizer="cma_es",
            optimizer_state=dict(state.optimizer_state),
            system_params=dict(state.system_params),
            api_context=resolved_api_context,
            iteration=state.iteration,
            global_iteration=state.global_iteration,
            chunk=int(chunk),
            stage=stage,
            random_state=dict(rng.bit_generator.state),
            random_state_source="numpy.bit_generator.state",
        )
        for observer in sinks.emit("on_checkpoint", event):
            if isinstance(observer, Trace):
                state.checkpoint_ids[str(label)] = observer.latest_checkpoint_id(str(label))

    if checkpoint_start:
        emit_checkpoint("chunk_start", "initial_valid_state")

    def close_events(reason: str, *, degenerate: bool = False) -> None:
        """Close evidence from existing CMA-ES state without extra evaluations."""

        state.stop_reason = reason
        if checkpoint_best and state.best_controls is not None and state.best_metrics is not None:
            emit_checkpoint(
                f"best_{config.accept_metric}",
                "terminal_selected_best",
                checkpoint_controls=state.best_controls,
                checkpoint_metrics=state.best_metrics,
            )
        if checkpoint_final:
            emit_checkpoint("chunk_end", reason)
        chunk_event = ChunkEvent(
            optimizer="cma_es",
            chunk=int(chunk),
            start_iteration=0,
            end_iteration=state.iteration,
            start_metrics=dict(initial.metrics),
            end_metrics=dict(state.metrics),
            system_params=dict(state.system_params),
            stage=stage,
            accepted=accepted_any,
            reason=reason,
        )
        certification: dict[str, Any] = {}
        if failed_generation_count:
            certification["cma_es_degeneracy"] = {
                "status": (
                    "all_generations_degenerate"
                    if not usable_generation_any
                    else "degeneracy_observed_after_usable_generation"
                ),
                "failed_generations": failed_generation_count,
                "first_usable_generation": first_usable_generation,
                "population_size": int(population),
            }
        terminal_event = TerminalEvent(
            optimizer="cma_es",
            status=classify_terminal_status(reason, degenerate=degenerate),
            reason=reason,
            final=_snapshot(evaluator, state, state.metrics),
            best=None if state.best_metrics is None else _snapshot(evaluator, state, state.best_metrics),
            iteration=state.iteration,
            global_iteration=state.global_iteration,
            stage=stage,
            chunk=int(chunk),
            system_params=dict(state.system_params),
            certification=certification,
        )
        sinks.emit("on_chunk", chunk_event)
        sinks.emit("on_terminal", terminal_event)

    for generation in range(int(stop_config.maxiter)):
        if requested_stop is not None:
            break
        before = _snapshot(evaluator, state, state.metrics)
        if config.variant == "isotropic":
            samples = mean[None, :] + float(np.mean(diagonal_sigma)) * rng.normal(size=(population, dimension))
        else:
            samples = mean[None, :] + rng.normal(size=(population, dimension)) * diagonal_sigma[None, :]
        samples[0] = mean

        evaluations: list[Evaluation | None] = []
        values = np.full(
            population,
            np.inf if config.accept_mode == "min" else -np.inf,
            dtype=float,
        )
        errors: list[str | None] = []
        for row in range(population):
            candidate = controls_from_flat_like(controls, samples[row], name=f"cma_es_candidate_{row}")
            try:
                outcome = evaluator.try_evaluate(candidate)
            except KeyboardInterrupt:
                close_events("interrupted")
                raise
            if outcome.ok and outcome.evaluation is not None:
                evaluations.append(outcome.evaluation)
                values[row] = _metric(outcome.evaluation.metrics, config.accept_metric)
                errors.append(None)
            else:
                evaluations.append(None)
                errors.append(outcome.error)

        successful_indices = np.asarray(
            [index for index, evaluation in enumerate(evaluations) if evaluation is not None],
            dtype=int,
        )
        if successful_indices.size:
            usable_generation_any = True
            if first_usable_generation is None:
                first_usable_generation = generation + 1
            local_order = _rank_indices(values[successful_indices], mode=config.accept_mode)
            order = successful_indices[local_order]
            best_index: int | None = int(order[0])
            usable_elite_count = min(int(elite_count), int(successful_indices.size))
            elites = samples[order[:usable_elite_count]]
            elite_mean = _weighted_elite_mean(elites)
            elite_spread = np.mean((elites - elite_mean[None, :]) ** 2, axis=0)
            next_diagonal_sigma = np.sqrt(
                (1.0 - float(config.covariance_lr)) * (diagonal_sigma * diagonal_sigma)
                + float(config.covariance_lr)
                * np.maximum(elite_spread, float(config.min_sigma) ** 2)
            )
            next_diagonal_sigma = np.maximum(next_diagonal_sigma, float(config.min_sigma))
            best_eval = evaluations[best_index]
            best_value: float | None = float(values[best_index])
        else:
            failed_generation_count += 1
            best_index = None
            best_eval = None
            best_value = None
            elite_mean = mean
            next_diagonal_sigma = diagonal_sigma
        best_controls = (
            controls_from_flat_like(controls, samples[best_index], name="cma_es_best")
            if best_eval is not None and best_index is not None
            else None
        )
        accepted = best_eval is not None and _improves(
            float(best_value),
            current_value,
            mode=config.accept_mode,
            tolerance=float(accept_tolerance),
        )
        improved = False
        if accepted:
            mean = elite_mean
            diagonal_sigma = next_diagonal_sigma
            state.update_current(
                best_controls,
                best_eval.metrics,
                step_size=float(np.mean(diagonal_sigma)),
                iteration_increment=1,
            )
            current_value = float(best_value)
            accepted_any = True
            improved = state.update_best_by_metric(
                metric=config.accept_metric,
                mode=config.accept_mode,
            )
        else:
            # Even rejected generations shrink/adapt the sampling radius slightly
            # around the existing mean, so repeated failures do not keep sampling the
            # same large cloud forever.
            diagonal_sigma = np.maximum(0.9 * diagonal_sigma, float(config.min_sigma))
            state.iteration += 1
            state.global_iteration += 1
            state.step_size = float(np.mean(diagonal_sigma))

        state.optimizer_state = {
            "variant": config.variant,
            "mean": mean.copy(),
            "diagonal_sigma": diagonal_sigma.copy(),
            "population_size": int(population),
            "elite_count": int(elite_count),
            "generation": int(generation + 1),
            "seed": seed,
            "best_population_value": best_value,
            "best_population_index": best_index,
        }
        iteration_event = IterationEvent(
            optimizer="cma_es",
            iteration=state.iteration,
            global_iteration=state.global_iteration,
            stage=stage,
            system_params=dict(state.system_params),
            before=before,
            trial=None if best_eval is None else _snapshot(evaluator, state, best_eval.metrics),
            after=_snapshot(evaluator, state, state.metrics),
            best_after=None
            if state.best_metrics is None
            else _snapshot(evaluator, state, state.best_metrics),
            technical={
                "variant": config.variant,
                "population_size": int(population),
                "elite_count": min(int(elite_count), int(successful_indices.size)),
                "sigma_mean": float(np.mean(diagonal_sigma)),
                "sigma_max": float(np.max(diagonal_sigma)),
                "best_population_value": best_value,
                "best_population_index": best_index,
                "failed_evaluations": int(sum(error is not None for error in errors)),
            },
            accepted=accepted,
            reason="accepted" if accepted else "rejected_no_improvement",
            chunk=int(chunk),
        )
        sinks.emit("on_iteration", iteration_event)
        if improved and checkpoint_best:
            emit_checkpoint(f"best_{config.accept_metric}", "new_best_metric")
        decision = tracker.check_metrics(state.metrics)
        if decision.stop:
            requested_stop = decision.reason or "stopped"
    stop_reason = requested_stop or (
        "maxiter" if usable_generation_any or int(stop_config.maxiter) == 0 else "no_usable_candidates"
    )
    close_events(stop_reason, degenerate=int(stop_config.maxiter) == 0 or not usable_generation_any)
    result = OptimizerResult.from_state(
        state,
        stop_reason=state.stop_reason,
        optimizer="cma_es",
        trace=active_trace,
        tracking_failures=sinks.failures,
        objective_metric=str(objective["metric_id"]),
    )
    return result
