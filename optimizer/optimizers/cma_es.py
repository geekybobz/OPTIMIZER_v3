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
- best-so-far controls are tracked even when the mean update is rejected.
- random seed controls reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from optimizer.controls import Controls
from optimizer.blackbox.run import BlackBoxRun, ensure_run
from optimizer.core.evaluate import SystemEvaluator
from optimizer.logs.trace import Trace
from optimizer.optimizers._common import controls_from_flat_like, require_finite, require_variant
from optimizer.result import Evaluation, OptimizerResult
from optimizer.state import RunState
from optimizer.system_olgs import validate_controls_for_system


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


def cma_es(
    system: Any,
    controls: Controls,
    *,
    variant: str = "diagonal",
    maxiter: int = 100,
    population_size: int | None = None,
    elite_fraction: float = 0.5,
    sigma: float = 0.1,
    min_sigma: float = 1.0e-8,
    covariance_lr: float = 0.2,
    seed: int | None = None,
    accept_metric: str = "J",
    accept_mode: str = "min",
    accept_tolerance: float = 0.0,
    trace: Trace | None = None,
    create_trace: bool = True,
    blackbox: BlackBoxRun | str | bool | None = None,
    blackbox_policy: Any | None = None,
    stage: str | None = None,
    use_cache: bool = True,
) -> OptimizerResult:
    """Run compact CMA-ES style derivative-free search."""

    config = CMAESConfig(
        variant=require_variant(variant, VALID_VARIANTS, family="cma_es"),
        population_size=population_size,
        elite_fraction=elite_fraction,
        sigma=sigma,
        min_sigma=min_sigma,
        covariance_lr=covariance_lr,
        seed=seed,
        accept_metric=accept_metric,
        accept_mode=accept_mode,
    )
    if int(maxiter) < 0:
        raise ValueError("maxiter must be >= 0.")
    require_finite("accept_tolerance", accept_tolerance, nonnegative=True)

    evaluator = SystemEvaluator(system, use_cache=use_cache)
    validate_controls_for_system(evaluator.system, controls)
    active_trace = trace if trace is not None else (Trace(run_id=uuid4().hex) if create_trace else None)
    active_blackbox = ensure_run(blackbox, policy=blackbox_policy)
    owns_blackbox = active_blackbox is not None and not isinstance(blackbox, BlackBoxRun)
    initial = evaluator.evaluate(controls)
    state = RunState.initial(
        controls.copy(name=controls.name),
        metrics=initial.metrics,
        optimizer_name="cma_es",
        step_size=float(config.sigma),
        trace_id=None if active_trace is None else active_trace.run_id,
    )
    if active_trace is not None:
        active_trace.checkpoint("chunk_start", state.controls, state, stage=stage)
    if active_blackbox is not None:
        active_blackbox.record_start(
            system=evaluator.system,
            controls=controls,
            metrics=initial.metrics,
            optimizer="cma_es",
            stage=stage,
            objective={"metric": accept_metric, "mode": accept_mode},
            config={
                "variant": config.variant,
                "maxiter": int(maxiter),
                "population_size": population_size,
                "elite_fraction": float(elite_fraction),
                "sigma": float(sigma),
                "min_sigma": float(min_sigma),
                "covariance_lr": float(covariance_lr),
                "seed": seed,
                "accept_metric": accept_metric,
                "accept_mode": accept_mode,
                "accept_tolerance": float(accept_tolerance),
                "use_cache": bool(use_cache),
            },
        )

    rng = np.random.default_rng(seed)
    dimension = controls.spec.size
    population = config.effective_population_size(dimension)
    elite_count = config.elite_count(population)
    mean = controls.flatten(copy=True).astype(float, copy=False)
    diagonal_sigma = np.full(dimension, float(config.sigma), dtype=float)
    current_value = _metric(initial.metrics, accept_metric)
    accepted_any = False

    for generation in range(int(maxiter)):
        if config.variant == "isotropic":
            samples = mean[None, :] + float(np.mean(diagonal_sigma)) * rng.normal(size=(population, dimension))
        else:
            samples = mean[None, :] + rng.normal(size=(population, dimension)) * diagonal_sigma[None, :]
        samples[0] = mean

        evaluations: list[Evaluation | None] = []
        values = np.full(population, np.inf if accept_mode == "min" else -np.inf, dtype=float)
        errors: list[str | None] = []
        for row in range(population):
            candidate = controls_from_flat_like(controls, samples[row], name=f"cma_es_candidate_{row}")
            outcome = evaluator.try_evaluate(candidate)
            if outcome.ok and outcome.evaluation is not None:
                evaluations.append(outcome.evaluation)
                values[row] = _metric(outcome.evaluation.metrics, accept_metric)
                errors.append(None)
            else:
                evaluations.append(None)
                errors.append(outcome.error)

        order = _rank_indices(values, mode=accept_mode)
        best_index = int(order[0])
        elites = samples[order[:elite_count]]
        elite_mean = _weighted_elite_mean(elites)
        elite_spread = np.mean((elites - elite_mean[None, :]) ** 2, axis=0)
        next_diagonal_sigma = np.sqrt(
            (1.0 - float(config.covariance_lr)) * (diagonal_sigma * diagonal_sigma)
            + float(config.covariance_lr) * np.maximum(elite_spread, float(config.min_sigma) ** 2)
        )
        next_diagonal_sigma = np.maximum(next_diagonal_sigma, float(config.min_sigma))

        best_eval = evaluations[best_index]
        best_value = float(values[best_index])
        previous_controls = state.controls
        previous_metrics = dict(state.metrics)
        previous_value = float(current_value)
        best_controls = (
            controls_from_flat_like(controls, samples[best_index], name="cma_es_best")
            if best_eval is not None
            else None
        )
        accepted = best_eval is not None and _improves(
            best_value,
            current_value,
            mode=accept_mode,
            tolerance=float(accept_tolerance),
        )
        if accepted:
            mean = elite_mean
            diagonal_sigma = next_diagonal_sigma
            state.update_current(
                best_controls,
                best_eval.metrics,
                step_size=float(np.mean(diagonal_sigma)),
                iteration_increment=1,
            )
            current_value = best_value
            accepted_any = True
            state.update_best_by_metric(metric=accept_metric, mode=accept_mode)
            if active_trace is not None:
                active_trace.checkpoint("latest", state.controls, state, stage=stage)
                active_trace.checkpoint(f"best_{accept_metric}", state.controls, state, stage=stage)
            if active_blackbox is not None:
                active_blackbox.record_checkpoint(
                    "latest",
                    state.controls,
                    metrics=state.metrics,
                    optimizer="cma_es",
                    iteration=state.iteration,
                    stage=stage,
                )
                active_blackbox.record_checkpoint(
                    f"best_{accept_metric}",
                    state.controls,
                    metrics=state.metrics,
                    optimizer="cma_es",
                    iteration=state.iteration,
                    stage=stage,
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
        if active_trace is not None:
            active_trace.record_iteration(
                optimizer="cma_es",
                iteration=state.iteration,
                global_iteration=state.global_iteration,
                metrics=state.metrics,
                system_params=state.system_params,
                technical={
                    "variant": config.variant,
                    "population_size": int(population),
                    "elite_count": int(elite_count),
                    "sigma_mean": float(np.mean(diagonal_sigma)),
                    "sigma_max": float(np.max(diagonal_sigma)),
                    "best_population_value": best_value,
                    "best_population_index": best_index,
                    "failed_evaluations": int(sum(error is not None for error in errors)),
                },
                stage=stage,
                accepted=accepted,
                reason="accepted" if accepted else "rejected_no_improvement",
            )
        if active_blackbox is not None:
            active_blackbox.record_iteration(
                optimizer="cma_es",
                iteration=state.iteration,
                global_iteration=state.global_iteration,
                metrics=state.metrics,
                previous_metrics=previous_metrics,
                trial_metrics=None if best_eval is None else best_eval.metrics,
                controls=state.controls,
                previous_controls=previous_controls,
                proposal_controls=best_controls,
                gradient=None,
                technical={
                    "variant": config.variant,
                    "population_size": int(population),
                    "elite_count": int(elite_count),
                    "sigma_mean": float(np.mean(diagonal_sigma)),
                    "sigma_max": float(np.max(diagonal_sigma)),
                    "best_population_value": best_value,
                    "best_population_index": best_index,
                    "failed_evaluations": int(sum(error is not None for error in errors)),
                    "acceptance": {
                        "accept_metric": accept_metric,
                        "accept_mode": accept_mode,
                        "current_value": previous_metrics.get(accept_metric),
                        "trial_value": best_value,
                        "improvement": float(previous_value - best_value) if accept_mode == "min" else float(best_value - previous_value),
                        "tolerance": float(accept_tolerance),
                    },
                },
                stage=stage,
                accepted=accepted,
                reason="accepted" if accepted else "rejected_no_improvement",
                accept_metric=accept_metric,
                accept_mode=accept_mode,
            )

    state.stop_reason = "maxiter"
    if active_trace is not None:
        active_trace.record_chunk(
            optimizer="cma_es",
            chunk=0,
            start_iteration=0,
            end_iteration=state.iteration,
            start_metrics=initial.metrics,
            end_metrics=state.metrics,
            system_params=state.system_params,
            stage=stage,
            accepted=accepted_any,
            reason=state.stop_reason,
        )
    if active_blackbox is not None:
        active_blackbox.record_chunk(
            optimizer="cma_es",
            chunk=0,
            start_iteration=0,
            end_iteration=state.iteration,
            start_metrics=initial.metrics,
            end_metrics=state.metrics,
            system_params=state.system_params,
            stage=stage,
            accepted=accepted_any,
            reason=state.stop_reason,
        )
    result = OptimizerResult.from_state(
        state,
        stop_reason=state.stop_reason,
        optimizer="cma_es",
        trace=active_trace,
        blackbox=active_blackbox,
    )
    if active_blackbox is not None and owns_blackbox:
        active_blackbox.close(result)
    return result
