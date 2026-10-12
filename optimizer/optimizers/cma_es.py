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
- compact numerical aggregates use the same result contract as engine-driven
  optimizers.

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
from optimizer.core.progress import _Progress
from optimizer.controls import Controls
from optimizer.core.evaluate import SystemEvaluator
from optimizer.optimizers._common import controls_from_flat_like, require_finite, require_variant
from optimizer.optimizers._common import coerce_warmstart
from optimizer.result import Evaluation, OptimizerResult
from optimizer.state import RunState, WarmStartState
from optimizer.core.stopping import StopTracker, StoppingConfig
from optimizer.system_olgs import get_system_spec, validate_controls_for_system
VALID_VARIANTS = ('diagonal', 'isotropic')

@dataclass(frozen=True)
class CMAESConfig:
    """Validated options for the compact CMA-ES implementation."""
    variant: str = 'diagonal'
    population_size: int | None = None
    elite_fraction: float = 0.5
    sigma: float = 0.1
    min_sigma: float = 1e-08
    covariance_lr: float = 0.2
    seed: int | None = None
    accept_metric: str = 'J'
    accept_mode: str = 'min'

    def __post_init__(self) -> None:
        require_variant(self.variant, VALID_VARIANTS, family='cma_es')
        require_finite('sigma', self.sigma, positive=True)
        require_finite('min_sigma', self.min_sigma, positive=True)
        require_finite('covariance_lr', self.covariance_lr, positive=True)
        if not 0.0 < float(self.elite_fraction) <= 1.0:
            raise ValueError('elite_fraction must satisfy 0 < elite_fraction <= 1.')
        if not 0.0 < float(self.covariance_lr) <= 1.0:
            raise ValueError('covariance_lr must satisfy 0 < covariance_lr <= 1.')
        if self.population_size is not None and int(self.population_size) < 2:
            raise ValueError('population_size must be >= 2.')
        if self.accept_mode not in {'min', 'max'}:
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
        raise KeyError(f'metrics do not include {key!r}.')
    value = np.asarray(metrics[key])
    if value.shape != ():
        raise ValueError(f'metric {key!r} must be scalar for CMA-ES ranking.')
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f'metric {key!r} must be finite.')
    return out

def _improves(candidate: float, current: float, *, mode: str, tolerance: float) -> bool:
    """Return whether candidate metric improves current metric."""
    if mode == 'min':
        return candidate <= current + float(tolerance)
    return candidate >= current - float(tolerance)

def _rank_indices(values: np.ndarray, *, mode: str) -> np.ndarray:
    """Return candidate indices sorted best to worst."""
    order = np.argsort(values)
    return order if mode == 'min' else order[::-1]

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

def _system_params(system: Any, explicit: Mapping[str, Any] | None) -> dict[str, Any]:
    if explicit is not None:
        return dict(explicit)
    params = getattr(system, 'params', None)
    if isinstance(params, Mapping):
        return dict(params)
    if params is not None and hasattr(params, '__dict__'):
        return dict(vars(params))
    return {}

def cma_es(system: Any, controls: Controls | None=None, *, variant: str='diagonal', maxiter: int=100, population_size: int | None=None, elite_fraction: float=0.5, sigma: float=0.1, min_sigma: float=1e-08, covariance_lr: float=0.2, seed: int | None=None, accept_metric: str | None=None, accept_mode: str | None=None, accept_tolerance: float=0.0, state: RunState | None=None, warmstart: Any | None=None, stopping: StoppingConfig | None=None, target_value: float | None=None, target_metric: str | None=None, target_mode: str | None=None, stall_patience: int | None=None, stall_tolerance: float=0.0, stall_metric: str | None=None, stall_mode: str | None=None, system_params: Mapping[str, Any] | None=None, stage: str | None=None, use_cache: bool=True) -> OptimizerResult:
    """Run compact CMA-ES style derivative-free search."""
    objective = get_system_spec(system)['objective']
    resolved_metric = str(objective['metric_id']) if accept_metric is None else str(accept_metric)
    resolved_mode = 'min' if objective['direction'] == 'minimize' else 'max'
    if accept_mode is not None:
        resolved_mode = str(accept_mode)
    config = CMAESConfig(variant=require_variant(variant, VALID_VARIANTS, family='cma_es'), population_size=population_size, elite_fraction=elite_fraction, sigma=sigma, min_sigma=min_sigma, covariance_lr=covariance_lr, seed=seed, accept_metric=resolved_metric, accept_mode=resolved_mode)
    if int(maxiter) < 0:
        raise ValueError('maxiter must be >= 0.')
    require_finite('accept_tolerance', accept_tolerance, nonnegative=True)
    previous_state = state
    warm = coerce_warmstart(warmstart, target_optimizer='cma_es')
    if previous_state is not None:
        controls = previous_state.controls.copy(name=previous_state.controls.name)
    elif warm is not None:
        controls = warm.controls.copy(name=warm.controls.name)
    if controls is None:
        raise ValueError('cma_es requires controls, state, or warmstart.')
    evaluator = SystemEvaluator(system, use_cache=use_cache)
    validate_controls_for_system(evaluator.system, controls)
    params = _system_params(evaluator.system, system_params)
    initial_outcome = evaluator.try_evaluate(controls)
    if not initial_outcome.ok or initial_outcome.evaluation is None:
        raise ValueError(f'initial controls failed evaluation: {initial_outcome.error}')
    initial = initial_outcome.evaluation
    state = RunState.initial(controls.copy(name=controls.name), metrics=initial.metrics, optimizer_name='cma_es', step_size=float(config.sigma), system_params=params)
    if previous_state is not None:
        state.global_iteration = previous_state.global_iteration
        if previous_state.optimizer_name == 'cma_es':
            state.optimizer_state = dict(previous_state.optimizer_state)
    elif warm is not None and warm.optimizer_state is not None:
        state.optimizer_state = dict(warm.optimizer_state)
    progress = _Progress(state.metrics, str(objective['metric_id']))
    rng = np.random.default_rng(seed)
    dimension = controls.spec.size
    population = config.effective_population_size(dimension)
    elite_count = config.elite_count(population)
    mean = np.asarray(state.optimizer_state.get('mean', controls.flatten(copy=True)), dtype=float).reshape(-1)
    diagonal_sigma = np.asarray(state.optimizer_state.get('diagonal_sigma', np.full(dimension, float(config.sigma))), dtype=float).reshape(-1)
    if mean.shape != (dimension,) or diagonal_sigma.shape != (dimension,):
        raise ValueError('CMA-ES warmstart state has the wrong dimension.')
    current_value = _metric(initial.metrics, config.accept_metric)
    accepted_any = False
    usable_generation_any = False
    failed_generation_count = 0
    first_usable_generation: int | None = None
    resolved_target_metric = config.accept_metric if target_metric is None else str(target_metric)
    resolved_target_mode = ('le' if config.accept_mode == 'min' else 'ge') if target_mode is None else str(target_mode)
    resolved_stall_metric = config.accept_metric if stall_metric is None else str(stall_metric)
    resolved_stall_mode = config.accept_mode if stall_mode is None else str(stall_mode)
    if stopping is None:
        stop_config = StoppingConfig(maxiter=maxiter, target_value=target_value, target_metric=resolved_target_metric, target_mode=resolved_target_mode, stall_patience=stall_patience, stall_tolerance=stall_tolerance, stall_metric=resolved_stall_metric, stall_mode=resolved_stall_mode)
    else:
        stop_config = replace(stopping, target_metric=resolved_target_metric if stopping.target_metric is None else stopping.target_metric, target_mode=resolved_target_mode if stopping.target_mode is None else stopping.target_mode, stall_metric=resolved_stall_metric if stopping.stall_metric is None else stopping.stall_metric, stall_mode=resolved_stall_mode if stopping.stall_mode is None else stopping.stall_mode)
    tracker = StopTracker(stop_config, initial_metrics=state.metrics)
    initial_stop = tracker.check_initial_metrics(state.metrics)
    requested_stop: str | None = initial_stop.reason if initial_stop.stop else None

    def finish_state(reason: str, *, degenerate: bool=False) -> None:
        """Close evidence from existing CMA-ES state without extra evaluations."""
        state.stop_reason = reason
        certification: dict[str, Any] = {}
        if failed_generation_count:
            certification['cma_es_degeneracy'] = {'status': 'all_generations_degenerate' if not usable_generation_any else 'degeneracy_observed_after_usable_generation', 'failed_generations': failed_generation_count, 'first_usable_generation': first_usable_generation, 'population_size': int(population)}
    for generation in range(int(stop_config.maxiter)):
        if requested_stop is not None:
            break
        if config.variant == 'isotropic':
            samples = mean[None, :] + float(np.mean(diagonal_sigma)) * rng.normal(size=(population, dimension))
        else:
            samples = mean[None, :] + rng.normal(size=(population, dimension)) * diagonal_sigma[None, :]
        samples[0] = mean
        evaluations: list[Evaluation | None] = []
        values = np.full(population, np.inf if config.accept_mode == 'min' else -np.inf, dtype=float)
        errors: list[str | None] = []
        for row in range(population):
            candidate = controls_from_flat_like(controls, samples[row], name=f'cma_es_candidate_{row}')
            try:
                outcome = evaluator.try_evaluate(candidate)
            except KeyboardInterrupt:
                finish_state('interrupted')
                raise
            if outcome.ok and outcome.evaluation is not None:
                evaluations.append(outcome.evaluation)
                values[row] = _metric(outcome.evaluation.metrics, config.accept_metric)
                errors.append(None)
            else:
                evaluations.append(None)
                errors.append(outcome.error)
        successful_indices = np.asarray([index for index, evaluation in enumerate(evaluations) if evaluation is not None], dtype=int)
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
            next_diagonal_sigma = np.sqrt((1.0 - float(config.covariance_lr)) * (diagonal_sigma * diagonal_sigma) + float(config.covariance_lr) * np.maximum(elite_spread, float(config.min_sigma) ** 2))
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
        best_controls = controls_from_flat_like(controls, samples[best_index], name='cma_es_best') if best_eval is not None and best_index is not None else None
        accepted = best_eval is not None and _improves(float(best_value), current_value, mode=config.accept_mode, tolerance=float(accept_tolerance))
        improved = False
        if accepted:
            mean = elite_mean
            diagonal_sigma = next_diagonal_sigma
            state.update_current(best_controls, best_eval.metrics, step_size=float(np.mean(diagonal_sigma)), iteration_increment=1)
            current_value = float(best_value)
            accepted_any = True
            improved = state.update_best_by_metric(metric=config.accept_metric, mode=config.accept_mode)
        else:
            diagonal_sigma = np.maximum(0.9 * diagonal_sigma, float(config.min_sigma))
            state.iteration += 1
            state.global_iteration += 1
            state.step_size = float(np.mean(diagonal_sigma))
        state.optimizer_state = {'variant': config.variant, 'mean': mean.copy(), 'diagonal_sigma': diagonal_sigma.copy(), 'population_size': int(population), 'elite_count': int(elite_count), 'generation': int(generation + 1), 'seed': seed, 'best_population_value': best_value, 'best_population_index': best_index}
        progress.observe(state.metrics, accepted=accepted)
        decision = tracker.check_metrics(state.metrics)
        if decision.stop:
            requested_stop = decision.reason or 'stopped'
    stop_reason = requested_stop or ('maxiter' if usable_generation_any or int(stop_config.maxiter) == 0 else 'no_usable_candidates')
    finish_state(stop_reason, degenerate=int(stop_config.maxiter) == 0 or not usable_generation_any)
    result = OptimizerResult.from_state(state, stop_reason=state.stop_reason, optimizer='cma_es', objective_metric=str(objective['metric_id']), diagnostics=progress.finish(evaluator.counters()))
    return result
