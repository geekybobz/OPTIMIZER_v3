"""Multistart orchestration and qualified-root ranking for OLSS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from optimizer.shooting.solvers import _solve_or_record
from optimizer.system_olss import MultiStartResult, ShootingResult, ShootingSpec, ShootingVariables
from optimizer.system_olss import get_shooting_system_spec, roots_equivalent_system
from optimizer.utils.parallel import ParallelConfig, parallel_map


@dataclass(frozen=True)
class _MultistartJob:
    factory: Callable[[], Any]
    seed: ShootingVariables
    options: dict[str, Any]


def _run_multistart_job(job: _MultistartJob) -> ShootingResult:
    return _solve_or_record(job.factory(), job.seed, job.options)


def _finite_or_inf(value: float) -> float:
    return float(value) if np.isfinite(value) else float("inf")


def _same_root(
    system: Any,
    left: ShootingResult,
    right: ShootingResult,
    tolerance: float,
) -> bool:
    declared = roots_equivalent_system(
        system,
        left.variables,
        right.variables,
        tolerance=tolerance,
    )
    if declared is not None:
        return declared
    scales = left.variables.spec.scales
    left_scaled = left.variables.flatten(copy=False) / scales
    right_scaled = right.variables.flatten(copy=False) / scales
    delta = left_scaled - right_scaled
    scale = max(1.0, np.linalg.norm(left_scaled), np.linalg.norm(right_scaled))
    return bool(np.linalg.norm(delta) <= float(tolerance) * scale)


def solve_shooting_multistart(
    system: Any,
    starts: Iterable[ShootingVariables],
    *,
    parallel: ParallelConfig | None = None,
    system_factory: Callable[[], Any] | None = None,
    dedup_tolerance: float = 1.0e-7,
    solver_options: Mapping[str, Any] | None = None,
) -> MultiStartResult:
    """Solve independent starts, deduplicate qualified roots, and rank them."""

    seeds = list(starts)
    if not seeds:
        raise ValueError("Multistart requires at least one seed.")
    dedup_tolerance = float(dedup_tolerance)
    if not np.isfinite(dedup_tolerance) or dedup_tolerance < 0.0:
        raise ValueError("dedup_tolerance must be finite and nonnegative.")
    cfg = parallel or ParallelConfig(backend="process" if system_factory is not None else "serial")
    if cfg.backend != "serial" and system_factory is None:
        raise ValueError("Parallel multistart requires system_factory so workers own system instances.")

    options = dict(solver_options or {})

    if cfg.backend == "serial":
        results = [
            _solve_or_record(system if system_factory is None else system_factory(), seed, options)
            for seed in seeds
        ]
    else:
        jobs = [_MultistartJob(system_factory, seed, options) for seed in seeds]
        results = parallel_map(_run_multistart_job, jobs, config=cfg)
    qualified = [result for result in results if result.qualified]
    unique: list[ShootingResult] = []
    for result in qualified:
        if not any(_same_root(system, result, existing, dedup_tolerance) for existing in unique):
            unique.append(result)

    spec = get_shooting_system_spec(system)
    selection = spec.get("selection")
    selection_source = None
    if selection is None:
        ranked = list(unique)
        metric = None
        direction = None
    else:
        metric = str(selection["metric_id"])
        direction = str(selection["direction"])
        sources = {_selection_value(result, metric)[1] for result in unique}
        selection_source = sources.pop() if len(sources) == 1 else ("mixed" if sources else None)
        ranked = sorted(
            unique,
            key=lambda result: _selection_value(result, metric)[0],
            reverse=direction == "maximize",
        )
    return MultiStartResult(
        results=results,
        qualified=qualified,
        unique_qualified=unique,
        ranked=ranked,
        selection_metric=metric,
        selection_direction=direction,
        selection_source=selection_source,
    )


def _selection_value(result: ShootingResult, metric: str) -> tuple[float, str]:
    """Rank on the verified metric when verification reports it."""

    verification = result.verification
    if verification is not None and metric in verification.metrics:
        return float(verification.metrics[metric]), "verification"
    return float(result.metrics[metric]), "working"


def random_shooting_seeds(
    spec: ShootingSpec,
    count: int,
    *,
    seed: int | None = None,
    center: ShootingVariables | None = None,
    radius: float = 1.0,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[ShootingVariables]:
    """Generate a reproducible scale-aware random cloud in shooting coordinates."""

    if int(count) < 1:
        raise ValueError("count must be >= 1.")
    if not np.isfinite(float(radius)) or float(radius) <= 0.0:
        raise ValueError("radius must be finite and positive.")
    origin = np.zeros(spec.size) if center is None else center.flatten(copy=False)
    if center is not None and center.spec != spec:
        raise ValueError("center must use the supplied ShootingSpec.")
    rng = np.random.default_rng(seed)
    values = origin + float(radius) * spec.scales * rng.normal(size=(int(count), spec.size))
    if bounds is not None:
        lower, upper = (np.asarray(item, dtype=float).reshape(-1) for item in bounds)
        if lower.shape != (spec.size,) or upper.shape != (spec.size,):
            raise ValueError("bounds must match the ShootingSpec size.")
        values = np.clip(values, lower, upper)
    return [ShootingVariables.from_flat(spec, row, name=f"random_seed_{index}") for index, row in enumerate(values)]


def perturb_shooting_seeds(
    anchor: ShootingVariables,
    count: int,
    *,
    relative_radius: float = 0.1,
    seed: int | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[ShootingVariables]:
    """Generate a scale-aware perturbation cloud around one known candidate."""

    return random_shooting_seeds(
        anchor.spec,
        count,
        seed=seed,
        center=anchor,
        radius=relative_radius,
        bounds=bounds,
    )


def refine_shooting_seeds(
    system: Any,
    starts: Iterable[ShootingVariables],
    *,
    elite_count: int = 3,
    perturbations_per_elite: int = 4,
    relative_radius: float = 0.05,
    seed: int | None = None,
    solver_options: Mapping[str, Any] | None = None,
    parallel: ParallelConfig | None = None,
    system_factory: Callable[[], Any] | None = None,
) -> MultiStartResult:
    """Solve a seed set, perturb its best candidates, and solve the expanded cloud."""

    initial = solve_shooting_multistart(
        system,
        starts,
        solver_options=solver_options,
        parallel=parallel,
        system_factory=system_factory,
    )
    candidates = initial.ranked or sorted(
        initial.results, key=lambda item: _finite_or_inf(item.residual_norm)
    )
    elites = candidates[: max(1, int(elite_count))]
    rng = np.random.default_rng(seed)
    bounds_hook = getattr(system, "shooting_bounds", None)
    bounds = bounds_hook() if callable(bounds_hook) else None
    expanded = [result.variables for result in elites]
    for result in elites:
        expanded.extend(
            perturb_shooting_seeds(
                result.variables,
                int(perturbations_per_elite),
                relative_radius=relative_radius,
                seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                bounds=bounds,
            )
        )
    return solve_shooting_multistart(
        system,
        expanded,
        solver_options=solver_options,
        parallel=parallel,
        system_factory=system_factory,
    )
