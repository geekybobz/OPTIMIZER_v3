"""Population evaluation owned by core; method adapters own sampling/ranking."""
from __future__ import annotations
from optimizer.core.evaluate import EvaluationOutcome, SystemEvaluator
from optimizer.core.runtime import EvaluatorBase


def _worker(item):
    factory, candidate = item
    runtime = SystemEvaluator(factory(), use_cache=False)
    outcome = runtime.try_evaluate(candidate)
    return outcome, runtime.counters()


def evaluate_population(runtime, reference, samples, *, parallel=None, system_factory=None):
    candidates = []
    outcomes = []
    positions = []
    for index, values in enumerate(samples):
        outcome = EvaluatorBase.attempt(reference.with_flat, values, name=f"cma_es_candidate_{index}")
        if outcome.ok:
            candidates.append(outcome.value)
            positions.append(index)
            outcomes.append(None)
        else:
            outcomes.append(EvaluationOutcome(None, False, "evaluation_failed", outcome.error))
    if parallel is None or parallel.backend == "serial" or parallel.resolved_workers() == 1:
        evaluated = [runtime.try_evaluate(candidate) for candidate in candidates]
    else:
        if not callable(system_factory):
            raise TypeError("Parallel population evaluation requires a system_factory.")
        from optimizer.utils.parallel import parallel_map
        values = parallel_map(_worker, [(system_factory, candidate) for candidate in candidates], config=parallel)
        evaluated = []
        for outcome, counters in values:
            evaluated.append(outcome)
            for key, value in counters.items():
                for _ in range(value):
                    runtime._increment(key)
    for position, outcome in zip(positions, evaluated):
        outcomes[position] = outcome
    return outcomes
