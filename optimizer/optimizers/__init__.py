"""Optimizer implementations for OPTIMIZER v4.

Why this package exists
-----------------------
The public API should stay direct, but the implementation still needs clean module
boundaries. This package contains first-order, derivative-free, projected,
proximal, and matrix-free second-order methods. Advanced methods use conditional
system capabilities while the four-hook OLGS core remains unchanged.

How it fits the architecture
----------------------------
- ``optimizer.library`` exposes these functions as ``opt.adam(...)`` and friends.
- every optimizer delegates its loop mechanics to ``core.engine.run_chunk``.
- method-specific memory lives in ``RunState.optimizer_state``.
- variants stay inside one family function instead of leaking wrapper classes.

Reviewer invariants
-------------------
- optimizer modules do not build physical objectives.
- optional numerical hooks are called through ``SystemEvaluator``.
- optimizer modules return the standard ``OptimizerResult``.
"""

from optimizer.optimizers.adam import adam
from optimizer.optimizers.adaptive import adagrad, rmsprop
from optimizer.optimizers.cma_es import cma_es
from optimizer.optimizers.lbfgs import lbfgs
from optimizer.optimizers.line_search import line_search
from optimizer.optimizers.momentum import momentum
from optimizer.optimizers.newton_cg import newton_cg
from optimizer.optimizers.nonlinear_cg import ncg, nonlinear_cg
from optimizer.optimizers.projected import projected_gradient
from optimizer.optimizers.proximal import proximal_gradient
from optimizer.optimizers.trust_region import trust_region
from optimizer.catalog import attach_namespace_helpers


attach_namespace_helpers(globals(), "optimizers")

__all__ = [
    "adagrad",
    "adam",
    "cma_es",
    "info",
    "lbfgs",
    "line_search",
    "list",
    "momentum",
    "newton_cg",
    "ncg",
    "nonlinear_cg",
    "projected_gradient",
    "proximal_gradient",
    "rmsprop",
    "trust_region",
]
