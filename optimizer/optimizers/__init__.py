"""First-order optimizer implementations for OPTIMIZER v3.

Why this package exists
-----------------------
The public API should stay direct, but the implementation still needs clean module
boundaries.  This package contains optimizers whose job is to move controls using
``system.gradient(...)`` and the shared engine.  Repair tools, projections,
Jacobian-based constraint handling, and diagnostics deliberately live outside this
package in later phases.

How it fits the architecture
----------------------------
- ``optimizer.library`` exposes these functions as ``opt.adam(...)`` and friends.
- every optimizer delegates its loop mechanics to ``core.engine.run_chunk``.
- method-specific memory lives in ``RunState.optimizer_state``.
- variants stay inside one family function instead of leaking wrapper classes.

Reviewer invariants
-------------------
- optimizer modules do not build physical objectives.
- optimizer modules do not call residual/Jacobian repair hooks.
- optimizer modules return the standard ``OptimizerResult``.
"""

from optimizer.optimizers.adam import adam
from optimizer.optimizers.adaptive import adagrad, rmsprop
from optimizer.optimizers.cma_es import cma_es
from optimizer.optimizers.lbfgs import lbfgs
from optimizer.optimizers.line_search import line_search
from optimizer.optimizers.momentum import momentum
from optimizer.optimizers.nonlinear_cg import ncg, nonlinear_cg

__all__ = [
    "adagrad",
    "adam",
    "cma_es",
    "lbfgs",
    "line_search",
    "momentum",
    "ncg",
    "nonlinear_cg",
    "rmsprop",
]
