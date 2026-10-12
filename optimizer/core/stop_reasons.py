"""Stop-reason registry shared by every OLGS and OLSS run.

Native stop names stay as each tool reports them; this table gives each name
one meaning, the families that use it, an outcome class, and whether the final
state is a valid starting point for another run. Diagnosis, the terminal
interface, MCP, and the documentation read this table instead of keeping their
own lists. A labelled reason such as ``stage_gate_failed:warmup`` is looked up
by the part before the colon.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

OUTCOMES = ("converged", "stopped", "failed", "interrupted")
FAMILIES = ("olgs", "olss")


@dataclass(frozen=True)
class StopReason:
    """Meaning and classification of one stop reason."""

    name: str
    meaning: str
    families: tuple[str, ...]
    outcome: str
    continuable: bool
    source: str

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {self.outcome!r} for stop reason {self.name!r}")
        if not self.families or any(family not in FAMILIES for family in self.families):
            raise ValueError(f"invalid families for stop reason {self.name!r}")

    @staticmethod
    def lookup(reason: str | None) -> "StopReason | None":
        """Return the registry entry for a native reason, or ``None`` when unregistered."""

        if reason is None:
            return None
        return STOP_REASONS.get(str(reason).split(":", 1)[0])

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "meaning": self.meaning,
            "families": list(self.families),
            "outcome": self.outcome,
            "continuable": self.continuable,
            "source": self.source,
        }


_BOTH = ("olgs", "olss")
_OLGS = ("olgs",)
_OLSS = ("olss",)

_ENTRIES = (
    # Shared budget, failure, and completion reasons.
    StopReason("maxiter", "The iteration budget was used up.", _BOTH, "stopped", True, "engine, root solvers, repair"),
    StopReason("nonfinite_trial", "Consecutive trials failed to evaluate: failed_trial_retries in the OLGS engine, every line-search trial of one iteration in OLSS.", _BOTH, "failed", True, "engine, root solvers"),
    StopReason("initial_evaluation_failed", "The starting point could not be evaluated; a coordinator records this for a seed or stage.", _BOTH, "failed", False, "CMA-ES, multistart, natural continuation"),
    StopReason("line_search_failed", "No backtracking step reduced the residual norm.", _BOTH, "failed", True, "residual repair, root solvers"),
    StopReason("completed", "Every requested stage or continuation step finished.", _BOTH, "stopped", True, "continuation"),
    StopReason("interrupted", "The run was interrupted; the last valid state is kept.", _OLGS, "interrupted", True, "engine, CMA-ES"),
    # OLGS engine and stopping rules.
    StopReason("target", "The watched metric reached target_value.", _OLGS, "stopped", True, "stopping rules"),
    StopReason("stall", "No improvement for stall_patience iterations.", _OLGS, "stopped", True, "stopping rules"),
    StopReason("nonfinite", "The current metrics contain a non-finite value.", _OLGS, "failed", False, "stopping rules"),
    StopReason("gradient_converged", "The gradient norm is at or below gradient_tolerance.", _OLGS, "converged", True, "engine"),
    StopReason("stationary_nonminimum", "The gradient tolerance is met but sampled curvature is negative: a saddle or a maximum.", _OLGS, "stopped", True, "engine"),
    StopReason("gradient_failed", "The system gradient failed at the current controls.", _OLGS, "failed", False, "engine"),
    StopReason("smooth_gradient_failed", "The smooth-gradient call of a proximal run failed.", _OLGS, "failed", False, "engine"),
    StopReason("proposal_failed", "The step function raised or proposed invalid controls.", _OLGS, "failed", True, "engine"),
    StopReason("stopped", "Fallback label for a stop without a named reason.", _OLGS, "stopped", True, "engine"),
    StopReason("projected_kkt_converged", "Projected gradient met its feasibility and projected-stationarity tolerances.", _OLGS, "converged", True, "projected_gradient"),
    StopReason("restoration_failed", "Projected gradient could not restore feasibility.", _OLGS, "failed", True, "projected_gradient"),
    StopReason("proximal_mapping_converged", "The proximal-gradient mapping norm is below tolerance.", _OLGS, "converged", True, "proximal_gradient"),
    StopReason("no_usable_candidates", "CMA-ES produced no finite candidate.", _OLGS, "failed", False, "cma_es"),
    StopReason("converged", "The residual norm is at or below the repair tolerance.", _OLGS, "converged", True, "residual repair"),
    StopReason("stage_gate_failed", "A continuation stage did not pass its gate; the stage label follows the colon.", _OLGS, "failed", True, "continuation"),
    # OLSS root solving and verification.
    StopReason("root_tolerance", "The residual norm reached the working tolerance.", _OLSS, "converged", True, "solve_shooting"),
    StopReason("singular_step", "The step could not be computed from the Jacobian.", _OLSS, "failed", True, "solve_shooting"),
    StopReason("least_squares_stationary", "The residual cannot be reduced further: a least-squares stationary point, not a root.", _OLSS, "stopped", True, "solve_shooting"),
    StopReason("bound_stalled", "Declared bounds blocked every step.", _OLSS, "stopped", True, "solve_shooting"),
    StopReason("jacobian_failed", "The Jacobian failed at an accepted point.", _OLSS, "failed", False, "solve_shooting, direct shooting"),
    StopReason("evaluation_failed", "Declared metrics failed at the converged point.", _OLSS, "failed", False, "solve_shooting"),
    StopReason("verification_failed", "Independent verification failed, raised, missed its tolerance, or failed a named check.", _OLSS, "failed", True, "solve_shooting, continuation"),
    StopReason("verification_not_independent", "Verification reused the working options, so it cannot qualify the root.", _OLSS, "failed", True, "solve_shooting"),
    StopReason("initial_feasibility_failed", "Direct shooting could not correct its start onto the feasible set.", _OLSS, "failed", False, "direct shooting"),
    StopReason("cost_stationary", "The projected cost gradient is below tolerance on the feasible set.", _OLSS, "converged", True, "direct shooting"),
    StopReason("cost_line_search_failed", "No corrected step reduced the cost.", _OLSS, "stopped", True, "direct shooting"),
    StopReason("cost_gradient_failed", "The cost gradient failed at the current point.", _OLSS, "failed", False, "direct shooting"),
    StopReason("seed_verification_failed", "A pseudo-arclength seed failed verification.", _OLSS, "failed", False, "pseudo-arclength continuation"),
    StopReason("target_landing_failed", "The corrector could not land exactly on the target value.", _OLSS, "failed", True, "pseudo-arclength continuation"),
    StopReason("minimum_step_reached", "The continuation step shrank below its minimum after failed attempts.", _OLSS, "stopped", True, "continuation"),
    StopReason("target_reached", "The continuation reached its target value exactly.", _OLSS, "stopped", True, "continuation"),
    StopReason("max_steps", "The continuation step budget was used up before the target.", _OLSS, "stopped", True, "natural continuation"),
)

STOP_REASONS: Mapping[str, StopReason] = MappingProxyType({entry.name: entry for entry in _ENTRIES})

if len(STOP_REASONS) != len(_ENTRIES):  # pragma: no cover - guards accidental duplicates
    raise RuntimeError("duplicate stop-reason names in the registry")
