"""User-style shooting loop on the Duffing OLSS system, library calls only.

Run from the V4 checkout root:

    PYTHONPATH=.:test_contained/review_2026_10_12/duffing_olss python -B \
        test_contained/review_2026_10_12/duffing_olss/run_loop.py
"""

import json
import time

import optimizer as opt
from system import DuffingShooting


def show(label, value):
    print(f"\n== {label}")
    print(json.dumps(value, indent=1, default=str)[:900])


t0 = time.perf_counter()
system = DuffingShooting(primary={"T": 2.0}, secondary={"epsilon": 0.0})
spec = system.shooting_spec()

# 1. seed and derivative check
z0 = opt.ShootingVariables.from_blocks(spec, {"initial_costate": [0.0, 0.0]})
jacobian_check = opt.verify_shooting_jacobian(system, z0)
show("verify_shooting_jacobian", {k: jacobian_check[k] for k in ("passed", "max_abs_error", "declared_source")})

# 2. a one-iteration chunk, diagnosis, geometry, then resume into one history
events = []
r1 = opt.solve_shooting(system, z0, maxiter=1, on_event=events.append)
show("diagnose_run after one iteration", opt.diagnose_run(r1))
show("shooting_geometry_probe", {k: v for k, v in opt.shooting_geometry_probe(system, r1.variables).items() if k != "counters"})
r2 = opt.solve_shooting(system, r1.variables, previous_result=r1, on_event=events.append)
show(
    "resumed solve",
    {
        "qualified": r2.qualified,
        "stop": r2.stop_reason,
        "history_iterations": [row["iteration"] for row in r2.history],
        "verification_tolerance": [r2.verification_tolerance, r2.verification_tolerance_source],
        "checks": [check.to_dict() for check in r2.verification.checks],
    },
)

# 3. natural continuation epsilon 0 -> 2; the system declares its verification tolerance
branch = opt.continue_parameter(system, r2.variables, parameter="epsilon", start=0.0, target=2.0, step_size=0.5)
show("continue_parameter", {"stop": branch.stop_reason, "reached": branch.reached_target,
                            "path": [(p.parameter, round(p.result.metrics["energy"], 6)) for p in branch.points]})

# 4. multistart at epsilon = 2 with 12 random seeds plus one seed that blows up
hard = system.with_secondary(epsilon=2.0)
seeds = opt.random_shooting_seeds(spec, 12, seed=7, radius=4.0)
seeds.append(opt.ShootingVariables.from_blocks(spec, {"initial_costate": [1.0e6, -1.0e6]}))
campaign = opt.solve_shooting_multistart(hard, seeds, solver_options={"maxiter": 30})
show(
    "multistart",
    {
        "qualified": len(campaign.qualified),
        "unique": len(campaign.unique_qualified),
        "best_energy": campaign.best.metrics["energy"],
        "ranked_on": campaign.selection_source,
        "stops": sorted({r.stop_reason for r in campaign.results}),
    },
)

# 5. strict JSON for every result
for result in (r2, branch, campaign):
    json.dumps(result.to_dict(), allow_nan=True if result is campaign else False)
print("\nresults serialize: ok (the failed seed carries NaN residuals)")
print(f"total wall time {time.perf_counter() - t0:.2f} s")
