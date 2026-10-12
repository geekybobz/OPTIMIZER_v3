"""User-style short-run loop on the qubit OLGS system, library calls only.

Run from the V4 checkout root:

    PYTHONPATH=.:test_contained/review_2026_10_12/qubit_olgs python -B \
        test_contained/review_2026_10_12/qubit_olgs/run_loop.py
"""

import json
import time

import numpy as np

import optimizer as opt
from system import QubitTransfer


def show(label, value):
    print(f"\n== {label}")
    print(json.dumps(value, indent=1, default=str)[:1500])


t0 = time.perf_counter()
system = QubitTransfer(primary={"control_dim": 100, "dt": 0.05}, secondary={"delta": 1.0, "energy_weight": 1e-3})

# 1. discovery and contract check
info = opt.system_info(system)
show("system_info capabilities", info["capabilities"])

# 2. initial guess and gradient check
u0 = opt.guesses.random_fourier_guess(system, amplitude=0.5, modes=4, seed=3)
check = opt.utils.verify_gradient(system, u0, eps=1e-6, directions=6)
show("verify_gradient", {k: check[k] for k in check if k in ("passed", "max_abs_error", "max_rel_error", "max_relative_error", "directions")})
print("J0 =", opt.evaluate(system, u0).metrics)

# 3. a short Adam chunk, then diagnose
r1 = opt.adam(system, u0, step_size=0.05, maxiter=10)
show("adam(10) diagnosis", opt.diagnose_run(r1))
print("stop:", r1.stop_reason, "| J:", r1.metrics["J"], "| fidelity:", r1.metrics["fidelity"])

# 4. local curvature and a quick method comparison from the same point
show("curvature_probe", {k: v for k, v in opt.curvature_probe(system, r1.controls, directions=6).items() if k != "weak_direction"})
cmp = opt.compare_methods(system, r1.controls, ("adam", "momentum", "line_search", "lbfgs"), iterations=10,
                          options={"adam": {"step_size": 0.05}, "momentum": {"step_size": 0.05}})
show("compare_methods", [{k: row[k] for k in ("method", "objective", "stop_reason", "iterations")} | {"class": row["diagnosis"]["classification"]} for row in cmp["ranking"]])

# 5. continue with the best-looking method, carrying the warm start
r2 = opt.lbfgs(system, r1.controls, warmstart=r1.warmstart(), maxiter=40)
show("lbfgs(40) after adam", {"stop": r2.stop_reason, "J": r2.metrics["J"], "fidelity": r2.metrics["fidelity"], "iterations": r2.iterations})
show("lbfgs diagnosis", opt.diagnose_run(r2))

# 6. continue the same method from its state (chunked run)
r3 = opt.lbfgs(system, state=r2.state, maxiter=20)
show("lbfgs(+20) continued", {"stop": r3.stop_reason, "J": r3.metrics["J"], "fidelity": r3.metrics["fidelity"]})

# 7. curriculum: change a secondary parameter and continue
system_b = system.with_secondary(energy_weight=1e-2)
r4 = opt.adam(system_b, r3.controls, step_size=0.02, maxiter=10)
show("adam(10) after with_secondary(energy_weight=1e-2)", {"stop": r4.stop_reason, "J": r4.metrics["J"], "energy": r4.metrics["energy"]})

# 8. results are JSON-safe
json.dumps(r4.to_dict() if hasattr(r4, "to_dict") else {}, default=None)
print("\nresult JSON-safe: ok")
print(f"total wall time {time.perf_counter() - t0:.2f} s")
