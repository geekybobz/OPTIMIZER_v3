"""Run the OLGS convergence matrix, workflow checks, physical studies, and stress cases.

Run from the V4 checkout with bytecode disabled:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:test_contained/olgs_convergence \
        python -B test_contained/olgs_convergence/run_matrix.py --json OUT --workers 8

Every number is numerical evidence for the declared configuration only; nothing
here ranks optimizers universally or certifies a physical result.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import optimizer as opt
import scenarios as S
from optimizer.utils.parallel import ParallelConfig, parallel_map

HERE = Path(__file__).resolve().parent
CONTAINED = HERE.parent

FIRST_ORDER = {
    "ls_fixed": ("line_search", {"variant": "fixed"}),
    "ls_normalized": ("line_search", {"variant": "normalized"}),
    "ls_backtracking": ("line_search", {"variant": "backtracking"}),
    "ls_armijo": ("line_search", {"variant": "armijo"}),
    "mom_heavy_ball": ("momentum", {"variant": "heavy_ball"}),
    "mom_nesterov": ("momentum", {"variant": "nesterov"}),
    "mom_restart": ("momentum", {"variant": "restart"}),
    "mom_clipped": ("momentum", {"variant": "clipped"}),
    "adam": ("adam", {"variant": "adam"}),
    "amsgrad": ("adam", {"variant": "amsgrad"}),
    "adamw": ("adam", {"variant": "adamw"}),
    "radam": ("adam", {"variant": "radam"}),
    "adabelief": ("adam", {"variant": "adabelief"}),
    "adagrad": ("adagrad", {}),
    "rmsprop": ("rmsprop", {}),
}
CURVATURE = {
    "lbfgs": ("lbfgs", {}),
    "ncg_fr": ("nonlinear_cg", {"variant": "fletcher_reeves"}),
    "ncg_pr": ("nonlinear_cg", {"variant": "polak_ribiere"}),
    "ncg_prplus": ("nonlinear_cg", {"variant": "polak_ribiere_plus"}),
    "ncg_hs": ("nonlinear_cg", {"variant": "hestenes_stiefel"}),
    "newton_cg": ("newton_cg", {}),
    "trust_region": ("trust_region", {}),
}
TUNED_STEP = 0.05  # README-style step for first-order methods

UNCONSTRAINED = [
    ("S0", "IsotropicQuadratic", {}, "zeros", 300),
    ("S0max", "IsotropicQuadraticMax", {}, "zeros", 300),
    ("S1-1e2", "RotatedQuadratic", {"condition": 1e2}, "zeros", 1000),
    ("S1-1e4", "RotatedQuadratic", {"condition": 1e4}, "zeros", 1000),
    ("S1-1e6", "RotatedQuadratic", {"condition": 1e6}, "zeros", 1000),
    ("S2", "ExtendedRosenbrock", {}, "rosen", 3000),
    ("S3-near0", "DoubleWell", {}, "near_zero", 500),
    ("S3-rand", "DoubleWell", {}, "uniform2", 500),
    ("S4-N11", "ChannelScale", {"N": 11}, "zeros", 1000),
    ("S4-N101", "ChannelScale", {"N": 101}, "zeros", 1000),
    ("S4-N1001", "ChannelScale", {"N": 1001}, "zeros", 1000),
]
CONSTRAINED = [
    ("S5a-box", "BoxQuadratic", {}, "zeros", 300),
    ("S5b-linear", "LinearEqualityQuadratic", {}, "zeros", 300),
    ("S5c-eq+bounds", "SimplexQuadratic", {}, "zeros", 300),
    ("S5d-sphere", "SphereQuadratic", {}, "ones_scaled", 300),
    ("S5e-inconsistent", "InconsistentEquality", {}, "zeros", 100),
]


def initial(system, start: str) -> np.ndarray:
    n = system.size
    if start == "zeros":
        return np.zeros(n)
    if start == "rosen":
        return system.start()
    if start == "near_zero":
        return 0.05 * np.ones(n)
    if start == "uniform2":
        return np.random.default_rng(11).uniform(-2.0, 2.0, n)
    if start == "ones_scaled":
        return 0.3 * np.ones(n)
    raise ValueError(start)


def _counters(result) -> dict:
    records = list(getattr(result.trace, "iteration_records", []) or []) if result.trace is not None else []
    if not records:
        return {"accepted": 0, "rejected": 0, "counters": {}}
    return {
        "accepted": sum(1 for r in records if r.accepted is True),
        "rejected": sum(1 for r in records if r.accepted is False),
        "counters": dict(records[-1].technical.get("counters", {})),
    }


def assess(system, x0: np.ndarray, x: np.ndarray) -> dict:
    """Oracle-based evidence in minimization form (f), independent of the optimizer."""

    oracle = system.oracle()
    f0, f1 = system.f(x0), system.f(x)
    out = {"f0": f0, "f": f1, "grad_norm": float(np.linalg.norm(system.g(x)))}
    if oracle["J_star"] is not None:
        out["gap"] = f1 - oracle["J_star"]
    if oracle["x_star"] is not None:
        out["dist"] = float(np.linalg.norm(x - oracle["x_star"]))
        out["converged"] = out["dist"] <= 1e-4 * max(1.0, float(np.linalg.norm(oracle["x_star"])))
        if np.any(oracle["x_star"] == 0.0):
            out["exact_zero_support_match"] = bool(np.array_equal(x == 0.0, oracle["x_star"] == 0.0))
    elif oracle["kind"].startswith("local"):
        eig = system.hessian_eigenvalues(x)
        out["min_hessian_eig"] = float(eig.min())
        out["converged"] = bool(np.all(np.abs(np.abs(x) - 1.0) < 1e-4) and eig.min() > 0)
        out["outcome"] = "local minimum" if out["converged"] else ("near saddle/maximum" if np.any(np.abs(x) < 0.5) else "not stationary")
    if hasattr(system, "residuals"):
        out["feasibility"] = float(np.linalg.norm(system.residuals(system.controls(x))))
    if hasattr(system, "control_bounds"):
        lo, hi = system.control_bounds()
        out["bound_violation"] = float(max(0.0, np.max(lo.reshape(-1) - x), np.max(x - hi.reshape(-1))))
    return out


def run_job(job: dict) -> dict:
    system = getattr(S, job["system"])(job["primary"])
    x0 = initial(system, job["start"])
    fn = getattr(opt.optimizers, job["family"])
    row = {k: job[k] for k in ("scenario", "config", "family")}
    t0 = time.perf_counter()
    try:
        result = fn(system, system.controls(x0), maxiter=job["maxiter"], **job["options"])
    except Exception as exc:  # noqa: BLE001
        row.update({"error": f"{type(exc).__name__}: {str(exc)[:160]}", "seconds": time.perf_counter() - t0})
        return row
    row["seconds"] = time.perf_counter() - t0
    x = result.controls.flatten()
    row.update({"stop": result.stop_reason, "iterations": result.iterations, "finite": bool(np.all(np.isfinite(x)))})
    row.update(_counters(result))
    row.update(assess(system, x0, x))
    try:
        json.dumps(result.to_dict(), default=str)
        row["serializable"] = True
    except Exception:  # noqa: BLE001
        row["serializable"] = False
    return row


def matrix_jobs() -> list[dict]:
    jobs = []
    for scenario, system, primary, start, maxiter in UNCONSTRAINED:
        for name, (family, options) in FIRST_ORDER.items():
            for label, extra in (("default", {}), ("step0.05", {"step_size": TUNED_STEP})):
                jobs.append({"scenario": scenario, "system": system, "primary": primary, "start": start, "maxiter": maxiter,
                             "config": f"{name}/{label}", "family": family, "options": {**options, **extra}})
        for name, (family, options) in CURVATURE.items():
            jobs.append({"scenario": scenario, "system": system, "primary": primary, "start": start, "maxiter": maxiter,
                         "config": name, "family": family, "options": dict(options)})
    for scenario, system, primary, start, maxiter in CONSTRAINED:
        jobs.append({"scenario": scenario, "system": system, "primary": primary, "start": start, "maxiter": maxiter,
                     "config": "projected_gradient", "family": "projected_gradient", "options": {}})
    for config, family, options in (("proximal_gradient", "proximal_gradient", {}), ("lbfgs (smooth method on L1)", "lbfgs", {}),
                                    ("adam/step0.05 (smooth method on L1)", "adam", {"step_size": TUNED_STEP})):
        jobs.append({"scenario": "S6-l1", "system": "LassoIdentity", "primary": {}, "start": "zeros", "maxiter": 500,
                     "config": config, "family": family, "options": options})
    for config, family, options in (("adam/step0.05", "adam", {"step_size": TUNED_STEP}), ("lbfgs", "lbfgs", {}),
                                    ("ls_armijo", "line_search", {"variant": "armijo"}), ("ncg_prplus", "nonlinear_cg", {}),
                                    ("mom_heavy_ball/step0.05", "momentum", {"step_size": TUNED_STEP})):
        jobs.append({"scenario": "X2-domain", "system": "RestrictedDomain", "primary": {}, "start": "zeros", "maxiter": 500,
                     "config": config, "family": family, "options": options})
        jobs.append({"scenario": "X3-bad-gradient", "system": "InconsistentGradient", "primary": {}, "start": "zeros", "maxiter": 500,
                     "config": config, "family": family, "options": options})
    return jobs


# --- S7 workflows --------------------------------------------------------------

def workflows() -> list[dict]:
    out = []
    ros = S.ExtendedRosenbrock()
    x0 = ros.controls(ros.start())

    a = opt.optimizers.adam(ros, x0, step_size=0.01, maxiter=300)
    warm = a.warmstart(target_optimizer="lbfgs")
    b = opt.optimizers.lbfgs(ros, warmstart=warm, maxiter=500)
    out.append({"workflow": "Adam -> L-BFGS (S2)", "handoff_J_equal": bool(np.isclose(a.metrics["J"], ros.evaluate(warm.controls)["J"])),
                "private_state_dropped": not bool(warm.optimizer_state), "final_dist": float(np.linalg.norm(b.controls.flatten() - 1.0)),
                "stop": b.stop_reason})

    rq = S.RotatedQuadratic({"condition": 1e4})
    z = rq.controls(np.zeros(rq.size))
    l = opt.optimizers.line_search(rq, z, variant="armijo", maxiter=100)
    m = opt.optimizers.lbfgs(rq, warmstart=l.warmstart(target_optimizer="lbfgs"), maxiter=400)
    out.append({"workflow": "line search -> L-BFGS (S1 cond 1e4)", "final_dist": float(np.linalg.norm(m.controls.flatten() - rq.c)), "stop": m.stop_reason})

    t = opt.optimizers.trust_region(ros, warmstart=a.warmstart(target_optimizer="trust_region"), maxiter=200)
    out.append({"workflow": "Adam -> trust region (S2, HVP)", "final_dist": float(np.linalg.norm(t.controls.flatten() - 1.0)), "stop": t.stop_reason})

    q = S.IsotropicQuadratic()
    zq = q.controls(np.zeros(q.size))
    one = opt.optimizers.adam(q, zq, step_size=0.01, maxiter=200)
    first = opt.optimizers.adam(q, zq, step_size=0.01, maxiter=100)
    second = opt.optimizers.adam(q, state=first.state, maxiter=100)
    out.append({"workflow": "Adam 100 + Adam 100 (same-method state) vs Adam 200",
                "max_diff": float(np.max(np.abs(second.controls.flatten() - one.controls.flatten()))),
                "global_iteration": getattr(second.state, "global_iteration", None)})

    lasso = S.LassoIdentity({}, {"lam": 2.0})
    stages = [opt.ContinuationStage(label=f"lam={lam}", optimizer="proximal_gradient", secondary_updates={"lam": lam},
                                    optimizer_options={"maxiter": 200}) for lam in (2.0, 1.0, 0.5)]
    try:
        cont = opt.continuation(lasso, lasso.controls(np.zeros(lasso.size)), stages=stages)
        final_sys = S.LassoIdentity({}, {"lam": 0.5})
        out.append({"workflow": "continuation lam 2 -> 1 -> 0.5 (S6, prox)", "stop": cont.stop_reason if hasattr(cont, "stop_reason") else "?",
                    "final_dist": float(np.linalg.norm(cont.controls.flatten() - final_sys.oracle()["x_star"])),
                    "stage_J": [round(float(sr.result.J), 6) for sr in cont.stages] if hasattr(cont, "stages") else None})
    except Exception as exc:  # noqa: BLE001
        out.append({"workflow": "continuation lam 2 -> 1 -> 0.5 (S6, prox)", "error": f"{type(exc).__name__}: {exc}"})

    coarse = S.ChannelScale({"N": 101})
    xc = opt.optimizers.lbfgs(coarse, coarse.controls(np.zeros(coarse.size)), maxiter=300)
    fine_factory = lambda _sys: S.ChannelScale({"N": 1001})  # noqa: E731

    def interpolate(prev, nxt, controls):
        old = controls.as_matrix()
        t_old, t_new = np.linspace(0, 1, prev.N), np.linspace(0, 1, nxt.N)
        return opt.Controls.from_matrix(nxt.control_spec(), np.vstack([np.interp(t_new, t_old, row) for row in old]))

    try:
        opt.continuation(coarse, xc.controls, stages=[opt.ContinuationStage(label="fine", optimizer="lbfgs", system_factory=fine_factory,
                                                                             optimizer_options={"maxiter": 10})])
        refused = False
    except ValueError:
        refused = True
    fine = opt.continuation(coarse, xc.controls, stages=[opt.ContinuationStage(label="fine", optimizer="lbfgs", system_factory=fine_factory,
                                                                                control_transfer=interpolate, optimizer_options={"maxiter": 300})])
    fs = S.ChannelScale({"N": 1001})
    out.append({"workflow": "grid transfer N=101 -> 1001 (S4)", "refused_without_transfer": refused,
                "final_dist": float(np.linalg.norm(fine.controls.flatten() - fs.target))})
    return out


# --- S8/S9 physical studies ------------------------------------------------------

def physical() -> list[dict]:
    rows = []
    sys.path.insert(0, str(CONTAINED / "ho_purification"))
    from system import HOPurificationTrackerSystem  # noqa: E402
    sys.path.pop(0)
    sys.modules.pop("system", None)
    sys.path.insert(0, str(CONTAINED / "second_order_pmp"))
    from system import SecondOrderPMPSystem  # noqa: E402
    sys.path.pop(0)

    cases = [
        ("S8 HO purification (N=24)", HOPurificationTrackerSystem({"N": 24}), dict(amplitude=1.0, modes=4, seed=1)),
        ("S9 robust second order (N=201)", SecondOrderPMPSystem({"N": 201}), dict(amplitude=3.0, modes=5, seed=2)),
    ]
    for label, system, guess in cases:
        u0 = opt.guesses.random_fourier_guess(system, **guess)
        check = opt.utils.verify_gradient(system, u0, directions=6, eps=1e-6)
        plans = {
            "ls_armijo": lambda u: opt.optimizers.line_search(system, u, variant="armijo", maxiter=300),
            "adam/step0.05": lambda u: opt.optimizers.adam(system, u, step_size=0.05, maxiter=300),
            "lbfgs": lambda u: opt.optimizers.lbfgs(system, u, maxiter=300),
            "adam(100) -> lbfgs(200)": lambda u: opt.optimizers.lbfgs(
                system, warmstart=opt.optimizers.adam(system, u, step_size=0.05, maxiter=100).warmstart(target_optimizer="lbfgs"), maxiter=200),
        }
        J0 = system.evaluate(u0)["J"]
        for name, run in plans.items():
            t0 = time.perf_counter()
            res = run(u0)
            dt = time.perf_counter() - t0
            m = res.metrics
            g = opt.gradient(system, res.controls).norm()
            row = {"case": label, "method": name, "gradient_check": bool(check.get("passed")), "J0": J0, "J": m["J"],
                   "grad_norm": float(g), "stop": res.stop_reason, "iterations": res.iterations, "seconds": round(dt, 2)}
            for key in ("fidelity", "robustness_penalty", "control_energy"):
                if key in m:
                    row[key] = m[key]
            rows.append(row)
    return rows


# --- extra checks ----------------------------------------------------------------

def large_dimension() -> list[dict]:
    big = S.LargeChannelScale()
    x0 = big.controls(np.zeros(big.size))
    rows = []
    for name, call in {
        "adam/step0.05": lambda: opt.optimizers.adam(big, x0, step_size=0.05, maxiter=20),
        "lbfgs": lambda: opt.optimizers.lbfgs(big, x0, maxiter=20),
        "ncg_prplus": lambda: opt.optimizers.nonlinear_cg(big, x0, maxiter=20),
        "ls_armijo": lambda: opt.optimizers.line_search(big, x0, variant="armijo", maxiter=20),
        "newton_cg": lambda: opt.optimizers.newton_cg(big, x0, maxiter=5),
    }.items():
        t0 = time.perf_counter()
        res = call()
        dt = time.perf_counter() - t0
        rows.append({"method": name, "variables": big.size, "iterations": res.iterations,
                     "ms_per_iteration": round(1e3 * dt / max(1, res.iterations), 1), "stop": res.stop_reason})
    return rows


def campaign_check(root: Path) -> list[dict]:
    rows = []
    system = S.ExtendedRosenbrock()
    guesses = opt.campaigns.random_smooth_guesses(system, 8, amplitude=1.0, seed=7)
    for backend in ("serial", "thread", "process"):
        camp = opt.campaigns.create(system, root / backend, system_factory=S.ExtendedRosenbrock)
        t0 = time.perf_counter()
        try:
            mode = camp.scout(guesses, method=opt.campaigns.MethodSpec("lbfgs", {"maxiter": 2000}),
                              parallel=None if backend == "serial" else ParallelConfig(backend=backend, workers=4))
            dt = time.perf_counter() - t0
            text = json.dumps(getattr(mode, "to_dict", lambda: {})(), default=str)
            rows.append({"backend": backend, "seconds": round(dt, 2), "ok": True, "summary_bytes": len(text)})
        except Exception as exc:  # noqa: BLE001
            rows.append({"backend": backend, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    t0 = time.perf_counter()
    jobs = matrix_jobs()
    matrix = parallel_map(run_job, jobs, config=ParallelConfig(backend="process", workers=args.workers))
    t_matrix = time.perf_counter() - t0
    report = {"matrix": matrix, "matrix_seconds": round(t_matrix, 1), "jobs": len(jobs)}
    report["workflows"] = workflows()
    report["physical"] = physical()
    report["large_dimension"] = large_dimension()
    with tempfile.TemporaryDirectory() as tmp:
        report["campaigns"] = campaign_check(Path(tmp))
    report["seconds"] = round(time.perf_counter() - t0, 1)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str))
    print(f"{len(jobs)} matrix jobs in {t_matrix:.1f}s; total {report['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
