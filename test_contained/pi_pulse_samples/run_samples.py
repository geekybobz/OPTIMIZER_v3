"""Run the pi-pulse sample tests PP1-PP8 and print a pass/fail table.

The tests have analytical answers; see REPORT.md for the model, the expected
values and the latest evidence. From ``stable/v4``:

    PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/optimizer_v4/bin/python -B \
        test_contained/pi_pulse_samples/run_samples.py

Exit status is 0 when every check passes. ``--json`` prints the rows as JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
V4 = HERE.parents[1]
sys.path.insert(0, str(V4))
sys.path.insert(0, str(HERE))

import optimizer as opt  # noqa: E402
from optimizer import guesses, optimizers, shooting, utils  # noqa: E402

import pi_pulse_olgs as G  # noqa: E402
import pi_pulse_olss as S  # noqa: E402

ALPHA, T, N = 0.01, 1.0, 100


def _soft_angle() -> float:
    """Root of sin(theta) = 4 alpha theta / T on (pi/2, pi), by bisection."""

    lo, hi = 0.5 * math.pi, math.pi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.sin(mid) - 4.0 * ALPHA * mid / T > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


THETA_SOFT = _soft_angle()
J_SOFT = math.cos(0.5 * THETA_SOFT) ** 2 + ALPHA * THETA_SOFT ** 2 / T
E_HARD = math.pi ** 2 / T
TIMES = (np.arange(N) + 0.5) * (T / N)

ROWS: list[dict[str, Any]] = []
VALUES: dict[tuple[str, str], float] = {}


def check(test: str, interface: str, name: str, ok: bool, observed: Any = None, expected: Any = None) -> None:
    ROWS.append({"test": test, "interface": interface, "check": name, "passed": bool(ok),
                 "observed": observed, "expected": expected})


def close(value: float, target: float, tol: float) -> bool:
    return abs(float(value) - float(target)) <= tol


def flatness(controls: Any) -> float:
    return float(np.std(controls.as_matrix(copy=False)[0]))


# --------------------------------------------------------------------------
# Python API
# --------------------------------------------------------------------------


def python_api() -> None:
    i = "python"
    for name, system in (("soft", G.soft()), ("hard", G.hard())):
        x = guesses.random_smooth_guess(system, amplitude=2.0, seed=3)
        opt.require_system(system)
        check("contract", i, f"olgs {name} verify_gradient", utils.verify_gradient(system, x)["passed"])
        if name == "hard":
            check("contract", i, "olgs hard verify_jacobian", utils.verify_jacobian(system, x)["passed"])
    for name, system in (("hard", S.hard()), ("soft", S.soft())):
        opt.system_olss.require_shooting_system(system)
        z = opt.ShootingVariables.from_flat(system.shooting_spec(), np.array([2.5]))
        report = shooting.verify_shooting_jacobian(system, z)
        check("contract", i, f"olss {name} verify_shooting_jacobian", report["passed"] and report["independent"],
              report["max_abs_error"])

    # PP1 gradient, soft
    g = G.soft()
    r1 = optimizers.adam(g, guesses.constant_guess(g, value=0.5), maxiter=100, step_size=0.05)
    r2 = optimizers.line_search(g, r1.controls, maxiter=60, step_size=50.0)
    VALUES[("PP1", i)] = r2.metrics["J"]
    check("PP1", i, "J = J*", close(r2.metrics["J"], J_SOFT, 1e-8), r2.metrics["J"], J_SOFT)
    check("PP1", i, "flat control", flatness(r2.controls) <= 1e-6, flatness(r2.controls), "<= 1e-6")

    # PP2 gradient, hard
    h = G.hard()
    r = optimizers.projected_gradient(h, guesses.gaussian_guess(h, amplitude=7.0), maxiter=30, step_size=25.0)
    VALUES[("PP2", i)] = r.metrics["energy"]
    check("PP2", i, "feasible", abs(r.metrics["c0_re"]) <= 1e-8, r.metrics["c0_re"], "<= 1e-8")
    check("PP2", i, "E = pi^2", close(r.metrics["energy"], E_HARD, 2e-7), r.metrics["energy"], E_HARD)
    check("PP2", i, "flat control", flatness(r.controls) <= 1e-6, flatness(r.controls), "<= 1e-6")

    # PP3 shooting, hard
    s = S.hard()
    seeds = shooting.random_shooting_seeds(s.shooting_spec(), 16, seed=1, radius=10.0)
    root = shooting.solve_shooting(s, seeds[0])
    check("PP3", i, "first seed qualifies", root.qualified and root.metrics["residual_norm"] <= 1e-10,
          root.metrics["residual_norm"], "<= 1e-10")
    ms = shooting.solve_shooting_multistart(s, seeds)
    best = ms.ranked[0]
    VALUES[("PP3", i)] = best.metrics["energy"]
    check("PP3", i, "best |z| = pi", close(abs(best.variables.flatten()[0]), math.pi, 1e-9),
          float(best.variables.flatten()[0]), "+-pi")
    check("PP3", i, "best E = pi^2", close(best.metrics["energy"], E_HARD, 1e-8), best.metrics["energy"], E_HARD)

    # PP4 shooting, soft
    ss = S.soft()
    ms = shooting.solve_shooting_multistart(ss, shooting.random_shooting_seeds(ss.shooting_spec(), 16, seed=1, radius=10.0))
    best = ms.ranked[0]
    roots = [float(r.variables.flatten()[0]) for r in ms.unique_qualified]
    VALUES[("PP4", i)] = best.metrics["soft_cost"]
    check("PP4", i, "best |z| = theta*", close(abs(best.variables.flatten()[0]), THETA_SOFT, 1e-9),
          float(best.variables.flatten()[0]), THETA_SOFT)
    check("PP4", i, "best J = J*", close(best.metrics["soft_cost"], J_SOFT, 1e-10), best.metrics["soft_cost"], J_SOFT)
    zero_rank = [k for k, r in enumerate(ms.ranked) if abs(r.variables.flatten()[0]) <= 1e-9]
    check("PP4", i, "z = 0 root qualifies and ranks lower", bool(zero_rank) and zero_rank[0] > 0,
          zero_rank, "rank > 0")

    # PP5 replay, hard and soft
    hard_root = shooting.solve_shooting(s, opt.ShootingVariables.from_flat(s.shooting_spec(), np.array([2.5])))
    soft_root = shooting.solve_shooting(ss, opt.ShootingVariables.from_flat(ss.shooting_spec(), np.array([2.5])))
    for label, system, root, target, mapping in (
        ("hard", s, hard_root, G.hard(), {"energy": "energy", "fidelity": "fidelity"}),
        ("soft", ss, soft_root, G.soft(), {"soft_cost": "J", "fidelity": "fidelity"}),
    ):
        report = shooting.compare_shooting_replay(system, root.variables, target, times=TIMES, metric_map=mapping)
        worst = max(c["absolute_difference"] for c in report["comparisons"])
        check("PP5", i, f"{label} replay", worst <= 1e-10, worst, "<= 1e-10")

    # PP6 traps: current behavior
    zero = optimizers.adam(g, guesses.constant_guess(g, value=0.0), maxiter=10, step_size=0.05)
    check("PP6", i, "zero start reports gradient_converged at J = 1",
          zero.stop_reason == "gradient_converged" and close(zero.metrics["J"], 1.0, 1e-12), zero.stop_reason)
    probe = utils.curvature_probe(g, zero.controls)
    check("PP6", i, "zero start curvature is negative", bool(probe["negative_curvature"]), probe["minimum_curvature"])
    weak = optimizers.projected_gradient(h, guesses.constant_guess(h, value=0.5), maxiter=100, step_size=25.0)
    check("PP6", i, "weak hard start lands on E = 9 pi^2", close(weak.metrics["energy"], 9 * E_HARD, 1e-6),
          weak.metrics["energy"], 9 * E_HARD)
    for label, system, target_z in (("hard", s, 3 * math.pi), ("soft", ss, 0.0)):
        res = shooting.solve_shooting(system, opt.ShootingVariables.from_flat(system.shooting_spec(), np.array([0.5])))
        z = float(res.variables.flatten()[0])
        check("PP6", i, f"{label} shooting from z0 = 0.5 lands on z = {target_z:.6g}",
              res.qualified and close(z, target_z, 1e-9), z, target_z)


# --------------------------------------------------------------------------
# Operation service (the layer the terminal and MCP call)
# --------------------------------------------------------------------------


def service_api() -> None:
    i = "service"
    service = opt.OperationService()

    def call(op: str, arguments: dict[str, Any] | None = None, **kw: Any):
        return service.execute(opt.OperationRequest(op, dict(arguments or {}), **kw))

    def head(context: str) -> str | None:
        return service.session._contexts[context].head

    def ref(response, *path):
        return opt.ObjectRef(response.object_refs[0].object_id, path=tuple(path))

    olgs, olss = str(HERE / "pi_pulse_olgs.py"), str(HERE / "pi_pulse_olss.py")

    # PP1
    call("session.load", {"name": "g_soft", "factory": olgs, "factory_name": "soft"})
    call("guesses.constant_guess", {"value": 0.5}, activate=True)
    r1 = call("optimizers.adam", {"maxiter": 100, "step_size": 0.05})
    before = head("g_soft")
    call("utils.diagnose_run", {"result": ref(r1)})
    call("utils.curvature_probe", {})
    check("PP1", i, "inspect calls keep the head", head("g_soft") == before)
    r2 = call("optimizers.line_search", {"maxiter": 60, "step_size": 50.0})
    after_r2 = head("g_soft")
    call("optimizers.adam", {"maxiter": 5, "step_size": 0.05}, activate=False)
    check("PP1", i, "activate=False keeps the head", head("g_soft") == after_r2)
    J = r2.result.metrics["J"]
    VALUES[("PP1", i)] = J
    check("PP1", i, "J = J*", r2.ok and close(J, J_SOFT, 1e-8), J, J_SOFT)

    # PP2
    call("session.load", {"name": "g_hard", "factory": olgs, "factory_name": "hard"})
    call("guesses.gaussian_guess", {"amplitude": 7.0}, activate=True)
    r = call("optimizers.projected_gradient", {"maxiter": 30, "step_size": 25.0})
    VALUES[("PP2", i)] = r.result.metrics["energy"]
    check("PP2", i, "feasible and E = pi^2", abs(r.result.metrics["c0_re"]) <= 1e-8
          and close(r.result.metrics["energy"], E_HARD, 2e-7), r.result.metrics["energy"], E_HARD)

    # PP3
    call("session.load", {"name": "s_hard", "factory": olss, "factory_name": "hard"})
    seeds = call("shooting.random_shooting_seeds", {"count": 16, "seed": 1, "radius": 10.0})
    call("session.use", {"value": ref(seeds, 0)})
    root = call("shooting.solve_shooting", {})
    check("PP3", i, "first seed qualifies", root.ok and root.result.qualified)
    ms = call("shooting.solve_shooting_multistart", {"starts": ref(seeds)})
    best = ms.result.ranked[0]
    VALUES[("PP3", i)] = best.metrics["energy"]
    check("PP3", i, "best |z| = pi, E = pi^2", close(abs(best.variables.flatten()[0]), math.pi, 1e-9)
          and close(best.metrics["energy"], E_HARD, 1e-8), best.metrics["energy"], E_HARD)
    activated = call("session.use", {"value": opt.ObjectRef(ms.object_refs[0].object_id, path=("ranked", 0))})
    check("PP3", i, "best root can be activated", activated.ok)

    # PP4
    call("session.load", {"name": "s_soft", "factory": olss, "factory_name": "soft"})
    seeds_soft = call("shooting.random_shooting_seeds", {"count": 16, "seed": 1, "radius": 10.0})
    ms = call("shooting.solve_shooting_multistart", {"starts": ref(seeds_soft)})
    best = ms.result.ranked[0]
    VALUES[("PP4", i)] = best.metrics["soft_cost"]
    check("PP4", i, "best J = J*", close(best.metrics["soft_cost"], J_SOFT, 1e-10), best.metrics["soft_cost"], J_SOFT)

    # PP5, hard replay with the shooting context selected
    call("session.select", {"name": "s_hard"})
    replay = call("shooting.compare_shooting_replay", {
        "shooting_system": "s_hard", "gradient_system": "g_hard", "times": TIMES.tolist(),
        "metric_map": {"energy": "energy", "fidelity": "fidelity"}})
    worst = max(c["absolute_difference"] for c in replay.result["comparisons"]) if replay.ok else math.inf
    check("PP5", i, "hard replay", worst <= 1e-10, worst, "<= 1e-10")

    # PP7, error codes; the session state must not change
    cases = (
        ("optimizers.adam", {"maxiter": 5}, "s_hard", "wrong_family"),
        ("shooting.solve_shooting", {"maxiter": 3}, "g_soft", "wrong_family"),
        ("optimizers.projected_gradient", {"maxiter": 5}, "g_soft", "capability_missing"),
        ("optimizers.adam", {"maxiterr": 5}, "g_soft", "invalid_arguments"),
        ("session.load", {"name": "g_soft", "factory": olgs, "factory_name": "soft"}, None, "context_exists"),
        ("session.use", {"value": opt.ObjectRef("0" * 32)}, None, "missing_object"),
        ("session.use", {"value": ref(seeds, 0)}, "g_soft", "incompatible_layout"),
    )
    for op, arguments, context, code in cases:
        response = call(op, arguments, context=context)
        error = response.error or {}
        unchanged = response.state_before == response.state_after
        check("PP7", i, f"{op} -> {code}", (not response.ok) and error.get("code") == code and unchanged,
              f"{error.get('code')}: {str(error.get('message'))[:70]}", code)


# --------------------------------------------------------------------------
# Terminal (subprocess, JSON output)
# --------------------------------------------------------------------------


def terminal_run(*, script: str | None = None, command: str | None = None) -> tuple[int, list[dict[str, Any]]]:
    argv = [sys.executable, "-B", "-m", "optimizer.cli", "--json"]
    argv += ["--script", str(HERE / "scripts" / script)] if script else ["--command", str(command)]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join(filter(None, (str(V4), os.environ.get("PYTHONPATH"))))}
    done = subprocess.run(argv, cwd=HERE, env=env, capture_output=True, text=True, timeout=900)
    responses = [json.loads(line) for line in done.stdout.splitlines() if line.lstrip().startswith("{")]
    return done.returncode, responses


def _summary(response: dict[str, Any]) -> Any:
    return (response.get("metadata") or {}).get("summary")


def _metrics(response: dict[str, Any]) -> dict[str, Any]:
    summary = _summary(response) or {}
    if "metrics" in summary:
        return summary["metrics"]
    fields = summary.get("fields") or {}
    return fields.get("metrics") or summary


def terminal_api() -> None:
    i = "terminal"

    code, out = terminal_run(script="pp1_gradient_soft.opt")
    ok = code == 0 and all(r["status"] == "ok" for r in out)
    search = next(r for r in out if r["operation"] == "optimizers.line_search")
    J = _metrics(search)["J"]
    VALUES[("PP1", i)] = J
    check("PP1", i, "script runs; J = J*", ok and close(J, J_SOFT, 1e-8), J, J_SOFT)
    check("PP1", i, "head unchanged by inspect and candidate runs", _metrics(out[-1])["J"] == J, _metrics(out[-1])["J"], J)

    code, out = terminal_run(script="pp2_gradient_hard.opt")
    m = _metrics(out[-1])
    VALUES[("PP2", i)] = m["energy"]
    check("PP2", i, "script runs; feasible and E = pi^2", code == 0 and abs(m["c0_re"]) <= 1e-8
          and close(m["energy"], E_HARD, 2e-7), m["energy"], E_HARD)

    code, out = terminal_run(script="pp3_shooting_hard.opt")
    m = _metrics(out[-1])
    VALUES[("PP3", i)] = m["energy"]
    check("PP3", i, "script runs; best E = pi^2", code == 0 and close(m["energy"], E_HARD, 1e-8), m["energy"], E_HARD)

    code, out = terminal_run(script="pp4_shooting_soft.opt")
    best = _metrics(next(r for r in out if r["operation"] == "inspect"))
    low = _metrics(out[-1])
    VALUES[("PP4", i)] = best["soft_cost"]
    check("PP4", i, "script runs; best J = J*", code == 0 and close(best["soft_cost"], J_SOFT, 1e-10),
          best["soft_cost"], J_SOFT)
    check("PP4", i, "seed 3 gives a qualified root with higher cost",
          low["residual_norm"] <= 1e-10 and low["soft_cost"] > J_SOFT + 1.0, low["soft_cost"])

    code, out = terminal_run(script="pp6_traps.opt")
    adam = next(r for r in out if r["operation"] == "optimizers.adam")
    curvature = _summary(next(r for r in out if r["operation"] == "utils.curvature_probe"))
    weak = _metrics(out[-1])
    check("PP6", i, "zero start reports gradient_converged at J = 1",
          _summary(adam)["stop_reason"] == "gradient_converged" and _metrics(adam)["J"] == 1.0,
          _summary(adam)["stop_reason"])
    check("PP6", i, "zero start curvature is negative", bool(curvature["negative_curvature"]),
          curvature["minimum_curvature"])
    check("PP6", i, "weak hard start lands on E = 9 pi^2", close(weak["energy"], 9 * E_HARD, 1e-6),
          weak["energy"], 9 * E_HARD)

    load_s = 'opt.session.load(name="s", factory="./pi_pulse_olss.py", factory_name="hard"); '
    load_g = 'opt.session.load(name="g", factory="./pi_pulse_olgs.py", factory_name="soft"); '
    cases = (
        (load_s + "opt.optimizers.adam(maxiter=5);", "wrong_family"),
        (load_g + "opt.shooting.solve_shooting(maxiter=3);", "wrong_family"),
        (load_g + "opt.optimizers.projected_gradient(maxiter=5);", "capability_missing"),
        (load_g + "opt.optimizers.adam(maxiterr=5);", "invalid_arguments"),
        (load_g + load_g, "context_exists"),
        (load_s + 'seeds = opt.shooting.random_shooting_seeds(count=2, seed=1); ' + load_g
         + "opt.session.use(value=seeds[0]);", "incompatible_layout"),
        ("import os;", None),
        ("opt.optimizers.adm(maxiter=5);", None),
        ('opt.session.load(name="g", factory="./pi_pulse_olgs.py") opt.evaluate();', None),
    )
    def meaningful(state: dict[str, Any] | None) -> dict[str, Any]:
        # Contexts, heads, bindings and revision; byte accounting is reported separately.
        return {k: v for k, v in (state or {}).items() if k != "retained_bytes"}

    for command, expected in cases:
        code, out = terminal_run(command=command)
        failed = [r for r in out if r["status"] != "ok"]
        last = failed[-1] if failed else {}
        got = (last.get("error") or {}).get("code")
        unchanged = bool(failed) and meaningful(last.get("state_before")) == meaningful(last.get("state_after"))
        passed = len(failed) == 1 and failed[0] is out[-1] and unchanged and (expected is None or got == expected)
        note = f"{got}; exit {code}"
        before, after = ((last.get(k) or {}).get("retained_bytes") for k in ("state_before", "state_after"))
        if before != after:
            note += f"; retained_bytes {before} -> {after}"
        check("PP7", i, f"{command[-48:]} -> {expected or 'parse/validation error'}", passed, note, expected)


def parity() -> None:
    for test in ("PP1", "PP2", "PP3", "PP4"):
        values = {k[1]: v for k, v in VALUES.items() if k[0] == test}
        if len(values) < 2:
            continue
        spread = max(values.values()) - min(values.values())
        scale = max(1.0, max(abs(v) for v in values.values()))
        check("PP8", "+".join(sorted(values)), f"{test} values agree", spread <= 1e-12 * scale, spread, "<= 1e-12 rel")
    check("PP8", "mcp", "MCP adapter", False, "not built", "pending")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="Print rows as JSON.")
    options = parser.parse_args()
    python_api()
    service_api()
    terminal_api()
    parity()
    pending = [row for row in ROWS if row["expected"] == "pending"]
    failed = [row for row in ROWS if not row["passed"] and row not in pending]
    if options.json:
        print(json.dumps({"rows": ROWS, "failed": len(failed), "pending": len(pending)}, default=str, indent=1))
    else:
        for row in ROWS:
            status = "PEND" if row in pending else ("PASS" if row["passed"] else "FAIL")
            observed = "" if row["observed"] is None else f"  [{row['observed']}]"
            print(f"{status}  {row['test']:8s} {row['interface']:24s} {row['check']}{observed}")
        print(f"\n{len(ROWS) - len(failed) - len(pending)} passed, {len(failed)} failed, {len(pending)} pending")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
