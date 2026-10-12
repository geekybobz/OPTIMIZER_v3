"""Structural probes on the qubit OLGS system: bounds, schedules, custom update,
failed trials, transactions, and diagnosis from compact numerical aggregates.

Probes (e) and (f) show the transaction and diagnosis fixes recorded in ../REPORT.md
(issues T-1 to T-3 and G-2); before the fixes they failed.
Run from the V4 checkout root:

    PYTHONPATH=.:test_contained/review_2026_10_12/qubit_olgs python -B \
        test_contained/review_2026_10_12/qubit_olgs/probes.py
"""

import json

import numpy as np

import optimizer as opt
from optimizer import ControlBounds, Controls
from system import QubitTransfer


def line(label, value):
    print(f"{label:58s} {value}")


system = QubitTransfer()
u0 = opt.guesses.random_fourier_guess(system, amplitude=0.5, modes=4, seed=3)


# (a) declared bounds honored by standard methods
class BoundedQubit(QubitTransfer):
    def control_bounds(self):
        spec = self.control_spec()
        return ControlBounds(Controls.constant(spec, -0.6), Controls.constant(spec, 0.6))


bounded = BoundedQubit()
for name in ("adam", "momentum", "line_search", "lbfgs", "nonlinear_cg"):
    kwargs = {"step_size": 0.2} if name in ("adam", "momentum") else {}
    res = getattr(opt, name)(bounded, u0 * 3.0, maxiter=15, **kwargs)
    line(f"(a) bounds |u|<=0.6, {name}: max|u| / stop", (round(float(np.max(np.abs(res.controls.as_matrix()))), 6), res.stop_reason))

# (b) schedules inside Adam and momentum
for schedule in (opt.CosineDecaySchedule(0.05, 20), opt.ExponentialDecaySchedule(0.05, 0.9), opt.LinearWarmupSchedule(0.005, 0.05, 5)):
    res = opt.adam(system, u0, step_size=schedule, maxiter=6)
    line(f"(b) adam + {type(schedule).__name__}: final step size", res.state.optimizer_state.get("step_size"))
res_m = opt.momentum(system, u0, step_size=opt.CosineDecaySchedule(0.05, 20), maxiter=4)
line("(b) momentum + CosineDecaySchedule: stop", res_m.stop_reason)


# (c) the user's own update rule through run_chunk (forward -> backward -> gradient -> update)
def plain_gradient_descent(ctx):
    return ctx.state.controls - 0.5 * ctx.gradient


res_c = opt.run_chunk(system, u0, step=plain_gradient_descent, optimizer_name="my_gd", maxiter=10)
line("(c) run_chunk(custom step): J start -> end, stop", (round(opt.evaluate(system, u0).metrics["J"], 5), round(res_c.metrics["J"], 5), res_c.stop_reason))
line("(c) custom step result type / diagnosable", (type(res_c).__name__, opt.diagnose_run(res_c)["classification"]))


# (d) a trial that fails (non-finite J) is rejected, not fatal
class FragileQubit(QubitTransfer):
    def evaluate(self, controls):
        if np.max(np.abs(controls.as_matrix())) > 2.0:
            return {"J": float("nan"), "fidelity": 0.0, "infidelity": 1.0, "energy": 0.0}
        return super().evaluate(controls)


res_d = opt.adam(FragileQubit(), u0, step_size=3.0, maxiter=8)
line("(d) adam with failing trials: stop / rejected attempts", (res_d.stop_reason, res_d.diagnostics["rejected"]))

# (e) transaction adapter: direct objects, JSON-like payload, invalid output
ok = opt.transact(system, opt.SystemRequest("evaluate", {"controls": u0}))
line("(e) transact evaluate(Controls): ok / J", (ok.ok, round(ok.result["J"], 6)))
grad = opt.transact(system, {"operation": "gradient", "payload": {"controls": u0}})
line("(e) transact gradient: ok / JSON-safe", (grad.ok, isinstance(json.dumps(grad.to_dict()), str)))
as_json = json.loads(json.dumps(opt.SystemRequest("evaluate", {"controls": u0}).to_dict()))
back = opt.transact(system, {"operation": as_json["operation"], "payload": as_json["payload"]})
line("(e) transact with a JSON round-tripped request: ok / error", (back.ok, (back.error or {}).get("message", "")[:70]))


class BrokenQubit(QubitTransfer):
    def evaluate(self, controls):
        return {"J": float("nan"), "fidelity": 2.0, "infidelity": -1.0, "energy": 0.0, "undeclared": 1}


bad = opt.transact(BrokenQubit(), opt.SystemRequest("evaluate", {"controls": u0}))
line("(e) transact on NaN J + undeclared metric: ok?", (bad.ok, bad.result))
try:
    opt.evaluate(BrokenQubit(), u0)
    line("(e) opt.evaluate on the same broken system", "accepted")
except Exception as exc:  # noqa: BLE001
    line("(e) opt.evaluate on the same broken system", f"rejected: {type(exc).__name__}")
vg = opt.transact(system, opt.SystemRequest("value_and_gradient", {"controls": u0}))
line("(e) transact value_and_gradient (optional hook)", (vg.ok, (vg.error or {}).get("message", "")))
ws = opt.transact(system, opt.SystemRequest("with_secondary", {"energy_weight": 0.5}))
line("(e) transact with_secondary: result type", type(ws.result).__name__)

# (f) diagnosis reads compact numerical aggregates, with no logger
res_f = opt.adam(system, u0, step_size=0.05, maxiter=5)
line("(f) adam -> diagnose_run family/class", (opt.diagnose_run(res_f)["family"], opt.diagnose_run(res_f)["classification"]))
