---
role: topic
parent: REPORT.md
summary: "Planned four-case installed-MCP π-pulse experiment, action derivation, control exchange and visible evidence."
read_when: "Read before the next π-pulse campaign; this is a plan, not an execution receipt."
status: draft
reviewed: 2026-10-13
---
# Four-case MCP π-pulse testing plan

The shared core is a working model. This campaign tests the globally installed
release through MCP before further numerical enhancements or skill design.
It is not executed by publication. Earlier [sample results](REPORT.md) used a
coarser grid and Python/service/terminal routes; they do not qualify this plan.

## Common physical problem

Use a resonant qubit with no drift, one real control and no amplitude bound:

\[
H(t)=\tfrac12u(t)\sigma_x,\quad i\dot\psi=H\psi,\quad
\psi(0)=|0\rangle,\quad \psi_{\rm target}=|1\rangle.
\]

Set \(T=1\), \(N=1000\), \(\Delta t=10^{-3}\). There are **1000 control
intervals and 1001 state boundaries**. Controls are constant on each interval.
Define angle \(\theta(t)=\int_0^t u(s)ds\), fidelity
\(F=\sin^2(\theta(T)/2)\), and full energy \(E=\int_0^T u^2dt\).
Both families report the same physical F, E and soft cost; solver-normalized
costs must be labelled separately.

| Case | Physical objective | Method | Qualification |
|---|---|---|---|
| Gradient soft | \(J=1-F+\alpha E\), \(\alpha=0.01\) | Short gradient chunks, then L-BFGS/line search if diagnostics warrant | Cost, continuum stationarity, curvature and oracle agreement |
| Gradient hard | Minimize E subject to \(c=\cos(\theta(T)/2)=0\) | Feasibility restoration and projected gradient | Feasibility AND projected stationarity AND minimum-branch energy |
| Shooting soft | Same J, via PMP transversality | Newton/LM on initial costate | Root residual, curvature, branch selection and independent physical replay |
| Shooting hard | Same equality-constrained E | Newton/LM on endpoint residual | Endpoint, branch energy and independent physical replay |

The four cases run sequentially. Begin with one worker and one numerical-library
thread. At every launch use at most `max(1, available_cpus - 3)` total worker
threads per coordinator; on the observed 10-core host the ceiling is 7. Set
OMP, OpenBLAS, MKL, BLIS, vecLib and NumExpr thread limits to 1. This safeguard
limits our jobs; it is not an operating-system reservation against other apps.

## Derive the action before implementing the gradient

The complex-state action for the soft problem is

\[
\mathcal A=\phi(\psi(T))+\int_0^T\left[
\alpha u^2+2\operatorname{Re}\langle\chi,\dot\psi+iH\psi\rangle
\right]dt,\qquad \phi=1-|\langle1|\psi(T)\rangle|^2.
\]

Here χ is a multiplier enforcing Schrödinger's equation. Varying χ gives the
forward dynamics. Integration by parts in the state variation gives
\(\dot\chi=-iH\chi\), with terminal value
\(\chi(T)=\langle1|\psi(T)\rangle|1\rangle\). It is integrated backward.
The continuous gradient density is

\[
g(t)=2\alpha u(t)-2\operatorname{Im}\langle\chi(t),
\tfrac12\sigma_x\psi(t)\rangle.
\]

For this commuting Hamiltonian the equivalent reduced action is

\[
\mathcal A_\theta=1-\sin^2(\theta(T)/2)+
\int_0^T[\alpha u^2+\lambda(u-\dot\theta)]dt.
\]

Thus \(\dot\theta=u\), \(\dot\lambda=0\),
\(\lambda(T)=-\tfrac12\sin\theta(T)\), and
\(g(t)=2\alpha u(t)-\tfrac12\sin\theta(T)\).
For interval variables the exact Euclidean derivative is

\[
\theta_T=\Delta t\sum_k u_k,\quad E=\Delta t\sum_k u_k^2,
\qquad \frac{\partial J}{\partial u_k}=
\Delta t[2\alpha u_k-\tfrac12\sin\theta_T].
\]

Build the project-owned OLGS system using exact interval propagation
\(U_k=\exp(-i\Delta t u_k\sigma_x/2)\), forward state storage, backward
adjoints and exact propagator derivatives. Use the reduced formula only as an
independent check. Confirm gradients with central directional finite differences
at two step sizes, nonconstant controls and several seeds before optimization.

For the hard problem use the reduced constrained action
\(E+\nu\cos(\theta_T/2)+\int\lambda(u-\dot\theta)dt\).
Then \(\nabla_k E=2\Delta t u_k\) and
\(\partial c/\partial u_k=-\tfrac12\Delta t\sin(\theta_T/2)\).
Require both restoration of c and descent in the feasible tangent space.
The soft penalty is not used as a substitute for the hard equality.

The discrete gradient contains Δt: changing N changes its Euclidean magnitude.
Use the continuum density and mesh-weighted norm for comparisons, alongside the
solver's documented scaled norm. Calibrate the initial step with a small pilot
chunk; do not copy the old N=100 step size or stopping threshold unchanged.

## Shooting derivation and analytical answers

For soft shooting rescale the objective by the positive constant \(1/(2\alpha)\):
\(C=E/2+(1-F)/(2\alpha)\). With maximized PMP Hamiltonian
\(\mathcal H=pu-u^2/2\), stationarity gives \(u=p\), \(\dot p=0\).
The variable is \(p_0\); \(\theta(T)=Tp_0\). Its residual and Jacobian are

\[
R_s=p_0-\frac{\sin(Tp_0)}{4\alpha},\qquad
R_s'=1-\frac{T\cos(Tp_0)}{4\alpha}.
\]

For hard shooting, \(R_h=\cos(Tp_0/2)\) and
\(R_h'=-T\sin(Tp_0/2)/2\). Compare analytical/variational Jacobians with
finite differences. Use 1000 integration steps initially, then 500/2000/4000.
Since this reduced ODE is constant, RK4 refinement alone is weak evidence:
independently replay recovered controls through the full complex qubit state,
check the closed-form answer, and record the verification method accurately.

The hard global answer **for this declared unbounded one-axis model** follows
from Cauchy–Schwarz: \(E\ge\theta_T^2/T\). Feasibility requires
\(\theta_T=(2m+1)\pi\), so minima are \(u=\pm\pi/T\),
\(E=\pi^2/T\). At T=1, E ≈ 9.8696044011 and F=1.
A ±3π root is feasible but costs nine times as much.

The soft problem also reduces globally to a scalar minimization because
constant u minimizes energy at fixed θ. At α=.01, T=1, its minimum lies at
\(\theta_*\approx\pm3.02047766\), solving \(\sin\theta_*=4\alpha\theta_*\).
The expected F ≈ .99633727, E ≈ 9.12328530 and J ≈ .09489558.
A scalar oracle must compare both signs, stationary points and boundaries on a
bounded search interval: an incumbent J bounds \(|\theta|\le\sqrt{JT/\alpha}\).
The zero control is a stationary root but has negative curvature in the uniform
control direction; it must not be accepted as a soft minimum.

## MCP setup and visible workflow

1. Record release version, commit, wheel checksum and `optimizer.__file__` from
   the global isolated launcher. Run from a separate temporary project folder;
   stage trusted, hashed physics factory files there. Do not import checkout
   optimizer source. Declare factories, conventions and initial seeds explicitly.
2. Connect the installed `optimizer-mcp` stdio launcher to the assistant client.
   Discover its six tools: attach, detach, list, info, execute and inspect. These
   are transport tools; execute dispatches documented native operations. The
   current chat has no optimizer-specific MCP tools exposed, so client connection
   and tool availability are a prerequisite. An official SDK transport test is
   reported separately from an assistant-host session.
3. Initialize an isolated named experiment, then attach with explicit coordinator
   ownership. Create four separate contexts and immutable starting seeds.
   Use `info` to inspect operation schemas and required capabilities before calls.
4. Make each request visible: operation, context, physical/grid parameters and
   returned identifier. Use `activate=false` for candidate computations. Inspect
   candidate metrics before explicitly activating accepted results.
5. Submit bounded chunks; inspect after each chunk. For longer jobs use submit
   plus short waits/polls, respecting the 45-second tool deadline. A disconnect
   does not cancel a job or roll it back. Display real run-record rows and status.
6. Save checkpoints and export histories/results, then detach cleanly and shut
   down the owned coordinator when finished. Label live exact resume separately
   from a disk physical restart. Do not call restored controls an exact resume.

```mermaid
flowchart LR
    A[Visible MCP request] --> B[Installed stdio adapter]
    B --> C[Named experiment and context]
    C --> D[Bounded numerical chunk]
    D --> E[Inspect metrics and run record]
    E --> F[Plots and candidate decision]
    F --> G[Checkpoint and typed history]
    F -->|Continue with recorded settings| D
```

Show a visible experiment panel or generated report fed by actual inspected data:
control versus time; state populations or Bloch trajectory; F/E/J versus iteration;
hard/root/stationarity residuals on log axes; accepted/rejected steps; evaluation
counts; elapsed time; job state; and the exact MCP request producing each change.
The MCP server itself supplies structured data, not a GUI. Use file/browser previews
for plots and the report, and visible tool calls for the transport. Never populate
plots with simulated progress presented as a live run. If full trajectories are
absent from an operation, obtain them through a declared factory/analysis hook or
saved verification artifact rather than inventing a native operation.

## Controls, grids and duration exchange

- Recover a shooting control and replay it on the 1000-interval gradient grid.
  Compare area, F, E and both objective conventions; then use it as a warm start.
- For a gradient control, use \(p_0=\theta_T/T\) as a shooting seed. A general
  nonconstant control is not a costate trajectory. Record this reduction and its
  lost variance; at fixed area the mean control has no higher energy.
- Compare N=500/1000/2000 and an independent N=4000 replay. Preserve interval
  areas under grid transfer, use explicit boundary/sample conventions, and check
  energies. Piecewise averaging can reduce energy; it need not preserve energy
  even when the pulse area is preserved. Repeat optimization after transfer.
- Test durations τ=.5 and 2 with
  \(u_\tau(t)=u(tT/\tau)T/\tau\). Area and F are preserved; energy becomes
  \(E_\tau=(T/\tau)E\), with \(\Delta t_\tau=\tau/1000\).
  This preserves hard optimality for a constant minimum-branch pulse. At fixed
  α it is only a soft warm start. To preserve the soft dimensionless tradeoff,
  use \(\alpha_\tau=\alpha\tau/T\), and test both choices explicitly.

## Acceptance and retained evidence

Use deterministic seeds including a nonconstant control, near-minimum controls,
zero soft control and a higher hard branch. Zero hard control has a singular
constraint derivative: test and report restoration failure rather than hiding it.
Compare positive and negative branches; choose by physical cost after qualification.

Initial targets: directional derivative relative error ≤1e-6 (with a denominator
floor), hard endpoint/root residual ≤1e-8, independent F agreement ≤1e-9,
J/E agreement with the selected analytical minimum ≤1e-6 absolute and continuum
stationarity norm ≤1e-6. First confirm these against propagation accuracy; any
revision must be recorded before accepting results, with its reason. For soft
shooting include curvature/branch checks; for hard gradient include feasibility
and projected stationarity. Do not use the tiny unweighted discrete gradient
alone as evidence when N=1000.

Retain one comparison table for the four cases, request/history files, system
hashes, seeds, solver settings, bounded run records, checkpoints, verification
outputs and plots. Exercise wrong control layouts and unavailable hooks, ensuring
structured errors leave active state unchanged; test save/reopen and strict JSON
inspection with paged arrays. Report failures and untested routes explicitly.

Success qualifies this elementary model and these pathways. It does not prove
unrestricted optimality for other physics or qualify the deferred article cases.
Record stronger direct shooting, sparse Jacobians, adaptive LM, exact bounded
projection and weighted proximal scaling as later enhancements. Skill design
uses the verified workflow after this campaign, as requested by the user.

---

[⌂ Home](../../README.md) · [Sample evidence](REPORT.md) · [MCP contract](../../optimizer/mcp/MCP.md)
