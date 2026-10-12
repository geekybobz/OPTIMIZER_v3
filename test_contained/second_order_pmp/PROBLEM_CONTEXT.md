---
role: topic
parent: ../../tests/TESTING_HUB.md
summary: "Contained second-order robust qubit-transfer system used as a tracker trial and campaign case."
read_when: "Read when running or changing the second-order PMP trial."
status: active
reviewed: 2026-10-12
---
# Second-order PMP tracker trial

This contained system is derived from
`/Users/billabobz/phd/rob_unversal_article/second_order/latex/pmp_secondorder.tex`.
It uses the Method 1 universal-robustness objective for a qubit transfer:

\[
H_0(t)=\frac12[u_x(t)\sigma_x+u_y(t)\sigma_y+u_z(t)\sigma_z],\qquad
F_k=\int_0^{t_f}\langle\psi_\perp(t)|\sigma_k|\psi_0(t)\rangle\,dt,
\]

\[
J=(1-\mathcal F_0)+\sum_{k=x,y,z}|F_k|^2+\lambda E.
\]

The default \(\lambda=0\) reproduces the active old source objective.  The
new adapter uses left piecewise-constant controls and an exact derivative of
that discrete objective.  The old `corrections()` update is not reused as an
unverified generic-optimizer gradient.

Finite-\(\alpha\) directional robustness is a post-run diagnostic, never a
hidden per-iteration tracker calculation.  This short run checks system and
tracker plumbing only; it is not a certificate of robust transfer, constrained
stationarity, or global optimality.

---

[⌂ Home](../../tests/TESTING_HUB.md)
