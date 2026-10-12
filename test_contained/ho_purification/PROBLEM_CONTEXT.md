---
role: topic
parent: ../../tests/TESTING_HUB.md
summary: "Contained harmonic-oscillator purification system and historical tracker-trial provenance."
read_when: "Read when running or changing the HO purification trial."
status: active
reviewed: 2026-10-12
---
# HO purification contained system

This contained test system is derived from
`/Users/billabobz/phd/HO_purification/Non-unitary_Control_Qubit_HO.tex` and
the implementation authority
`/Users/billabobz/phd/HO_purification/systems/ho_purification_v3/system.py`.

It evolves the closed qubit--oscillator state under

\[
H(t)=\frac{\omega_A(t)}{2}\sigma_z\otimes I_B+
\omega_B I_A\otimes b^\dagger b+g\,\sigma_x\otimes(b+b^\dagger),
\]

with one control channel `omega_A`.  The tracked objective is
\(J=1-F+\lambda_u E_u\), where
\(F=\operatorname{Tr}[(|\psi_{tg}\rangle\langle\psi_{tg}|\otimes I_B)\rho(t_f)]\)
and \(E_u=\frac{\Delta t}{2}\sum_j\omega_{A,j}^2\).

The adapter uses exact Fréchet derivatives of each discrete propagator.  Its
trial supports Fock and diagonal thermal oscillator preparations only; it is
not a replacement for the full project system or its cutoff/ledger campaign.
The historical tracked run was an engineering/tracker demonstration, not a physical
claim about purification, thermodynamic work, reachability, or optimality.

The tracker-specific harnesses are retired and preserved under workspace
`temp/task5-retired-logging-docs/`. Numerical system equations and saved scientific
results are preserved; current logging uses frontend command history only.

---

[⌂ Home](../../tests/TESTING_HUB.md)
