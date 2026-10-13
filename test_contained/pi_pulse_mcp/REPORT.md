---
role: deep
parent: ../../tests/TESTING_HUB.md
summary: "Installed-MCP four-case π-pulse results, real progress evidence, retained failures and client connection boundary."
read_when: "Read to assess the 2026-10-13 MCP π-pulse campaign and reproduce its settings."
status: active
reviewed: 2026-10-13
---
# Installed MCP π-pulse campaign — 2026-10-13

All four main cases passed their physical acceptance targets through the
installed 4.0.6 MCP server. The final ledger has **103 successful checks and two
initial verification-configuration failures**, both resolved by recorded retries.
It does not erase those failures or count the initial pilot as a successful run.

Source release: `7edaa3051d8cc39694a45c09e257a35b7515e64e`.
Execution imported `Optimizer/.../site-packages/optimizer`, using the isolated
global `optimizer-mcp` launcher and an experiment/factories outside the checkout.
Project physics stays in the self-contained [OLGS](pi_pulse_olgs.py) and
[OLSS](pi_pulse_olss.py) test systems. Library source and the installed wheel were
not changed by this testing phase.

## Main results

The [action derivation and acceptance plan](../pi_pulse_samples/MCP_TEST_PLAN.md)
defines H=uσx/2, transfer |0⟩→|1⟩, T=1, N=1000, dt=.001 and α=.01.
There are 1000 interval controls and 1001 state boundaries.

| Case | Fidelity | Full energy | Soft cost | Stationarity density norm |
|---|---:|---:|---:|---:|
| Gradient soft | 0.996337270290 | 9.123285304565 | 0.094895582756 | 6.84e-7 |
| Shooting soft | 0.996337270290 | 9.123285303396 | 0.094895582744 | 5.79e-15 |
| Gradient hard | 1.000000000000 | 9.869604401089 | 0.098696044011 | 8.43e-8 projected |
| Shooting hard | 1.000000000000 | 9.869604401090 | 0.098696044011 | 8.88e-16 projected |

For hard cases the minimized objective is E, not the diagnostic soft-cost column.
The hard answer is u=±π, E=π². The soft answer has |u|≈3.02047766146; its
less-than-unit fidelity is the expected optimum of the declared penalty tradeoff.
These analytical claims apply to this unbounded, drift-free one-axis model.

The gradient system uses exact interval propagators and backward adjoints. Its
MCP-returned discrete gradients agree with the reduced action formula to ≤1e-12;
finite-difference checks at eps=1e-4 and 1e-5 passed on nonconstant controls.
Hard constraint derivatives were also checked at both steps. Independent full
complex-state RK4 replay used four substeps per control interval and agreed in
fidelity within 1e-9. Gradient acceptance used continuum gradient-density norms,
not the small unweighted discrete norm caused by dt=.001.

Gradient soft used two ten-iteration Adam chunks, then 100-iteration Armijo
chunks with step_size=500, normalize=false and gradient_tolerance=1e-12. Hard
used five-iteration projected-gradient chunks with step_size=250, damping=1e-12,
feasibility_tolerance=1e-10 and stationarity_tolerance=1e-12. Candidate operations
used activate=false; inspected controls were explicitly selected afterward.
Shooting used Newton, maxiter=30, tolerance=1e-11 and separate verification.
The retained request ledger owns exact settings for every call.

## Other completed checks

- Both shooting signs were compared. The zero soft root is numerically stationary
  but has negative curvature and J=1; it was rejected as a minimum. The hard 3π
  branch was rejected because its energy is nine times the minimum.
- Shooting controls were recovered and replayed in gradient systems, then used
  as gradient warm starts. Gradient pulse-area means supplied shooting seeds.
  That reduction is model-specific, not a general waveform-to-costate inverse.
- Constant recovered controls were transferred to N=500/2000/4000 using explicit
  cell-center grids and hold extrapolation. F and E agreed with the oracle.
  This checks constant-control grid transfer; it does not establish general
  area conservation for the library's linear interpolation of arbitrary pulses.
- Shooting working integrations used 500/1000/2000/4000 steps. The full-state
  verifier used distinct settings, including 16000 steps for the accepted
  4000-step retries. Constant reduced PMP dynamics alone were not treated as
  a strong independent integrator test.
- Durations .5 and 2 preserved area/F and scaled E by 1/T. At fixed α the scaled
  soft control was reoptimized; setting α_T=αT preserved the original tradeoff.
- Wrong-family, missing-capability and incompatible-layout requests returned
  structured errors and left active state unchanged. The hard zero start
  returned restoration_failed with c0=1, and was not activated as feasible.
- MCP detach/reconnect retained accepted controls. Pinned physical checkpoints,
  coordinator shutdown/reopen and byte-equal restored control arrays passed.
  This is saved physical-data recovery, not an exact live solver-resume claim.
- The final journal contains 356 submitted MCP records with structured expressions
  and empty raw text. CLI initialization and coordinator lifecycle entries are
  recorded separately. All numerical work was sequential with six native thread
  limits set to one; the observed ten-CPU host had a seven-slot admission ceiling.

## Findings and resolved setup failures

**Verification options must differ.** Two supplementary calls used steps=4000
for both working and verification options. Their residuals were about 4e-12 and
4e-14, but the library correctly returned verification_not_independent under
its current option-difference policy. Both passed after requesting a 16000-step
full-state verifier. The two original failed assertions remain in the ledger,
with resolved_by links to the passing retries. The library's independence_source
records system_default or explicit_options; it overrides the hook's own label.
Changing that conservative policy is a separate design decision. Independent
algorithm identity should always be stated alongside configuration provenance.

**Retention is explicit.** Initial keep_recent=20 pruned an old result payload
before control exchange, despite a remaining name pointing to it. Larger
retention cannot resurrect a pruned payload. Transfers resumed from active
physical heads; keep_recent=256 and pinned accepted checkpoints supported the
remaining workflow. A retained binding is not a promise of permanently retained
numerical bytes. Pin results needed for later comparisons before they are pruned.

**An initial soft pilot missed its budget target.** Short L-BFGS chunks with
step_size=500 and physical control restarts reached J≈.0949947 and density≈.0060,
which did not meet acceptance. Scale-calibrated Armijo chunks passed. This is
not a controlled comparison of algorithms and does not establish a solver bug.
Pilot records are kept separately from the accepted campaign.

The test driver also initially used the wrong checkpoint operation, sent an
activation flag to lifecycle calls, assumed compact error replies included state
snapshots, and omitted required residual metrics from its new verifier. Those
were driver/fixture mistakes and were corrected. A deliberate shadow optimizer.py
also triggered the source dependency audit; trusted factories were moved to a
separate folder. All rejected requests remain visible in the progress/request
ledger. No optimizer source fix or extra deployment was performed.

## Native assistant connection boundary

The installed server is registered automatically in Codex as optimizer, with an
absolute launcher, cwd=/Users/billabobz, six numerical thread limits set to one,
and TMPDIR=/private/tmp for consistent Unix-socket identity checks.
The official MCP SDK drove real attach/execute/inspect requests against that
installed server; it did not call solvers directly in this test driver.

This running chat did not expose native optimizer MCP tools after registration.
No callable app-reload tool was available, and computer-use access to Codex and
Terminal was blocked. Native assistant-host reload/tool discovery remains
unverified. The [official client setup](https://learn.chatgpt.com/docs/extend/mcp?surface=app)
describes saving a server and selecting Restart. These transport campaign results
must not be relabelled as native assistant-host acceptance.

A separate client that omits the coordinator's temporary-directory environment
can fail socket identity checks. An exploratory reconnect demonstrated this;
full matching environment succeeded. The registered server and accepted
campaign now explicitly use /private/tmp. Only this campaign's coordinators
were shut down; the read-only localhost progress server may remain available.

## Evidence and reproduction

Local evidence is under runs/20261013/: results.json, checks.json, provenance.json,
requests.jsonl, progress.log, per-chunk controls/records, plots.png, pilot/,
factory-snapshot/ and experiment-snapshot/. Runs are ignored by Git. Runtime
socket credentials are omitted from the saved experiment snapshot. Source paths
in the recipes still refer to the isolated /tmp factories; the snapshot is not
advertised as a portable exact restart.

The executed campaign used run_mcp.py, then continue_mcp.py, finish_mcp.py,
supplement_mcp.py and retry_verification_mcp.py as recorded in the progress ledger.
The final driver files include fixes discovered during testing; their archived
hashes identify the final harness, not every earlier executed variant. Effective
MCP requests and exact physics-file hashes are the authoritative execution record.
To reproduce, stage the two physics files under /tmp/optimizer-pi-pulse-mcp-20261013/factories,
initialize a fresh named experiment through the global CLI, set PI_MCP_EXPERIMENT
accordingly and run the driver with the installed Python -I -B. Use the six
thread limits and matching TMPDIR=/private/tmp; never import checkout optimizer
source. The supplemental verifier must use distinct working/verification settings.

These elementary tests support designing the next skill around explicit schemas,
short inspected runs, resource bounds, retention, control exchange and independent
qualification. Stronger numerical solvers and article qualification remain
separate deferred work.

---

[⌂ Home](../../README.md) · [Testing](../../tests/TESTING_HUB.md) · [Plan](../pi_pulse_samples/MCP_TEST_PLAN.md)
