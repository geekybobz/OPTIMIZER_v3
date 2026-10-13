---
role: topic
parent: README.md
summary: "Release verification scope and the authority for exact artifact provenance."
read_when: "Read when checking what a release verifies and where its hashes live."
status: active
reviewed: 2026-10-13
---
# Release evidence

Version **4.0.10** combines the current numerical contracts, foreground service,
structured CLI, persistent experiments, control transforms, bounded jobs and
standard-I/O MCP adapter with explicit commit-bound wheel deployment. It adds
the [shared numerical core](docs/SHARED_NUMERICAL_CORE.md), scale-aware solver
coordinates, bounded run records and the available-cores-minus-three resource
ceiling. [Core verification](test_contained/shared_core/REPORT.md) records its
parity checks and deferred numerical enhancements. The shared
[Bloch renderer](optimizer/visualization/VISUALIZATION_HUB.md) adds camera controls,
PNG/PDF exports and typed PlotData persistence for Python, CLI and MCP, with
OLGS/OLSS system-owned plot data.

The [four-case installed-MCP π-pulse campaign](test_contained/pi_pulse_mcp/REPORT.md)
completed soft/hard gradient and shooting checks. Its retained results were
measured on 4.0.6; publishing the harness does not imply rerunning that scientific
campaign on each release. It used the official MCP SDK transport. A subsequent
native Codex MCP smoke check on installed 4.0.10 passed experiment attachment,
renderer discovery, retained PlotData reference export, receipt inspection/pinning
and coordinator shutdown. Its bounded evidence is attached as `native-chat-smoke.json`
to the release; it is not a native rerun of the four-case numerical campaign.
Fresh-host/Claude skill acceptance remains separately unverified.

Exact commit, wheel/source SHA-256, installed-package tests and dependency
constraints are external release artifacts, published with the
[GitHub release](https://github.com/geekybobz/OPTIMIZER_v3/releases/tag/v4.0.10).
`RELEASE_MANIFEST.json` describes the release policy; it is not a wheel receipt.
The local installation receipt is `~/.local/share/optimizer/install.json`.

## Details

The [release procedure](docs/RELEASE_PROCEDURE.md) owns build, candidate
verification, promotion, publication and recovery. Its checks distinguish:

- Full source tests, generated API freshness, Markdown inventory/navigation and
  whitespace validation.
- Installed-wheel numerical, persistence, CLI, coordinator and official MCP SDK
  tests, excluding only the three checkout documentation/context checks.
- Installed file hashes, dependency consistency and version/import location.
- Global launcher and internal-child isolation from cwd and `PYTHONPATH`.
- Public remote branch/tag and attached release artifacts after publication.

The source and installed test counts belong to the accompanying verification
artifact; they are not copied into several documentation pages. Candidate
verification reuses the target environment's scientific dependencies and does
not establish cross-platform bitwise reproducibility.

The physical article comparison campaign remains deferred for separate review.
Library tests are not scientific benchmark qualification. Live job resume,
physical disk restart, per-coordinator resource safeguards and typed MCP replay
limits retain their documented meanings.

Legacy working trees were removed from the active repository after verified Git
bundles and a local context backup were saved outside it. Historical revisions
remain in Git history; current project integrations follow [Migration](docs/MIGRATION.md).

---

[⌂ Home](README.md) · [Next →](docs/RELEASE_PROCEDURE.md)
