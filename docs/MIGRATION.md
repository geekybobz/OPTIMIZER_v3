---
role: topic
parent: ../README.md
summary: "Use the current system contracts and retire legacy trees and logger integrations."
read_when: "Read when converting an older project or looking for the former workspace layout."
status: active
reviewed: 2026-10-13
---
# Migration to the current optimizer

**Convert legacy integrations to the current OLGS or OLSS contracts. Do not add
legacy implementations or compatibility trees back into the active repository.**
The repository root is the maintained optimizer. There is no separate stable,
registry or legacy checkout in the active layout. Git history retains previous
versions; local recovery backups are kept outside the repository.

## Details

### Paths and environments

The former `stable/v4/` contents now live at the repository root. Update shell
scripts, editor workspaces and source references to that root. `Optimizer` is the
standard Conda environment; use the [installation guide](INSTALLATION.md) to
replace old editable or stale environments. A renamed environment alone does
not update its installed package or notebook kernel.

### Replace old integrations

1. Identify whether your project owns objective gradients (OLGS) or shooting
   residual equations (OLSS).
2. Implement the current [OLGS](../optimizer/system_olgs/CONTRACT.md) or
   [OLSS](../optimizer/system_olss/CONTRACT.md) contract from the project's theory.
3. Validate shapes, layouts and derivative capabilities before optimization.
4. Replace tracker/logger calls with explicit data inspection in Python, or
   frontend command history plus numerical checkpoints in CLI/MCP.
5. Check a short run before migrating expensive campaigns.

The removed tracker API has no automatic compatibility layer. Its retirement
and replacement are documented in [command history](../optimizer/history/HISTORY.md).
Ordinary Python creates no automatic history writer. CLI and MCP retain requests
and outcomes, including rollback, failed commands and inspection. Checkpoints,
inputs, source/dependency identity and environment are needed alongside history
for numerical reproduction.

### Existing experiments

Moving source files or changing package versions can invalidate exact restoration.
Keep original artifacts. Use the documented `controls_only` or `variables_only`
opening mode for an explicit physical restart when exact provenance no longer
matches. This resets method memory and is not an exact continuation claim.

The old GitHub repository name is retained so URLs continue to work. The current
`main` branch and versioned releases are authoritative. Historical plans are
context; current contract documentation describes implemented behavior.

---

[⌂ Home](../README.md) · [Next →](INSTALLATION.md)
