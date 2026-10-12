---
role: topic
parent: README.md
summary: "Maintainer rules for source ownership, generated docs and explicit release promotion."
read_when: "Read before changing this repository."
status: active
reviewed: 2026-10-13
---
# Maintainer instructions

The active optimizer is this repository root. Convert older integrations to the
current OLGS/OLSS contracts; see [migration](docs/MIGRATION.md). Do not restore
legacy, registry or stable checkout trees.

Keep development separate from the installed release. A pull or commit must not
automatically deploy code. Follow [release procedure](docs/RELEASE_PROCEDURE.md)
for explicit commit-bound wheel promotion. Preserve project-owned physics in
self-contained OLGS and OLSS systems.

Use the documented generators for API_REFERENCE.md and INVENTORY.md. Local
`.ai-memory/` is private project context, managed separately from public docs.

---

[⌂ Home](README.md)
