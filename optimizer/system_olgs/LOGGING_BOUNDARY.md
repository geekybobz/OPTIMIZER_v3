---
role: topic
parent: SYSTEM_OLGS_HUB.md
summary: Numerical system ownership and frontend command-history boundaries.
read_when: Read before adding persistence or interaction recording to a system.
status: active
reviewed: 2026-10-12
---

# OLGS interaction boundary

OLGS owns physics, metrics, analytical derivatives and its declared numerical
capabilities. It has no required logging hook. Static objective/metric metadata
belongs to `system_spec()`; optional cache hooks remain numerical lifecycle tools.

Ordinary Python numerical calls create no automatic log files. CLI and future
MCP interactions use the shared [command recorder](../history/HISTORY.md).
[Checkpoint persistence](../persistence/PERSISTENCE.md) owns saved numerical state,
source identity and explicit retention. The optimizer engine retains native method
memory and compact numerical aggregates, without recording interaction history.

## Connections

- Parent: [OLGS hub](SYSTEM_OLGS_HUB.md)
- Related: [Contract](CONTRACT.md)
- Related: [Lifecycle](LIFECYCLE.md)

---

[⌂ Home](SYSTEM_OLGS_HUB.md)
