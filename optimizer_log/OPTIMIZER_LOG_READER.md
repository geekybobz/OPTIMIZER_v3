# Standalone optimization-log reader

`optimizer_log` is the read-only boundary for tracker run files. It reads and
validates receipts, timelines, live state, terminal state, writer ownership,
checkpoint references, and compact metric coverage without importing the
numerical optimizer or a scientific system.

## Flow

- [`reader.py`](reader.py) validates supported schemas and causal sequences.
- Full reads return byte-accurate continuation offsets.
- Incremental reads distinguish incomplete tails from permanent corruption.
- Writer inspection is read-only and never sends a process signal.
- Unknown event kinds from a supported schema are reported separately so newer
  writers do not destroy the readable prefix.

Run discovery, a portal UI, storage retention, modes, and verification scans are
deliberately outside this reader foundation.

---

[⌂ Home](../README.md)
