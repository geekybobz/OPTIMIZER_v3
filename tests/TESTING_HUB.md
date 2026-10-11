---
title: TESTING_HUB
type: module_hub
module: tests
tags:
  - optimizer
  - tests
  - hub
---

# Testing Hub

The library test suite, its fixtures, and the tracker trials.

## Run the suite

```bash
MPLCONFIGDIR=/tmp/optimizer-mpl python -B -m unittest discover -s tests
```

Run it from the checkout root inside the `optimizer_v4` environment. `-B` keeps
bytecode out of the checkout.

## Test files

| Area | Files |
|---|---|
| Controls, results, and state | `test_controls.py`, `test_result_state.py` |
| Public API and discovery | `test_public_api.py`, `test_namespaces.py`, `test_catalog.py`, `test_optimizer_context.py` |
| Optimizers and engine | `test_optimizers.py`, `test_advanced_optimizers.py`, `test_engine.py`, `test_numerical_capabilities.py` |
| Guesses and utilities | `test_guesses.py`, `test_utils_tools.py` |
| System contract and visualization | `test_system_contract.py`, `test_system_spec_visualization.py` |
| Traces, tracker, and campaigns | `test_trace_checkpoint.py`, `test_tracker.py`, `test_tracker_bridge.py`, `test_campaigns.py` |
| Documentation | `test_document_links.py`: every relative link in the Markdown resolves |

## Worked example

[fixtures/quadratic_system.py](fixtures/quadratic_system.py) is a complete,
minimal OLGS system used across the suite. Read it next to the
[OLGS Contract](../optimizer/system_olgs/CONTRACT.md).

## Tracker trials

Small contained systems that exercise the tracker on real physics:

- [HO purification trial](../test_contained/ho_purification/PROBLEM_CONTEXT.md)
- [Second-order PMP trial](../test_contained/second_order_pmp/PROBLEM_CONTEXT.md)

Local exploration notebooks may live under `tests/system_check/`, which Git
ignores.

## Connections

- Parent: [the V4 entry](../README.md)
- Related: [OLGS Contract](../optimizer/system_olgs/CONTRACT.md), [Utils Documentation Hub](../optimizer/utils/UTILS_HUB.md)

---

[⌂ Home](../README.md)
