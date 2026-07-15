# `constraints/`: Bounds And Projections

Status: review draft.
Last updated: 2026-07-15.

## Purpose

Constraints handle generic control restrictions that are not the system objective.

Examples:

```text
amplitude bounds
norm caps
projection onto simple feasible sets
smoothness limits
```

## Not The Same As System Residuals

System residuals are analytical physical constraints:

```text
terminal leakage
F_k(T)
exact equality conditions
```

Constraints are generic control-space restrictions:

```text
u_min <= u <= u_max
||u|| <= R
bandwidth cap
```

## Public Use

```python
bounds = opt.Bounds(low=-1.0, high=1.0)
result = opt.adam(sys, controls, constraints=[bounds])
```

or:

```python
controls = opt.project_bounds(controls, low=-1.0, high=1.0)
```

## First Constraints

Implement first:

```text
box bounds
max norm
fixed norm
simple projection
```

