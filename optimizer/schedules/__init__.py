"""Step-size schedule helpers.

Why this package exists
-----------------------
Training often needs consistent step-size behavior across optimizers and repair
loops: constant values, shrink on rejection, growth after success, and lower/upper
limits.  Schedules provide that policy without owning control updates.

How it fits the architecture
----------------------------
- optimizers can use schedules to update ``state.step_size``.
- repair tools can use the same shrink/grow vocabulary.
- schedules are plain data objects and do not call ``system.evaluate``.
"""

from optimizer.schedules.step_size import AdaptiveStepSchedule, ConstantSchedule

__all__ = ["AdaptiveStepSchedule", "ConstantSchedule"]
