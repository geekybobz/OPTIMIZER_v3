"""Single OLGS base class for new systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from optimizer.controls import ControlSpec, Controls
from optimizer.system_olgs.contract import get_secondary_update_hook, probe_system
from optimizer.system_olgs.validation import validate_controls_for_system


class OLGS(ABC):
    """Open Loop Gradient System base class.

    Subclassing is recommended for new systems, but not required for compatibility.
    Existing systems can satisfy the same contract structurally.
    """

    @abstractmethod
    def control_spec(self) -> ControlSpec:
        """Return the named control layout expected by this system."""

    @abstractmethod
    def evaluate(self, controls: Controls) -> dict[str, Any]:
        """Return optimizer-facing metrics with finite scalar key ``J``."""

    @abstractmethod
    def gradient(self, controls: Controls) -> Controls:
        """Return analytical ``dJ/du`` in the same control layout."""

    @abstractmethod
    def with_secondary(self, **updates: Any) -> "OLGS":
        """Return an equivalent system with updated secondary params."""

    def with_params(self, **updates: Any) -> "OLGS":
        """Compatibility alias during migration to ``with_secondary``."""

        return self.with_secondary(**updates)

    def validate_controls(self, controls: Controls) -> ControlSpec:
        """Validate controls against this system's control spec."""

        return validate_controls_for_system(self, controls)

    def forward_prop(self, controls: Controls) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement forward_prop(...).")

    def back_prop(self, controls: Controls) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement back_prop(...).")

    def metrics(self) -> dict[str, Any]:
        latest = getattr(self, "_latest_metrics", None)
        if latest is None:
            raise RuntimeError("No latest metrics are available. Run evaluate(...) first.")
        return dict(latest)

    def describe(self) -> dict[str, Any]:
        """Return compact system information suitable for notebooks and logs."""

        params = {}
        for name in ("primary", "secondary", "params"):
            value = getattr(self, name, None)
            if value is not None:
                if hasattr(value, "to_dict") and callable(value.to_dict):
                    params[name] = value.to_dict()
                elif hasattr(value, "__dict__"):
                    params[name] = dict(vars(value))
                else:
                    params[name] = value
        return {
            "kind": "OLGS",
            "class": self.__class__.__name__,
            "control_spec": self.control_spec().to_dict(),
            "params": params,
            "hooks": probe_system(self).to_dict(),
        }

    def cache_reset(self) -> None:
        """Placeholder hook for future logging/blackbox cache integration."""

        for name in ("_latest_controls", "_latest_metrics", "_latest_state"):
            if hasattr(self, name):
                delattr(self, name)

    def cache_status(self) -> dict[str, Any]:
        """Placeholder cache status until logging/blackbox ownership is redesigned."""

        return {
            "latest_controls": hasattr(self, "_latest_controls"),
            "latest_metrics": hasattr(self, "_latest_metrics"),
            "latest_state": hasattr(self, "_latest_state"),
            "owner": "placeholder",
        }


def with_secondary(system: Any, **updates: Any) -> Any:
    """Update a system through ``with_secondary`` or migration ``with_params``."""

    return get_secondary_update_hook(system)(**updates)
