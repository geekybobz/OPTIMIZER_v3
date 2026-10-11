"""Base class for self-contained open-loop shooting systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np

from optimizer.system_olss.results import ShootingVerification
from optimizer.system_olss.variables import ShootingSpec, ShootingVariables


class OLSS(ABC):
    """Open Loop Shooting System base class."""

    @abstractmethod
    def system_spec(self) -> Mapping[str, Any]:
        """Return the static OLSS declaration."""

    @abstractmethod
    def shooting_spec(self) -> ShootingSpec:
        """Return the ordered shooting-variable layout."""

    @abstractmethod
    def residuals(
        self,
        variables: ShootingVariables,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        """Return the complete unscaled residual vector."""

    @abstractmethod
    def jacobian(
        self,
        variables: ShootingVariables,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Any:
        """Return the derivative of residuals with respect to the shooting vector."""

    @abstractmethod
    def with_secondary(self, **updates: Any) -> "OLSS":
        """Return an equivalent system with updated staged parameters."""

    @staticmethod
    def _residual_metrics(residuals: np.ndarray) -> dict[str, float]:
        residuals = np.asarray(residuals, dtype=float).reshape(-1)
        return {
            "residual_norm": float(np.linalg.norm(residuals)),
            "residual_max_abs": float(np.max(np.abs(residuals))) if residuals.size else 0.0,
        }

    def evaluate(
        self,
        variables: ShootingVariables,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return self._residual_metrics(self.residuals(variables, options=options))

    def verify(
        self,
        variables: ShootingVariables,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> ShootingVerification:
        residuals = np.asarray(self.residuals(variables, options=options), dtype=float).reshape(-1)
        return ShootingVerification(
            residuals=residuals,
            metrics=self._residual_metrics(residuals),
            options=dict(options or {}),
        )

    def cache_reset(self) -> None:
        for name in ("_latest_variables", "_latest_residuals", "_latest_jacobian"):
            if hasattr(self, name):
                delattr(self, name)

    def cache_status(self) -> dict[str, Any]:
        return {
            "latest_variables": hasattr(self, "_latest_variables"),
            "latest_residuals": hasattr(self, "_latest_residuals"),
            "latest_jacobian": hasattr(self, "_latest_jacobian"),
        }
