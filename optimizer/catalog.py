"""Programmatic catalog for exploring OPTIMIZER v4.

Why this file exists
--------------------
Users often arrive at the library from a notebook or a closed-loop agent state where
they do not remember the available methods, the expected inputs, or the right next
tool.  Long documentation helps eventually, but interactive work needs compact,
structured answers available through Python calls.

This module provides that discovery layer.  It keeps normal execution APIs unchanged:
``opt.optimizers.adam(...)`` still runs Adam.  The catalog adds machine-friendly
``dict`` payloads by default and readable summaries when ``h=True`` is requested.

How it fits the architecture
----------------------------
- every public export has exactly one ``CatalogItem`` declaration here: its
  namespace (group), aliases, system family, and whether the package root also
  exports it (``root``).
- namespaces build ``__all__`` with ``exports(group)``; ``opt.methods()``, the
  optimizer-name lookup of continuation and campaigns, and ``API_REFERENCE.md``
  are built from the same declarations.
- role namespaces can expose ``list(...)`` and ``info(name, ...)`` from this module.
- public callables can receive an ``.info(...)`` attribute without wrapping execution.
- metadata is curated where reflection cannot know workflow meaning.
- signatures are discovered lazily from the already-importable public objects.

What this file deliberately does not do
---------------------------------------
It does not inspect or document project ``system.py`` files.  Systems own physics and
metric names; this catalog describes the reusable optimizer library surface.
"""

from __future__ import annotations

import builtins
import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Mapping

from optimizer.families import FAMILY_NAMES


TextMap = Mapping[str, str]


def _tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(value) for value in values)


def _dict(values: Mapping[str, str] | None) -> dict[str, str]:
    return {str(key): str(value) for key, value in dict(values or {}).items()}


def _contains(value: str, needle: str) -> bool:
    return needle.lower() in str(value).lower()


@dataclass(frozen=True)
class CatalogItem:
    """The one declaration of a public tool, class, constant, or alias.

    Its group is the namespace that exports it; ``root`` also exports it from the
    package root, and ``family`` names the system families (``olgs``, ``olss``)
    it serves.  Namespace exports, ``opt.methods()``, the optimizer-name lookup,
    and the API reference are all built from these declarations.
    """

    name: str
    group: str
    kind: str
    summary: str
    inputs: TextMap = field(default_factory=dict)
    returns: TextMap = field(default_factory=dict)
    requires: tuple[str, ...] = ()
    variants: tuple[str, ...] = ()
    best_for: tuple[str, ...] = ()
    not_for: tuple[str, ...] = ()
    watch_out: tuple[str, ...] = ()
    related: tuple[str, ...] = ()
    example: str | None = None
    module: str | None = None
    object_name: str | None = None
    status: str = "implemented"
    phase: str | None = None
    aliases: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    family: tuple[str, ...] = ()
    root: bool = False

    @property
    def id(self) -> str:
        return f"{self.group}.{self.name}"

    @property
    def return_type(self) -> str:
        if "type" in self.returns:
            return str(self.returns["type"])
        if self.returns:
            return ", ".join(str(value) for value in self.returns.values())
        return "unknown"

    def public_object(self) -> Any | None:
        module_name = _GROUP_MODULES.get(self.group)
        if module_name is None:
            return None
        try:
            module = importlib.import_module(module_name)
        except Exception:
            return None
        return getattr(module, self.object_name or self.name, None)

    def signature(self) -> str | None:
        obj = self.public_object()
        if obj is None or not callable(obj):
            return None
        try:
            return str(inspect.signature(obj))
        except (TypeError, ValueError):
            return None

    def short_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "summary": self.summary,
            "requires": builtins.list(self.requires),
            "returns": self.return_type,
            "status": self.status,
            "family": builtins.list(self.family),
        }
        if self.root:
            payload["root"] = True
        if self.variants:
            payload["variants"] = builtins.list(self.variants)
        if self.aliases:
            payload["aliases"] = builtins.list(self.aliases)
        if self.tags:
            payload["tags"] = builtins.list(self.tags)
        return payload

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "group": self.group,
            "kind": self.kind,
            "status": self.status,
            "summary": self.summary,
            "signature": self.signature(),
            "inputs": _dict(self.inputs),
            "returns": _dict(self.returns),
            "requires": builtins.list(self.requires),
            "variants": builtins.list(self.variants),
            "best_for": builtins.list(self.best_for),
            "not_for": builtins.list(self.not_for),
            "watch_out": builtins.list(self.watch_out),
            "related": builtins.list(self.related),
            "example": self.example,
            "module": self.module,
            "phase": self.phase,
            "aliases": builtins.list(self.aliases),
            "tags": builtins.list(self.tags),
            "notes": builtins.list(self.notes),
            "family": builtins.list(self.family),
            "root": True if self.root else None,
        }
        return {key: value for key, value in payload.items() if value not in (None, [], {}, "")}

    def render_human(self) -> str:
        lines = [self.id, f"Summary: {self.summary}"]
        signature = self.signature()
        if signature:
            lines.append(f"Signature: {self.name}{signature}")
        if self.family:
            lines.append("Family: " + ", ".join(self.family))
        if self.root:
            lines.append(f"Root shortcut: opt.{self.object_name or self.name}")
        if self.requires:
            lines.append("Requires: " + ", ".join(self.requires))
        if self.variants:
            lines.append("Variants: " + ", ".join(self.variants))
        if self.inputs:
            lines.append("Inputs:")
            lines.extend(f"- {key}: {value}" for key, value in self.inputs.items())
        if self.returns:
            lines.append("Returns:")
            lines.extend(f"- {key}: {value}" for key, value in self.returns.items())
        if self.best_for:
            lines.append("Use when:")
            lines.extend(f"- {value}" for value in self.best_for)
        if self.not_for:
            lines.append("Avoid when:")
            lines.extend(f"- {value}" for value in self.not_for)
        if self.watch_out:
            lines.append("Watch out:")
            lines.extend(f"- {value}" for value in self.watch_out)
        if self.related:
            lines.append("Related: " + ", ".join(self.related))
        if self.example:
            lines.append(f"Example: {self.example}")
        if self.module:
            lines.append(f"Module: {self.module}")
        return "\n".join(lines)


@dataclass(frozen=True)
class CatalogGroup:
    """Metadata and item lookup for one role namespace."""

    name: str
    summary: str
    common_inputs: TextMap = field(default_factory=dict)
    common_returns: TextMap = field(default_factory=dict)
    workflow: tuple[str, ...] = ()
    items: tuple[CatalogItem, ...] = ()

    def list(
        self,
        *,
        h: bool = False,
        kind: str | None = None,
        requires: str | None = None,
        returns: str | None = None,
        tag: str | None = None,
    ) -> dict[str, Any] | str:
        filtered = [
            item
            for item in self.items
            if _item_matches(item, kind=kind, requires=requires, returns=returns, tag=tag)
        ]
        if h:
            return self._render_list(filtered, kind=kind, requires=requires, returns=returns, tag=tag)
        return {
            "group": self.name,
            "summary": self.summary,
            "common_inputs": _dict(self.common_inputs),
            "common_returns": _dict(self.common_returns),
            "workflow": builtins.list(self.workflow),
            "filters": {
                "kind": kind,
                "requires": requires,
                "returns": returns,
                "tag": tag,
            },
            "items": [item.short_dict() for item in filtered],
        }

    def info(self, name: str, *, h: bool = False) -> dict[str, Any] | str:
        item = self.get(name)
        return item.render_human() if h else item.to_dict()

    def get(self, name: str) -> CatalogItem:
        query = str(name)
        for item in self.items:
            if item.name == query or query in item.aliases or item.id == query:
                return item
        raise KeyError(f"Unknown {self.name} catalog item {query!r}.")

    def _render_list(
        self,
        items: list[CatalogItem],
        *,
        kind: str | None,
        requires: str | None,
        returns: str | None,
        tag: str | None,
    ) -> str:
        filters = {
            key: value
            for key, value in {
                "kind": kind,
                "requires": requires,
                "returns": returns,
                "tag": tag,
            }.items()
            if value is not None
        }
        lines = [self.name, f"Summary: {self.summary}"]
        if filters:
            lines.append("Filters: " + ", ".join(f"{key}={value}" for key, value in filters.items()))
        if self.common_inputs:
            lines.append("Common inputs:")
            lines.extend(f"- {key}: {value}" for key, value in self.common_inputs.items())
        if self.common_returns:
            lines.append("Common returns:")
            lines.extend(f"- {key}: {value}" for key, value in self.common_returns.items())
        if self.workflow:
            lines.append("Typical workflow:")
            lines.extend(f"- {value}" for value in self.workflow)
        lines.append("Items:")
        for item in items:
            suffix = []
            if item.variants:
                suffix.append("variants: " + ", ".join(item.variants))
            if item.requires:
                suffix.append("requires: " + ", ".join(item.requires))
            suffix.append("returns: " + item.return_type)
            lines.append(f"- {item.name}: {item.summary} ({'; '.join(suffix)})")
        return "\n".join(lines)


class CatalogItemView:
    """Attribute view used for calls such as ``opt.catalog.optimizers.adam.info()``."""

    def __init__(self, item: CatalogItem) -> None:
        self._item = item

    def info(self, *, h: bool = False) -> dict[str, Any] | str:
        return self._item.render_human() if h else self._item.to_dict()


class CatalogGroupView:
    """Interactive view over a catalog group."""

    def __init__(self, group_name: str) -> None:
        self._group_name = group_name

    @property
    def _group(self) -> CatalogGroup:
        return group(self._group_name)

    def list(
        self,
        *,
        h: bool = False,
        kind: str | None = None,
        requires: str | None = None,
        returns: str | None = None,
        tag: str | None = None,
    ) -> dict[str, Any] | str:
        return self._group.list(h=h, kind=kind, requires=requires, returns=returns, tag=tag)

    def info(self, name: str, *, h: bool = False) -> dict[str, Any] | str:
        return self._group.info(name, h=h)

    def __getattr__(self, name: str) -> CatalogItemView:
        try:
            return CatalogItemView(self._group.get(name))
        except KeyError as exc:
            raise AttributeError(name) from exc


def _item_matches(
    item: CatalogItem,
    *,
    kind: str | None,
    requires: str | None,
    returns: str | None,
    tag: str | None,
) -> bool:
    if kind is not None and item.kind != kind and kind not in item.tags:
        return False
    if requires is not None and not any(_contains(value, requires) for value in item.requires):
        return False
    if returns is not None and not _contains(item.return_type, returns):
        return False
    if tag is not None and tag not in item.tags:
        return False
    return True


def _make_item(
    name: str,
    group: str,
    kind: str,
    summary: str,
    *,
    inputs: TextMap | None = None,
    returns: TextMap | None = None,
    requires: Any = None,
    variants: Any = None,
    best_for: Any = None,
    not_for: Any = None,
    watch_out: Any = None,
    related: Any = None,
    example: str | None = None,
    module: str | None = None,
    object_name: str | None = None,
    phase: str | None = None,
    aliases: Any = None,
    tags: Any = None,
    notes: Any = None,
    family: Any = None,
    root: bool = False,
) -> CatalogItem:
    families = _tuple(family) or _GROUP_FAMILIES[group]
    unknown = set(families) - set(FAMILY_NAMES)
    if unknown:
        raise ValueError(f"Catalog item {group}.{name} names unknown families {sorted(unknown)}.")
    return CatalogItem(
        name=name,
        group=group,
        kind=kind,
        summary=summary,
        inputs=_dict(inputs),
        returns=_dict(returns),
        requires=_tuple(requires),
        variants=_tuple(variants),
        best_for=_tuple(best_for),
        not_for=_tuple(not_for),
        watch_out=_tuple(watch_out),
        related=_tuple(related),
        example=example,
        module=module,
        object_name=object_name,
        phase=phase,
        aliases=_tuple(aliases),
        tags=_tuple(tags),
        notes=_tuple(notes),
        family=families,
        root=bool(root),
    )


# Family of every declaration in a group unless the declaration names its own.
_BOTH = ("olgs", "olss")
_GROUP_FAMILIES: dict[str, tuple[str, ...]] = {
    "optimizers": ("olgs",),
    "guesses": ("olgs",),
    "utils": ("olgs",),
    "schedules": ("olgs",),
    "base": ("olgs",),
    "core": ("olgs",),
    "campaigns": ("olgs",),
    "tracker": ("olgs",),
    "visualization": _BOTH,
    "system_olgs": ("olgs",),
    "system_olss": ("olss",),
    "shooting": ("olss",),
    "optimizer_log": ("olgs",),
}


_GROUP_MODULES = {
    "optimizers": "optimizer.optimizers",
    "guesses": "optimizer.guesses",
    "utils": "optimizer.utils",
    "schedules": "optimizer.schedules",
    "base": "optimizer",
    "core": "optimizer.core",
    "campaigns": "optimizer.campaigns",
    "tracker": "optimizer.tracker",
    "visualization": "optimizer.visualization",
    "system_olgs": "optimizer.system_olgs",
    "system_olss": "optimizer.system_olss",
    "shooting": "optimizer.shooting",
    "optimizer_log": "optimizer_log",
}


_OPTIMIZER_COMMON_INPUTS = {
    "system": "Object implementing the optimizer system contract. Most methods need evaluate(...) and gradient(...).",
    "controls": "Starting Controls. Optimizers can instead start from compatible state or warmstart data.",
    "maxiter": "Short chunk iteration budget. Longer workflows should chain chunks and warmstarts.",
    "step_size": "Initial scalar update size when supported.",
    "warmstart": "OptimizerResult, RunState, or WarmStartState used to continue a compatible method.",
    "accept_metric": "Scalar metric used for accept/reject decisions, usually J.",
    "gradient_tolerance": "Visible stationarity threshold for gradient-based methods; set None to disable.",
}


_RESULT_RETURN = {
    "type": "OptimizerResult",
    "fields": "controls, metrics, J, stop_reason, iterations, optimizer, state, trace, checkpoint_ids",
    "handoff": "Call result.warmstart(target_optimizer=...) before chaining methods.",
}


_GROUPS: dict[str, CatalogGroup] = {}


def _build_groups() -> dict[str, CatalogGroup]:
    optimizers_group = CatalogGroup(
        name="optimizers",
        summary="Methods that move Controls to improve system metrics.",
        common_inputs=_OPTIMIZER_COMMON_INPUTS,
        common_returns=_RESULT_RETURN,
        workflow=(
            "Start from opt.guesses.* or an existing Controls object.",
            "Run short chunks and inspect result.objective_value, result.metrics, and result.trace.",
            "Use result.warmstart(...) when chaining optimizers.",
        ),
        items=(
            _make_item(
                "adam",
                "optimizers",
                "optimizer",
                "Adaptive first-order optimizer with first and second moment state.",
                inputs={
                    "system": "Must provide evaluate(...) and gradient(...).",
                    "controls": "Starting Controls; optional when state or warmstart supplies controls.",
                    "variant": "adam, amsgrad, adamw, radam, or adabelief.",
                    "maxiter": "Number of chunk iterations.",
                    "step_size": "Learning rate as a scalar or serializable schedule object.",
                    "beta1/beta2/eps": "Moment and denominator stabilization parameters.",
                    "weight_decay": "Decoupled decay for adamw or explicit decay.",
                    "max_step_norm": "Optional L2 clipping of the applied update.",
                    "warmstart": "Carries Adam moment state only when target_optimizer is adam.",
                    "accept": "Optional guard; by default every finite Adam-family trial is accepted.",
                    "gradient_tolerance": "Gradient-norm stationarity stop; default 1e-8.",
                },
                returns=_RESULT_RETURN,
                requires=("system.evaluate", "system.gradient", "Controls"),
                variants=("adam", "amsgrad", "adamw", "radam", "adabelief"),
                best_for=("rough first optimization", "uneven gradient scales", "warmstarting before polishing"),
                not_for=("systems without analytical gradient", "hard residual repair by itself"),
                watch_out=("step_size controls the real behavior", "custom guards can reject and preserve moment state"),
                related=("utils.verify_gradient", "optimizers.lbfgs", "guesses.random_fourier_guess"),
                example="result = opt.optimizers.adam(system, controls, step_size=0.05, maxiter=50)",
                module="optimizer/optimizers/adam.py",
                phase="Phase 7",
                tags=("gradient", "adaptive", "first_order", "warmstart"),
                root=True,
            ),
            _make_item(
                "momentum",
                "optimizers",
                "optimizer",
                "Low-memory gradient optimizer with a velocity buffer.",
                inputs={
                    "system": "Must provide evaluate(...) and gradient(...).",
                    "controls": "Starting Controls; optional with state or warmstart.",
                    "variant": "heavy_ball, nesterov, restart, or clipped.",
                    "momentum": "Velocity carry coefficient in [0, 1).",
                    "step_size": "Gradient step scale as a scalar or serializable schedule object.",
                    "max_step_norm": "Optional update clipping; default active for clipped variant.",
                    "accept": "Optional guard; by default every finite momentum trial is accepted.",
                    "gradient_tolerance": "Gradient-norm stationarity stop; default 1e-8.",
                },
                returns=_RESULT_RETURN,
                requires=("system.evaluate", "system.gradient", "Controls"),
                variants=("heavy_ball", "nesterov", "restart", "clipped"),
                best_for=("simple gradient baselines", "smoothing zigzag motion", "debuggable first-order behavior"),
                watch_out=("large finite steps are standard behavior unless a custom guard is supplied", "restart variant resets velocity after rejection"),
                related=("optimizers.line_search", "optimizers.adam", "utils.diagnostic_report"),
                example="result = opt.optimizers.momentum(system, controls, variant='nesterov', step_size=0.03, maxiter=20)",
                module="optimizer/optimizers/momentum.py",
                phase="Phase 7",
                tags=("gradient", "first_order", "low_memory"),
                root=True,
            ),
            _make_item(
                "line_search",
                "optimizers",
                "optimizer",
                "Gradient descent family with fixed, normalized, backtracking, and Armijo steps.",
                inputs={
                    "system": "Must provide evaluate(...) and gradient(...).",
                    "controls": "Starting Controls; optional with state or warmstart.",
                    "variant": "fixed, normalized, backtracking, or armijo.",
                    "step_size": "Initial trial step.",
                    "shrink/grow": "Backtracking and next-step adaptation factors.",
                    "max_backtracks": "Trial attempts per iteration for backtracking/Armijo.",
                    "accept": "Optional custom accept function such as opt.metric_guard(...).",
                    "gradient_tolerance": "Gradient-norm stationarity stop; default 1e-8.",
                },
                returns=_RESULT_RETURN,
                requires=("system.evaluate", "system.gradient", "Controls"),
                variants=("fixed", "backtracking", "normalized", "armijo"),
                best_for=("auditable descent baseline", "step-size debugging", "guarded metric acceptance"),
                watch_out=("fixed steps can reject if too large", "Armijo uses directional derivative thresholds"),
                related=("core.metric_guard", "optimizers.momentum", "utils.verify_gradient"),
                example="result = opt.optimizers.line_search(system, controls, variant='backtracking', step_size=1.0, maxiter=10)",
                module="optimizer/optimizers/line_search.py",
                phase="Phase 7",
                tags=("gradient", "line_search", "first_order", "guardable"),
                root=True,
            ),
            _make_item(
                "adagrad",
                "optimizers",
                "optimizer",
                "Adaptive first-order method with cumulative squared-gradient scaling.",
                inputs={
                    "system": "Must provide evaluate(...) and gradient(...).",
                    "step_size": "Global learning-rate multiplier.",
                    "eps": "Denominator stabilizer.",
                    "initial_accumulator": "Initial per-coordinate squared-gradient history.",
                    "max_step_norm": "Optional update clipping.",
                    "gradient_tolerance": "Gradient-norm stationarity stop; default 1e-8.",
                },
                returns=_RESULT_RETURN,
                requires=("system.evaluate", "system.gradient", "Controls"),
                best_for=("sparse or uneven gradient coordinates", "simple adaptive scaling without momentum"),
                watch_out=("accumulator only grows, so later steps can become very small"),
                related=("optimizers.rmsprop", "optimizers.adam"),
                example="result = opt.optimizers.adagrad(system, controls, step_size=0.05, maxiter=30)",
                module="optimizer/optimizers/adaptive.py",
                phase="Phase 7",
                tags=("gradient", "adaptive", "first_order"),
                root=True,
            ),
            _make_item(
                "rmsprop",
                "optimizers",
                "optimizer",
                "Adaptive first-order method with moving-average squared-gradient scaling.",
                inputs={
                    "system": "Must provide evaluate(...) and gradient(...).",
                    "step_size": "Global learning-rate multiplier.",
                    "decay": "Moving-average memory coefficient in [0, 1).",
                    "eps": "Denominator stabilizer.",
                    "max_step_norm": "Optional update clipping.",
                    "gradient_tolerance": "Gradient-norm stationarity stop; default 1e-8.",
                },
                returns=_RESULT_RETURN,
                requires=("system.evaluate", "system.gradient", "Controls"),
                best_for=("nonstationary gradient scales", "adaptive scaling with less memory than Adam"),
                watch_out=("decay controls how quickly the scale estimate reacts"),
                related=("optimizers.adagrad", "optimizers.adam"),
                example="result = opt.optimizers.rmsprop(system, controls, step_size=0.02, decay=0.8, maxiter=30)",
                module="optimizer/optimizers/adaptive.py",
                phase="Phase 7",
                tags=("gradient", "adaptive", "first_order"),
                root=True,
            ),
            _make_item(
                "lbfgs",
                "optimizers",
                "optimizer",
                "Limited-memory quasi-Newton polish method using accepted curvature history.",
                inputs={
                    "system": "Must provide evaluate(...) and gradient(...).",
                    "controls": "Usually a reasonably good starting Controls object.",
                    "history_size": "Number of accepted curvature pairs to keep.",
                    "step_size": "Scalar multiplier for the L-BFGS direction.",
                    "curvature_eps": "Minimum positive s dot y required to store a pair.",
                    "max_step_norm": "Optional step clipping.",
                    "reject_shrink": "Step reduction after a rejected proposal.",
                    "gradient_tolerance": "Gradient-norm stationarity stop; default 1e-8.",
                },
                returns=_RESULT_RETURN,
                requires=("system.evaluate", "system.gradient", "Controls"),
                best_for=("polishing after Adam or line search", "smooth deterministic objectives", "low-memory curvature use"),
                not_for=("very rough global exploration", "systems without gradient"),
                watch_out=("bad curvature pairs are skipped", "too-large step_size can still reject"),
                related=("optimizers.adam", "optimizers.nonlinear_cg", "utils.diagnostic_report"),
                example="result = opt.optimizers.lbfgs(system, r1.controls, warmstart=r1.warmstart('lbfgs'), maxiter=20)",
                module="optimizer/optimizers/lbfgs.py",
                phase="Phase 7",
                tags=("gradient", "quasi_newton", "polish", "warmstart"),
                root=True,
            ),
            _make_item(
                "nonlinear_cg",
                "optimizers",
                "optimizer",
                "Low-memory nonlinear conjugate-gradient optimizer.",
                inputs={
                    "system": "Must provide evaluate(...) and gradient(...).",
                    "variant": "fletcher_reeves, polak_ribiere, polak_ribiere_plus, or hestenes_stiefel.",
                    "step_size": "Scalar update multiplier.",
                    "restart_on_nondescent": "Restart to steepest descent when conjugate direction is not descent.",
                    "max_step_norm": "Optional update clipping.",
                    "reject_shrink": "Step reduction and direction restart after rejection.",
                    "gradient_tolerance": "Gradient-norm stationarity stop; default 1e-8.",
                },
                returns=_RESULT_RETURN,
                requires=("system.evaluate", "system.gradient", "Controls"),
                variants=("fletcher_reeves", "polak_ribiere", "polak_ribiere_plus", "hestenes_stiefel"),
                best_for=("large control vectors", "middle ground between gradient descent and L-BFGS", "low-memory deterministic runs"),
                watch_out=("does not do a full Wolfe line search", "may restart often near noisy gradients"),
                related=("optimizers.lbfgs", "optimizers.line_search"),
                example="result = opt.optimizers.nonlinear_cg(system, controls, step_size=0.05, maxiter=25)",
                module="optimizer/optimizers/nonlinear_cg.py",
                phase="Phase 7",
                aliases=("ncg",),
                tags=("gradient", "conjugate_gradient", "low_memory"),
                root=True,
            ),
            _make_item(
                "ncg",
                "optimizers",
                "alias",
                "Short alias for nonlinear_cg.",
                inputs={"args/kwargs": "Forwarded directly to nonlinear_cg."},
                returns=_RESULT_RETURN,
                requires=("system.evaluate", "system.gradient", "Controls"),
                related=("optimizers.nonlinear_cg",),
                example="result = opt.optimizers.ncg(system, controls, maxiter=20)",
                module="optimizer/optimizers/nonlinear_cg.py",
                phase="Phase 7",
                tags=("gradient", "alias", "conjugate_gradient"),
                root=True,
            ),
            _make_item(
                "cma_es",
                "optimizers",
                "optimizer",
                "Derivative-free population search with isotropic or diagonal sampling radius.",
                inputs={
                    "system": "Needs evaluate(...). It does not use gradient during iterations.",
                    "controls": "Starting mean Controls; optional when state or warmstart supplies it.",
                    "variant": "diagonal or isotropic.",
                    "population_size": "Candidates sampled per generation; default scales with dimension.",
                    "elite_fraction": "Fraction retained to update the mean.",
                    "sigma": "Initial sampling radius.",
                    "seed": "Random seed for reproducibility.",
                    "state/warmstart": "Compatible continuation inputs using the standard optimizer call shape.",
                    "stopping/target/stall": "Standard structured stopping configuration or explicit target/stall options.",
                },
                returns=_RESULT_RETURN,
                requires=("system.evaluate", "Controls, state, or warmstart"),
                variants=("diagonal", "isotropic"),
                best_for=("rough global search", "systems with unreliable gradients", "escaping poor starts"),
                not_for=("fast polishing near a good deterministic solution", "very high-dimensional expensive evaluations without budget"),
                watch_out=("uses many evaluations per iteration", "implementation is compact CMA-ES style, not full research CMA-ES"),
                related=("guesses.random_fourier_guess", "optimizers.adam", "utils.metric_report"),
                example="result = opt.optimizers.cma_es(system, controls, population_size=24, sigma=0.15, seed=7, maxiter=6)",
                module="optimizer/optimizers/cma_es.py",
                phase="Phase 7",
                tags=("derivative_free", "population", "random", "global_search"),
                root=True,
            ),
            _make_item(
                "projected_gradient",
                "optimizers",
                "optimizer",
                "Equality- and bound-constrained descent with restoration and projected-KKT stopping.",
                inputs={
                    "system": "Base OLGS plus residuals(...), jacobian(...), and optional control_bounds().",
                    "residual_name": "Named hard residual set, default hard.",
                    "feasibility_tolerance": "Scaled residual norm required in the feasible phase.",
                    "stationarity_tolerance": "Projected-KKT norm stopping threshold.",
                },
                returns=_RESULT_RETURN,
                requires=("system.evaluate", "system.gradient", "system.residuals", "system.jacobian", "Controls"),
                best_for=("hard equality constraints", "box-bounded feasible polishing"),
                watch_out=("general nonlinear inequalities are not supported", "finite-difference Jacobians are not used implicitly"),
                related=("utils.project_gradient", "utils.repair_newton"),
                example="result = opt.optimizers.projected_gradient(system, controls, maxiter=50)",
                module="optimizer/optimizers/projected.py",
                tags=("gradient", "constraints", "projected_kkt", "feasibility"),
                root=True,
                phase="Numerical capabilities",
            ),
            _make_item(
                "newton_cg",
                "optimizers",
                "optimizer",
                "Matrix-free Newton method using conjugate gradients and analytical Hessian-vector products.",
                inputs={"system": "Base OLGS plus hvp(controls, vector).", "cg_maxiter": "Inner linear-solve budget.", "gradient_tolerance": "Gradient-norm stop with curvature classification.", "reject_shrink": "Step reduction after rejection; the configured full step is restored after acceptance."},
                returns=_RESULT_RETURN,
                requires=("system.evaluate", "system.gradient", "system.hvp", "Controls"),
                best_for=("smooth deterministic objectives", "large controls with matrix-free curvature"),
                watch_out=("a finite-difference HVP is never selected implicitly",),
                related=("optimizers.trust_region", "optimizers.lbfgs"),
                example="result = opt.optimizers.newton_cg(system, controls, maxiter=20)",
                module="optimizer/optimizers/newton_cg.py",
                tags=("gradient", "second_order", "hvp", "conjugate_gradient"),
                root=True,
                phase="Numerical capabilities",
            ),
            _make_item(
                "trust_region",
                "optimizers",
                "optimizer",
                "Steihaug trust-region method with model-ratio acceptance and matrix-free HVPs.",
                inputs={"system": "Base OLGS plus hvp(controls, vector).", "initial_radius": "Initial L2 trust radius.", "gradient_tolerance": "Gradient-norm stop with curvature classification."},
                returns=_RESULT_RETURN,
                requires=("system.evaluate", "system.gradient", "system.hvp", "Controls"),
                best_for=("indefinite or locally inaccurate quadratic models", "guarded second-order steps"),
                watch_out=("currently a minimization method",),
                related=("optimizers.newton_cg",),
                example="result = opt.optimizers.trust_region(system, controls, initial_radius=1.0, maxiter=20)",
                module="optimizer/optimizers/trust_region.py",
                tags=("gradient", "second_order", "hvp", "trust_region"),
                root=True,
                phase="Numerical capabilities",
            ),
            _make_item(
                "proximal_gradient",
                "optimizers",
                "optimizer",
                "Composite-objective descent using a system smooth gradient and proximal operator.",
                inputs={"system": "Base OLGS plus smooth_gradient(...) and prox(...).", "mapping_tolerance": "Proximal-gradient mapping threshold."},
                returns=_RESULT_RETURN,
                requires=("system.evaluate", "system.smooth_gradient", "system.prox", "Controls"),
                best_for=("L1 regularization", "total variation or other available proximal operators"),
                watch_out=("evaluate must return the total composite J",),
                example="result = opt.optimizers.proximal_gradient(system, controls, step_size=0.1, maxiter=50)",
                module="optimizer/optimizers/proximal.py",
                tags=("composite", "proximal", "nonsmooth"),
                root=True,
                phase="Numerical capabilities",
            ),
        ),
    )

    guesses_group = CatalogGroup(
        name="guesses",
        summary="Initial Controls generators and restart helpers.",
        common_inputs={
            "target": "A system with control_spec() or a ControlSpec.",
            "amplitude": "Scalar, per-channel mapping, or per-channel sequence for generated channels.",
            "channels": "Optional channel name or names to fill; unselected channels stay zero.",
            "endpoint": "free/hold keeps endpoints, zero forces selected endpoints to zero.",
            "envelope": "Optional shaping envelope such as hann, sin2, gaussian, or smoothstep.",
            "scale": "Amplitude convention: max_abs, l2, energy, or none.",
            "seed": "Random seed where supported.",
        },
        common_returns={
            "type": "Controls",
            "fields": "spec, matrix values, name, and meta describing the guess family.",
        },
        workflow=(
            "Use deterministic guesses for debugging and reproducibility.",
            "Use random_smooth_guess or random_fourier_guess for multi-start exploration.",
            "Use scale_guess, mix_guess, and perturb_guess around known useful controls.",
        ),
        items=(
            _make_item(
                "constant_guess",
                "guesses",
                "guess",
                "Selected-channel constant Controls.",
                inputs={
                    "value": "Scalar, channel mapping, or per-channel sequence.",
                    "channels": "Optional channels to fill.",
                    "endpoint": "free/hold or zero.",
                },
                returns={"type": "Controls", "meaning": "Constant selected channels."},
                requires=("ControlSpec or system.control_spec",),
                best_for=("simple offsets", "channel sign tests", "known constant baselines"),
                example="controls = opt.guesses.constant_guess(system, value={'ux': 0.2, 'uy': 0.0, 'uz': -0.1})",
                module="optimizer/guesses/simple.py",
                phase="Phase 9",
                tags=("deterministic", "simple"),
                root=True,
            ),
            _make_item(
                "ramp_guess",
                "guesses",
                "guess",
                "Monotone ramp between per-channel start and stop values.",
                inputs={
                    "start/stop": "Scalar, channel mapping, or sequence.",
                    "kind": "linear, quadratic, or smoothstep.",
                    "channels": "Optional channels to fill.",
                },
                returns={"type": "Controls", "meaning": "Smooth monotone selected-channel ramp."},
                requires=("ControlSpec or system.control_spec",),
                best_for=("testing endpoint behavior", "slow turn-on controls", "simple curriculum starts"),
                example="controls = opt.guesses.ramp_guess(system, start=0.0, stop={'ux': 0.3})",
                module="optimizer/guesses/simple.py",
                phase="Phase 9",
                tags=("deterministic", "simple", "smooth"),
                root=True,
            ),
            _make_item(
                "sine_guess",
                "guesses",
                "guess",
                "Sine-wave guess with channel-wise frequency and phase.",
                inputs={
                    "amplitude": "Output amplitude under the selected scale convention.",
                    "frequency": "Cycles over the normalized control interval.",
                    "phase": "Phase shift in radians.",
                    "envelope/endpoint/scale": "Common shaping controls.",
                },
                returns={"type": "Controls", "meaning": "Smooth periodic selected-channel pulse."},
                requires=("ControlSpec or system.control_spec",),
                best_for=("single-frequency starts", "phase/frequency experiments", "smooth deterministic controls"),
                example="controls = opt.guesses.sine_guess(system, amplitude=0.2, frequency=2, endpoint='zero')",
                module="optimizer/guesses/harmonic.py",
                phase="Phase 9",
                tags=("deterministic", "harmonic", "smooth"),
                root=True,
            ),
            _make_item(
                "cosine_guess",
                "guesses",
                "guess",
                "Cosine-wave guess with channel-wise frequency and phase.",
                inputs={
                    "amplitude": "Output amplitude under the selected scale convention.",
                    "frequency": "Cycles over the normalized control interval.",
                    "phase": "Phase shift in radians.",
                    "envelope/endpoint/scale": "Common shaping controls.",
                },
                returns={"type": "Controls", "meaning": "Smooth periodic selected-channel pulse."},
                requires=("ControlSpec or system.control_spec",),
                best_for=("even-symmetry pulse starts", "smooth deterministic controls"),
                example="controls = opt.guesses.cosine_guess(system, amplitude=0.2, frequency=1)",
                module="optimizer/guesses/harmonic.py",
                phase="Phase 9",
                tags=("deterministic", "harmonic", "smooth"),
                root=True,
            ),
            _make_item(
                "gaussian_guess",
                "guesses",
                "guess",
                "Localized Gaussian pulse guess.",
                inputs={
                    "amplitude": "Peak scale under the selected convention.",
                    "center": "Pulse center in normalized time.",
                    "width": "Positive normalized width.",
                    "envelope/endpoint/scale": "Common shaping controls.",
                },
                returns={"type": "Controls", "meaning": "Localized selected-channel pulse."},
                requires=("ControlSpec or system.control_spec",),
                best_for=("localized pulses", "smooth starts with low endpoint disturbance"),
                watch_out=("width must be positive",),
                example="controls = opt.guesses.gaussian_guess(system, amplitude=0.2, width=0.15)",
                module="optimizer/guesses/harmonic.py",
                phase="Phase 9",
                tags=("deterministic", "localized", "smooth"),
                root=True,
            ),
            _make_item(
                "sinc_guess",
                "guesses",
                "guess",
                "Centered sinc pulse with controllable side-lobe width.",
                inputs={
                    "amplitude": "Output amplitude under the selected scale convention.",
                    "center": "Pulse center in normalized time.",
                    "width": "Positive side-lobe width parameter.",
                },
                returns={"type": "Controls", "meaning": "Sinc-shaped selected-channel pulse."},
                requires=("ControlSpec or system.control_spec",),
                best_for=("band-limited style starts", "localized pulse experiments with side lobes"),
                watch_out=("width must be positive",),
                example="controls = opt.guesses.sinc_guess(system, amplitude=0.25, width=5.0)",
                module="optimizer/guesses/harmonic.py",
                phase="Phase 9",
                tags=("deterministic", "localized", "harmonic"),
                root=True,
            ),
            _make_item(
                "fourier_guess",
                "guesses",
                "guess",
                "Low-frequency Fourier-series deterministic guess.",
                inputs={
                    "modes": "Number of sine modes.",
                    "frequency_base": "Base frequency multiplier.",
                    "coefficients/phases": "Optional mode coefficients and phases.",
                    "decay": "flat, 1/k, or 1/k2 coefficient template when coefficients are omitted.",
                },
                returns={"type": "Controls", "meaning": "Smooth Fourier-series selected-channel pulse."},
                requires=("ControlSpec or system.control_spec",),
                best_for=("structured smooth starts", "low-frequency pulse families", "basis-style experiments"),
                watch_out=("modes must be >= 1", "frequency_base must be positive"),
                related=("guesses.random_fourier_guess", "utils.control_spectrum"),
                example="controls = opt.guesses.fourier_guess(system, amplitude=0.2, modes=4, decay='1/k2')",
                module="optimizer/guesses/harmonic.py",
                phase="Phase 9",
                tags=("deterministic", "fourier", "smooth", "low_frequency"),
                root=True,
            ),
            _make_item(
                "random_guess",
                "guesses",
                "guess",
                "Raw random control guess.",
                inputs={
                    "distribution": "uniform, normal, or rademacher.",
                    "amplitude": "Scalar/mapping/sequence, or two-number random amplitude range.",
                    "seed": "Random seed for reproducibility.",
                },
                returns={"type": "Controls", "meaning": "Random selected-channel pulse."},
                requires=("ControlSpec or system.control_spec",),
                best_for=("stress tests", "rough random baselines", "checking optimizer robustness"),
                not_for=("physically smooth starts unless envelope/smoothing is added",),
                related=("guesses.random_smooth_guess", "guesses.random_fourier_guess"),
                example="controls = opt.guesses.random_guess(system, amplitude=0.2, distribution='normal', seed=4)",
                module="optimizer/guesses/random.py",
                phase="Phase 9",
                tags=("random", "baseline"),
                root=True,
            ),
            _make_item(
                "random_smooth_guess",
                "guesses",
                "guess",
                "Random guess smoothed by a Gaussian correlation kernel.",
                inputs={
                    "correlation": "Smoothing width; <=1 is interpreted as fraction of control_dim.",
                    "distribution": "uniform or normal raw samples before smoothing.",
                    "envelope": "Defaults to hann.",
                    "seed": "Random seed for reproducibility.",
                },
                returns={"type": "Controls", "meaning": "Smooth random selected-channel pulse."},
                requires=("ControlSpec or system.control_spec",),
                best_for=("multi-start exploration", "smoother random pulses", "less high-frequency roughness than random_guess"),
                related=("utils.smoothness_report", "guesses.random_fourier_guess"),
                example="controls = opt.guesses.random_smooth_guess(system, amplitude=0.2, correlation=0.2, seed=4)",
                module="optimizer/guesses/random.py",
                phase="Phase 9",
                tags=("random", "smooth", "multi_start"),
                root=True,
            ),
            _make_item(
                "random_fourier_guess",
                "guesses",
                "guess",
                "Random low-frequency Fourier-series guess.",
                inputs={
                    "modes": "Number of random Fourier modes.",
                    "coefficient_scale": "Standard deviation scale for random coefficients.",
                    "decay": "flat, 1/k, or 1/k2 random coefficient damping.",
                    "seed": "Random seed for reproducibility.",
                },
                returns={"type": "Controls", "meaning": "Smooth low-frequency random pulse."},
                requires=("ControlSpec or system.control_spec",),
                best_for=("default multi-start guess", "smooth global exploration", "starting Adam or CMA-ES"),
                related=("optimizers.adam", "optimizers.cma_es", "utils.control_spectrum"),
                example="controls = opt.guesses.random_fourier_guess(system, amplitude=0.1, modes=5, seed=1)",
                module="optimizer/guesses/random.py",
                phase="Phase 9",
                tags=("random", "fourier", "smooth", "multi_start", "low_frequency"),
                root=True,
            ),
            _make_item(
                "scale_guess",
                "guesses",
                "restart_helper",
                "Rescale existing Controls without mutating them.",
                inputs={
                    "controls": "Existing Controls to rescale.",
                    "amplitude": "Target amplitude convention.",
                    "offset": "Optional per-channel offset.",
                    "channels": "Optional channels to retain/rescale.",
                },
                returns={"type": "Controls", "meaning": "Rescaled copy of existing controls."},
                requires=("Controls",),
                best_for=("trying lower amplitudes", "normalizing a previous pulse", "restart preparation"),
                example="scaled = opt.guesses.scale_guess(controls, amplitude=0.1)",
                module="optimizer/guesses/composite.py",
                phase="Phase 9",
                tags=("composite", "restart", "rescale"),
                root=True,
            ),
            _make_item(
                "mix_guess",
                "guesses",
                "restart_helper",
                "Weighted mix of compatible Controls objects.",
                inputs={
                    "guesses": "Sequence of Controls or first Controls argument.",
                    "weights": "Optional weights; defaults to equal average.",
                },
                returns={"type": "Controls", "meaning": "Weighted linear combination with shared ControlSpec."},
                requires=("Controls",),
                best_for=("combining useful starts", "averaging nearby candidates", "manual basis experiments"),
                watch_out=("all guesses must share keys and control_dim",),
                example="mixed = opt.guesses.mix_guess([base, perturb], weights=[0.8, 0.2])",
                module="optimizer/guesses/composite.py",
                phase="Phase 9",
                tags=("composite", "restart"),
                root=True,
            ),
            _make_item(
                "perturb_guess",
                "guesses",
                "restart_helper",
                "Add a generated perturbation to existing Controls.",
                inputs={
                    "controls": "Base Controls.",
                    "kind": "random_fourier, random_smooth, random, or fourier.",
                    "amplitude": "Perturbation amplitude.",
                    "seed": "Random seed for random perturbation kinds.",
                    "kwargs": "Forwarded to the selected perturbation generator.",
                },
                returns={"type": "Controls", "meaning": "Base controls plus generated perturbation."},
                requires=("Controls",),
                best_for=("local restarts around a good pulse", "escaping a shallow local basin", "small random exploration"),
                related=("guesses.random_fourier_guess", "optimizers.adam"),
                example="trial = opt.guesses.perturb_guess(best.controls, amplitude=0.01, kind='random_fourier', seed=2, modes=3)",
                module="optimizer/guesses/composite.py",
                phase="Phase 9",
                tags=("composite", "restart", "random"),
                root=True,
            ),
        ),
    )

    utils_group = CatalogGroup(
        name="utils",
        summary="Diagnostics, derivative checks, residual geometry, repair, guards, and spectrum tools.",
        common_inputs={
            "system": "Required for tools that evaluate metrics, gradients, residuals, or Jacobians.",
            "controls": "Controls object to inspect, verify, project, or repair.",
            "residuals": "Named residual hook, default hard, where relevant.",
            "eps": "Finite-difference scale for derivative and Jacobian fallback tools.",
        },
        common_returns={
            "dict": "Most diagnostics return JSON-friendly dictionaries.",
            "Controls": "Projection and finite-difference gradient tools can return Controls.",
            "RepairResult": "repair_newton returns a rich object with to_dict(...).",
        },
        workflow=(
            "Run verify_gradient before trusting optimizer behavior on a new system.",
            "Use diagnostic_report when a run gets stuck or metrics look wrong.",
            "Use geometry_probe, project_gradient, and repair_newton around residual-constrained workflows.",
        ),
        items=(
            _make_item(
                "metric_report",
                "utils",
                "diagnostic",
                "Evaluate and summarize current system metrics.",
                inputs={"system": "Must provide evaluate(...).", "controls": "Controls to evaluate."},
                returns={"type": "dict", "fields": "kind, metrics, raw_metric_keys"},
                requires=("system.evaluate", "Controls"),
                best_for=("quick metric snapshot", "checking objective names", "lightweight logs"),
                example="report = opt.utils.metric_report(system, controls)",
                module="optimizer/utils/diagnostics.py",
                phase="Phase 11",
                tags=("diagnostic", "metrics"),
                root=True,
            ),
            _make_item(
                "diagnostic_report",
                "utils",
                "diagnostic",
                "Metrics plus control, residual, and optional gradient summaries.",
                inputs={
                    "system": "Usually provides evaluate(...), gradient(...), and optional residuals(...).",
                    "controls": "Controls to inspect.",
                    "residuals": "Residual hook name or None to skip residuals.",
                    "include_gradient": "Whether to include gradient norm summaries.",
                },
                returns={"type": "dict", "fields": "system_hooks, control_spec, controls, metrics, gradient, residuals"},
                requires=("system.evaluate", "Controls"),
                best_for=("getting unstuck", "post-run inspection", "checking residual availability and gradient scale"),
                related=("utils.geometry_probe", "utils.verify_gradient", "utils.control_spectrum"),
                example="report = opt.utils.diagnostic_report(system, result.controls)",
                module="optimizer/utils/diagnostics.py",
                phase="Phase 11",
                tags=("diagnostic", "metrics", "residual", "gradient"),
                root=True,
            ),
            _make_item(
                "geometry_probe",
                "utils",
                "residual_geometry",
                "Residual/Jacobian rank, conditioning, and repair feasibility summary.",
                inputs={
                    "system": "Must provide residuals(...); jacobian(...) is optional when fallback=True.",
                    "controls": "Controls at which to inspect local geometry.",
                    "fallback": "Use finite-difference Jacobian if analytical hook is missing.",
                    "rcond": "Optional rank cutoff.",
                },
                returns={"type": "dict", "fields": "residual_norm, jacobian_source, rank, nullspace_dimension, condition_number"},
                requires=("system.residuals", "Controls"),
                best_for=("checking if repair/projection is locally possible", "diagnosing rank deficiency", "understanding hard residual constraints"),
                related=("utils.repair_newton", "utils.nullspace_basis", "utils.project_gradient"),
                example="geometry = opt.utils.geometry_probe(system, controls, eps=1e-6)",
                module="optimizer/utils/diagnostics.py",
                phase="Phase 11",
                tags=("residual", "jacobian", "geometry", "diagnostic"),
                root=True,
            ),
            _make_item(
                "finite_difference_gradient",
                "utils",
                "derivative",
                "Full coordinate finite-difference gradient for a scalar metric.",
                inputs={
                    "system": "Must provide evaluate(...).",
                    "controls": "Controls to perturb coordinate by coordinate.",
                    "metric": "Scalar metric to differentiate, default J.",
                    "method": "central or forward.",
                },
                returns={"type": "Controls", "meaning": "Finite-difference gradient in the same ControlSpec layout."},
                requires=("system.evaluate", "Controls"),
                best_for=("small debugging systems", "checking full gradient arrays"),
                not_for=("large control vectors in normal loops",),
                watch_out=("cost is one or two evaluations per control coordinate",),
                related=("utils.verify_gradient",),
                example="fd = opt.utils.finite_difference_gradient(system, controls, eps=1e-6)",
                module="optimizer/utils/derivatives.py",
                phase="Phase 11",
                tags=("derivative", "finite_difference", "gradient"),
                root=True,
            ),
            _make_item(
                "verify_gradient",
                "utils",
                "derivative",
                "Directional finite-difference check of the analytical gradient.",
                inputs={
                    "system": "Must provide evaluate(...) and gradient(...).",
                    "controls": "Controls at which to check the gradient.",
                    "directions": "Number of random directions or explicit direction matrix.",
                    "rtol/atol": "Pass thresholds for relative/absolute error.",
                    "seed": "Random seed for generated directions.",
                },
                returns={"type": "dict", "fields": "passed, max_absolute_error, max_relative_error, directions"},
                requires=("system.evaluate", "system.gradient", "Controls"),
                best_for=("first check on a new system", "debugging optimizer failures", "detecting sign/scale/channel-order mistakes"),
                watch_out=("directional checks are cheaper than full finite differences but not exhaustive",),
                related=("utils.finite_difference_gradient", "optimizers.adam"),
                example="check = opt.utils.verify_gradient(system, controls, eps=1e-6, directions=8)",
                module="optimizer/utils/derivatives.py",
                phase="Phase 11",
                tags=("derivative", "finite_difference", "gradient", "check"),
                root=True,
            ),
            _make_item(
                "finite_difference_jacobian",
                "utils",
                "derivative",
                "Full finite-difference Jacobian of system residuals.",
                inputs={
                    "system": "Must provide residuals(...).",
                    "controls": "Controls to perturb coordinate by coordinate.",
                    "residuals": "Residual hook name, default hard.",
                    "method": "central or forward.",
                },
                returns={"type": "np.ndarray", "shape": "(n_residuals, controls.spec.size)"},
                requires=("system.residuals", "Controls"),
                best_for=("small systems without analytical jacobian", "repair/projection debugging"),
                watch_out=("expensive for large control vectors",),
                related=("utils.verify_jacobian", "utils.geometry_probe"),
                example="jac = opt.utils.finite_difference_jacobian(system, controls, eps=1e-6)",
                module="optimizer/utils/derivatives.py",
                phase="Phase 11",
                tags=("derivative", "finite_difference", "jacobian", "residual"),
                root=True,
            ),
            _make_item(
                "verify_jacobian",
                "utils",
                "derivative",
                "Directional finite-difference check of analytical residual Jacobian.",
                inputs={
                    "system": "Must provide residuals(...) and usually jacobian(...).",
                    "controls": "Controls at which to check the Jacobian.",
                    "fallback": "Allow finite-difference source when analytical jacobian is missing.",
                    "directions": "Number of random directions or explicit direction matrix.",
                },
                returns={"type": "dict", "fields": "passed, jacobian_source, max_absolute_error, max_relative_error"},
                requires=("system.residuals", "Controls"),
                best_for=("validating residual Jacobian hooks", "repair/projection debugging"),
                related=("utils.finite_difference_jacobian", "utils.geometry_probe"),
                example="check = opt.utils.verify_jacobian(system, controls, eps=1e-6, directions=8)",
                module="optimizer/utils/derivatives.py",
                phase="Phase 11",
                tags=("derivative", "finite_difference", "jacobian", "residual", "check"),
                root=True,
            ),
            _make_item(
                "nullspace_basis",
                "utils",
                "residual_geometry",
                "Basis columns for the local residual-Jacobian nullspace.",
                inputs={
                    "system": "Must provide residuals(...); jacobian(...) optional with fallback.",
                    "controls": "Controls at which to compute the local basis.",
                    "rcond": "Optional rank cutoff.",
                },
                returns={"type": "np.ndarray", "shape": "(controls.spec.size, nullspace_dimension)"},
                requires=("system.residuals", "Controls"),
                best_for=("understanding feasible first-order directions", "projected method experiments"),
                related=("utils.project_gradient", "utils.geometry_probe"),
                example="basis = opt.utils.nullspace_basis(system, controls, eps=1e-6)",
                module="optimizer/utils/geometry.py",
                phase="Phase 11",
                tags=("residual", "jacobian", "geometry", "projection"),
                root=True,
            ),
            _make_item(
                "project_gradient",
                "utils",
                "projection",
                "Remove local residual row-space component from a gradient-like Controls object.",
                inputs={
                    "system": "Must provide residuals(...); jacobian(...) optional with fallback.",
                    "controls": "Point defining the residual Jacobian.",
                    "gradient": "Controls-shaped gradient/direction to project.",
                    "return_info": "Return dict with geometry and projected Controls when True.",
                },
                returns={"type": "Controls | dict", "meaning": "Projected gradient or info dict containing projected_gradient."},
                requires=("system.residuals", "Controls"),
                best_for=("first-order constrained descent experiments", "checking if a gradient changes hard residuals"),
                watch_out=("does not choose step size or run an optimizer",),
                related=("utils.nullspace_basis", "utils.repair_newton"),
                example="info = opt.utils.project_gradient(system, controls, gradient, return_info=True)",
                module="optimizer/utils/geometry.py",
                phase="Phase 11",
                tags=("residual", "jacobian", "geometry", "projection"),
                root=True,
            ),
            _make_item(
                "repair_newton",
                "utils",
                "repair",
                "Newton/LM-style residual repair that moves controls to reduce a named residual vector.",
                inputs={
                    "system": "Must provide residuals(...); jacobian(...) optional with fallback.",
                    "controls": "Controls to repair.",
                    "method": "newton, lm, or damped.",
                    "tolerance": "Residual-norm convergence threshold.",
                    "damping": "LM/damped solve regularization.",
                    "line_search": "Shrink repair delta until residual norm improves.",
                },
                returns={"type": "RepairResult", "fields": "controls, residual_norm, converged, iterations, history, to_dict(...)"},
                requires=("system.residuals", "Controls"),
                variants=("newton", "lm", "damped"),
                best_for=("restoring hard residual feasibility", "post-optimizer repair", "local residual solves"),
                not_for=("optimizing J, fidelity, or energy directly",),
                watch_out=("finite-difference fallback can be expensive", "line search can fail when local model is poor"),
                related=("utils.geometry_probe", "utils.project_gradient", "utils.verify_jacobian"),
                example="fixed = opt.utils.repair_newton(system, controls, maxiter=4, tolerance=1e-8)",
                module="optimizer/utils/repairs.py",
                phase="Phase 11",
                tags=("residual", "repair", "jacobian"),
                root=True,
            ),
            _make_item(
                "control_spectrum",
                "utils",
                "spectrum",
                "FFT-based channel-wise amplitude and power summaries.",
                inputs={
                    "controls": "Controls to analyze.",
                    "dt": "Sample spacing; defaults to controls.spec.dt or 1.0.",
                    "high_frequency_cutoff": "Optional frequency threshold for high-power fraction.",
                },
                returns={"type": "dict", "fields": "frequencies, dominant_frequency, total_power, amplitude, power"},
                requires=("Controls",),
                best_for=("seeing frequency content", "checking high-frequency artifacts", "comparing smooth/random guesses"),
                related=("utils.smoothness_report", "guesses.fourier_guess"),
                example="spectrum = opt.utils.control_spectrum(controls, high_frequency_cutoff=5.0)",
                module="optimizer/utils/spectrum.py",
                phase="Phase 11",
                tags=("spectrum", "diagnostic", "controls"),
                root=True,
            ),
            _make_item(
                "smoothness_report",
                "utils",
                "spectrum",
                "Finite-difference roughness, jump, and total-variation summaries.",
                inputs={"controls": "Controls to analyze."},
                returns={"type": "dict", "fields": "first_difference_norm, second_difference_norm, total_variation, max_jump"},
                requires=("Controls",),
                best_for=("checking pulse roughness", "comparing raw vs smooth guesses", "spotting large jumps"),
                related=("utils.control_spectrum", "guesses.random_smooth_guess"),
                example="smoothness = opt.utils.smoothness_report(controls)",
                module="optimizer/utils/spectrum.py",
                phase="Phase 11",
                tags=("spectrum", "smoothness", "diagnostic", "controls"),
                root=True,
            ),
            _make_item(
                "RepairResult",
                "utils",
                "result_object",
                "Result object returned by repair_newton.",
                inputs={"constructor": "Usually not called directly; produced by repair_newton."},
                returns={"type": "RepairResult", "fields": "controls, residuals, residual_norm, converged, history, to_dict(...)"},
                requires=("Controls",),
                best_for=("inspecting repair outcome", "compact repair logs via to_dict(...)"),
                related=("utils.repair_newton",),
                example="payload = fixed.to_dict(include_controls=False)",
                module="optimizer/utils/repairs.py",
                phase="Phase 11",
                tags=("result", "repair"),
                root=True,
            ),
            _make_item("diagnose_run", "utils", "diagnostic", "Classify a short OLGS or OLSS history as converging, slow, stalled, oscillating, diverging, or repeating rejections.", module="optimizer/utils/run_diagnostics.py", tags=("diagnostic", "short_run", "history"), root=True, family=("olgs", "olss")),
            _make_item("curvature_probe", "utils", "diagnostic", "Probe local curvature, negative-curvature evidence, and a suggested gradient step.", module="optimizer/utils/run_diagnostics.py", tags=("diagnostic", "curvature", "hvp"), root=True),
            _make_item("compare_methods", "utils", "diagnostic", "Compare several optimizers for a short in-memory budget from identical controls.", module="optimizer/utils/run_diagnostics.py", tags=("diagnostic", "comparison", "short_run"), root=True),
            _make_item("shooting_geometry_probe", "utils", "diagnostic", "Report OLSS Jacobian rank, condition, weak variable blocks, and residual groups.", module="optimizer/utils/run_diagnostics.py", tags=("diagnostic", "shooting", "jacobian"), root=True, family=("olss",)),
        ),
    )

    schedules_group = CatalogGroup(
        name="schedules",
        summary="Small step-size policy objects for optimizer or repair workflows.",
        common_inputs={
            "current": "Current step size passed to update(...), when available.",
            "accepted": "Boolean outcome used by adaptive schedules.",
        },
        common_returns={"type": "schedule object", "methods": "initial() and update(...) return finite float step sizes."},
        workflow=(
            "Use constant schedules when step size should stay fixed.",
            "Use adaptive schedules to shrink on rejection and optionally grow on acceptance.",
        ),
        items=(
            _make_item(
                "ConstantSchedule",
                "schedules",
                "schedule",
                "Schedule that always returns the same step size.",
                inputs={"value": "Finite positive scalar step size."},
                returns={"type": "ConstantSchedule", "methods": "initial() and update(...)"},
                requires=("finite scalar value",),
                best_for=("fixed-step experiments", "consistent repair/optimizer policy"),
                example="schedule = opt.schedules.ConstantSchedule(0.25)",
                module="optimizer/schedules/step_size.py",
                tags=("schedule", "step_size"),
                root=True,
            ),
            _make_item(
                "AdaptiveStepSchedule",
                "schedules",
                "schedule",
                "Shrink/grow step-size policy with optional min/max clamps.",
                inputs={
                    "initial_step": "Initial positive step.",
                    "shrink": "Factor in (0, 1) used after rejection.",
                    "grow": "Positive multiplier used after acceptance.",
                    "min_step/max_step": "Lower/upper clamps.",
                },
                returns={"type": "AdaptiveStepSchedule", "methods": "initial() and update(current, accepted=...)"},
                requires=("finite scalar parameters",),
                best_for=("accept/reject driven step control", "manual closed-loop experiments"),
                example="schedule = opt.schedules.AdaptiveStepSchedule(initial_step=1.0, shrink=0.5, grow=1.2)",
                module="optimizer/schedules/step_size.py",
                tags=("schedule", "step_size", "adaptive"),
                root=True,
            ),
            _make_item("ExponentialDecaySchedule", "schedules", "schedule", "Exponential accepted-iteration step decay with a floor.", module="optimizer/schedules/step_size.py", tags=("schedule", "decay"), root=True),
            _make_item("LinearWarmupSchedule", "schedules", "schedule", "Linear warmup from an initial to a target step.", module="optimizer/schedules/step_size.py", tags=("schedule", "warmup"), root=True),
            _make_item("CosineDecaySchedule", "schedules", "schedule", "Cosine step decay over a fixed iteration horizon.", module="optimizer/schedules/step_size.py", tags=("schedule", "cosine", "decay"), root=True),
        ),
    )

    base_group = CatalogGroup(
        name="base",
        summary="Base data objects and top-level entry points: Controls, results, warmstart state, trace, and bound context utilities.",
        common_inputs={
            "system": "Project physics object satisfying the optimizer contract.",
            "controls": "Named dense control matrix represented by Controls.",
        },
        common_returns={
            "Evaluation": "Single evaluate(...) snapshot.",
            "OptimizerResult": "Standard optimizer result.",
            "Trace": "In-memory run records and checkpoints.",
        },
        workflow=(
            "Build or receive Controls matching system.control_spec().",
            "Call evaluate/gradient or a role namespace method.",
            "Carry OptimizerResult, WarmStartState, or Trace across chunks.",
        ),
        items=(
            _make_item(
                "ControlSpec",
                "base",
                "data_object",
                "Named control layout: channel keys, control_dim, dtype, dt, and metadata.",
                inputs={"keys": "Unique channel names.", "control_dim": "Number of time samples."},
                returns={"type": "ControlSpec", "fields": "keys, shape, size, dt, meta, to_dict()"},
                requires=("channel names", "control dimension"),
                best_for=("declaring the control layout expected by a system",),
                example="spec = opt.ControlSpec(('ux', 'uy', 'uz'), 33)",
                module="optimizer/controls.py",
                tags=("controls", "data"),
            ),
            _make_item(
                "Controls",
                "base",
                "data_object",
                "Named control values backed by a dense (n_controls, control_dim) matrix.",
                inputs={"spec": "ControlSpec.", "values": "Matrix with shape spec.shape."},
                returns={"type": "Controls", "methods": "as_matrix, flatten, channel, norm, max_abs, arithmetic"},
                requires=("ControlSpec", "finite matrix"),
                best_for=("all optimizer, guess, diagnostic, and repair calls",),
                example="controls = opt.Controls.zeros(system.control_spec())",
                module="optimizer/controls.py",
                tags=("controls", "data"),
            ),
            _make_item(
                "evaluate",
                "base",
                "helper",
                "Evaluate a system/Controls pair and wrap metrics in Evaluation.",
                inputs={"system": "Must provide evaluate(...).", "controls": "Controls matching system.control_spec()."},
                returns={"type": "Evaluation", "fields": "controls, metrics, J, to_dict(...)"},
                requires=("system.evaluate", "Controls"),
                best_for=("quick objective/metric snapshots",),
                example="evaluation = opt.evaluate(system, controls)",
                module="optimizer/core/evaluate.py",
                phase="Phase 5",
                tags=("evaluation", "metrics"),
                family=("olgs", "olss"),
            ),
            _make_item(
                "gradient",
                "base",
                "helper",
                "Call and validate the system analytical gradient.",
                inputs={"system": "Must provide gradient(...).", "controls": "Controls matching system.control_spec()."},
                returns={"type": "Controls", "meaning": "Analytical gradient in the same layout."},
                requires=("system.gradient", "Controls"),
                best_for=("manual gradient inspection", "custom StepProposal functions"),
                related=("utils.verify_gradient",),
                example="grad = opt.gradient(system, controls)",
                module="optimizer/core/evaluate.py",
                phase="Phase 5",
                tags=("gradient", "controls"),
            ),
            _make_item(
                "value_and_gradient",
                "base",
                "helper",
                "Call and validate a combined objective/analytical-gradient system capability.",
                inputs={"system": "Must provide value_and_gradient(...).", "controls": "Compatible Controls."},
                returns={"type": "tuple[Evaluation, Controls]"},
                requires=("system.value_and_gradient", "Controls"),
                best_for=("systems that reuse an expensive forward propagation",),
                example="evaluation, gradient = opt.value_and_gradient(system, controls)",
                module="optimizer/core/evaluate.py",
                tags=("evaluation", "gradient", "capability"),
                phase="Numerical capabilities",
            ),
            _make_item(
                "OptimizerResult",
                "base",
                "result_object",
                "Standard return object from optimizer calls.",
                inputs={"constructor": "Usually produced by optimizer methods."},
                returns={"type": "OptimizerResult", "fields": "controls, metrics, J, stop_reason, iterations, state, trace, warmstart()"},
                requires=("Controls", "metrics with J"),
                best_for=("inspecting run outcome", "chaining via warmstart", "exporting to_dict(...)"),
                example="warm = result.warmstart(target_optimizer='lbfgs')",
                module="optimizer/result.py",
                tags=("result", "warmstart"),
            ),
            _make_item(
                "WarmStartState",
                "base",
                "data_object",
                "Safe handoff state for starting or continuing an optimizer.",
                inputs={"result_or_state": "Usually created from result.warmstart(...) or opt.warmstart(...)."},
                returns={"type": "WarmStartState", "fields": "controls, metrics, step_size, compatible optimizer_state"},
                requires=("Controls",),
                best_for=("chaining optimizer chunks", "transferring compatible optimizer memory"),
                example="second = opt.optimizers.adam(system, warmstart=first.warmstart('adam'), maxiter=20)",
                module="optimizer/state.py",
                phase="Phase 3",
                tags=("warmstart", "state"),
            ),
            _make_item(
                "SystemRequest",
                "base",
                "data_object",
                "Family-aware operation request used by shared core and future CLI/MCP adapters.",
                inputs={"operation": "Required OLGS or OLSS method name.", "payload/options": "Structured arguments.", "family": "Optional olgs/olss assertion."},
                returns={"type": "SystemRequest", "fields": "request_id, family, operation, payload, options, to_dict()"},
                related=("base.SystemResponse", "base.transact"),
                module="optimizer/transactions.py",
                tags=("transaction", "cli", "mcp", "system"),
                family=("olgs", "olss"),
            ),
            _make_item(
                "SystemResponse",
                "base",
                "data_object",
                "Stable success/error response envelope for one system transaction.",
                returns={"type": "SystemResponse", "fields": "request_id, family, operation, ok, result, error, to_dict()"},
                related=("base.SystemRequest", "base.transact"),
                module="optimizer/transactions.py",
                tags=("transaction", "cli", "mcp", "system"),
                family=("olgs", "olss"),
            ),
            _make_item(
                "SystemTransactionAdapter",
                "base",
                "adapter",
                "Bind one OLGS or OLSS system to the shared request/response execution shape.",
                inputs={"system": "Contract-valid OLGS or OLSS object."},
                returns={"type": "SystemTransactionAdapter", "methods": "execute(request) -> SystemResponse"},
                related=("base.SystemRequest", "base.transact"),
                module="optimizer/transactions.py",
                tags=("transaction", "adapter", "system"),
                family=("olgs", "olss"),
            ),
            _make_item(
                "transact",
                "base",
                "adapter",
                "Execute a direct system operation through a family-aware structured envelope.",
                inputs={"system": "Contract-valid OLGS or OLSS object.", "request": "SystemRequest or equivalent mapping."},
                returns={"type": "SystemResponse", "meaning": "JSON-safe result or structured failure."},
                related=("base.SystemRequest", "base.SystemResponse"),
                example="response = opt.transact(system, opt.SystemRequest('evaluate', {'controls': controls}))",
                module="optimizer/transactions.py",
                tags=("transaction", "cli", "mcp", "system"),
                family=("olgs", "olss"),
            ),
            _make_item(
                "Trace",
                "base",
                "data_object",
                "In-memory run ledger for iteration records, chunk records, events, and checkpoints.",
                inputs={"run_id": "String id for a run ledger."},
                returns={"type": "Trace", "methods": "record_iteration, record_chunk, checkpoint, restore, to_dict"},
                requires=("run id",),
                best_for=("closed-loop inspection", "rollback checkpoints", "technical logs"),
                example="trace = opt.trace('my-run')",
                module="optimizer/logs/trace.py",
                phase="Phase 4",
                tags=("trace", "checkpoint", "logs"),
            ),
            _make_item(
                "context",
                "base",
                "helper",
                "Bind a system once for notebook/curriculum convenience.",
                inputs={"system": "System object to bind.", "trace": "Optional Trace or run id.", "defaults": "Default kwargs for bound runs."},
                returns={"type": "OptimizerContext", "methods": "ctx.adam, ctx.evaluate, ctx.with_secondary, ctx.*_guess"},
                requires=("system contract",),
                best_for=("notebooks", "curriculum workflows", "reducing repeated system arguments"),
                example="ctx = opt.context(system, trace='run'); result = ctx.adam(controls, maxiter=10)",
                module="optimizer/library.py",
                phase="Phase 6",
                aliases=("bind",),
                tags=("context", "notebook"),
            ),
            _make_item(
                "system_info",
                "base",
                "helper",
                "Return declared system capabilities and optional static metadata without evaluating controls.",
                inputs={"system": "Any system object; required and optional hooks are probed structurally."},
                returns={"type": "dict", "schema": "optimizer.system-info/v1"},
                best_for=("checking a system API before an optimization run", "building generic tooling"),
                watch_out=("describe() and tracking_spec() must themselves be static hooks.",),
                example="system_info = opt.system_info(system)",
                module="optimizer/library.py",
                phase="Phase 13",
                tags=("system", "api", "introspection"),
                family=("olgs", "olss"),
            ),
            _make_item(
                "plot_data",
                "base",
                "helper",
                "Request one explicit system-owned visualization data product without rendering it.",
                inputs={
                    "system": "OLGS system exposing the optional plot_data hook.",
                    "controls": "Validated Controls.",
                    "kind": "Declared visualization kind such as bloch.",
                },
                returns={"type": "PlotData", "fields": "kind, named layers, annotations, metadata"},
                requires=("system.plot_data", "declared system_spec visualization"),
                best_for=("explicit trajectory inspection", "renderer-independent system data"),
                watch_out=("This can run a full trajectory and is never called automatically by tracking.",),
                example="data = opt.plot_data(system, controls, kind='bloch')",
                module="optimizer/core/evaluate.py",
                phase="System data products",
                tags=("system", "visualization", "data"),
            ),
            _make_item(
                "context_info",
                "base",
                "helper",
                "Return the current registered optimizer, OLGS contract, and tracker capability surface.",
                inputs={"system": "Optional system whose declared static metadata should be included."},
                returns={"type": "dict", "schema": "optimizer.context/v1"},
                best_for=("agent handoff", "API discovery", "checking tracker readiness"),
                watch_out=("It reports capabilities only; it does not run an optimization or evaluate a system.",),
                related=("base.system_info",),
                example="snapshot = opt.context_info(system)",
                module="optimizer/introspection.py",
                phase="Phase 13",
                tags=("api", "introspection", "tracker"),
                family=("olgs", "olss"),
            ),
            _make_item(
                "continuation",
                "base",
                "coordinator",
                "Run explicit system-parameter stages with safe control transfer and stage gates.",
                inputs={
                    "system": "Initial base OLGS system.",
                    "controls": "Initial Controls.",
                    "stages": "ContinuationStage objects or equivalent mappings.",
                    "control_transfer": "Required callable when ControlSpec layout or dt changes.",
                },
                returns={"type": "ContinuationResult", "fields": "stages, final_result, final_system, stop_reason"},
                requires=("system.with_secondary", "one or more optimizer stages"),
                best_for=("penalty curricula", "homotopy stages", "explicit grid refinement"),
                watch_out=("optimizer-private memory is reset between stages", "new-stage controls are re-evaluated"),
                example="result = opt.continuation(system, controls, stages=[opt.ContinuationStage(...)])",
                module="optimizer/continuation.py",
                tags=("continuation", "curriculum", "coordinator"),
                phase="Numerical capabilities",
            ),
            _make_item("to_plain", "base", "helper", "Convert a library value to plain, strict-JSON data under the one serialization policy; non-finite floats become \"nan\", \"inf\", or \"-inf\".", returns={"type": "JSON data"}, related=("base.transact",), example="payload = opt.to_plain(result)", module="optimizer/serialization.py", family=("olgs", "olss"), tags=("serialization", "json")),
            _make_item("system_family", "base", "helper", "Return \"olgs\" or \"olss\" from a system's static system_spec(); a missing family means OLGS.", returns={"type": "str"}, requires=("system.system_spec",), example="family = opt.system_family(system)", module="optimizer/families.py", family=("olgs", "olss"), tags=("system", "family", "introspection")),
            _make_item("control_spec", "base", "helper", "Build a ControlSpec from channel keys and a control dimension.", returns={"type": "ControlSpec"}, example="spec = opt.control_spec(('ux', 'uy'), 33)", module="optimizer/library.py", tags=("controls", "data")),
            _make_item("controls", "base", "helper", "Build Controls from a matrix-like payload for a ControlSpec.", returns={"type": "Controls"}, module="optimizer/library.py", tags=("controls", "data")),
            _make_item("zeros", "base", "helper", "Build zero Controls for a ControlSpec.", returns={"type": "Controls"}, module="optimizer/library.py", tags=("controls", "data")),
            _make_item("constant", "base", "helper", "Build constant Controls for a ControlSpec.", returns={"type": "Controls"}, module="optimizer/library.py", tags=("controls", "data")),
            _make_item("Evaluation", "base", "result_object", "Snapshot of one system evaluation: controls, metrics, and J.", module="optimizer/result.py", tags=("evaluation", "result")),
            _make_item("RunState", "base", "data_object", "Mutable state of one active optimization run: controls, metrics, step size, and optimizer memory.", module="optimizer/state.py", tags=("state", "warmstart")),
            _make_item("warmstart", "base", "helper", "Create a safe WarmStartState from a result or active run state, optionally for a target optimizer.", returns={"type": "WarmStartState"}, module="optimizer/state.py", phase="Phase 3", tags=("warmstart", "state")),
            _make_item("trace", "base", "helper", "Create an in-memory Trace ledger with the given or a random run id.", returns={"type": "Trace"}, module="optimizer/logs/trace.py", phase="Phase 4", tags=("trace", "logs")),
            _make_item("Checkpoint", "base", "data_object", "Restorable snapshot of controls plus optional run state in a Trace.", module="optimizer/logs/checkpoint.py", tags=("trace", "checkpoint")),
            _make_item("ChunkRecord", "base", "data_object", "In-memory Trace record for one optimizer chunk.", module="optimizer/logs/records.py", tags=("trace", "logs")),
            _make_item("IterationRecord", "base", "data_object", "In-memory Trace record for one optimizer iteration.", module="optimizer/logs/records.py", tags=("trace", "logs")),
            _make_item("EventRecord", "base", "data_object", "In-memory Trace event for actions outside iterations, such as rollback.", module="optimizer/logs/records.py", tags=("trace", "logs")),
            _make_item("parallel_map", "base", "helper", "Map a function over items with local parallel execution configured by ParallelConfig.", returns={"type": "list"}, module="optimizer/utils/parallel.py", phase="Phase 5", family=("olgs", "olss"), tags=("parallel",)),
            _make_item("ParallelConfig", "base", "data_object", "Configuration for local map-style parallel execution.", module="optimizer/utils/parallel.py", family=("olgs", "olss"), tags=("parallel", "config")),
            _make_item("OptimizerContext", "base", "data_object", "Bound-system facade returned by context() for notebook and curriculum workflows.", module="optimizer/library.py", tags=("context", "notebook")),
            _make_item("OptimizerLibrary", "base", "facade", "Facade class behind the root shortcuts; library is its default instance.", module="optimizer/library.py", family=("olgs", "olss"), tags=("facade",)),
            _make_item("library", "base", "object", "Default OptimizerLibrary instance that serves the root shortcuts.", module="optimizer/library.py", family=("olgs", "olss"), tags=("facade",)),
            _make_item("methods", "base", "helper", "Return the status record of every public root function, built from the catalog declarations.", returns={"type": "dict[str, MethodInfo]"}, module="optimizer/library.py", family=("olgs", "olss"), tags=("api", "introspection")),
            _make_item("MethodInfo", "base", "data_object", "Status record for one public root function: name, status, source module, and phase.", module="optimizer/library.py", family=("olgs", "olss"), tags=("api", "introspection")),
            _make_item("ContinuationStage", "base", "data_object", "One declared system transition and bounded optimizer segment of an OLGS continuation plan.", module="optimizer/continuation.py", tags=("continuation", "curriculum")),
            _make_item("StageGate", "base", "data_object", "Explicit evidence required before an OLGS continuation advances to the next stage.", module="optimizer/continuation.py", tags=("continuation", "gate")),
            _make_item("StageResult", "base", "result_object", "One completed continuation stage: its system, optimizer result, gate verdict, and control transfer.", module="optimizer/continuation.py", tags=("continuation", "result")),
            _make_item("ContinuationResult", "base", "result_object", "Result chain of an OLGS continuation plan: stages, final result, final system, and stop reason.", module="optimizer/continuation.py", tags=("continuation", "result")),
            _make_item("constant_schedule", "base", "helper", "Create a ConstantSchedule.", returns={"type": "ConstantSchedule"}, module="optimizer/library.py", tags=("schedule", "step_size")),
            _make_item("adaptive_step_schedule", "base", "helper", "Create an AdaptiveStepSchedule from keyword arguments.", returns={"type": "AdaptiveStepSchedule"}, module="optimizer/library.py", tags=("schedule", "step_size", "adaptive")),
            _make_item("exponential_decay_schedule", "base", "helper", "Create an ExponentialDecaySchedule from keyword arguments.", returns={"type": "ExponentialDecaySchedule"}, module="optimizer/library.py", tags=("schedule", "decay")),
            _make_item("linear_warmup_schedule", "base", "helper", "Create a LinearWarmupSchedule from keyword arguments.", returns={"type": "LinearWarmupSchedule"}, module="optimizer/library.py", tags=("schedule", "warmup")),
            _make_item("cosine_decay_schedule", "base", "helper", "Create a CosineDecaySchedule from keyword arguments.", returns={"type": "CosineDecaySchedule"}, module="optimizer/library.py", tags=("schedule", "cosine", "decay")),
        ),
    )

    core_group = CatalogGroup(
        name="core",
        summary="Shared chunk engine: the proposal contract, validated system evaluation, acceptance guards, and stopping rules.",
        common_inputs={
            "system": "Project physics object satisfying the optimizer contract.",
            "controls": "Named dense control matrix represented by Controls.",
        },
        common_returns={
            "StepProposal": "Trial update returned by one optimizer iteration.",
            "AcceptanceDecision": "Engine verdict on whether a trial replaces the current controls.",
            "OptimizerResult": "Standard optimizer result returned by run_chunk.",
        },
        workflow=(
            "Write a step callable mapping StepContext to StepProposal.",
            "Pass it to run_chunk with an optimizer_name and an iteration budget.",
            "Tighten acceptance with metric_guard and stopping with StoppingConfig.",
        ),
        items=(
            _make_item(
                "run_chunk",
                "core",
                "engine",
                "Shared optimizer engine for custom StepProposal functions.",
                inputs={
                    "system": "Must provide evaluate(...) and usually gradient(...).",
                    "controls/state/warmstart": "Starting point for the chunk.",
                    "step": "Callable from StepContext to StepProposal or Controls.",
                    "optimizer_name": "Name recorded in result and trace.",
                    "maxiter": "Chunk iteration budget.",
                },
                returns=_RESULT_RETURN,
                requires=("system.evaluate", "system.gradient", "Controls", "step callable"),
                best_for=("custom optimizer experiments", "shared tracing/checkpoint behavior", "advanced users"),
                watch_out=(
                    "attempted iterations are counted whether a trial is accepted or rejected",
                    "pass either stopping=StoppingConfig(...) or the loose target/stall arguments, not both",
                ),
                related=("core.StepContext", "core.StepProposal", "core.metric_guard"),
                example="result = opt.run_chunk(system, controls, step=my_step, optimizer_name='custom', maxiter=5)",
                module="optimizer/core/engine.py",
                phase="Phase 5",
                tags=("engine", "advanced"),
                root=True,
            ),
            _make_item(
                "StepContext",
                "core",
                "data_object",
                "Read-only bundle the engine hands to a proposal function each iteration.",
                returns={
                    "fields": "system, evaluator, state, evaluation, gradient, iteration, stage",
                    "meaning": "Everything a method needs to propose the next trial controls.",
                },
                requires=("run_chunk",),
                best_for=("writing a custom step callable",),
                related=("core.StepProposal", "core.run_chunk"),
                example="def my_step(ctx): return opt.StepProposal(ctx.state.controls - 0.01 * ctx.gradient)",
                module="optimizer/core/engine.py",
                phase="Phase 5",
                tags=("engine", "contract"),
                root=True,
            ),
            _make_item(
                "StepProposal",
                "core",
                "data_object",
                "Trial update returned by one optimizer iteration.",
                inputs={
                    "controls": "Proposed trial controls.",
                    "step_size": "Optional step size to retain on the chunk state.",
                    "optimizer_state": "Optimizer-private state update applied after the proposal.",
                    "optimizer_state_on_accept/on_reject": "Branch-specific state so moments advance only on accepted moves.",
                    "technical": "Free-form payload recorded in the iteration log.",
                },
                requires=("Controls",),
                best_for=("returning a trial from a custom step callable",),
                watch_out=("branch-specific fields win over the plain optimizer_state field",),
                related=("core.StepContext", "core.AcceptanceDecision"),
                example="proposal = opt.StepProposal(trial_controls, step_size=0.05)",
                module="optimizer/core/engine.py",
                phase="Phase 5",
                tags=("engine", "contract"),
                root=True,
            ),
            _make_item(
                "AcceptanceDecision",
                "core",
                "data_object",
                "Engine verdict on whether a trial replaces the current controls.",
                returns={
                    "fields": "accepted, reason, technical",
                    "meaning": "Recorded on the iteration log with the reason string.",
                },
                requires=("Evaluation",),
                best_for=("writing a custom accept function",),
                related=("core.metric_guard", "core.run_chunk"),
                example="return opt.AcceptanceDecision(accepted=True, reason='accepted')",
                module="optimizer/core/engine.py",
                phase="Phase 5",
                tags=("engine", "acceptance"),
                root=True,
            ),
            _make_item(
                "SystemEvaluator",
                "core",
                "helper",
                "Validation and cache wrapper around an optimizer-facing system.",
                inputs={
                    "system": "Object satisfying the optimizer system contract.",
                    "use_cache": "Reuse evaluations when numeric control content is unchanged.",
                },
                returns={
                    "type": "SystemEvaluator",
                    "meaning": "evaluate/gradient raise; try_evaluate/try_gradient return structured outcomes.",
                },
                requires=("system.evaluate", "system.gradient"),
                best_for=("batch probes that should not repeat expensive propagation",),
                not_for=("finite-difference fallbacks, which live in utils.derivatives",),
                related=("core.run_chunk", "utils.verify_gradient"),
                example="ev = opt.SystemEvaluator(system); snapshot = ev.evaluate(controls)",
                module="optimizer/core/evaluate.py",
                phase="Phase 5",
                tags=("evaluation", "cache"),
                root=True,
            ),
            _make_item(
                "StoppingConfig",
                "core",
                "data_object",
                "Declarative stopping rules for one chunk: budget, target, stall, and finiteness.",
                inputs={
                    "maxiter": "Attempted engine iterations.",
                    "target_value/target_metric/target_mode": "Stop once a metric reaches a threshold.",
                    "stall_patience/stall_tolerance/stall_metric": "Stop when a metric stops improving.",
                    "check_finite": "Stop before target/stall logic when metrics go non-finite.",
                },
                requires=("scalar metrics",),
                best_for=("reusing one stopping policy across chunks",),
                related=("core.run_chunk",),
                example="result = opt.run_chunk(system, controls, step=my_step, optimizer_name='custom', maxiter=50, stopping=opt.StoppingConfig(maxiter=50, target_value=1e-6))",
                module="optimizer/core/stopping.py",
                phase="Phase 5",
                tags=("stopping", "budget"),
                root=True,
            ),
            _make_item(
                "metric_guard",
                "core",
                "guard",
                "Build a reusable multi-metric accept/reject guard for run_chunk or optimizer accept hooks.",
                inputs={
                    "improve": "Metric that should improve or not get worse, default J.",
                    "mode": "min or max.",
                    "require": "Mapping of metric -> (operator, threshold[, tolerance]).",
                    "tolerance": "Default tolerance for improve and two-field requirements.",
                },
                returns={"type": "MetricGuard", "meaning": "Callable accept function returning AcceptanceDecision."},
                requires=("scalar metrics",),
                best_for=("preventing one metric from improving while another breaks", "custom acceptance in line_search/run_chunk"),
                related=("optimizers.line_search", "core.run_chunk", "core.AcceptanceDecision"),
                example="guard = opt.core.metric_guard(improve='J', require={'fidelity': ('>=', 0.0)})",
                module="optimizer/core/guards.py",
                phase="Phase 11",
                tags=("guard", "acceptance", "metrics"),
                root=True,
            ),
            _make_item("StopReason", "core", "data_object", "Meaning, outcome class, families, and continuability of one stop reason; StopReason.lookup(reason) also resolves labelled reasons such as stage_gate_failed:<label>.", related=("core.STOP_REASONS", "utils.diagnose_run"), module="optimizer/core/stop_reasons.py", family=("olgs", "olss"), tags=("stopping", "registry")),
            _make_item("STOP_REASONS", "core", "registry", "Read-only registry of every stop reason the library assigns, keyed by name.", returns={"type": "Mapping[str, StopReason]"}, related=("core.StopReason",), example="opt.core.STOP_REASONS['maxiter'].outcome", module="optimizer/core/stop_reasons.py", family=("olgs", "olss"), tags=("stopping", "registry")),
            _make_item("StopDecision", "core", "data_object", "Decision returned by stopping checks: whether to stop and the stop reason.", module="optimizer/core/stopping.py", tags=("stopping",)),
            _make_item("StopTracker", "core", "helper", "Stateful stopping helper for one optimizer chunk.", module="optimizer/core/stopping.py", tags=("stopping",)),
            _make_item("MetricGuard", "core", "guard", "Callable multi-metric accept rule built by metric_guard().", related=("core.metric_guard",), module="optimizer/core/guards.py", tags=("guard", "acceptance", "metrics"), root=True),
            _make_item("default_accept", "core", "guard", "Default acceptance: accept trial controls when the selected metric does not get worse.", module="optimizer/core/engine.py", tags=("guard", "acceptance")),
            _make_item("EvaluationOutcome", "core", "data_object", "Structured result of an evaluation attempt that may fail; returned by SystemEvaluator.try_evaluate.", module="optimizer/core/evaluate.py", tags=("evaluation", "outcome")),
            _make_item("GradientOutcome", "core", "data_object", "Structured result of a gradient attempt that may fail; returned by SystemEvaluator.try_gradient.", module="optimizer/core/evaluate.py", tags=("gradient", "outcome")),
            _make_item("ValueGradientOutcome", "core", "data_object", "Structured result of a combined value and gradient attempt that may fail.", module="optimizer/core/evaluate.py", tags=("evaluation", "gradient", "outcome")),
            _make_item("RunObserver", "core", "protocol", "Passive sink for the engine lifecycle events.", module="optimizer/core/observer.py", tags=("observer", "events")),
            _make_item("ObserverAdapter", "core", "base_class", "No-op observer base class; override only the event methods you need.", module="optimizer/core/observer.py", tags=("observer", "events")),
            _make_item("SnapshotFacts", "core", "data_object", "Compact facts known at one point of the engine lifecycle.", module="optimizer/core/observer.py", tags=("observer", "events")),
            _make_item("RunStartEvent", "core", "data_object", "The first validated state of one engine chunk, passed to observers.", module="optimizer/core/observer.py", tags=("observer", "events")),
            _make_item("IterationEvent", "core", "data_object", "One attempted outer iteration, including a rejected or failed trial, passed to observers.", module="optimizer/core/observer.py", tags=("observer", "events")),
            _make_item("CheckpointEvent", "core", "data_object", "Engine-selected reproducibility point carrying already-owned state, passed to observers.", module="optimizer/core/observer.py", tags=("observer", "events", "checkpoint")),
            _make_item("ChunkEvent", "core", "data_object", "A completed engine chunk, passed to observers and the in-memory compatibility trace.", module="optimizer/core/observer.py", tags=("observer", "events")),
            _make_item("TerminalEvent", "core", "data_object", "The final fact produced by one engine chunk, passed to observers.", module="optimizer/core/observer.py", tags=("observer", "events")),
        ),
    )

    campaigns_group = CatalogGroup(
        name="campaigns",
        summary="Durable multi-stage coordination using objective and provenance declared by system_spec().",
        common_inputs={"system": "OLGS system with static system_spec().", "root": "New durable campaign directory."},
        common_returns={"type": "Campaign or ModeResult", "evidence": "per-stage tracker journals plus campaign receipt/timeline."},
        workflow=(
            "Create a campaign from one declared base system.",
            "Scout or compare compatible candidates, then refine or validate the selected candidate.",
            "Use Campaign.open(..., system_resolver=...) to reload durable evidence.",
        ),
        items=(
            _make_item("Campaign", "campaigns", "coordinator", "Campaign object with durable stage, system-fingerprint, and selection evidence.", module="optimizer/campaigns/__init__.py", tags=("campaign", "provenance", "multistart")),
            _make_item("create", "campaigns", "factory", "Create a new campaign rooted at a new directory.", inputs={"system": "Declared OLGS system.", "root": "Nonexistent campaign path."}, returns={"type": "Campaign"}, requires=("system.system_spec",), module="optimizer/campaigns/__init__.py", tags=("campaign", "create")),
            _make_item("open", "campaigns", "factory", "Open saved campaign evidence with a caller-owned system resolver.", inputs={"root": "Existing campaign path.", "system_resolver": "Callable from saved SystemReference to matching system."}, returns={"type": "Campaign"}, requires=("system resolver", "verified checkpoint artifacts"), module="optimizer/campaigns/__init__.py", tags=("campaign", "replay", "warmstart")),
            _make_item("MethodSpec", "campaigns", "data_object", "Declared optimizer name and bounded options for one campaign stage.", module="optimizer/campaigns/__init__.py", tags=("campaign", "method")),
            _make_item("SelectionSpec", "campaigns", "data_object", "System-owned metric and direction used for ranking; never inferred from J.", module="optimizer/campaigns/__init__.py", tags=("campaign", "objective", "system_spec")),
            _make_item("SystemReference", "campaigns", "data_object", "Digest of the normalized system-owned semantic specification for one stage.", module="optimizer/campaigns/__init__.py", tags=("campaign", "provenance", "system_spec")),
            _make_item("CandidateRef", "campaigns", "data_object", "One selected stage result tied to an exact system and checkpoint.", module="optimizer/campaigns/__init__.py", tags=("campaign", "provenance")),
            _make_item("ModeResult", "campaigns", "result_object", "Candidates produced by a campaign mode, sorted by the system-owned objective.", module="optimizer/campaigns/__init__.py", tags=("campaign", "result")),
            _make_item("ValidationResult", "campaigns", "result_object", "Explicit validation of a candidate against the system that produced it.", module="optimizer/campaigns/__init__.py", tags=("campaign", "validation")),
            _make_item("random_smooth_guesses", "campaigns", "generator", "Generate count random smooth guesses with consecutive seeds.", returns={"type": "tuple[Controls, ...]"}, related=("guesses.random_smooth_guess",), module="optimizer/campaigns/__init__.py", tags=("campaign", "guess", "random")),
        ),
    )

    tracker_group = CatalogGroup(
        name="tracker",
        summary="Passive durable run evidence; it records engine facts and never calls system numerics.",
        common_inputs={"receipt": "Declared metric semantics and source references.", "run_dir": "One owned durable run directory."},
        common_returns={"type": "RunJournal, reader records, or checkpoint data"},
        workflow=("Create one RunJournal per run.", "Attach JournalObserver to a supported optimizer.", "Read receipt/timeline/terminal through the standalone reader APIs."),
        items=(
            _make_item("RunJournal", "tracker", "writer", "Append-only receipt, timeline, checkpoint, live, and terminal writer.", module="optimizer/tracker/journal.py", tags=("tracker", "journal", "durable")),
            _make_item("JournalObserver", "tracker", "observer", "Passive bridge from already-computed core facts to a RunJournal.", module="optimizer/tracker/bridge.py", tags=("tracker", "observer", "passive")),
            _make_item("CheckpointPolicy", "tracker", "policy", "Bound durable control artifacts while retaining every scalar iteration record.", module="optimizer/tracker/bridge.py", tags=("tracker", "checkpoint", "retention")),
            _make_item("read_timeline", "tracker", "reader", "Validate and read append-only tracker events without optimizer imports.", module="optimizer/tracker/reader.py", tags=("tracker", "reader", "read_only")),
            _make_item("load_checkpoint", "tracker", "reader", "Load a SHA-256 verified control artifact without a system object.", module="optimizer/tracker/artifacts.py", tags=("tracker", "checkpoint", "replay")),
            _make_item("RunReceipt", "tracker", "data_object", "Small immutable receipt for one run, without redefining its problem.", module="optimizer/tracker/model.py", tags=("tracker", "receipt")),
            _make_item("MetricDescriptor", "tracker", "data_object", "Declared meaning of one scalar metric saved by the tracker.", module="optimizer/tracker/model.py", tags=("tracker", "metrics")),
            _make_item("SourceReference", "tracker", "data_object", "Small pointer to an authority owned outside the tracker.", module="optimizer/tracker/model.py", tags=("tracker", "provenance")),
            _make_item("ScalarSnapshot", "tracker", "data_object", "Compact state view at one causal point in a run.", module="optimizer/tracker/model.py", tags=("tracker", "snapshot")),
            _make_item("IterationRecord", "tracker", "data_object", "Raw before, proposal, trial, and after record for one outer iteration.", module="optimizer/tracker/model.py", tags=("tracker", "journal")),
            _make_item("ChunkRecord", "tracker", "data_object", "One completed optimizer chunk with its resolved method and system facts.", module="optimizer/tracker/model.py", tags=("tracker", "journal")),
            _make_item("InterventionRecord", "tracker", "data_object", "An action between normal iterations, linked to its numerical evidence.", module="optimizer/tracker/model.py", tags=("tracker", "journal")),
            _make_item("AnalysisRecord", "tracker", "data_object", "A derived, explicitly scheduled analysis of a raw iteration window.", module="optimizer/tracker/model.py", tags=("tracker", "analysis")),
            _make_item("LiveSnapshot", "tracker", "data_object", "Replaceable progress view for a local observer; not the scientific source.", module="optimizer/tracker/model.py", tags=("tracker", "live")),
            _make_item("TerminalRecord", "tracker", "data_object", "The selected terminal result and any explicitly requested certificate.", module="optimizer/tracker/model.py", tags=("tracker", "terminal")),
            _make_item("CheckpointReference", "tracker", "data_object", "Small JSON-safe pointer to a separately stored control artifact.", module="optimizer/tracker/artifacts.py", tags=("tracker", "checkpoint")),
            _make_item("CheckpointData", "tracker", "data_object", "Loaded controls plus caller-supplied JSON metadata.", module="optimizer/tracker/artifacts.py", tags=("tracker", "checkpoint")),
            _make_item("TimelineReadResult", "tracker", "data_object", "Complete timeline records and, if present, one ignored damaged tail line.", module="optimizer_log/reader.py", tags=("tracker", "reader")),
            _make_item("follow_timeline", "tracker", "reader", "Read complete appended events and distinguish partial tails from corruption.", module="optimizer_log/reader.py", tags=("tracker", "reader", "live")),
            _make_item("read_receipt", "tracker", "reader", "Read the stored run receipt exactly as written.", module="optimizer_log/reader.py", tags=("tracker", "reader")),
            _make_item("read_live", "tracker", "reader", "Read the current replaceable observer snapshot, if the run has started.", module="optimizer_log/reader.py", tags=("tracker", "reader", "live")),
            _make_item("read_terminal", "tracker", "reader", "Read the direct terminal-result index, if the run has ended.", module="optimizer_log/reader.py", tags=("tracker", "reader")),
            _make_item("read_writer", "tracker", "reader", "Read explicit writer ownership and heartbeat state, if present.", module="optimizer_log/reader.py", tags=("tracker", "reader", "liveness")),
            _make_item("inspect_writer", "tracker", "reader", "Report local writer liveness without taking ownership or modifying files.", module="optimizer_log/reader.py", tags=("tracker", "reader", "liveness")),
            _make_item("metric_coverage", "tracker", "reader", "Compare declared metrics with observed and explicitly omitted metrics.", module="optimizer_log/reader.py", tags=("tracker", "reader", "metrics")),
            _make_item("project_timeline", "tracker", "reader", "Create a generic read-only projection for API or interface consumers.", module="optimizer_log/reader.py", tags=("tracker", "reader")),
            _make_item("validate_event", "tracker", "validator", "Validate one causal event envelope without importing optimizer code.", module="optimizer_log/reader.py", tags=("tracker", "validation")),
            _make_item("validate_receipt", "tracker", "validator", "Validate the generic receipt fields required by read-only consumers.", module="optimizer_log/reader.py", tags=("tracker", "validation")),
            _make_item("validate_timeline", "tracker", "validator", "Validate the one-start, one-terminal lifecycle of a complete prefix.", module="optimizer_log/reader.py", tags=("tracker", "validation")),
        ),
    )

    visualization_group = CatalogGroup(
        name="visualization",
        summary="Render explicit system-owned plot data; visualization never invents a trajectory.",
        common_inputs={"data": "PlotData returned by a system plot_data capability."},
        common_returns={"type": "matplotlib figure/axes or saved image path"},
        workflow=("Request system-owned plot data explicitly.", "Validate/coerce the data.", "Render or save without rerunning the system."),
        items=(
            _make_item("PlotData", "visualization", "data_object", "Named layers and metadata for one explicit visualization product.", module="optimizer/visualization/models.py", tags=("visualization", "data"), root=True),
            _make_item("plot_bloch", "visualization", "renderer", "Render validated Bloch layers without calling the system.", module="optimizer/visualization/bloch.py", tags=("visualization", "bloch", "renderer")),
            _make_item("draw_timeseries", "visualization", "renderer", "Draw named per-component channels vs a scalar parameter onto an existing 2D axis; shared by plot_bloch's projections subplot and the standalone panel functions.", module="optimizer/visualization/bloch.py", tags=("visualization", "timeseries", "renderer")),
            _make_item("draw_sphere", "visualization", "renderer", "Draw a light unit-sphere scaffold on an existing 3D axis.", module="optimizer/visualization/bloch.py", tags=("visualization", "bloch", "renderer")),
            _make_item("style_context", "visualization", "helper", "Apply the compact serif plotting style without mutating global rcParams.", module="optimizer/visualization/bloch.py", tags=("visualization", "style")),
            _make_item("validate_bloch_data", "visualization", "validator", "Validate generic plot data for rendering inside the Bloch ball.", module="optimizer/visualization/bloch.py", tags=("visualization", "bloch", "validation")),
            _make_item("plot_bloch_panel", "visualization", "renderer", "Render one standalone Bloch-sphere panel (single figure, no projections subplot).", module="optimizer/visualization/panels.py", tags=("visualization", "bloch", "panel")),
            _make_item("plot_projection_panel", "visualization", "renderer", "Render one standalone Bloch-component-vs-parameter panel from validated Bloch trajectory data.", module="optimizer/visualization/panels.py", tags=("visualization", "bloch", "panel")),
            _make_item("plot_controls_panel", "visualization", "renderer", "Render one standalone control-amplitude-vs-parameter panel; uses CONTROL_COLORS and skips the Bloch-ball check.", module="optimizer/visualization/panels.py", tags=("visualization", "controls", "panel")),
            _make_item("validate_controls_data", "visualization", "validator", "Validate generic plot data for a control-amplitude-vs-parameter panel (no Bloch-ball check).", module="optimizer/visualization/panels.py", tags=("visualization", "controls", "validation")),
            _make_item("save_figure", "visualization", "helper", "Save a renderer-produced figure to an explicit path.", module="optimizer/visualization/bloch.py", tags=("visualization", "export")),
            _make_item("save_panel", "visualization", "helper", "Save one panel figure as .pdf and .png from a shared path stem at one dpi.", module="optimizer/visualization/panels.py", tags=("visualization", "export", "panel")),
            _make_item("PlotLayer", "visualization", "data_object", "One named data layer with generic coordinates and presentation hints.", module="optimizer/visualization/models.py", tags=("visualization", "data"), root=True),
            _make_item("coerce_plot_data", "visualization", "helper", "Normalize a PlotData instance or an equivalent mapping into PlotData.", module="optimizer/visualization/models.py", tags=("visualization", "data")),
        ),
    )

    system_group = CatalogGroup(
        name="system_olgs",
        summary="Mandatory static system semantics and capability-gated numerical hooks owned by each problem.",
        common_inputs={"system": "Problem-owned OLGS system object."},
        common_returns={"type": "validated system specification, controls, or capability result"},
        workflow=("Declare system_spec and control_spec.", "Expose evaluate/gradient/with_secondary.", "Add optional capabilities only when the system owns their meaning."),
        items=(
            _make_item("get_system_spec", "system_olgs", "helper", "Return and validate the required static system semantic declaration.", module="optimizer/system_olgs/contract.py", tags=("system", "system_spec", "static"), root=True),
            _make_item("require_system", "system_olgs", "validator", "Validate the required OLGS contract before an optimizer starts.", module="optimizer/system_olgs/contract.py", tags=("system", "contract"), root=True),
            _make_item("with_secondary", "system_olgs", "helper", "Request a system-owned parameter-updated copy for explicit continuation.", module="optimizer/system_olgs/contract.py", tags=("system", "continuation"), root=True),
            _make_item("optional_control_bounds", "system_olgs", "capability", "Read optional system-declared elementwise bounds.", module="optimizer/system_olgs/validation.py", tags=("system", "bounds", "capability"), root=True),
            _make_item("OLGS", "system_olgs", "base_class", "Base class for a self-contained Open Loop Gradient System.", module="optimizer/system_olgs/olgs.py", tags=("system", "contract"), root=True),
            _make_item("OLGSystem", "system_olgs", "protocol", "Protocol for the required OLGS public surface.", aliases=("OptimizerSystem",), module="optimizer/system_olgs/contract.py", tags=("system", "contract", "protocol"), root=True),
            _make_item("SYSTEM_SPEC_SCHEMA", "system_olgs", "constant", "Schema identifier of the OLGS system_spec() declaration.", module="optimizer/system_olgs/specs.py", tags=("system_spec", "schema"), root=True),
            _make_item("basic_system_spec", "system_olgs", "helper", "Build the minimum complete specification for a new OLGS system.", module="optimizer/system_olgs/specs.py", tags=("system_spec",), root=True),
            _make_item("metric_spec", "system_olgs", "helper", "Build one normalized OLGS metric declaration for system_spec().", module="optimizer/system_olgs/specs.py", tags=("system_spec", "metric"), root=True),
            _make_item("validate_system_spec", "system_olgs", "validator", "Validate and normalize a static OLGS system_spec() mapping.", module="optimizer/system_olgs/specs.py", tags=("system_spec", "validation"), root=True),
            _make_item("validate_declared_metrics", "system_olgs", "validator", "Validate metric names and declared shapes without changing system values.", module="optimizer/system_olgs/specs.py", tags=("system_spec", "metric", "validation")),
            _make_item("TRACKING_SPEC_SCHEMA", "system_olgs", "constant", "Schema identifier of the optional tracking specification.", module="optimizer/system_olgs/contract.py", tags=("tracking", "schema"), root=True),
            _make_item("TrackingSpec", "system_olgs", "type_alias", "Type of the optional static tracking specification mapping.", module="optimizer/system_olgs/contract.py", tags=("tracking", "type"), root=True),
            _make_item("Metrics", "system_olgs", "type_alias", "Type of an OLGS metrics mapping from metric name to value.", module="optimizer/system_olgs/contract.py", tags=("metrics", "type")),
            _make_item("get_tracking_spec", "system_olgs", "helper", "Return optional static tracking metadata declared by a system.", module="optimizer/system_olgs/contract.py", tags=("tracking", "static"), root=True),
            _make_item("contract_info", "system_olgs", "helper", "Return the generic OLGS contract without inspecting a concrete system.", module="optimizer/system_olgs/contract.py", tags=("contract", "introspection"), root=True),
            _make_item("describe_system", "system_olgs", "helper", "Describe declared OLGS system capabilities without evaluating controls.", module="optimizer/system_olgs/contract.py", tags=("system", "introspection"), root=True),
            _make_item("SystemProbe", "system_olgs", "data_object", "Presence of required, recommended, and optional OLGS hooks.", module="optimizer/system_olgs/contract.py", tags=("system", "introspection"), root=True),
            _make_item("probe_system", "system_olgs", "helper", "Inspect OLGS hook presence without calling expensive methods.", module="optimizer/system_olgs/contract.py", tags=("system", "introspection"), root=True),
            _make_item("get_secondary_update_hook", "system_olgs", "helper", "Return the system hook that applies secondary-parameter updates.", module="optimizer/system_olgs/contract.py", tags=("system", "continuation")),
            _make_item("ControlBounds", "system_olgs", "data_object", "Elementwise lower and upper control limits in the system control layout.", module="optimizer/system_olgs/contract.py", tags=("bounds",), root=True),
            _make_item("ValueGradient", "system_olgs", "data_object", "Validated system-owned objective metrics and analytical gradient pair.", module="optimizer/system_olgs/contract.py", tags=("gradient", "capability"), root=True),
            _make_item("SupportsValueAndGradient", "system_olgs", "protocol", "Optional capability for sharing work between value and gradient calls.", module="optimizer/system_olgs/contract.py", tags=("capability", "protocol"), root=True),
            _make_item("SupportsBounds", "system_olgs", "protocol", "Optional elementwise control-bound capability.", module="optimizer/system_olgs/contract.py", tags=("capability", "protocol", "bounds"), root=True),
            _make_item("SupportsEqualityConstraints", "system_olgs", "protocol", "Optional hard equality-constraint capability: residuals and their Jacobian.", module="optimizer/system_olgs/contract.py", tags=("capability", "protocol", "constraints"), root=True),
            _make_item("SupportsHVP", "system_olgs", "protocol", "Optional Hessian-vector product capability for second-order methods.", module="optimizer/system_olgs/contract.py", tags=("capability", "protocol", "hvp"), root=True),
            _make_item("SupportsProximalObjective", "system_olgs", "protocol", "Optional composite-objective capability for proximal methods.", module="optimizer/system_olgs/contract.py", tags=("capability", "protocol", "proximal"), root=True),
            _make_item("SupportsControlTransfer", "system_olgs", "protocol", "Optional system-owned feasibility-preserving control transfer.", module="optimizer/system_olgs/contract.py", tags=("capability", "protocol", "continuation"), root=True),
            _make_item("SupportsPlotData", "system_olgs", "protocol", "Optional explicit data-product capability for visualization tools.", module="optimizer/system_olgs/contract.py", tags=("capability", "protocol", "visualization"), root=True),
            _make_item("validate_control_spec", "system_olgs", "validator", "Call and validate system.control_spec().", module="optimizer/system_olgs/validation.py", tags=("controls", "validation")),
            _make_item("validate_controls_for_system", "system_olgs", "validator", "Validate that controls match the system control specification.", module="optimizer/system_olgs/validation.py", tags=("controls", "validation"), root=True),
            _make_item("validate_metrics", "system_olgs", "validator", "Validate and normalize an evaluation metrics mapping.", module="optimizer/system_olgs/validation.py", tags=("metrics", "validation"), root=True),
            _make_item("evaluate_system", "system_olgs", "helper", "Validate controls, evaluate the system, and normalize its metrics.", module="optimizer/system_olgs/validation.py", tags=("evaluation", "validation"), root=True),
            _make_item("gradient_system", "system_olgs", "helper", "Validate controls and return a gradient matching the system layout.", module="optimizer/system_olgs/validation.py", tags=("gradient", "validation"), root=True),
            _make_item("value_and_gradient_system", "system_olgs", "capability", "Call and validate the optional combined objective and gradient hook.", module="optimizer/system_olgs/validation.py", tags=("gradient", "capability"), root=True),
            _make_item("optional_residuals", "system_olgs", "capability", "Call the optional residual hook and validate a finite 1D vector.", module="optimizer/system_olgs/validation.py", tags=("residual", "capability"), root=True),
            _make_item("optional_jacobian", "system_olgs", "capability", "Call the optional Jacobian hook and validate a finite 2D matrix.", module="optimizer/system_olgs/validation.py", tags=("jacobian", "capability"), root=True),
            _make_item("optional_hvp", "system_olgs", "capability", "Call the optional Hessian-vector product hook and validate the Controls output.", module="optimizer/system_olgs/validation.py", tags=("hvp", "capability"), root=True),
            _make_item("optional_hessian", "system_olgs", "capability", "Call the optional dense Hessian hook and validate its shape.", module="optimizer/system_olgs/validation.py", tags=("hessian", "capability"), root=True),
            _make_item("optional_smooth_gradient", "system_olgs", "capability", "Call and validate the smooth-part gradient of a composite objective.", module="optimizer/system_olgs/validation.py", tags=("proximal", "capability"), root=True),
            _make_item("optional_prox", "system_olgs", "capability", "Call and validate the optional proximal operator.", module="optimizer/system_olgs/validation.py", tags=("proximal", "capability"), root=True),
            _make_item("optional_control_transfer", "system_olgs", "capability", "Call one explicit system-owned feasibility-preserving transfer hook.", module="optimizer/system_olgs/validation.py", tags=("continuation", "capability"), root=True),
            _make_item("optional_plot_data", "system_olgs", "capability", "Call and validate an explicit system-owned visualization data product.", module="optimizer/system_olgs/validation.py", tags=("visualization", "capability"), root=True),
            _make_item("finite_difference_jacobian", "system_olgs", "derivative", "Finite-difference Jacobian of system.residuals; the utils version adds named residual sets.", related=("utils.finite_difference_jacobian",), module="optimizer/system_olgs/derivatives.py", tags=("derivative", "finite_difference", "jacobian")),
            _make_item("finite_difference_hessian", "system_olgs", "derivative", "Dense finite-difference Hessian of a scalar metric.", module="optimizer/system_olgs/derivatives.py", tags=("derivative", "finite_difference", "hessian"), root=True),
            _make_item("finite_difference_hvp", "system_olgs", "derivative", "Finite-difference Hessian-vector product from gradient calls.", module="optimizer/system_olgs/derivatives.py", tags=("derivative", "finite_difference", "hvp"), root=True),
            _make_item("get_jacobian", "system_olgs", "derivative", "Return (jacobian, source) from the analytical hook or finite differences.", module="optimizer/system_olgs/derivatives.py", tags=("derivative", "jacobian")),
            _make_item("get_hvp", "system_olgs", "derivative", "Return (hvp, source) from the analytical hook or finite differences.", module="optimizer/system_olgs/derivatives.py", tags=("derivative", "hvp")),
            _make_item("get_hessian", "system_olgs", "derivative", "Return (hessian, source) from the analytical hook or finite differences.", module="optimizer/system_olgs/derivatives.py", tags=("derivative", "hessian")),
            _make_item("SystemEvaluation", "system_olgs", "data_object", "Optional structured evaluation output: controls, validated metrics, and system-owned data.", module="optimizer/system_olgs/results.py", tags=("result", "lifecycle")),
            _make_item("ForwardResult", "system_olgs", "data_object", "Optional structured forward-propagation output: controls and system-owned data.", module="optimizer/system_olgs/results.py", tags=("result", "lifecycle")),
            _make_item("BackwardResult", "system_olgs", "data_object", "Optional structured backward-propagation output: controls and system-owned data.", module="optimizer/system_olgs/results.py", tags=("result", "lifecycle")),
            _make_item("GradientResult", "system_olgs", "data_object", "Optional structured gradient output: controls, gradient, optional metrics, and system-owned data.", module="optimizer/system_olgs/results.py", tags=("result", "lifecycle")),
            _make_item("SimulationResult", "system_olgs", "data_object", "Optional structured simulation output: controls, metrics, trajectories, and system-owned data.", module="optimizer/system_olgs/results.py", tags=("result", "lifecycle")),
        ),
    )

    shooting_system_group = CatalogGroup(
        name="system_olss",
        summary="Construction types, validation, and discovery for project-owned OLSS contracts.",
        common_inputs={"system": "Problem-owned OLSS system object."},
        common_returns={"type": "validated OLSS declaration or shooting data object"},
        workflow=("Declare system_spec and shooting_spec.", "Implement residuals, jacobian, evaluate, verify, and with_secondary.", "Use optional capabilities only when the project owns their meaning."),
        items=(
            _make_item("OLSS", "system_olss", "base_class", "Base class for a self-contained Open Loop Shooting System.", module="optimizer/system_olss/olss.py", tags=("shooting", "system", "contract"), root=True),
            _make_item("ShootingBlock", "system_olss", "data_object", "One named, scaled block in a shooting vector.", module="optimizer/system_olss/variables.py", tags=("shooting", "variables"), root=True),
            _make_item("ShootingSpec", "system_olss", "data_object", "Ordered layout of shooting-variable blocks.", module="optimizer/system_olss/variables.py", tags=("shooting", "variables"), root=True),
            _make_item("ShootingVariables", "system_olss", "data_object", "Finite values attached to a ShootingSpec.", module="optimizer/system_olss/variables.py", tags=("shooting", "variables"), root=True),
            _make_item("VerificationCheck", "system_olss", "data_object", "Named independent acceptance check included in OLSS qualification.", module="optimizer/system_olss/results.py", tags=("shooting", "verification"), root=True),
            _make_item("basic_shooting_system_spec", "system_olss", "helper", "Build and validate the static OLSS declaration.", module="optimizer/system_olss/specs.py", tags=("shooting", "system_spec"), root=True),
            _make_item("shooting_metric_spec", "system_olss", "helper", "Build one OLSS metric declaration for system_spec().", module="optimizer/system_olss/specs.py", tags=("shooting", "system_spec", "metric"), root=True),
            _make_item("ShootingEvaluation", "system_olss", "data_object", "Candidate, residual vector, declared metrics, and working options of one evaluation.", module="optimizer/system_olss/results.py", tags=("shooting", "result"), root=True),
            _make_item("ShootingVerification", "system_olss", "data_object", "Independent verification residuals, metrics, options, independence source, and named checks.", module="optimizer/system_olss/results.py", tags=("shooting", "verification", "result"), root=True),
            _make_item("ShootingResult", "system_olss", "data_object", "One root solve: candidate, residuals, metrics, convergence, verification, qualification, stop reason, error, tolerance source, counters, and history.", module="optimizer/system_olss/results.py", tags=("shooting", "result"), root=True),
            _make_item("MultiStartResult", "system_olss", "data_object", "All multistart results, qualified and unique roots, and the ranking with its selection source.", module="optimizer/system_olss/results.py", tags=("shooting", "multistart", "result"), root=True),
            _make_item("ContinuationPoint", "system_olss", "data_object", "One verified pseudo-arclength branch point with its tangent.", module="optimizer/system_olss/results.py", tags=("shooting", "continuation", "result"), root=True),
            _make_item("ShootingContinuationResult", "system_olss", "data_object", "Pseudo-arclength branch points, step history, stop reason, and verification tolerance.", module="optimizer/system_olss/results.py", tags=("shooting", "continuation", "result"), root=True),
            _make_item("ParameterContinuationPoint", "system_olss", "data_object", "One natural-continuation stage: parameter value, attempted step, and its shooting result.", module="optimizer/system_olss/results.py", tags=("shooting", "continuation", "result"), root=True),
            _make_item("ParameterContinuationResult", "system_olss", "data_object", "Natural-continuation stages, per-attempt step history, stop reason, and target status.", module="optimizer/system_olss/results.py", tags=("shooting", "continuation", "result"), root=True),
            _make_item("OpenLoopShootingSystem", "system_olss", "protocol", "Protocol for the seven required OLSS calls.", module="optimizer/system_olss/contract.py", tags=("shooting", "contract", "protocol"), root=True),
            _make_item("SHOOTING_SYSTEM_SPEC_SCHEMA", "system_olss", "constant", "Schema identifier of the OLSS system_spec() declaration.", module="optimizer/system_olss/specs.py", tags=("shooting", "system_spec", "schema"), root=True),
            _make_item("validate_shooting_system_spec", "system_olss", "validator", "Validate and normalize a static OLSS system_spec() mapping.", module="optimizer/system_olss/specs.py", tags=("shooting", "system_spec", "validation"), root=True),
            _make_item("get_shooting_system_spec", "system_olss", "helper", "Return and validate the static OLSS system_spec() declaration.", module="optimizer/system_olss/contract.py", tags=("shooting", "system_spec"), root=True),
            _make_item("validate_shooting_spec", "system_olss", "validator", "Call shooting_spec() and check it against the declared variable size.", module="optimizer/system_olss/contract.py", tags=("shooting", "variables", "validation")),
            _make_item("require_shooting_system", "system_olss", "validator", "Validate the required OLSS contract before a solver starts.", module="optimizer/system_olss/contract.py", tags=("shooting", "contract"), root=True),
            _make_item("ShootingSystemProbe", "system_olss", "data_object", "Presence of required and optional OLSS hooks, with required_ok.", module="optimizer/system_olss/contract.py", tags=("shooting", "introspection"), root=True),
            _make_item("probe_shooting_system", "system_olss", "helper", "Inspect OLSS hook presence without calling numerical methods.", module="optimizer/system_olss/contract.py", tags=("shooting", "introspection"), root=True),
            _make_item("describe_shooting_system", "system_olss", "helper", "Describe an OLSS system's declarations and hook presence without evaluating it.", module="optimizer/system_olss/contract.py", tags=("shooting", "introspection"), root=True),
            _make_item("shooting_contract_info", "system_olss", "helper", "Return the generic OLSS contract: required and optional calls.", module="optimizer/system_olss/contract.py", tags=("shooting", "contract", "introspection"), root=True),
            _make_item("ShootingOptions", "system_olss", "type_alias", "Type of the per-call OLSS numerical options mapping.", module="optimizer/system_olss/contract.py", tags=("shooting", "type")),
            _make_item("ShootingMetrics", "system_olss", "type_alias", "Type of an OLSS metrics mapping from metric name to value.", module="optimizer/system_olss/contract.py", tags=("shooting", "type")),
            _make_item("validate_variables_for_shooting_system", "system_olss", "validator", "Check that ShootingVariables match the system's shooting_spec().", module="optimizer/system_olss/validation.py", tags=("shooting", "variables", "validation")),
            _make_item("residuals_system", "system_olss", "helper", "Call residuals and validate a finite vector of the declared size.", module="optimizer/system_olss/validation.py", tags=("shooting", "residual", "validation")),
            _make_item("jacobian_system", "system_olss", "helper", "Call jacobian and validate a finite dense matrix of the declared shape.", module="optimizer/system_olss/validation.py", tags=("shooting", "jacobian", "validation")),
            _make_item("residuals_and_jacobian_system", "system_olss", "helper", "Call the optional combined hook, or residuals and jacobian separately, and validate both.", module="optimizer/system_olss/validation.py", tags=("shooting", "residual", "jacobian", "validation")),
            _make_item("evaluate_shooting_system", "system_olss", "helper", "Call evaluate and check the metrics against the declaration.", module="optimizer/system_olss/validation.py", tags=("shooting", "evaluation", "validation")),
            _make_item("verify_shooting_system", "system_olss", "helper", "Call verify and normalize it into a ShootingVerification with its independence evidence.", module="optimizer/system_olss/validation.py", tags=("shooting", "verification", "validation")),
            _make_item("roots_equivalent_system", "system_olss", "capability", "Call the optional system-owned root-equivalence predicate; None means use the numerical default.", module="optimizer/system_olss/validation.py", tags=("shooting", "multistart", "capability")),
            _make_item("cost_gradient_system", "system_olss", "capability", "Call and validate the optional cost_gradient hook used by direct shooting.", module="optimizer/system_olss/validation.py", tags=("shooting", "cost", "capability")),
            _make_item("shooting_bounds_system", "system_olss", "capability", "Read and validate optional elementwise shooting-variable bounds.", module="optimizer/system_olss/validation.py", tags=("shooting", "bounds", "capability")),
        ),
    )

    shooting_group = CatalogGroup(
        name="shooting",
        summary="Root solving, verification, multistart, continuation, and direct shooting for OLSS systems.",
        common_inputs={
            "system": "Project-owned OLSS object with declared shooting variables, residuals, Jacobian, and verification.",
            "variables": "ShootingVariables matching system.shooting_spec().",
            "options": "System-owned per-call numerical accuracy settings.",
        },
        common_returns={
            "ShootingResult": "Root, residual evidence, verification, metrics, counters, and qualification status.",
            "MultiStartResult": "All starts plus deduplicated and ranked qualified roots.",
            "ShootingContinuationResult": "Verified branch points and corrector evidence.",
        },
        workflow=(
            "Inspect the static OLSS declaration with opt.system_info(system).",
            "Check derivatives with verify_shooting_jacobian.",
            "Solve and independently verify candidates before ranking them.",
        ),
        items=(
            _make_item(
                "solve_shooting",
                "shooting",
                "solver",
                "Solve one OLSS residual equation and independently verify the candidate.",
                inputs={"system": "OLSS system.", "initial": "Initial ShootingVariables.", "verification_options": "Tighter system evaluation settings."},
                returns={"type": "ShootingResult"},
                requires=("OLSS required contract",),
                best_for=("single shooting", "multiple shooting", "rectangular residual systems"),
                watch_out=("solver convergence and verification are separate",),
                example="result = opt.shooting.solve_shooting(system, z0, verification_options={'rtol': 1e-11})",
                module="optimizer/shooting/solvers.py",
                tags=("shooting", "root", "verification"),
                root=True,
                phase="OLSS",
            ),
            _make_item(
                "solve_shooting_multistart",
                "shooting",
                "campaign",
                "Solve independent seeds, deduplicate qualified roots, and rank a declared selection metric.",
                inputs={"starts": "Iterable of ShootingVariables.", "system_factory": "Required for parallel workers."},
                returns={"type": "MultiStartResult"},
                requires=("OLSS required contract",),
                best_for=("multiple root branches", "seed clouds"),
                example="campaign = opt.shooting.solve_shooting_multistart(system, seeds)",
                module="optimizer/shooting/campaigns.py",
                tags=("shooting", "multistart", "parallel", "ranking"),
                root=True,
                phase="OLSS",
            ),
            _make_item(
                "continue_shooting",
                "shooting",
                "coordinator",
                "Follow a one-dimensional OLSS root branch with pseudo-arclength continuation.",
                inputs={"previous/current": "Two verified neighboring branch points.", "step_size": "Arclength predictor size."},
                returns={"type": "ShootingContinuationResult"},
                requires=("Jacobian numerical rank = variable count - 1; redundant residual rows are allowed",),
                best_for=("continuation through folds",),
                example="branch = opt.shooting.continue_shooting(system, z_prev, z_now, steps=20, step_size=0.1)",
                module="optimizer/shooting/continuation.py",
                tags=("shooting", "continuation", "pseudo_arclength", "fold"),
                root=True,
                phase="OLSS",
            ),
            _make_item(
                "solve_direct_shooting",
                "shooting",
                "solver",
                "Optimize a declared selection cost along the feasible OLSS residual set.",
                inputs={"system": "OLSS system declaring selection metric and cost_gradient.", "initial": "Initial ShootingVariables."},
                returns={"type": "ShootingResult"},
                requires=("selection metric", "cost_gradient"),
                best_for=("underdetermined shooting", "cost-based feasible selection"),
                watch_out=("qualification is not a global optimality certificate",),
                example="result = opt.shooting.solve_direct_shooting(system, z0)",
                module="optimizer/shooting/solvers.py",
                tags=("shooting", "cost", "constraints"),
                root=True,
                phase="OLSS",
            ),
            _make_item(
                "verify_shooting_jacobian",
                "shooting",
                "derivative",
                "Compare the declared OLSS Jacobian with explicit central finite differences.",
                inputs={"system": "OLSS system.", "variables": "Point to check.", "eps": "Relative perturbation scale."},
                returns={"type": "dict", "fields": "passed, errors, comparison_steps, step_max_abs_errors, analytical, numerical"},
                requires=("OLSS residuals and Jacobian",),
                best_for=("system bring-up", "sensitivity debugging"),
                example="check = opt.shooting.verify_shooting_jacobian(system, z0)",
                module="optimizer/shooting/derivatives.py",
                tags=("shooting", "jacobian", "derivative", "verification"),
                root=True,
                phase="OLSS",
            ),
            _make_item("continue_parameter", "shooting", "coordinator", "Natural continuation over one secondary parameter with adaptive steps and exact target landing.", module="optimizer/shooting/continuation.py", tags=("shooting", "continuation", "parameter"), root=True, phase="OLSS"),
            _make_item("random_shooting_seeds", "shooting", "generator", "Generate scale-aware random shooting seeds.", module="optimizer/shooting/campaigns.py", tags=("shooting", "seed", "random"), root=True, phase="OLSS"),
            _make_item("perturb_shooting_seeds", "shooting", "generator", "Generate a scale-aware perturbation cloud around an anchor.", module="optimizer/shooting/campaigns.py", tags=("shooting", "seed", "perturbation"), root=True, phase="OLSS"),
            _make_item("refine_shooting_seeds", "shooting", "campaign", "Solve, select elites, perturb, and refine a shooting seed campaign.", module="optimizer/shooting/campaigns.py", tags=("shooting", "seed", "refine"), root=True, phase="OLSS"),
            _make_item("compare_shooting_replay", "shooting", "diagnostic", "Replay OLSS-recovered controls in OLGS and compare declared metrics.", module="optimizer/shooting/replay.py", tags=("shooting", "olgs", "replay"), root=True, family=("olgs", "olss"), phase="OLSS"),
            _make_item("finite_difference_shooting_jacobian", "shooting", "derivative", "Central-difference reference Jacobian of the OLSS residual vector; does not replace the declared Jacobian.", module="optimizer/shooting/derivatives.py", tags=("shooting", "jacobian", "derivative"), root=True, phase="OLSS"),
            _make_item("ShootingEvaluator", "shooting", "runtime", "Validated, cached, and counted OLSS evaluation shared by the shooting tools.", module="optimizer/shooting/evaluator.py", tags=("shooting", "evaluation", "cache"), root=True, phase="OLSS"),
        ),
    )

    reader_group = CatalogGroup(
        name="optimizer_log",
        summary="Standalone, read-only tracker reader package; it is not yet a portal UI or command-line application.",
        common_inputs={"run_dir": "Existing tracker run directory."},
        common_returns={"type": "validated receipt, records, live state, or terminal state"},
        workflow=("Locate a known run directory.", "Read or follow its timeline.", "Build user interfaces only above these read-only APIs."),
        items=(
            _make_item("read_timeline", "optimizer_log", "reader", "Read validated tracker events without importing optimizer numerics.", module="optimizer_log/reader.py", tags=("reader", "tracker", "read_only")),
            _make_item("follow_timeline", "optimizer_log", "reader", "Incrementally follow appended complete JSONL events.", module="optimizer_log/reader.py", tags=("reader", "tracker", "live")),
            _make_item("inspect_writer", "optimizer_log", "reader", "Inspect writer ownership/liveness without taking ownership.", module="optimizer_log/reader.py", tags=("reader", "tracker", "liveness")),
            _make_item("TimelineReadResult", "optimizer_log", "data_object", "Complete timeline records and, if present, one ignored damaged tail line.", module="optimizer_log/reader.py", tags=("reader", "tracker")),
            _make_item("read_receipt", "optimizer_log", "reader", "Read the stored run receipt exactly as written.", module="optimizer_log/reader.py", tags=("reader", "tracker")),
            _make_item("read_live", "optimizer_log", "reader", "Read the current replaceable observer snapshot, if the run has started.", module="optimizer_log/reader.py", tags=("reader", "tracker", "live")),
            _make_item("read_terminal", "optimizer_log", "reader", "Read the direct terminal-result index, if the run has ended.", module="optimizer_log/reader.py", tags=("reader", "tracker")),
            _make_item("read_writer", "optimizer_log", "reader", "Read explicit writer ownership and heartbeat state, if present.", module="optimizer_log/reader.py", tags=("reader", "tracker", "liveness")),
            _make_item("metric_coverage", "optimizer_log", "reader", "Compare declared metrics with observed and explicitly omitted metrics.", module="optimizer_log/reader.py", tags=("reader", "tracker", "metrics")),
            _make_item("project_timeline", "optimizer_log", "reader", "Create a generic read-only projection for API or interface consumers.", module="optimizer_log/reader.py", tags=("reader", "tracker")),
            _make_item("validate_event", "optimizer_log", "validator", "Validate one causal event envelope without importing optimizer code.", module="optimizer_log/reader.py", tags=("reader", "validation")),
            _make_item("validate_receipt", "optimizer_log", "validator", "Validate the generic receipt fields required by read-only consumers.", module="optimizer_log/reader.py", tags=("reader", "validation")),
            _make_item("validate_timeline", "optimizer_log", "validator", "Validate the one-start, one-terminal lifecycle of a complete prefix.", module="optimizer_log/reader.py", tags=("reader", "validation")),
        ),
    )

    return {
        "optimizers": optimizers_group,
        "guesses": guesses_group,
        "utils": utils_group,
        "schedules": schedules_group,
        "base": base_group,
        "core": core_group,
        "campaigns": campaigns_group,
        "tracker": tracker_group,
        "visualization": visualization_group,
        "system_olgs": system_group,
        "system_olss": shooting_system_group,
        "shooting": shooting_group,
        "optimizer_log": reader_group,
    }


def _catalog() -> dict[str, CatalogGroup]:
    global _GROUPS
    if not _GROUPS:
        _GROUPS = _build_groups()
    return _GROUPS


def group(name: str) -> CatalogGroup:
    try:
        return _catalog()[str(name)]
    except KeyError as exc:
        choices = ", ".join(sorted(_catalog()))
        raise KeyError(f"Unknown catalog group {name!r}; choices: {choices}.") from exc


def groups(*, h: bool = False) -> dict[str, Any] | str:
    payload = {
        name: {
            "summary": item.summary,
            "count": len(item.items),
            "common_returns": _dict(item.common_returns),
        }
        for name, item in _catalog().items()
    }
    if not h:
        return payload
    lines = ["catalog groups"]
    for name, data in payload.items():
        lines.append(f"- {name}: {data['summary']} ({data['count']} items)")
    return "\n".join(lines)


def list(*, h: bool = False) -> dict[str, Any] | str:
    if h:
        lines = ["catalog"]
        for name in _catalog():
            group_payload = group(name).list(h=True)
            lines.append(group_payload)
        return "\n\n".join(lines)
    return {
        "groups": groups(),
        "items": {
            name: group_obj.list()["items"]
            for name, group_obj in _catalog().items()
        },
    }


def info(name: str, *, h: bool = False) -> dict[str, Any] | str:
    if str(name) in _catalog():
        return group(str(name)).list(h=h)
    item = _resolve_item(name)
    return item.render_human() if h else item.to_dict()


def search(query: str, *, h: bool = False) -> dict[str, Any] | str:
    needle = str(query).lower()
    matches = [
        item
        for group_obj in _catalog().values()
        for item in group_obj.items
        if _matches_query(item, needle)
    ]
    if h:
        lines = [f"search: {query}"]
        lines.extend(f"- {item.id}: {item.summary}" for item in matches)
        return "\n".join(lines)
    return {
        "query": query,
        "count": len(matches),
        "items": [item.short_dict() for item in matches],
    }


def path(name: str, *, h: bool = False) -> dict[str, Any] | str:
    key = str(name).lower()
    if key not in _PATHS:
        choices = ", ".join(sorted(_PATHS))
        raise KeyError(f"Unknown catalog path {name!r}; choices: {choices}.")
    payload = _PATHS[key]
    if not h:
        return {
            "name": key,
            "goal": payload["goal"],
            "steps": builtins.list(payload["steps"]),
            "related": builtins.list(payload.get("related", ())),
        }
    lines = [key, f"Goal: {payload['goal']}", "Steps:"]
    lines.extend(f"- {step}" for step in payload["steps"])
    if payload.get("related"):
        lines.append("Related: " + ", ".join(payload["related"]))
    return "\n".join(lines)


def exports(group_name: str) -> builtins.list[str]:
    """Return the sorted public names a namespace exports, built from its declarations.

    The names are the declared names and their aliases. The package root
    (``base``) also exports every declaration marked ``root``. Discovery helpers
    such as ``info`` and ``list`` are not tools; each namespace adds its own.
    """

    return sorted(_namespace_items(group_name))


def _namespace_items(group_name: str) -> dict[str, CatalogItem]:
    """Map each public name of a namespace to the one declaration behind it."""

    if group_name == "base":
        items = [
            item
            for group_obj in _catalog().values()
            for item in group_obj.items
            if group_obj.name == "base" or item.root
        ]
    else:
        items = builtins.list(group(group_name).items)
    names: dict[str, CatalogItem] = {}
    for item in items:
        name = item.object_name or item.name
        if name in names:
            raise ValueError(
                f"Two declarations export {name!r} from the {group_name} namespace: {names[name].id} and {item.id}."
            )
        names[name] = item
    for item in items:
        for alias in item.aliases:
            names.setdefault(alias, item)
    return names


def attach_namespace_helpers(namespace: dict[str, Any], group_name: str) -> None:
    """Attach ``list``, ``info``, and per-callable ``.info`` helpers to a module."""

    group_obj = group(group_name)

    def namespace_list(
        *,
        h: bool = False,
        kind: str | None = None,
        requires: str | None = None,
        returns: str | None = None,
        tag: str | None = None,
    ) -> dict[str, Any] | str:
        return group_obj.list(h=h, kind=kind, requires=requires, returns=returns, tag=tag)

    def namespace_info(name: str, *, h: bool = False) -> dict[str, Any] | str:
        return group_obj.info(name, h=h)

    namespace_list.__name__ = "list"
    namespace_info.__name__ = "info"
    namespace["list"] = namespace_list
    namespace["info"] = namespace_info

    for name, item in _namespace_items(group_name).items():
        obj = namespace.get(name)
        if obj is not None:
            _attach_info(obj, item)


def attach_root_helpers(namespace: dict[str, Any]) -> None:
    """Attach ``.info`` helpers to the root exports: base declarations and root shortcuts."""

    for name, item in _namespace_items("base").items():
        obj = namespace.get(name)
        if obj is not None:
            _attach_info(obj, item)


def _attach_info(obj: Any, item: CatalogItem) -> None:
    # Only library functions and classes carry .info. Protocols are skipped: on
    # Python 3.11 a later class attribute becomes a required protocol member.
    # Constants and typing aliases are skipped: a typing alias forwards
    # setattr to its origin class, such as collections.abc.Mapping.
    if not (inspect.isfunction(obj) or inspect.isclass(obj)) or getattr(obj, "_is_protocol", False):
        return

    def item_info(*, h: bool = False) -> dict[str, Any] | str:
        return item.render_human() if h else item.to_dict()

    item_info.__name__ = "info"
    item_info.__doc__ = f"Return catalog metadata for {item.id}."
    if isinstance(obj, type):
        setattr(obj, "info", staticmethod(item_info))
    else:
        setattr(obj, "info", item_info)


def _resolve_item(name: str) -> CatalogItem:
    query = str(name)
    if "." in query:
        group_name, item_name = query.split(".", 1)
        return group(group_name).get(item_name)
    matches = []
    for group_obj in _catalog().values():
        for item in group_obj.items:
            if item.name == query or query in item.aliases:
                matches.append(item)
    if not matches:
        raise KeyError(f"Unknown catalog item {query!r}.")
    if len(matches) > 1:
        # A bare name means the root export when there is one, and any of the
        # declarations when they all describe the same re-exported object.
        root_item = _namespace_items("base").get(query)
        if root_item is not None and root_item in matches:
            return root_item
        objects = [item.public_object() for item in matches]
        if objects[0] is not None and all(obj is objects[0] for obj in objects):
            return matches[0]
        choices = ", ".join(item.id for item in matches)
        raise KeyError(f"Ambiguous catalog item {query!r}; use one of: {choices}.")
    return matches[0]


def _matches_query(item: CatalogItem, needle: str) -> bool:
    haystack = [
        item.id,
        item.name,
        item.kind,
        item.summary,
        item.return_type,
        " ".join(item.requires),
        " ".join(item.variants),
        " ".join(item.best_for),
        " ".join(item.not_for),
        " ".join(item.watch_out),
        " ".join(item.related),
        " ".join(item.aliases),
        " ".join(item.tags),
        " ".join(item.notes),
        " ".join(item.family),
    ]
    haystack.extend(item.inputs.keys())
    haystack.extend(item.inputs.values())
    haystack.extend(item.returns.keys())
    haystack.extend(item.returns.values())
    return any(needle in str(value).lower() for value in haystack)


_PATHS: dict[str, dict[str, Any]] = {
    "beginner": {
        "goal": "Find the public surface and run a small, inspectable optimization.",
        "steps": (
            "opt.info(h=True)",
            "opt.list(h=True)",
            "opt.guesses.list(h=True)",
            "controls = opt.guesses.random_fourier_guess(system, amplitude=0.1, modes=5, seed=1)",
            "check = opt.utils.verify_gradient(system, controls)",
            "result = opt.optimizers.adam(system, controls, step_size=0.05, maxiter=20)",
            "report = opt.utils.diagnostic_report(system, result.controls)",
        ),
        "related": ("guesses.random_fourier_guess", "utils.verify_gradient", "optimizers.adam"),
    },
    "debug_gradient": {
        "goal": "Check whether a new system gradient is trustworthy before blaming an optimizer.",
        "steps": (
            "controls = opt.guesses.random_smooth_guess(system, amplitude=0.1, seed=1)",
            "report = opt.utils.diagnostic_report(system, controls)",
            "check = opt.utils.verify_gradient(system, controls, eps=1e-6, directions=8)",
            "if needed on small systems: fd = opt.utils.finite_difference_gradient(system, controls)",
        ),
        "related": ("utils.verify_gradient", "utils.finite_difference_gradient", "utils.diagnostic_report"),
    },
    "repair_residuals": {
        "goal": "Understand and reduce named hard residual violations.",
        "steps": (
            "geometry = opt.utils.geometry_probe(system, controls, residuals='hard', eps=1e-6)",
            "fixed = opt.utils.repair_newton(system, controls, residuals='hard', maxiter=4, tolerance=1e-8)",
            "payload = fixed.to_dict(include_controls=False)",
            "report = opt.utils.diagnostic_report(system, fixed.controls)",
        ),
        "related": ("utils.geometry_probe", "utils.repair_newton", "utils.project_gradient"),
    },
    "optimize_then_polish": {
        "goal": "Use a robust rough optimizer first, then polish with curvature information.",
        "steps": (
            "controls = opt.guesses.random_fourier_guess(system, amplitude=0.1, modes=5, seed=1)",
            "r1 = opt.optimizers.adam(system, controls, step_size=0.05, maxiter=50)",
            "r2 = opt.optimizers.lbfgs(system, r1.controls, warmstart=r1.warmstart(target_optimizer='lbfgs'), maxiter=20)",
            "report = opt.utils.diagnostic_report(system, r2.controls)",
        ),
        "related": ("optimizers.adam", "optimizers.lbfgs", "utils.diagnostic_report"),
    },
}


optimizers = CatalogGroupView("optimizers")
guesses = CatalogGroupView("guesses")
utils = CatalogGroupView("utils")
schedules = CatalogGroupView("schedules")
base = CatalogGroupView("base")
core = CatalogGroupView("core")
campaigns = CatalogGroupView("campaigns")
tracker = CatalogGroupView("tracker")
visualization = CatalogGroupView("visualization")
system_olgs = CatalogGroupView("system_olgs")
system_olss = CatalogGroupView("system_olss")
shooting = CatalogGroupView("shooting")
optimizer_log = CatalogGroupView("optimizer_log")


__all__ = [
    "CatalogGroup",
    "CatalogGroupView",
    "CatalogItem",
    "CatalogItemView",
    "attach_namespace_helpers",
    "attach_root_helpers",
    "base",
    "campaigns",
    "core",
    "exports",
    "group",
    "groups",
    "guesses",
    "info",
    "list",
    "optimizers",
    "optimizer_log",
    "path",
    "schedules",
    "search",
    "shooting",
    "system_olgs",
    "system_olss",
    "tracker",
    "utils",
    "visualization",
]
