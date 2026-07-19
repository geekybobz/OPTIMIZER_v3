"""Programmatic catalog for exploring OPTIMIZER v3.

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
    """Curated metadata for one public tool, class, or alias."""

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
        }
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
        }
        return {key: value for key, value in payload.items() if value not in (None, [], {}, "")}

    def render_human(self) -> str:
        lines = [self.id, f"Summary: {self.summary}"]
        signature = self.signature()
        if signature:
            lines.append(f"Signature: {self.name}{signature}")
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
) -> CatalogItem:
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
    )


_GROUP_MODULES = {
    "optimizers": "optimizer.optimizers",
    "guesses": "optimizer.guesses",
    "utils": "optimizer.utils",
    "schedules": "optimizer.schedules",
    "core": "optimizer",
}


_OPTIMIZER_COMMON_INPUTS = {
    "system": "Object implementing the optimizer system contract. Most methods need evaluate(...) and gradient(...).",
    "controls": "Starting Controls. Engine-based optimizers can instead start from state or warmstart; cma_es requires controls.",
    "maxiter": "Short chunk iteration budget. Longer workflows should chain chunks and warmstarts.",
    "step_size": "Initial scalar update size when supported.",
    "warmstart": "OptimizerResult, RunState, or WarmStartState used to continue a compatible method.",
    "accept_metric": "Scalar metric used for accept/reject decisions, usually J.",
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
            "Run short chunks and inspect result.J, result.metrics, and result.trace.",
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
                    "step_size": "Learning rate. Start small when gradients are large.",
                    "beta1/beta2/eps": "Moment and denominator stabilization parameters.",
                    "weight_decay": "Decoupled decay for adamw or explicit decay.",
                    "max_step_norm": "Optional L2 clipping of the applied update.",
                    "warmstart": "Carries Adam moment state only when target_optimizer is adam.",
                },
                returns=_RESULT_RETURN,
                requires=("system.evaluate", "system.gradient", "Controls"),
                variants=("adam", "amsgrad", "adamw", "radam", "adabelief"),
                best_for=("rough first optimization", "uneven gradient scales", "warmstarting before polishing"),
                not_for=("systems without analytical gradient", "hard residual repair by itself"),
                watch_out=("step_size controls the real behavior", "accepted Adam state advances only on accepted trials"),
                related=("utils.verify_gradient", "optimizers.lbfgs", "guesses.random_fourier_guess"),
                example="result = opt.optimizers.adam(system, controls, step_size=0.05, maxiter=50)",
                module="optimizer/optimizers/adam.py",
                phase="Phase 7",
                tags=("gradient", "adaptive", "first_order", "warmstart"),
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
                    "step_size": "Gradient step scale.",
                    "max_step_norm": "Optional update clipping; default active for clipped variant.",
                },
                returns=_RESULT_RETURN,
                requires=("system.evaluate", "system.gradient", "Controls"),
                variants=("heavy_ball", "nesterov", "restart", "clipped"),
                best_for=("simple gradient baselines", "smoothing zigzag motion", "debuggable first-order behavior"),
                watch_out=("large step_size can reject and stall", "restart variant resets velocity after rejection"),
                related=("optimizers.line_search", "optimizers.adam", "utils.diagnostic_report"),
                example="result = opt.optimizers.momentum(system, controls, variant='nesterov', step_size=0.03, maxiter=20)",
                module="optimizer/optimizers/momentum.py",
                phase="Phase 7",
                tags=("gradient", "first_order", "low_memory"),
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
            ),
            _make_item(
                "cma_es",
                "optimizers",
                "optimizer",
                "Derivative-free population search with isotropic or diagonal sampling radius.",
                inputs={
                    "system": "Needs evaluate(...). It does not use gradient during iterations.",
                    "controls": "Required starting mean Controls.",
                    "variant": "diagonal or isotropic.",
                    "population_size": "Candidates sampled per generation; default scales with dimension.",
                    "elite_fraction": "Fraction retained to update the mean.",
                    "sigma": "Initial sampling radius.",
                    "seed": "Random seed for reproducibility.",
                },
                returns=_RESULT_RETURN,
                requires=("system.evaluate", "Controls"),
                variants=("diagonal", "isotropic"),
                best_for=("rough global search", "systems with unreliable gradients", "escaping poor starts"),
                not_for=("fast polishing near a good deterministic solution", "very high-dimensional expensive evaluations without budget"),
                watch_out=("uses many evaluations per iteration", "implementation is compact CMA-ES style, not full research CMA-ES"),
                related=("guesses.random_fourier_guess", "optimizers.adam", "utils.metric_report"),
                example="result = opt.optimizers.cma_es(system, controls, population_size=24, sigma=0.15, seed=7, maxiter=6)",
                module="optimizer/optimizers/cma_es.py",
                phase="Phase 7",
                tags=("derivative_free", "population", "random", "global_search"),
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
            ),
            _make_item(
                "metric_guard",
                "utils",
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
                related=("optimizers.line_search", "core.run_chunk"),
                example="guard = opt.utils.metric_guard(improve='J', require={'fidelity': ('>=', 0.0)})",
                module="optimizer/core/guards.py",
                phase="Phase 11",
                tags=("guard", "acceptance", "metrics"),
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
            ),
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
            ),
        ),
    )

    core_group = CatalogGroup(
        name="core",
        summary="Core data objects, engine helpers, blackbox ledgers, trace, checkpoints, and bound context utilities.",
        common_inputs={
            "system": "Project physics object satisfying the optimizer contract.",
            "controls": "Named dense control matrix represented by Controls.",
        },
        common_returns={
            "Evaluation": "Single evaluate(...) snapshot.",
            "OptimizerResult": "Standard optimizer result.",
            "Trace": "In-memory run records and checkpoints.",
            "BlackBoxRun": "Durable numeric run ledger with selected array artifacts.",
        },
        workflow=(
            "Build or receive Controls matching system.control_spec().",
            "Call evaluate/gradient or a role namespace method.",
            "Carry OptimizerResult, WarmStartState, Trace, or BlackBoxRun across chunks.",
        ),
        items=(
            _make_item(
                "ControlSpec",
                "core",
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
                "core",
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
                "core",
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
            ),
            _make_item(
                "gradient",
                "core",
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
                related=("core.StepContext", "core.StepProposal", "utils.metric_guard"),
                example="result = opt.run_chunk(system, controls, step=my_step, optimizer_name='custom', maxiter=5)",
                module="optimizer/core/engine.py",
                phase="Phase 5",
                tags=("engine", "advanced"),
            ),
            _make_item(
                "OptimizerResult",
                "core",
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
                "core",
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
                "Trace",
                "core",
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
                "BlackBoxRun",
                "core",
                "data_object",
                "Durable numeric blackbox writer for blackbox.json, ledger.jsonl, and selected arrays/*.npz artifacts.",
                inputs={"run_dir": "Run folder.", "policy": "minimal, standard, full, or policy mapping."},
                returns={"type": "BlackBoxRun", "methods": "record_iteration, record_chunk, snapshot, record_repair, close, records"},
                requires=("writable run folder",),
                best_for=("closed-loop numeric audit trails", "post-run diagnostics", "TensorBoard-like dashboard data source"),
                related=("core.diagnostics", "optimizers.adam", "utils.repair_newton"),
                example="box = opt.blackbox.start(run_dir); result = opt.adam(system, controls, blackbox=box)",
                module="optimizer/blackbox/run.py",
                phase="Phase 12",
                aliases=("blackbox_start",),
                tags=("blackbox", "ledger", "diagnostics", "artifacts"),
            ),
            _make_item(
                "diagnostics",
                "core",
                "helper",
                "Read a blackbox run folder and return structured historical diagnostics without extra system calls.",
                inputs={"run_dir": "Folder containing blackbox.json and ledger.jsonl.", "details": "summary, gradient, decisions, repairs, or thresholds."},
                returns={"type": "dict", "fields": "manifest, latest_iteration, analysis, optional detail tables"},
                requires=("blackbox run folder",),
                best_for=("closed-loop review", "gradient trend inspection", "accept/reject and repair analysis"),
                related=("core.BlackBoxRun", "utils.diagnostic_report"),
                example="report = opt.diagnostics(run_dir, details='gradient')",
                module="optimizer/blackbox/analysis.py",
                phase="Phase 12",
                tags=("blackbox", "diagnostics", "analysis"),
            ),
            _make_item(
                "context",
                "core",
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
        ),
    )

    return {
        "optimizers": optimizers_group,
        "guesses": guesses_group,
        "utils": utils_group,
        "schedules": schedules_group,
        "core": core_group,
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

    for item in group_obj.items:
        obj = namespace.get(item.object_name or item.name)
        if obj is not None:
            _attach_info(obj, item)


def attach_root_helpers(namespace: dict[str, Any]) -> None:
    """Attach ``.info`` helpers to top-level shortcut functions and classes."""

    for group_obj in _catalog().values():
        for item in group_obj.items:
            obj = namespace.get(item.object_name or item.name)
            if obj is not None:
                _attach_info(obj, item)


def _attach_info(obj: Any, item: CatalogItem) -> None:
    def item_info(*, h: bool = False) -> dict[str, Any] | str:
        return item.render_human() if h else item.to_dict()

    item_info.__name__ = "info"
    item_info.__doc__ = f"Return catalog metadata for {item.id}."
    if isinstance(obj, type):
        setattr(obj, "info", staticmethod(item_info))
    else:
        try:
            setattr(obj, "info", item_info)
        except (AttributeError, TypeError):
            return


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
            "opt.catalog.groups(h=True)",
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
core = CatalogGroupView("core")


__all__ = [
    "CatalogGroup",
    "CatalogGroupView",
    "CatalogItem",
    "CatalogItemView",
    "attach_namespace_helpers",
    "attach_root_helpers",
    "core",
    "group",
    "groups",
    "guesses",
    "info",
    "list",
    "optimizers",
    "path",
    "schedules",
    "search",
    "utils",
]
