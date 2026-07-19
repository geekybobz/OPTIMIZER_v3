"""Blackbox run writer.

``BlackBoxRun`` is the durable numeric ledger used by optimizers and repair tools.  It
does not run evaluations, gradients, residuals, or Jacobians.  Callers pass values they
already computed, and the writer stores compact summaries plus selected array
artifacts.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from optimizer.blackbox.analysis import analyze_path
from optimizer.blackbox.artifacts import artifact_id, save_array, save_controls
from optimizer.blackbox.policy import BlackBoxPolicy
from optimizer.blackbox.reader import latest_record, read_manifest, read_records
from optimizer.blackbox.records import (
    SCHEMA_VERSION,
    controls_summary,
    gradient_summary,
    json_safe,
    metrics_summary,
    record_base,
    scalar_float,
    system_summary,
    utc_now,
)
from optimizer.controls import Controls


FAILURE_REASONS = {"gradient_failed", "proposal_failed", "nonfinite_trial", "line_search_failed"}


def default_run_dir() -> Path:
    """Return a default ignored run folder under ``runs/``."""

    stamp = utc_now().replace("-", "").replace(":", "").replace("Z", "")
    return Path("runs") / f"blackbox_{stamp}_{uuid4().hex[:8]}"


@dataclass
class BlackBoxRun:
    """Append-only numeric blackbox for one optimization run."""

    run_dir: Path | str
    run_id: str | None = None
    policy: BlackBoxPolicy | str | Mapping[str, Any] | None = None
    overwrite: bool = False
    manifest: dict[str, Any] = field(init=False)
    seq: int = field(init=False, default=0)
    _last_gradient_flat: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.policy = BlackBoxPolicy.from_value(self.policy)
        if self.overwrite and self.run_dir.exists():
            shutil.rmtree(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "arrays").mkdir(parents=True, exist_ok=True)
        if (self.run_dir / "blackbox.json").exists() and not self.overwrite:
            self.manifest = read_manifest(self.run_dir)
            self.run_id = str(self.manifest.get("run_id", self.run_id or uuid4().hex))
            last = latest_record(self.run_dir)
            self.seq = int(last.get("seq", 0)) if last else int(self.manifest.get("counts", {}).get("records", 0))
        else:
            self.run_id = self.run_id or uuid4().hex
            now = utc_now()
            self.manifest = {
                "schema": SCHEMA_VERSION,
                "run_id": self.run_id,
                "created_utc": now,
                "updated_utc": now,
                "status": "initialized",
                "run_dir": str(self.run_dir),
                "policy": self.policy.to_dict(),
                "system": {},
                "objective": {},
                "initial": {},
                "best": {},
                "final": {},
                "counts": {
                    "records": 0,
                    "iterations": 0,
                    "chunks": 0,
                    "checkpoints": 0,
                    "thresholds": 0,
                    "repairs": 0,
                    "analysis": 0,
                    "artifacts": 0,
                },
                "artifacts": {},
            }
            self._write_manifest()

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "blackbox.json"

    @property
    def ledger_path(self) -> Path:
        return self.run_dir / "ledger.jsonl"

    def _write_manifest(self) -> None:
        self.manifest["updated_utc"] = utc_now()
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(json_safe(self.manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.manifest_path)

    def _append(self, kind: str, *, write_manifest: bool = True, **fields: Any) -> dict[str, Any]:
        self.seq += 1
        record = record_base(self.seq, kind, run_id=str(self.run_id))
        record.update(json_safe(fields))
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

        counts = self.manifest.setdefault("counts", {})
        counts["records"] = int(counts.get("records", 0)) + 1
        count_key = {
            "iteration": "iterations",
            "chunk": "chunks",
            "checkpoint": "checkpoints",
            "threshold": "thresholds",
            "repair": "repairs",
            "window_analysis": "analysis",
        }.get(kind)
        if count_key is not None:
            counts[count_key] = int(counts.get(count_key, 0)) + 1
        if write_manifest and counts["records"] % int(self.policy.manifest_every) == 0:
            self._write_manifest()
        return record

    def _register_artifact(self, label: str, entry: dict[str, Any]) -> dict[str, Any]:
        artifacts = self.manifest.setdefault("artifacts", {})
        artifacts[str(label)] = entry
        counts = self.manifest.setdefault("counts", {})
        counts["artifacts"] = len(artifacts)
        return entry

    def record_start(
        self,
        *,
        system: Any | None = None,
        controls: Controls | None = None,
        metrics: Mapping[str, Any] | None = None,
        optimizer: str | None = None,
        stage: str | None = None,
        objective: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Record run start and initial controls/metrics."""

        if self.manifest.get("status") != "initialized":
            return None
        self.manifest["status"] = "running"
        self.manifest["system"] = system_summary(system)
        self.manifest["objective"] = json_safe(dict(objective or {}))
        initial: dict[str, Any] = {
            "metrics": metrics_summary(metrics),
            "controls": controls_summary(controls),
            "optimizer": optimizer,
            "stage": stage,
        }
        if controls is not None and self.policy.save_initial:
            entry = save_controls(
                self.run_dir,
                artifact_id("controls", "initial", seq=self.seq + 1, iteration=0),
                controls,
                metadata={"label": "initial", "stage": stage, "optimizer": optimizer},
            )
            self._register_artifact("initial_controls", entry)
            initial["controls_ref"] = entry["path"]
        self.manifest["initial"] = json_safe(initial)
        metric = str((objective or {}).get("metric", "J"))
        value = scalar_float(metrics.get(metric)) if metrics and metric in metrics else None
        if value is not None:
            self.manifest["best"] = {
                "metric": metric,
                "mode": str((objective or {}).get("mode", "min")),
                "value": value,
                "iteration": 0,
                "controls_ref": initial.get("controls_ref"),
            }
        self._write_manifest()
        return self._append(
            "start",
            system=self.manifest["system"],
            optimizer=optimizer,
            stage=stage,
            objective=self.manifest["objective"],
            config=dict(config or {}),
            metadata=dict(metadata or {}),
            metrics=initial["metrics"],
            controls=initial["controls"],
            artifacts={"controls": initial.get("controls_ref")},
        )

    def _best_improved(self, metrics: Mapping[str, Any], *, metric: str, mode: str) -> tuple[bool, float | None]:
        value = scalar_float(metrics.get(metric)) if metric in metrics else None
        if value is None:
            return False, None
        best = self.manifest.get("best", {})
        previous = best.get("value")
        if previous is None:
            return True, value
        previous_value = float(previous)
        improved = value < previous_value if mode == "min" else value > previous_value
        return bool(improved), value

    def _threshold_flags(
        self,
        *,
        metrics: dict[str, Any],
        gradient: dict[str, Any],
        controls: dict[str, Any],
    ) -> list[dict[str, Any]]:
        flags: list[dict[str, Any]] = []
        rel_grad = gradient.get("rel_delta_norm")
        if rel_grad is not None and float(rel_grad) >= float(self.policy.gradient_rel_delta_threshold):
            flags.append(
                {
                    "code": "gradient_spike",
                    "metric": "gradient.rel_delta_norm",
                    "value": float(rel_grad),
                    "threshold": float(self.policy.gradient_rel_delta_threshold),
                    "ratio": float(rel_grad) / max(float(self.policy.gradient_rel_delta_threshold), 1.0),
                }
            )
        rel_dj = metrics.get("rel_dJ")
        if rel_dj is not None and float(rel_dj) >= float(self.policy.metric_regression_rel_threshold):
            flags.append(
                {
                    "code": "metric_regression",
                    "metric": "metrics.rel_dJ",
                    "value": float(rel_dj),
                    "threshold": float(self.policy.metric_regression_rel_threshold),
                    "ratio": float(rel_dj) / max(float(self.policy.metric_regression_rel_threshold), 1.0),
                }
            )
        if self.policy.control_max_abs_threshold is not None:
            max_abs = controls.get("max_abs")
            threshold = float(self.policy.control_max_abs_threshold)
            if max_abs is not None and float(max_abs) >= threshold:
                flags.append(
                    {
                        "code": "control_max_abs_threshold",
                        "metric": "controls.max_abs",
                        "value": float(max_abs),
                        "threshold": threshold,
                        "ratio": float(max_abs) / max(threshold, np.finfo(float).tiny),
                    }
                )
        return flags

    def _decision_payload(
        self,
        *,
        accepted: bool | None,
        reason: str | None,
        technical: Mapping[str, Any] | None,
        accept_metric: str,
        trial_metrics: Mapping[str, Any] | None,
        previous_metrics: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        acceptance = dict((technical or {}).get("acceptance", {})) if isinstance(technical, Mapping) else {}
        current_value = acceptance.get("current_value")
        trial_value = acceptance.get("trial_value")
        if current_value is None and previous_metrics and accept_metric in previous_metrics:
            current_value = scalar_float(previous_metrics[accept_metric])
        if trial_value is None and trial_metrics and accept_metric in trial_metrics:
            trial_value = scalar_float(trial_metrics[accept_metric])
        delta = None
        if current_value is not None and trial_value is not None:
            delta = float(trial_value) - float(current_value)
        return {
            "accepted": accepted,
            "code": reason,
            "metric": accept_metric,
            "current": current_value,
            "trial": trial_value,
            "delta": delta,
            "tolerance": acceptance.get("tolerance"),
            "raw": acceptance,
        }

    def _step_payload(self, technical: Mapping[str, Any] | None) -> dict[str, Any]:
        technical = dict(technical or {})
        proposal = dict(technical.get("proposal", {})) if isinstance(technical.get("proposal"), Mapping) else {}
        line_search = {}
        acceptance = technical.get("acceptance")
        if isinstance(acceptance, Mapping) and isinstance(acceptance.get("line_search"), Mapping):
            line_search = dict(acceptance["line_search"])
        source = {**proposal, **line_search}
        return {
            "step_size": source.get("step_size", source.get("accepted_step_size")),
            "proposal_delta_norm": source.get("applied_step_norm", source.get("raw_step_norm")),
            "raw_step_norm": source.get("raw_step_norm"),
            "applied_step_norm": source.get("applied_step_norm"),
            "clipped": source.get("clipped"),
            "line_search_attempts": len(source.get("attempts", [])) if isinstance(source.get("attempts"), list) else None,
        }

    def record_iteration(
        self,
        *,
        optimizer: str,
        iteration: int,
        global_iteration: int | None = None,
        metrics: Mapping[str, Any] | None = None,
        previous_metrics: Mapping[str, Any] | None = None,
        trial_metrics: Mapping[str, Any] | None = None,
        controls: Controls | None = None,
        previous_controls: Controls | np.ndarray | None = None,
        proposal_controls: Controls | None = None,
        gradient: Controls | None = None,
        technical: Mapping[str, Any] | None = None,
        stage: str | None = None,
        accepted: bool | None = None,
        reason: str | None = None,
        accept_metric: str = "J",
        accept_mode: str = "min",
    ) -> dict[str, Any] | None:
        """Append one numeric optimizer iteration record."""

        metrics_payload = metrics_summary(metrics, previous=previous_metrics)
        gradient_payload = gradient_summary(gradient, previous_flat=self._last_gradient_flat)
        controls_payload = controls_summary(controls, previous=previous_controls)
        proposal_payload = controls_summary(proposal_controls, previous=previous_controls)
        flags = self._threshold_flags(metrics=metrics_payload, gradient=gradient_payload, controls=controls_payload)

        improved, best_value = self._best_improved(metrics or {}, metric=accept_metric, mode=accept_mode)
        best_ref = None
        if improved and controls is not None:
            self.manifest["best"] = {
                "metric": accept_metric,
                "mode": accept_mode,
                "value": best_value,
                "iteration": int(iteration),
            }
            if self.policy.save_best:
                entry = save_controls(
                    self.run_dir,
                    artifact_id("controls", f"best_{accept_metric}", seq=self.seq + 1, iteration=iteration),
                    controls,
                    metadata={"label": f"best_{accept_metric}", "optimizer": optimizer, "stage": stage},
                )
                self._register_artifact(f"best_{accept_metric}", entry)
                best_ref = entry["path"]
                self.manifest["best"]["controls_ref"] = best_ref

        failure_ref = None
        if reason in FAILURE_REASONS and self.policy.save_failures and (proposal_controls or controls) is not None:
            failure_controls = proposal_controls or controls
            if failure_controls is not None:
                entry = save_controls(
                    self.run_dir,
                    artifact_id("controls", f"failure_{reason}", seq=self.seq + 1, iteration=iteration),
                    failure_controls,
                    metadata={"reason": reason, "optimizer": optimizer, "stage": stage},
                )
                self._register_artifact(f"failure_{reason}_{iteration}", entry)
                failure_ref = entry["path"]

        threshold_artifacts: dict[str, str] = {}
        if flags and self.policy.save_threshold_arrays and gradient is not None:
            entry = save_array(
                self.run_dir,
                artifact_id("gradient", "threshold", seq=self.seq + 1, iteration=iteration),
                gradient.flatten(copy=True),
                metadata={"flags": flags, "optimizer": optimizer, "stage": stage},
            )
            self._register_artifact(f"threshold_gradient_{iteration}", entry)
            threshold_artifacts["gradient"] = entry["path"]

        if gradient is not None:
            self._last_gradient_flat = gradient.flatten(copy=True)

        if not self.policy.should_record_iteration(iteration) and not flags and failure_ref is None and best_ref is None:
            self._write_manifest()
            return None

        record = self._append(
            "iteration",
            optimizer=optimizer,
            stage=stage,
            i=int(iteration),
            global_i=None if global_iteration is None else int(global_iteration),
            metrics=metrics_payload,
            trial_metrics=metrics_summary(trial_metrics),
            gradient=gradient_payload,
            controls=controls_payload,
            proposal=proposal_payload,
            step=self._step_payload(technical),
            decision=self._decision_payload(
                accepted=accepted,
                reason=reason,
                technical=technical,
                accept_metric=accept_metric,
                trial_metrics=trial_metrics,
                previous_metrics=previous_metrics,
            ),
            technical=json_safe(dict(technical or {})),
            flags=flags,
            artifacts={
                "best": best_ref,
                "failure": failure_ref,
                **threshold_artifacts,
            },
        )

        for flag in flags:
            self._append(
                "threshold",
                optimizer=optimizer,
                stage=stage,
                i=int(iteration),
                code=flag["code"],
                metric=flag["metric"],
                value=flag["value"],
                threshold=flag["threshold"],
                ratio=flag["ratio"],
                artifacts=threshold_artifacts,
            )

        if self.policy.should_analyze(iteration):
            analysis = analyze_path(self.run_dir, window=int(self.policy.analysis_every), policy=self.policy)
            analysis_payload = dict(analysis)
            analysis_payload.pop("kind", None)
            self._append("window_analysis", **analysis_payload)
        return record

    def record_chunk(
        self,
        *,
        optimizer: str,
        chunk: int,
        start_iteration: int,
        end_iteration: int,
        start_metrics: Mapping[str, Any] | None = None,
        end_metrics: Mapping[str, Any] | None = None,
        system_params: Mapping[str, Any] | None = None,
        stage: str | None = None,
        accepted: bool | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Append one chunk-level numeric record."""

        return self._append(
            "chunk",
            optimizer=optimizer,
            chunk=int(chunk),
            start_i=int(start_iteration),
            end_i=int(end_iteration),
            stage=stage,
            accepted=accepted,
            reason=reason,
            start_metrics=metrics_summary(start_metrics),
            end_metrics=metrics_summary(end_metrics, previous=start_metrics),
            system_params=dict(system_params or {}),
        )

    def record_checkpoint(
        self,
        label: str,
        controls: Controls,
        *,
        metrics: Mapping[str, Any] | None = None,
        optimizer: str | None = None,
        iteration: int | None = None,
        stage: str | None = None,
        force: bool = False,
    ) -> dict[str, Any] | None:
        """Record a checkpoint and optionally save a full controls artifact."""

        entry = None
        if self.policy.should_save_checkpoint(label, iteration, force=force):
            entry = save_controls(
                self.run_dir,
                artifact_id("controls", label, seq=self.seq + 1, iteration=iteration),
                controls,
                metadata={"label": label, "optimizer": optimizer, "stage": stage, "iteration": iteration},
            )
            self._register_artifact(str(label), entry)
        if label == "latest" and entry is None:
            return None
        return self._append(
            "checkpoint",
            label=str(label),
            optimizer=optimizer,
            stage=stage,
            i=iteration,
            metrics=metrics_summary(metrics),
            controls=controls_summary(controls),
            artifact=None if entry is None else entry["path"],
        )

    def snapshot(
        self,
        label: str,
        controls: Controls,
        *,
        metrics: Mapping[str, Any] | None = None,
        optimizer: str | None = None,
        iteration: int | None = None,
        stage: str | None = None,
    ) -> dict[str, Any] | None:
        """Force-save a named controls snapshot."""

        return self.record_checkpoint(
            label,
            controls,
            metrics=metrics,
            optimizer=optimizer,
            iteration=iteration,
            stage=stage,
            force=True,
        )

    def record_repair(
        self,
        *,
        method: str,
        residual_name: str,
        before_controls: Controls,
        after_controls: Controls,
        result: Any,
        stage: str | None = None,
        metrics_before: Mapping[str, Any] | None = None,
        metrics_after: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a repair before/after numeric record."""

        history = list(getattr(result, "history", []))
        before_norm = history[0].get("residual_norm") if history else None
        before_max = history[0].get("residual_max_abs") if history else None
        after_norm = float(getattr(result, "residual_norm"))
        delta_norm = None if before_norm is None else after_norm - float(before_norm)
        rel_reduction = None
        if before_norm is not None:
            rel_reduction = 1.0 - after_norm / max(abs(float(before_norm)), np.finfo(float).tiny)

        artifact = None
        if self.policy.save_repairs:
            entry = save_controls(
                self.run_dir,
                artifact_id("controls", "repair", seq=self.seq + 1, iteration=getattr(result, "iterations", None)),
                after_controls,
                metadata={"method": method, "residual_name": residual_name, "stage": stage},
            )
            self._register_artifact(f"repair_{self.seq + 1}", entry)
            artifact = entry["path"]

        return self._append(
            "repair",
            method=method,
            stage=stage,
            residual={
                "name": residual_name,
                "norm_before": before_norm,
                "norm_after": after_norm,
                "delta_norm": delta_norm,
                "rel_reduction": rel_reduction,
                "max_abs_before": before_max,
                "max_abs_after": float(getattr(result, "residual_max_abs")),
            },
            metrics={
                "before": metrics_summary(metrics_before),
                "after": metrics_summary(metrics_after, previous=metrics_before),
            },
            controls=controls_summary(after_controls, previous=before_controls),
            iterations=int(getattr(result, "iterations")),
            converged=bool(getattr(result, "converged")),
            stop_reason=str(getattr(result, "stop_reason")),
            jacobian_source=str(getattr(result, "jacobian_source")),
            artifact=artifact,
        )

    def close(self, result: Any | None = None, *, status: str = "completed") -> dict[str, Any]:
        """Mark a run complete and record final result data."""

        controls = getattr(result, "controls", None)
        metrics = getattr(result, "metrics", None)
        stop_reason = getattr(result, "stop_reason", None)
        iterations = getattr(result, "iterations", None)
        final: dict[str, Any] = {
            "metrics": metrics_summary(metrics),
            "stop_reason": stop_reason,
            "iterations": iterations,
            "optimizer": getattr(result, "optimizer", None),
        }
        artifact = None
        if isinstance(controls, Controls) and self.policy.save_final:
            entry = save_controls(
                self.run_dir,
                artifact_id("controls", "final", seq=self.seq + 1, iteration=iterations),
                controls,
                metadata={"label": "final", "status": status, "stop_reason": stop_reason},
            )
            self._register_artifact("final_controls", entry)
            artifact = entry["path"]
            final["controls_ref"] = artifact
        self.manifest["final"] = json_safe(final)
        self.manifest["status"] = str(status)
        self._write_manifest()
        return self._append("end", status=status, final=final, artifact=artifact)

    def to_dict(self) -> dict[str, Any]:
        """Return the current manifest."""

        return dict(self.manifest)

    def records(self, *, kind: str | None = None) -> list[dict[str, Any]]:
        """Read records from this run's ledger."""

        return read_records(self.run_dir, kind=kind)


def ensure_run(value: Any, *, policy: BlackBoxPolicy | str | Mapping[str, Any] | None = None) -> BlackBoxRun | None:
    """Normalize a blackbox argument into a ``BlackBoxRun``."""

    if value is None or value is False:
        return None
    if isinstance(value, BlackBoxRun):
        return value
    if value is True:
        return BlackBoxRun(default_run_dir(), policy=policy)
    if isinstance(value, (str, Path)):
        return BlackBoxRun(value, policy=policy)
    raise TypeError("blackbox must be a BlackBoxRun, path, True, or None.")


def start(
    run_dir: Path | str | None = None,
    *,
    run_id: str | None = None,
    policy: BlackBoxPolicy | str | Mapping[str, Any] | None = None,
    overwrite: bool = False,
    **start_kwargs: Any,
) -> BlackBoxRun:
    """Create a blackbox run and optionally record start metadata."""

    run = BlackBoxRun(run_dir or default_run_dir(), run_id=run_id, policy=policy, overwrite=overwrite)
    if start_kwargs:
        run.record_start(**start_kwargs)
    return run
