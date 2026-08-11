"""Create a small, capability-selected project OLGS system scaffold."""

from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path
import re
from typing import Sequence


_CAPABILITIES = ("constraints", "second-order", "proximal")


def _identifier(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")
    if not normalized or normalized[0].isdigit():
        raise ValueError(f"Invalid Python class name {value!r}.")
    return normalized


def _guidance(profile: str, capabilities: Sequence[str]) -> str:
    blocks: list[str] = []
    if profile == "qubit":
        blocks.append(
            '''    # Optional qubit data product. Uncomment only after propagation and
    # basis/state-to-Bloch conversion are implemented.
    # def plot_data(self, controls: Controls, *, kind: str, **options: Any):
    #     if kind != "bloch":
    #         raise KeyError(f"Unsupported plot-data kind: {kind}")
    #     # Return PlotData(kind="bloch", layers=(PlotLayer(...),)).
    #     raise NotImplementedError("TODO: return generic named Bloch-coordinate layers.")'''
        )
    if "constraints" in capabilities:
        blocks.append(
            '''    # Optional equality-constraint capability. Uncomment both hooks only
    # after the residual definition and analytical Jacobian are complete.
    # def residuals(self, controls: Controls, *, name: str = "hard"): ...
    # def jacobian(self, controls: Controls, *, name: str = "hard"): ...'''
        )
    if "second-order" in capabilities:
        blocks.append(
            '''    # Optional matrix-free analytical second-order capability.
    # def hvp(self, controls: Controls, vector: Controls): ...'''
        )
    if "proximal" in capabilities:
        blocks.append(
            '''    # Optional composite-objective capability. Uncomment as one bundle.
    # def smooth_gradient(self, controls: Controls) -> Controls: ...
    # def prox(self, controls: Controls, step_size: float) -> Controls: ...'''
        )
    return "\n\n".join(blocks) or "    # No optional capability hooks selected."


def _template(name: str) -> str:
    return files("optimizer.system_olgs.templates").joinpath(name).read_text(encoding="utf-8")


def scaffold_system(
    destination: str | Path,
    *,
    class_name: str = "System",
    system_id: str | None = None,
    system_name: str | None = None,
    profile: str = "base",
    capabilities: Sequence[str] = (),
    force: bool = False,
    with_test: bool = True,
) -> tuple[Path, ...]:
    """Create a system module and optional initial test without silent overwrite."""

    if profile not in {"base", "qubit"}:
        raise ValueError("profile must be 'base' or 'qubit'.")
    unknown = sorted(set(capabilities) - set(_CAPABILITIES))
    if unknown:
        raise ValueError(f"Unknown scaffold capabilities: {', '.join(unknown)}.")
    class_name = _identifier(class_name)
    destination = Path(destination).expanduser()
    system_path = destination if destination.suffix == ".py" else destination / "system.py"
    test_path = system_path.with_name("test_system.py")
    targets = (system_path, test_path) if with_test else (system_path,)
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite existing scaffold files: " + ", ".join(str(path) for path in existing)
        )

    chosen_id = str(system_id or system_path.parent.name or class_name).replace("_", "-")
    chosen_name = str(system_name or chosen_id.replace("-", " ").title())
    visualizations = (
        '{"bloch": {"provider": "plot_data", "status": "planned", '
        '"description": "Generic named layers of Bloch coordinates", "automatic": False}}'
        if profile == "qubit"
        else "{}"
    )
    replacements = {
        "__CLASS_NAME__": class_name,
        "__SYSTEM_ID__": chosen_id,
        "__SYSTEM_NAME__": chosen_name,
        "__SYSTEM_TYPE__": "qubit" if profile == "qubit" else "generic",
        "__VISUALIZATIONS__": visualizations,
        "__CAPABILITY_GUIDANCE__": _guidance(profile, tuple(capabilities)),
    }

    def render(text: str) -> str:
        for key, value in replacements.items():
            text = text.replace(key, value)
        return text

    rendered = [render(_template("system.py.tmpl"))]
    if with_test:
        rendered.append(render(_template("test_system.py.tmpl")))
    system_path.parent.mkdir(parents=True, exist_ok=True)
    for path, content in zip(targets, rendered):
        path.write_text(content, encoding="utf-8")
    return tuple(path.resolve() for path in targets)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", help="Target system.py path or project directory.")
    parser.add_argument("--class-name", default="System")
    parser.add_argument("--system-id")
    parser.add_argument("--system-name")
    parser.add_argument("--profile", choices=("base", "qubit"), default="base")
    parser.add_argument("--capability", action="append", choices=_CAPABILITIES, default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        created = scaffold_system(
            args.destination,
            class_name=args.class_name,
            system_id=args.system_id,
            system_name=args.system_name,
            profile=args.profile,
            capabilities=args.capability,
            force=args.force,
            with_test=not args.no_test,
        )
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    for path in created:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
