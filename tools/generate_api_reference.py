"""Generate the Markdown API reference from OPTIMIZER's runtime catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "API_REFERENCE.md"

# Document this checkout, never an installed release of the package.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _bullets(values: Iterable[Any]) -> list[str]:
    return [f"- {_text(value)}" for value in values]


def _mapping_table(values: Mapping[str, Any]) -> list[str]:
    rows = ["| Name | Meaning |", "|---|---|"]
    rows.extend(f"| `{_text(key)}` | {_text(value)} |" for key, value in values.items())
    return rows


def render() -> str:
    import optimizer as opt

    listing = opt.list()
    groups = listing["groups"]
    items = listing["items"]
    lines = [
        "---",
        "role: generated",
        "parent: README.md",
        "summary: Generated exact signatures and catalog semantics for every public OPTIMIZER tool.",
        "read_when: Looking up a callable signature, default, requirement, return, or runtime identifier.",
        "status: active",
        "reviewed: 2026-10-12",
        "---",
        "",
        "<!-- Generated file. Do not edit directly. Source: optimizer/catalog.py and runtime signatures. Regenerate with: python tools/generate_api_reference.py -->",
        "",
        "# API Reference",
        "",
        "This file is generated from `optimizer.catalog` and live Python signatures.",
        "Contract pages explain meaning and invariants; this reference owns exact callable",
        "surfaces and defaults.",
        "",
        "## Groups",
        "",
        "| Group | Entries | Purpose |",
        "|---|---:|---|",
    ]
    for group, metadata in groups.items():
        lines.append(
            f"| [`{_text(group)}`](#{_text(group).replace('_', '-')}) | {metadata['count']} | {_text(metadata['summary'])} |"
        )
    for group, entries in items.items():
        lines.extend(["", f"## {group.replace('_', ' ')}", "", _text(groups[group]["summary"])])
        for item in entries:
            detail = opt.info(item["id"])
            lines.extend(["", f"### `{detail['id']}`", "", _text(detail["summary"])])
            if signature := detail.get("signature"):
                lines.extend(["", "```python", f"{detail['name']}{signature}", "```"])
            for heading, key in (
                ("Inputs", "inputs"),
                ("Returns", "returns"),
            ):
                if values := detail.get(key):
                    lines.extend(["", f"#### {heading}", "", *_mapping_table(values)])
            for heading, key in (
                ("Requires", "requires"),
                ("Variants", "variants"),
                ("Best for", "best_for"),
                ("Not for", "not_for"),
                ("Watch out", "watch_out"),
                ("Related", "related"),
            ):
                if values := detail.get(key):
                    lines.extend(["", f"#### {heading}", "", *_bullets(values)])
            if example := detail.get("example"):
                lines.extend(["", "#### Example", "", "```python", str(example), "```"])
            if module := detail.get("module"):
                lines.extend(["", f"Source: `{_text(module)}`."])
    lines.extend(["", "---", "", "[⌂ Home](README.md)", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when API_REFERENCE.md is stale")
    args = parser.parse_args()
    content = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != content:
            print(f"stale generated reference: {OUTPUT}")
            return 1
        print(f"current generated reference: {OUTPUT}")
        return 0
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
