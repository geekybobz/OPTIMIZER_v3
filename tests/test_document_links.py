"""Source-checkout documentation hygiene checks."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


@unittest.skipUnless((ROOT / ".git").exists(), "requires the source checkout")
class DocumentationLinkTests(unittest.TestCase):
    def test_retired_v4_path_is_absent_and_relative_links_resolve(self) -> None:
        failures: list[str] = []
        names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.md"], cwd=ROOT, text=True).splitlines()
        for name in names:
            path = ROOT / name
            text = path.read_text(encoding="utf-8")
            if "/OPTIMIZER/v4" in text:
                failures.append(f"retired path: {path.relative_to(ROOT)}")
            for target in LINK.findall(text):
                target = target.strip().strip("<>")
                target = target.split("#", 1)[0].strip()
                if not target or target.startswith(("/", "http:", "https:", "mailto:")):
                    continue
                destination = (path.parent / target).resolve()
                if not destination.exists():
                    failures.append(f"broken link: {path.relative_to(ROOT)} -> {target}")
        self.assertEqual(failures, [])
