"""Regression checks for artifact integrity and installed import isolation."""
from pathlib import Path
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('release_tool', ROOT / 'tools/release.py')
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def test_full_commit_required_before_any_artifact_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'release'
            with self.assertRaisesRegex(ValueError, 'full reviewed'):
                release.build('main', output)
            self.assertFalse(output.exists())

    def test_changed_wheel_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ('test.whl', 'source.tar'): (directory / name).write_bytes(b'approved')
            value = {'schema': 'optimizer.release/1', 'wheel': {'file': 'test.whl', 'sha256': release.digest(directory / 'test.whl')},
                'source': {'file': 'source.tar', 'sha256': release.digest(directory / 'source.tar')}}
            release.write_json(directory / 'release.json', value)
            release.load_release(directory)
            (directory / 'test.whl').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                release.load_release(directory)

    def test_archive_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); archive = directory / 'source.tar'
            with tarfile.open(archive, 'w') as tar:
                entry = tarfile.TarInfo('../escape'); entry.size = 1
                tar.addfile(entry, io.BytesIO(b'x'))
            target = directory / 'unpacked'; target.mkdir()
            with self.assertRaisesRegex(ValueError, 'Unsafe archive'):
                release.extract(archive, target)
            self.assertFalse((directory / 'escape').exists())

    def test_internal_child_ignores_shadow_package_and_pythonpath(self):
        from optimizer._process import _command
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); (directory / 'optimizer').mkdir()
            (directory / 'optimizer/__init__.py').write_text("raise RuntimeError('SHADOW PACKAGE')")
            result = subprocess.run(_command('optimizer.cli', '--help'), cwd=directory,
                env=dict(os.environ, PYTHONPATH=temporary), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)

    def test_global_launcher_ignores_shadow_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'optimizer.py').write_text("raise RuntimeError('SHADOW PACKAGE')")
            command = directory / 'optimizer-command'
            command.write_text(release.launcher(sys.executable, 'optimizer.cli')); command.chmod(0o755)
            result = subprocess.run([str(command), '--help'], cwd=directory,
                env=dict(os.environ, PYTHONPATH=temporary), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)

    def test_documented_first_run_and_reopen(self):
        from optimizer._process import _command
        source = (ROOT / 'docs/GETTING_STARTED.md').read_text()
        commands = re.findall(r'```python\n(.*?)```', source, re.S)
        self.assertEqual(len(commands), 2)
        with tempfile.TemporaryDirectory() as temporary:
            shutil.copy2(ROOT / 'tests/fixtures/quadratic_system.py', Path(temporary) / 'system.py')
            for command in commands:
                result = subprocess.run(_command('optimizer.cli', '--json', '--command', command),
                    cwd=temporary, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                records = [json.loads(line) for line in result.stdout.splitlines()]
                self.assertTrue(records)
                self.assertTrue(all(row['status'] == 'ok' for row in records), result.stdout)
            history = Path(temporary) / '.optimizer/first-run/history.jsonl'
            self.assertTrue(history.exists())
