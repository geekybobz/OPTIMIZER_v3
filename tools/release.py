#!/usr/bin/env python3
"""Explicit, commit-bound wheel promotion; see docs/RELEASE_PROCEDURE.md."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
STATE = Path.home() / '.local/share/optimizer'
EXCLUDED = ('test_api_reference.py', 'test_document_links.py', 'test_optimizer_context.py')


def run(*args, **kwargs):
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def extract(archive, destination):
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()) or not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsafe archive member: {member.name}')
        source.extractall(destination)


def load_release(directory):
    directory = Path(directory).expanduser().resolve()
    value = json.loads((directory / 'release.json').read_text())
    if value['schema'] != 'optimizer.release/1':
        raise ValueError('Unknown release schema')
    for key in ('wheel', 'source'):
        name = value[key]['file']
        if Path(name).name != name or digest(directory / name) != value[key]['sha256']:
            raise ValueError(f'{key} checksum mismatch')
    return directory, value


def build(approved_ref, output):
    if not re.fullmatch(r'[0-9a-f]{40}', approved_ref):
        raise ValueError('--approved-ref requires the full reviewed 40-character commit SHA')
    actual = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', approved_ref + '^{commit}'], text=True).strip()
    if actual != approved_ref:
        raise ValueError('Approved commit does not match')
    directory = Path(output or STATE / 'releases' / actual).expanduser().resolve()
    if directory.exists():
        raise ValueError(f'Release output already exists: {directory}; choose a new --output')
    directory.mkdir(parents=True)
    archive = directory / 'source.tar'
    run('git', '-C', ROOT, 'archive', '--format=tar', '--output', archive, actual)
    with tempfile.TemporaryDirectory(prefix='optimizer-build-') as temporary:
        source = Path(temporary); extract(archive, source)
        run(sys.executable, '-I', '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation', '-w', directory, source)
        import tomllib
        version = tomllib.loads((source / 'pyproject.toml').read_text())['project']['version']
        files = {str(p.relative_to(source / 'optimizer')): digest(p)
                 for p in (source / 'optimizer').rglob('*') if p.is_file()}
    wheels = list(directory.glob('*.whl'))
    if len(wheels) != 1: raise ValueError('Expected exactly one wheel')
    manifest = {'schema': 'optimizer.release/1', 'commit': actual, 'version': version,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'wheel': {'file': wheels[0].name, 'sha256': digest(wheels[0])},
        'source': {'file': archive.name, 'sha256': digest(archive)},
        'package_files': files}
    write_json(directory / 'release.json', manifest)
    print(f'Release built: {directory}', flush=True)
    return directory


def inspect_python(python, manifest):
    code = '''import hashlib, importlib.metadata as m, json, pathlib, optimizer, sys
p=pathlib.Path(optimizer.__file__).parent
files={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file() and f.suffix in ('.py','.tmpl')}
import scipy, matplotlib, rich, mcp
print(json.dumps({'version':optimizer.__version__, 'package':str(p), 'prefix':sys.prefix, 'files':files, 'dependencies':sorted({d.metadata['Name']+'=='+d.version for d in m.distributions() if d.metadata['Name']})}))
'''
    report = json.loads(subprocess.check_output([str(python), '-I', '-B', '-c', code], text=True))
    expected = {k: v for k, v in manifest['package_files'].items() if Path(k).suffix in ('.py', '.tmpl')}
    if report['version'] != manifest['version'] or report['files'] != expected:
        raise ValueError('Installed package differs from the approved release')
    if not Path(report['package']).is_relative_to(Path(report['prefix'])):
        raise ValueError('Package is not installed inside the selected environment')
    run(python, '-I', '-m', 'pip', 'check')
    return report


def verify(directory):
    directory, manifest = load_release(directory)
    with tempfile.TemporaryDirectory(prefix='optimizer-wheel-test-') as temporary:
        work = Path(temporary)
        # Reuse scientific dependencies, but install the wheel in its own site-packages.
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(work / 'env')
        python = work / 'env/bin/python'
        run(python, '-I', '-m', 'pip', 'install', '--ignore-installed', '--no-deps', directory / manifest['wheel']['file'])
        run(python, '-I', '-m', 'pip', 'install', str(directory / manifest['wheel']['file']) + '[mcp,scientific,dev]')
        report = inspect_python(python, manifest)
        source = work / 'source'; source.mkdir(); extract(directory / manifest['source']['file'], source)
        shutil.rmtree(source / 'optimizer')
        # Tests that inspect package fixtures see the installed package, never source.
        (source / 'optimizer').symlink_to(report['package'], target_is_directory=True)
        command = [python, '-I', '-B', '-m', 'pytest', '-q', 'tests',
                   *['--ignore=tests/' + name for name in EXCLUDED]]
        result = run(*command, cwd=source, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout, end='')
        write_json(directory / 'verification.json', {'schema': 'optimizer.release-verification/1',
            'commit': manifest['commit'], 'wheel_sha256': manifest['wheel']['sha256'],
            'tested_utc': datetime.now(timezone.utc).isoformat(),
            'installed_package': report['package'], 'test_output': result.stdout,
            'excluded_checkout_tests': list(EXCLUDED)})
        (directory / 'constraints.txt').write_text('\n'.join(
            x for x in report['dependencies'] if not x.lower().startswith('optimizer-v4==')) + '\n')
    return manifest


def launcher(python, module):
    return '#!/bin/sh\n# Managed OPTIMIZER release launcher; ignores cwd and PYTHONPATH.\nexec ' + shlex.quote(str(python)) + ' -I -B -m ' + module + ' "$@"\n'


def install(directory, global_bin):
    # Every promotion validates the artifact in a candidate environment first.
    manifest = verify(directory)
    directory = Path(directory).expanduser().resolve()
    old = STATE / 'install.json'
    if old.exists():
        shutil.copy2(old, STATE / ('install-previous-' + datetime.now().strftime('%Y%m%dT%H%M%S') + '.json'))
    run(sys.executable, '-I', '-m', 'pip', 'install', '--force-reinstall', '--no-deps', directory / manifest['wheel']['file'])
    run(sys.executable, '-I', '-m', 'pip', 'install', '-c', directory / 'constraints.txt',
        str(directory / manifest['wheel']['file']) + '[mcp,scientific,dev]')
    report = inspect_python(sys.executable, manifest)
    run(sys.executable, '-I', '-m', 'ipykernel', 'install', '--user',
        '--name', 'optimizer', '--display-name', 'Optimizer')
    kernel_dir = Path(subprocess.check_output([sys.executable, '-I', '-c',
        "from jupyter_client.kernelspec import KernelSpecManager; print(KernelSpecManager().get_kernel_spec('optimizer').resource_dir)"], text=True).strip())
    kernel = json.loads((kernel_dir / 'kernel.json').read_text())
    kernel['argv'].insert(1, '-I')
    kernel['argv'].append('--IPKernelApp.ignore_cwd=True')
    write_json(kernel_dir / 'kernel.json', kernel)
    entries = {'optimizer': 'optimizer.cli', 'optimizer-mcp': 'optimizer.mcp',
               'optimizer-system': 'optimizer.system_olgs.scaffold'}
    for parent in {Path(sys.executable).parent, Path(global_bin).expanduser().resolve()}:
        parent.mkdir(parents=True, exist_ok=True)
        for name, module in entries.items():
            target = parent / name
            # Never follow an old symlink when publishing the new launcher.
            temporary = parent / (name + '.optimizer-new')
            temporary.write_text(launcher(sys.executable, module)); temporary.chmod(0o755)
            temporary.replace(target)
    write_json(old, {'schema': 'optimizer.install/1', 'commit': manifest['commit'],
        'version': manifest['version'], 'wheel': str(directory / manifest['wheel']['file']),
        'wheel_sha256': manifest['wheel']['sha256'], 'python': sys.executable,
        'package': report['package'], 'global_bin': str(Path(global_bin).expanduser().resolve()),
        'installed_utc': datetime.now(timezone.utc).isoformat()})
    print(f'Installed OPTIMIZER {manifest["version"]} from {manifest["commit"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('build'); p.add_argument('--approved-ref', required=True); p.add_argument('--output')
    for name in ('verify', 'install'):
        p = commands.add_parser(name); p.add_argument('--release', required=True)
        if name == 'install': p.add_argument('--global-bin', default=str(Path.home() / '.local/bin'))
    commands.add_parser('status')
    args = parser.parse_args()
    try:
        if args.command == 'build': build(args.approved_ref, args.output)
        elif args.command == 'verify': verify(args.release)
        elif args.command == 'install': install(args.release, args.global_bin)
        else:
            receipt = json.loads((STATE / 'install.json').read_text())
            _, manifest = load_release(Path(receipt['wheel']).parent)
            report = inspect_python(sys.executable, manifest)
            receipt['verified_package'] = report['package']
            print(json.dumps(receipt, indent=2, sort_keys=True))
    except (ValueError, OSError, subprocess.CalledProcessError, KeyError) as exc:
        if isinstance(exc, subprocess.CalledProcessError) and exc.stdout: print(exc.stdout, file=sys.stderr)
        parser.exit(1, f'Release operation failed: {exc}\n')


if __name__ == '__main__': main()
