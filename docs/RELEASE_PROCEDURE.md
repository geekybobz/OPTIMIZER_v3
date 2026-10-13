---
role: topic
parent: ../README.md
summary: "Promote an approved commit into an isolated global wheel, verify provenance and roll back."
read_when: "Read before publishing, updating or restoring the installed optimizer."
status: active
reviewed: 2026-10-13
---
# Release procedure

Development and installation are separate copies. Editing this checkout changes
the development copy. The global commands execute a wheel installed in
`Optimizer/site-packages`. A wheel is a packaged snapshot of code, not a link to
the checkout. Only explicit release promotion changes the global copy.

The release sequence is **review → commit → build wheel → test installed wheel →
promote → publish**. `git commit`, `git pull` and `update.sh` never deploy a release.
Use a new version/tag for changed released bytes. Keep the previous wheel,
manifest and installation receipt for recovery.

## Details

### 1. Review and commit source

Work on a branch, review the diff, and run the source checks from the checkout:

```bash
conda run -n Optimizer python -B -m pytest -q tests
conda run -n Optimizer python tools/generate_api_reference.py --check
mdp inventory . --check
mdp verify . --require-properties
git diff --check
```

The API generator requires checkout import resolution; use `PYTHONPATH=.` if
your Python invocation does not put the checkout first. Source checks test
maintenance code intentionally. They do not demonstrate the installed release.
Keep native numerical threads at one during verification; independent heavy
runs stay sequential and coordinator admission remains available CPUs minus
three (minimum one). Ignored experiments and draft Markdown are not release
inputs. If they interfere with Markdown checks, run those checks in a temporary
`git archive` extraction of the reviewed commit before building; preserve the
live files and check the inventory shipped in that archive.
Commit the reviewed change and record `git rev-parse HEAD`. This is the full
approved SHA. Approval is a human/project decision; the tool validates that the
SHA exists, not that a human performed the review.

### 2. Build only committed bytes

```bash
conda run -n Optimizer python -I tools/release.py build --approved-ref FULL_APPROVED_SHA
```

The builder uses `git archive` for exactly that commit, extracts it outside the
checkout, and builds a wheel there. Uncommitted changes and ignored files cannot
enter the archive. It retains:

| File | Purpose |
|---|---|
| `optimizer_v4-VERSION-py3-none-any.whl` | Installable code snapshot |
| `source.tar` | Exact committed source and tests |
| `release.json` | Commit, version, wheel/source SHA-256 and package file hashes |
| `verification.json` | Generated after installed-package checks succeed |
| `constraints.txt` | Dependency versions observed during candidate verification |

Default artifact location is `~/.local/share/optimizer/releases/FULL_APPROVED_SHA/`.
Use `--output /new/path` for another location. Existing output is never replaced.
The external manifest binds the wheel to the commit without putting a commit's
own hash inside itself. Obtain artifacts from the trusted repository release;
a checksum detects changed bytes, not the identity of their publisher.

### 3. Test the installed candidate

```bash
conda run -n Optimizer python -I tools/release.py verify --release /absolute/release/directory
```

Verification checks archive/wheel hashes, creates a temporary virtual environment,
installs the wheel and extras, checks installed code hashes/import location and
`pip check`, and executes the committed tests against that installed package.
It reuses scientific dependencies from the selected environment. Three checkout
checks (generated API, documentation links, context script) remain source checks.
No optimizer source tree is available as a fallback during numerical tests.
The test summary and dependency constraints are retained beside the wheel.

### 4. Promote explicitly

First save work and shut down persistent coordinators with `opt.jobs.shutdown();`.
Exit Python processes and notebook kernels using the old release. A wheel update
does not upgrade code already loaded into a running process.

```bash
conda run -n Optimizer python -I tools/release.py install --release /absolute/release/directory
conda run -n Optimizer python -I tools/release.py status
```

Promotion repeats candidate verification before changing the target package,
installs the wheel non-editably, checks its bytes and dependencies, and writes
isolated launchers in the environment and `~/.local/bin`, and registers an
isolated **Optimizer** notebook kernel. The install receipt
records the commit, wheel checksum, package path and Python interpreter. Previous
receipts are retained. Constraints from candidate verification guide dependency
installation; this is an observed environment record, not a cross-platform lock.

Pip changes an environment in place. This is not an atomic environment swap:
if installation fails, use the retained previous artifact to recover. Candidate
verification reduces this risk but cannot prevent disk or package-manager failures.
For zero-downtime upgrades, prepare a separate environment and switch launchers
only after verification; the standard procedure requires stopped consumers.

The launchers use `python -I`, which excludes the current folder, `PYTHONPATH`
and user-site packages. Internal workers explicitly use the parent's actual
package location. Normal scripts follow Python import rules; use `python -I`
where appropriate and inspect `optimizer.__file__` in notebooks.

### 5. Publish and verify availability

Push the reviewed commit to `main`, create a version tag on that exact commit,
and attach the wheel, source archive, manifest, verification and constraints to
a GitHub release. Confirm remote `main`, tag and release assets after publication.
The default branch gives new clones current source; release assets give users a
fixed package. Public availability does not automatically update anyone's machine.

### Roll back the installed release

Run the promotion command with the previous release directory. It verifies the
retained wheel and reinstalls those bytes. Then restart clients and kernels.
Environment dependencies can also change across releases; use the old constraints
or a saved Conda environment when an exact environment rollback is required.
Experiment-level `undo` or checkpoint restoration is a separate operation and
never changes the installed library.

### Check isolation yourself

Record the installed version and path with `python -I`. Edit a source file in
this checkout and repeat the installed check from another project folder. The
installed file must still reside in `Optimizer/.../site-packages/optimizer` and
its hash must remain unchanged. Global commands also ignore a project-local
`optimizer.py`; a deliberately shadowing ordinary Python import may still use it.

---

[⌂ Home](../README.md) · [Next →](INSTALLATION.md)
