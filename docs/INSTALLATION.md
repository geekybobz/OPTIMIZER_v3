---
role: topic
parent: ../README.md
summary: "Install a committed release in Optimizer and use it from any project folder."
read_when: "Read when installing, checking global commands, or configuring notebooks."
status: active
reviewed: 2026-10-13
---
# Installation

The standard Conda environment is named **`Optimizer`**. It contains Python 3.11,
scientific libraries, notebook support, the CLI and the optional MCP dependency.
Install a normal wheel, never an editable package, for your everyday runtime.
The public distribution name remains `optimizer-v4`; the Python import remains
`optimizer`. The repository name on GitHub remains `OPTIMIZER_v3` for existing links.

“Global” here means available from any folder for your user account. It does not
install on other computers or for every operating-system account. Anyone with
access to the public repository can follow the same procedure.

## Details

### Install a selected release

Conda and Git must already be available. Select a reviewed release tag, resolve
its complete commit, and pass that exact value to the installer:

```bash
git clone https://github.com/geekybobz/OPTIMIZER_v3.git OPTIMIZER
cd OPTIMIZER
git checkout v4.0.5
git rev-parse HEAD
./install.sh --approved-ref FULL_40_CHARACTER_COMMIT_FROM_THE_PREVIOUS_COMMAND
```

Replace the uppercase placeholder with the printed SHA. The installer creates
`Optimizer` if needed, builds the selected committed source, tests the installed
wheel in a temporary environment, and promotes it. It installs numerical,
visualization, development-test and MCP extras. Internet access is needed for
missing dependencies. Set `CONDA_EXE` if `conda` is not discoverable.

An existing release directory is never overwritten. To rebuild a selected
commit, supply `--release-dir /new/empty/path`. To reinstall a saved artifact,
follow the [release procedure](RELEASE_PROCEDURE.md).

### Use from any folder

```bash
conda activate Optimizer
optimizer --help
optimizer-system --help
optimizer-mcp --help
python -I -c 'import optimizer; print(optimizer.__version__, optimizer.__file__)'
```

The installer also creates user launchers under `~/.local/bin`. Add that directory
to your shell's `PATH` to run these three commands without activating Conda:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Add that line once to your shell configuration for persistence. The launchers
use the selected environment's absolute Python path and isolated import mode.
Both global and environment launchers ignore the current folder and `PYTHONPATH`
when importing the optimizer. Background workers use their parent's loaded
package. Explicit project factories are still loaded from your chosen paths.

### Notebooks and ordinary Python

```bash
conda run -n Optimizer python -m ipykernel install --user --name optimizer --display-name Optimizer
```

The installer registers this kernel automatically and enables isolated imports.
If registering it manually with the command above, add `-I` after Python in its
`kernel.json` argv and `--IPKernelApp.ignore_cwd=True` after the existing arguments.
Choose the **Optimizer** kernel. A new kernel process is required after a release
change. Notebook/Python imports follow normal Python import rules: avoid naming
project files `optimizer.py` or adding a development checkout to `sys.path` when
you intend to use the installed package. `python -I` excludes such source paths;
Jupyter kernels can use isolated mode as described in the release procedure.

Verify the live installed bytes against the receipt with:

```bash
conda run -n Optimizer python -I tools/release.py status
```

The receipt lives at `~/.local/share/optimizer/install.json`; release wheels,
source archives, checksums and test evidence live under its `releases/` directory.
The [release procedure](RELEASE_PROCEDURE.md) explains updates and rollback.
`uninstall.sh --remove-launchers` removes only managed user launchers; it preserves
experiments, environments and saved artifacts.

---

[⌂ Home](../README.md) · [Next →](GETTING_STARTED.md)
