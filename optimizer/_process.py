"""Start internal workers from the package that owns the calling process."""
from pathlib import Path
import sys


def _command(module, *arguments):
    # -I excludes cwd, user-site and PYTHONPATH. The only added location is the
    # actual loaded package's parent (site-packages for a deployed wheel).
    bootstrap = (
        "import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); "
        "runpy.run_module(sys.argv.pop(1), run_name='__main__')"
    )
    return [sys.executable, "-I", "-B", "-c", bootstrap,
            str(Path(__file__).resolve().parent.parent), module,
            *map(str, arguments)]
