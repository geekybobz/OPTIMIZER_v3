# V4 release evidence

This is the current release-evidence record for the single live V4 checkout.
Earlier candidate reports are superseded and are not release inputs.

| Item | Evidence |
|---|---|
| Intended tag | `v4.0.2` |
| Source revision | `a876e7d35c0403d85df9d17e7f8d1cbe50b87460` |
| Package | `optimizer-v4` 4.0.2 |
| Runtime | non-editable wheel in Conda environment `optimizer_v4` |
| Source tests | 142 passed: `MPLCONFIGDIR=/tmp/optimizer-mpl conda run --no-capture-output -n optimizer_v4 python -m unittest discover -s tests` |
| Installed-wheel tests | 142 passed through `install.sh --run-tests` |
| Numerical regressions | L-BFGS and Nesterov maximize paths; native/descent conversion; ordinary and combined non-`J` objectives |

The annotated Git tag is the immutable binding between this tracked evidence and the
release revision. The manifest intentionally does not try to embed the SHA of the
commit that contains it, because that would be a circular Git object identity.
