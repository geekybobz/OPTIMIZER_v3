# V4 release evidence

This is the current release-evidence record for the single live V4 checkout.
Earlier candidate reports are superseded and are not release inputs.

| Item | Evidence |
|---|---|
| Intended tag | `v4.0.3` |
| Release implementation revision | `9b15ecc08a75e85e3fee5dafd850585dfcb0f0cd` |
| Package | `optimizer-v4` 4.0.3 |
| Runtime | non-editable wheel in Conda environment `optimizer_v4` |
| Source tests | 150 passed: `MPLCONFIGDIR=/tmp/optimizer-mpl conda run --no-capture-output -n optimizer_v4 python -B -m unittest discover -s tests` |
| Installed-wheel tests | 150 passed through `install.sh --run-tests` |
| Numerical regressions | maximize/minimize HVP symmetry for Newton-CG; objective-aware non-`J` diagnostics/finite differences; explicit stopping resolution; public evaluator caching; defensive result/cache/control boundaries |
| Release hygiene | preflight import and all checkout subprocess tests use `-B`; clean repeated preflight is required |

The annotated Git tag is the immutable binding between this tracked evidence and the
release revision. The manifest intentionally does not try to embed the SHA of the
commit that contains it, because that would be a circular Git object identity.
