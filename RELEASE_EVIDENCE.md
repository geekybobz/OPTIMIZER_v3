# V4 release evidence

This is the current release-evidence record for the single live V4 checkout.
Earlier release records are superseded and are not release inputs.

| Item | Evidence |
|---|---|
| Intended tag | `v4.0.4` |
| Release implementation revision | `a661e5990f2f5c20cc2547c5152a9a908deb64b7` |
| Package | `optimizer-v4` 4.0.4 |
| Runtime | non-editable wheel in Conda environment `optimizer_v4`; the same wheel is kept in `release_artifacts/v4.0.4/` |
| Source tests | 159 passed: `/Users/billabobz/miniconda3/bin/conda run --no-capture-output -n optimizer_v4 python -B -m unittest discover -s tests` |
| Installed-wheel tests | 159 passed with `optimizer` imported from site-packages and only the checkout's `tests/` folder on `PYTHONPATH` |
| Changes since v4.0.3 | Bloch and sphere rendering fixes and optional reference-axis stubs; completed V4 docs with the runtime catalog declared as the public API; refreshed context command |
| Release hygiene | build and cache residue cleared; preflight must pass before tagging |

The annotated Git tag is the immutable binding between this tracked evidence and the
release revision. The manifest intentionally does not try to embed the SHA of the
commit that contains it, because that would be a circular Git object identity.

---

[⌂ Home](README.md)
