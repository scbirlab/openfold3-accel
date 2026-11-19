# Notes for AI agents

- Layout: `src/openfold3_accel/{_patch,_compat,_cache,patches/,bench/}`; tests in `tests/`.
- Never loosen fingerprint gating to make a test pass; regenerate with `scripts/fingerprint_versions.py` after reading the upstream diff.
- Never push, tag or publish; maintainers do that.
- Caches live only inside `_cache.rollout_scope()` and are keyed on tensor identity+`_version`; do not persist them.
- Equivalence tests must use non-trivial weights (see `tests/conftest.py`).
- Benchmarks must stay synthetic, offline, and write only to `--out-dir`.
