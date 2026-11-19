# Contributing

```bash
pip install -e ".[dev]"   # plus torch and openfold3
pytest
ruff check src tests scripts
```

Rules: exact groups must stay equivalent to upstream (tests randomise zero-init weights so they are not
vacuous); approximate behaviour must be opt-in; every patch target is guarded by a source fingerprint; every
change must be reversible with `disable()`. New OpenFold3 versions: see `docs/maintaining.md`.
