# Maintaining

See `CONTRIBUTING.md` and `AGENTS.md` in the repository root. Supporting a new OpenFold3 release:
run `python scripts/fingerprint_versions.py --releases X.Y.Z`, read the diff of every changed target,
and only then add its fingerprint to `FINGERPRINTS` in `_compat.py` and the CI matrix.
