#!/usr/bin/env python
"""Print the source fingerprint of every patch target for one or more OpenFold3 source trees.

Usage::

    python scripts/fingerprint_versions.py --releases 0.4.0 0.4.5 0.5.1     # download wheels from PyPI
    python scripts/fingerprint_versions.py --tree /path/to/openfold-3        # a checkout (dev)
    python scripts/fingerprint_versions.py --installed                       # the importable install

Sources are parsed with ``ast`` (nothing is imported or executed). Add *only* fingerprints of
versions whose upstream diff you have read, then update ``FINGERPRINTS`` in ``_compat.py``.
The helper ``--emit`` prints a ready-to-paste dictionary.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from openfold3_accel._compat import fingerprint, fingerprint_source  # noqa: E402
from openfold3_accel.patches import group_defs  # noqa: E402


def all_specs():
    seen = {}
    for _summary, make in group_defs().values():
        for p in make():
            seen[p.name] = p.spec
    return seen


def source_of(tree_root: Path, spec) -> str | None:
    module, cls, attr = spec
    path = tree_root / (module.replace(".", "/") + ".py")
    if not path.exists():
        return None
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    mod = ast.parse(text)
    scope = mod.body
    if cls:
        node = next((n for n in scope if isinstance(n, ast.ClassDef) and n.name == cls), None)
        if node is None:
            return None
        scope = node.body
    fn = next(
        (
            n
            for n in scope
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == attr
        ),
        None,
    )
    if fn is None:
        return None
    start = min([fn.lineno] + [d.lineno for d in fn.decorator_list])
    return "".join(lines[start - 1 : fn.end_lineno])


def tree_fingerprints(root: Path) -> dict[str, str | None]:
    out = {}
    for name, spec in all_specs().items():
        src = source_of(root, spec)
        out[name] = None if src is None else fingerprint_source(src)
    return out


def installed_fingerprints() -> dict[str, str | None]:
    out = {}
    for name, (module, cls, attr) in all_specs().items():
        owner = importlib.import_module(module)
        owner = getattr(owner, cls) if cls else owner
        out[name] = fingerprint(owner.__dict__[attr])
    return out


def fetch_release(version: str, dest: Path) -> Path | None:
    cp = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            f"openfold3=={version}",
            "--no-deps",
            "--only-binary=:all:",
            "-d",
            str(dest),
            "-q",
        ],
        capture_output=True,
        text=True,
    )
    wheels = sorted(dest.glob(f"openfold3-{version}-*.whl"))
    if cp.returncode or not wheels:
        print(
            f"  could not download openfold3=={version}: {cp.stderr.strip()[-200:]}",
            file=sys.stderr,
        )
        return None
    root = dest / f"src-{version}"
    with zipfile.ZipFile(wheels[0]) as z:
        z.extractall(root)
    return root


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--releases", nargs="*", default=[])
    ap.add_argument("--tree", nargs="*", default=[])
    ap.add_argument("--installed", action="store_true")
    ap.add_argument("--emit", action="store_true", help="print a FINGERPRINTS dict literal")
    a = ap.parse_args(argv)

    results: dict[str, dict[str, str | None]] = {}
    if a.installed:
        results["installed"] = installed_fingerprints()
    for t in a.tree:
        results[f"tree:{t}"] = tree_fingerprints(Path(t))
    if a.releases:
        with tempfile.TemporaryDirectory() as tmp:
            for v in a.releases:
                root = fetch_release(v, Path(tmp))
                if root:
                    results[v] = tree_fingerprints(root)

    names = list(all_specs())
    width = max(len(n) for n in names)
    print(f"{'target':{width}}  " + "  ".join(f"{k[:14]:14}" for k in results))
    for n in names:
        print(f"{n:{width}}  " + "  ".join(f"{(r.get(n) or '-'):14}" for r in results.values()))
    if a.emit:
        merged: dict[str, set[str]] = defaultdict(set)
        for r in results.values():
            for n, fp in r.items():
                if fp:
                    merged[n].add(fp)
        print("\nFINGERPRINTS = {")
        for n in names:
            print(f'    "{n}": frozenset({{{", ".join(repr(x) for x in sorted(merged[n]))}}}),')
        print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
