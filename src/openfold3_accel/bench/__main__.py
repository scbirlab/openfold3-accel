"""``openfold3-accel-bench``: compare original vs accelerated OpenFold3 on this machine.

Safe by design: synthetic inputs, random weights, no network, nothing written outside ``--out-dir``
(default: a temp directory). Each mode runs in its own subprocess so patches never leak between runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from . import report
from .core import MODES


def _run_worker(mode, device, args, out):
    cmd = [
        sys.executable,
        "-m",
        "openfold3_accel.bench.core",
        "--mode",
        mode,
        "--device",
        device,
        "--n-token",
        str(args.n_token),
        "--steps",
        str(args.steps),
        "--blocks",
        str(args.blocks),
        "--repeats",
        str(args.repeats),
        "--out",
        str(out),
    ]
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    if cp.returncode or not out.exists():
        return {"mode": mode, "device": device, "error": (cp.stderr or cp.stdout)[-400:]}
    return json.loads(out.read_text())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="openfold3-accel-bench", description=__doc__)
    ap.add_argument("--n-token", type=int, default=128, help="synthetic tokens (default 128)")
    ap.add_argument(
        "--steps", type=int, default=10, help="diffusion steps (default 10; real runs use 200)"
    )
    ap.add_argument("--blocks", type=int, default=24, help="diffusion transformer blocks")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument(
        "--modes",
        nargs="*",
        default=[m for m in MODES if not m.startswith("init:")],
        choices=MODES,
        help="default skips the slow init:* startup modes (full-size model construction, about a minute each)",
    )
    ap.add_argument(
        "--no-cuda", action="store_true", help="skip the CUDA comparison even if available"
    )
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--label", default="", help="free-text label for the report (e.g. runner name)")
    a = ap.parse_args(argv)

    out_dir = a.out_dir or Path(tempfile.mkdtemp(prefix="of3accel-bench-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for mode in a.modes:
        print(f"[cpu] {mode} ...", flush=True)
        runs.append(_run_worker(mode, "cpu", a, out_dir / f"cpu-{mode.replace(':', '-')}.json"))
    try:
        import torch

        cuda = torch.cuda.is_available() and not a.no_cuda
    except Exception:  # noqa: BLE001
        cuda = False
    if cuda:
        for mode in ("original", "exact"):
            if mode in a.modes:
                print(f"[cuda] {mode} ...", flush=True)
                runs.append(_run_worker(mode, "cuda", a, out_dir / f"cuda-{mode}.json"))
    result = report.summarise(runs, label=a.label)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    (out_dir / "report.html").write_text(report.render([result]))
    print(report.text_summary(result))
    print(f"\nresults: {out_dir / 'results.json'}\nreport:  {out_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
