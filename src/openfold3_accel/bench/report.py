"""Summaries and a self-contained HTML report (no external assets) for benchmark results."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


def _rms(a, b):
    n = min(len(a), len(b))
    if not n:
        return float("nan")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a[:n], b[:n])) / n)


def _aligned_rmsd(a, b):
    """RMSD after optimal rigid superposition (Kabsch) of two flat xyz lists; None without numpy."""
    try:
        import numpy as np
    except ImportError:
        return None
    n = min(len(a), len(b)) // 3 * 3
    if n < 9:
        return None
    A = np.asarray(a[:n]).reshape(-1, 3)
    B = np.asarray(b[:n]).reshape(-1, 3)
    A = A - A.mean(0)
    B = B - B.mean(0)
    U, _s, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return float(np.sqrt(((A @ R - B) ** 2).sum(1).mean()))


def summarise(runs, label=""):
    ok = [r for r in runs if "error" not in r]
    ref = {r["device"]: r for r in ok if r["mode"] in ("original", "init original")}
    rows = []
    for r in runs:
        row = {k: r.get(k) for k in ("mode", "device", "seconds_median", "error")}
        base = ref.get(r["device"])
        if "error" not in r and base:
            row["speedup"] = base["seconds_median"] / r["seconds_median"]
            row["rms_coord_diff_A"] = _rms(r["coords"], base["coords"])
            row["aligned_rmsd_A"] = _aligned_rmsd(r["coords"], base["coords"])
            row["max_coord_diff_A"] = max(
                (abs(x - y) for x, y in zip(r["coords"], base["coords"])), default=float("nan")
            )
        rows.append(row)
    meta = next((r for r in ok), {})
    return {
        "label": label,
        "rows": rows,
        "meta": {
            k: meta.get(k)
            for k in (
                "python",
                "torch",
                "threads",
                "platform",
                "machine",
                "cpu_count",
                "n_token",
                "steps",
                "blocks",
                "cuda",
            )
        },
    }


def text_summary(res):
    lines = [
        f"{'device':6} {'mode':11} {'median s':>9} {'speedup':>8} {'rms diff (A)':>13} {'aligned':>10}"
    ]
    for r in res["rows"]:
        if r.get("error"):
            lines.append(f"{r['device']:6} {r['mode']:11} ERROR {r['error'][:80]!r}")
            continue
        lines.append(
            f"{r['device']:6} {r['mode']:11} {r['seconds_median']:9.2f} "
            f"{r.get('speedup', float('nan')):7.2f}x {r.get('rms_coord_diff_A', float('nan')):13.2e}"
        )
    return "\n".join(lines)


_CSS = "body{font:15px system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border-bottom:1px solid #ccc;padding:.35rem .6rem;text-align:right}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}.bar{background:#2a6fdb;height:.8rem}.bad{color:#b00020}code{background:#eee;padding:0 .25rem}"


def render(results):
    out = [
        "<!doctype html><meta charset=utf-8><title>openfold3-accel benchmark</title>",
        f"<style>{_CSS}</style>",
        "<h1>openfold3-accel benchmark report</h1>",
        "<p>Original vs accelerated OpenFold3 diffusion rollout on synthetic inputs with random (non-trivial) weights. "
        "<b>exact</b> is expected to match upstream to float rounding; <b>exact+bf16</b> is opt-in and approximate.</p>",
    ]
    for res in results:
        m = res["meta"]
        out.append(f"<h2>{html.escape(res['label'] or 'run')}</h2>")
        out.append(
            "<p><code>"
            + html.escape(json.dumps({k: v for k, v in m.items() if v is not None}))
            + "</code></p>"
        )
        out.append(
            "<table><tr><th>device</th><th>mode</th><th>median (s)</th><th>speedup</th><th>RMS coord diff (Å)</th><th>aligned RMSD (Å)</th><th></th></tr>"
        )
        top = max([r.get("speedup") or 1 for r in res["rows"]] + [1])
        for r in res["rows"]:
            if r.get("error"):
                out.append(
                    f"<tr><td>{r['device']}</td><td>{r['mode']}</td><td colspan=5 class=bad>{html.escape(r['error'][:200])}</td></tr>"
                )
                continue
            sp = r.get("speedup", 1.0)
            out.append(
                f"<tr><td>{r['device']}</td><td>{r['mode']}</td><td>{r['seconds_median']:.2f}</td><td>{sp:.2f}×</td>"
                f"<td>{r.get('rms_coord_diff_A', float('nan')):.2e}</td>"
                f"<td>{(r.get('aligned_rmsd_A') if r.get('aligned_rmsd_A') is not None else float('nan')):.2e}</td><td style='width:30%'><div class=bar style='width:{100 * sp / top:.0f}%'></div></td></tr>"
            )
        out.append("</table>")
    return "\n".join(out)


def main(argv=None) -> int:
    """Merge several ``results.json`` files into one HTML report (used by CI)."""
    ap = argparse.ArgumentParser(prog="openfold3-accel-report")
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("report.html"))
    a = ap.parse_args(argv)
    res = [json.loads(p.read_text()) for p in sorted(a.results)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(render(res))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
