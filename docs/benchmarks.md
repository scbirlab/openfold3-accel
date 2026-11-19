# Benchmarks

Run on your own machine (synthetic inputs, random weights, no network, nothing written outside `--out-dir`):

```bash
openfold3-accel-bench --n-token 128 --steps 10
```

The CI report (speedup and predictive equivalence across Python versions and runners) is published at
<a href="benchmarks/report/">benchmarks/report/</a> (raw HTML).

Preliminary numbers (synthetic, diffusion rollout only; see [Prior art](prior_art.md) for caveats) from the development machine (2 vCPU Xeon with AMX, torch CPU, N=96 tokens, 6 steps, 24 blocks):
exact 1.21× (0.0 Å difference), exact+bf16 1.82× (0.06 Å RMS). Small shared runners are noisy.


## Per-lever results (2 vCPU Xeon with AMX, synthetic random-weight model, N=128 tokens, 6 steps, 24 blocks)

Single machine, single run set, synthetic weights: treat as indicative, not as a claim for real targets.

| Mode | Median s | Speedup | RMS diff vs original |
|---|---|---|---|
| original | 10.81 | 1.00× | 0 |
| `diffusion_cache` (incl. pre-summed bias) | 9.04 | 1.20× | 0 (within 1e-5) |
| + `fused_attention` | 9.80 | 1.10× | within 1e-5 |
| + `bf16_diffusion` | 4.30 | 2.51× | 0.29 Å |
| + `fused_attention` + `bf16_diffusion` | 3.99 | 2.71× | 0.29 Å (fused defers to upstream under bf16; difference is noise) |
| model construction, original → `fast_init` | 43.0 → 9.7 | 4.45× | not applicable |
