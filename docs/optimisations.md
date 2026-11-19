# What is optimised (and what is not)

## `diffusion_cache` (default, exact)
OpenFold3 0.5.1 recomputes, on every one of the 200 diffusion steps × 5 samples, quantities that do not
depend on the step:

- pair conditioning over N×N (relative-position concat, LayerNorm, Linear, two transition blocks);
- `LayerNorm(z)` and the per-block pair-bias projections in every diffusion transformer block;
- reference-conformer atom features and atom-pair terms in the atom encoder;
- the `s`-dependent half of adaptive LayerNorm and transition gates;
- key/value projections of atom-local attention on 4×-duplicated key blocks.

They are computed once per rollout. The mask and pair biases are also pre-summed into one tensor per block (one fewer full-size add per head, block and step; rounding differs from upstream at about 1 ulp, hence "within 1e-5" rather than bitwise). In our tests the output matches upstream within 1e-5 on valid atoms (0.0 difference in the benchmark).

## `bf16_diffusion` (opt-in, approximate)
After caching, runtime on CPU is dominated by fp32 GEMMs in the token-level transformer. On CPUs with
AMX/AVX512-BF16 bf16 GEMMs are several times faster. Not enabled by default because results change
(~0.06–0.3 Å RMS on the synthetic benchmark, depending on size; validate on your own targets).

The group also holds PyTorch's autocast weight-cast cache open for the whole rollout. Autocast only caches
fp32→bf16 weight casts for leaf tensors that require grad, so under `no_grad` every weight was re-cast on every
call (about a fifth of the bf16 profile). The package raises `requires_grad` on the diffusion weights for the
duration of the rollout (no graph is recorded, grad is off) and restores it afterwards. Measured on the
synthetic benchmark (2 vCPU): 4.78 s → 4.20 s (1.14×) on top of plain bf16.

## `fast_init` (opt-in)
Constructing OpenFold3 draws every weight with `scipy.stats.truncnorm`, then a checkpoint overwrites them. This
group replaces `trunc_normal_init_` with PyTorch's native truncated-normal sampler (same distribution, same
std and truncation points; a different random stream, so seeded from-scratch initialisation no longer matches
upstream, hence opt-in). Full-size model construction: 43.0 s → 9.7 s on 2 vCPU. Enable it **before** building the model.

## `fused_attention` (opt-in, no measured gain)
Routes the attention primitive through `scaled_dot_product_attention` with the biases as an additive mask
(inspired by the fused attention paths of Anthropic's kit, which target GPU). On 2 vCPU in fp32 it was
*not faster* (9.8 s vs 9.0 s) and under bf16 it defers to upstream (upstream asks for an fp32 softmax there). It is
kept for other CPUs and PyTorch builds where the fused kernel may win; measure before using it.

## Negative results
(Observations from preliminary measurements on two cores; no profiling traces or ablations are published here.)

- Fusing Q/K/V (and similar) projections: ~0% on 2 cores; the GEMMs are compute-bound.
- Trimul / SwiGLU fusion in the trunk: the trunk is a small fraction of CPU wall time.
- A C++ extension: not justified by measurement; the remaining time is in MKL/oneDNN GEMMs.

## Literature
GPU-oriented work (cuEquivariance fused triangle kernels, CUDA graphs, MSA on GPU) does not transfer to CPU;
the transferable principles are static-work caching, fewer/cheaper diffusion steps and reduced precision.
Fewer diffusion steps is a user-level setting and is not changed by this package.
