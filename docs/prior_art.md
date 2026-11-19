# Prior art and what is (and is not) new

The core optimisations here are **not novel**; several projects published or shipped the same ideas. We checked
the sources below directly (Boltz tag v0.4.1, Protenix main, and the Anthropic OpenFold3 kit), and our
implementation was written independently of them.

| Work | Overlap |
|---|---|
| **Boltz-1** (Nov 2024), preprint §3.4 and `model_cache` in `boltz/model/modules/diffusion.py` | The diffusion attention biases and the encoder's intermediate atom single/pair tensors are independent of timestep and noisy coordinates, so they are cached across steps. This is the central idea of `diffusion_cache`. |
| **ByteDance Protenix**, `enable_diffusion_shared_vars_cache` in `protenix/model/protenix.py` | The conditioned token-pair tensor and static atom-pair/atom-single tensors are prepared once and passed to the sampler. |
| **Anthropic's OpenFold3 / OpenBind-0 kits** (`anthropics/uplifting-biomolecular-modeling`, `openfold3/CHANGES.md`) | Same model, GPU-oriented, much broader: `paircache`, `atom_hoist`, `apb_hoist` hoist the same step-invariant work; the kit also covers template de-duplication, bf16 weight-cast caching, CUDA graphs and fused kernels. Its fused atom-window path computes keys/values once per atom, like our project-then-gather. |
| **Intel Open Omics** (AlphaFold2 on CPU, bf16), **DeepCache**, **DCFold** | Adjacent: CPU bf16, reuse of slowly changing activations, and fewer denoiser evaluations. |

What this package adds is packaging, not new algorithms: a CPU-first, import-time patch layer with
method-level source fingerprints, in-process reversal (`disable()` / `disabled()`), a runtime self-check,
status reporting, and a user-runnable benchmark with predictive-equivalence reporting. If you need maximum
GPU throughput, use the kit above; if you need a small, reversible, CPU-oriented layer on stock OpenFold3
0.5.x, this is for that.

## What we took from the kit
Independent reimplementations, with the kit as the source of the ideas (it is Apache-2.0; no code was copied):
bias pre-summing, bf16 weight-cast caching and fast initialisation. We did not adopt template de-duplication
(the trunk is a small part of CPU time), CUDA graphs or fused CUDA kernels (GPU only). Per-lever 2-vCPU numbers
are in the benchmarks page.

## Evidence status

The measurements in these docs are **preliminary**: synthetic inputs, a randomly initialised (non-trivial)
model, the diffusion rollout only (not end-to-end), on one 2-vCPU AMX Xeon. They show the mechanism works and
that the exact group reproduces upstream on that setup. They are not a validated speed-up for real
predictions with trained weights; run `openfold3-accel-bench` and compare real targets before relying on them.
CI measures speed and equivalence on shared runners, which are noisy.

"Exact" means algebraically equivalent and, in our tests, equal within `atol=rtol=1e-5` on valid atoms (often
bit-identical); GEMMs over different row counts can round differently.

## Known limitations

- Supported: `openfold3>=0.5.0,<0.6` (fingerprints verified for 0.5.0 and 0.5.1). 0.4.x differs in several
  patched functions and is not supported.
- Under `torch.inference_mode()` (what OpenFold3's `Trainer.predict` uses) tensors have no version counter, so the
  cache identifies static tensors by identity only; it cannot detect in-place mutation of such tensors. OpenFold3
  does not mutate its rollout conditioning tensors in place.
- Caching one dense pair bias per block grows quadratically with token count; the cache budget
  (`OPENFOLD3_ACCEL_CACHE_MB`, default 25% of RAM up to 8 GiB) bounds it, and over-budget entries are simply recomputed.
- The fast atom path needs un-batched, shared atom-mask geometry; otherwise upstream code runs.
- `bf16_diffusion` only activates on CPUs with AMX/AVX512-BF16 (otherwise it warns and runs fp32; override with
  `OPENFOLD3_ACCEL_BF16_FORCE=1`).
- The import-time self-check is only a smoke test of the cheap row-wise layers and the cache mechanics; full-path equivalence is
  covered by the test suite.
