# openfold3-accel

Drop-in, **reversible** inference acceleration for [OpenFold3](https://github.com/aqlaboratory/openfold-3),
focused on multicore CPU.

```python
import openfold3_accel  # that's it
```

- **Exact by default.** The default group removes work that is repeated on every diffusion step; outputs
  match upstream (0.0 difference on valid atoms in our tests).
- **Opt-in groups:** `bf16_diffusion` (AMX/AVX512-BF16 CPUs, with weight-cast caching), `fast_init` (faster model construction), `fused_attention` (SDPA; no measured gain on 2 vCPU).
- **Safe.** Patches apply only to fingerprint-verified OpenFold3 sources, only with grads off and in `eval()`;
  otherwise upstream code runs. `disable()` / `disabled()` revert in-process. Checkpoints unaffected.
- **Preliminary evidence:** on one 2-vCPU AMX Xeon, a synthetic rollout (random weights, diffusion only) ran 1.2× (exact)
  and 1.8× (exact+bf16) faster. Not an end-to-end or trained-weights result; run `openfold3-accel-bench` yourself.
- **Prior art:** the core caching ideas exist in Boltz-1, Protenix and Anthropic's OpenFold3 kit; see the
  docs' *Prior art* page. This package's contribution is a small, reversible, CPU-first packaging of them.

Docs: https://scbirlab.github.io/openfold3-accel/ · Licence: Apache-2.0

Install: `pip install openfold3-accel openfold3`. Supports `openfold3>=0.5.0,<0.6`, Python 3.12–3.14.
Note: on CPU-only hosts, set `use_triton_triangle_kernels: false` in the runner config (upstream default
tries Triton and fails without a GPU driver).
