# Changelog

## 0.1.0
- Review round 2: `disable()` is atomic (dry run first; if third-party code wrapped any installed patch, nothing is reverted and groups stay `active`, with a warning); bf16 wrappers also require the weights to be on CPU, so accelerator runs never enter CPU autocast or touch `requires_grad`; tests for third-party wrapping, re-apply failure and the off-CPU bypass.
- Review fixes: selective `disable()` of an inner group now rebuilds the wrapper stack (and never reports a still-installed patch as inactive); the self-check preserves torch/NumPy/Python RNG state; bf16 rollouts hold a lock and have an `OPENFOLD3_ACCEL_NO_CASTCACHE=1` opt-out; tests added for every enable/disable order, RNG preservation, threads, odd sizes, batch>1, masked atoms and a tiny cache budget.
- Kit-inspired additions (independent implementations; see prior art): pre-summed attention biases (default group), bf16 autocast weight-cast cache, opt-in `fast_init` and `fused_attention` groups; bench gains `exact+fused`, `exact+fused+bf16`, `init:*` modes.
- Group revert now unwinds in reverse order so stacked wrappers restore cleanly.
- Fix: cache no longer crashes under `torch.inference_mode()` (Lightning `Trainer.predict`); tensors without version counters are matched by identity.
- Real import-time self-check (replaces a stub); bf16 group now checks for AMX/AVX512-BF16 and falls back to fp32 with a warning.
- Supported range narrowed to `openfold3>=0.5.0,<0.6` (0.4.x fingerprints differ); CI covers 0.5.0 and 0.5.1.
- Tests assert every cache path is exercised and cover inference mode; bench reports Kabsch-aligned RMSD.
- Docs: prior-art page and evidence caveats.
- Tracing/export (ONNX, TorchScript, `torch.export`, `torch.compile`) bypass: upstream code runs.
- Initial release: diffusion_cache (exact), bf16_diffusion (opt-in).
