# Usage

```python
import openfold3_accel as acc

print(acc.status())  # what is active, and why anything was skipped
acc.disable()  # restore upstream callables (weights are never modified; see the thread-safety note for `bf16_diffusion`)
with acc.disabled():
    ...  # A/B inside one process
acc.enable(["diffusion_cache", "bf16_diffusion"])  # opt-in approximate group
```

Environment variables: `OPENFOLD3_ACCEL_DISABLE=1`, `OPENFOLD3_ACCEL_GROUPS=a,b`, `OPENFOLD3_ACCEL_SKIP=a`,
`OPENFOLD3_ACCEL_CACHE_MB=<n>`.

**Checkpoints.** Weights and `state_dict` layouts are never changed and derived caches are never
persisted, so existing checkpoints load unchanged. The patches act only when gradients are off and the
module is in `eval()`; training and activation checkpointing run upstream code.

**Export / tracing.** While tracing or exporting (ONNX, `torch.jit.trace`, `torch.export`, `torch.compile`) the
caches step aside and upstream OpenFold3 code runs. `yunta-af2-accel` is unaffected (JAX `jit` is not intercepted).


## Thread safety and reproducibility
- Enabling runs a small numerical smoke test; it saves and restores the torch, NumPy and Python global RNG
  states, so `import openfold3_accel` does not change a seeded process's random stream (tested).
- `diffusion_cache` is rollout-scoped (a `ContextVar`) and safe for concurrent inference threads.
- `bf16_diffusion` temporarily sets `requires_grad=True` on the diffusion weights for the duration of a rollout
  (it lets autocast cache the bf16 weight casts) and restores it afterwards. bf16 rollouts are therefore
  serialised by a process-wide lock, and other threads must not read or change those flags meanwhile. If that is
  unacceptable, set `OPENFOLD3_ACCEL_NO_CASTCACHE=1` to skip the toggle (slower, ~14% on our benchmark).
- The documented thread guarantee is concurrent bf16 rollouts through the patched entry point only, not general thread-safe access to a shared model. For a shared-model inference server use `OPENFOLD3_ACCEL_NO_CASTCACHE=1` unless access is externally synchronised.
- If other code has wrapped a patched function, `disable()` changes nothing and warns (atomic), so a group is never left half-installed.
- Enabling and disabling groups is lock-protected. Groups that wrap the same function are removed
  last-in-first-out, and `disable(["diffusion_cache"])` while `bf16_diffusion` is active rebuilds the stack, so a
  disabled group is truly off (regression-tested over every enable/disable order).
