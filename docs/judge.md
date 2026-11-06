# Judge and evidence

Faster-FermiNet claims two things for Fe, Kr, Cd and Xe: a faster training step, and results that are
**bit-for-bit identical** to the original FermiNet. They are measured under different XLA settings, and
the difference matters.

## Default-mode timing vs deterministic-mode bit-wise verification

| | Timing (speed claim) | Bit-wise verification (identity claim) |
|---|---|---|
| XLA flags | **default** (what upstream FermiNet uses) | `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, for **both** trees |
| What is compared | wall time per training step | every step: energy bits, `pmove`, SHA-256 of every leaf of walkers, network parameters, KFAC optimizer state, MCMC width |
| Run length | 18 iterations (compile + 2 warm-up + 3 × 5 timed) | 8 steps, and 110 steps (crosses the MCMC width adaptation at step 100) |
| Result | 1.244–1.263× faster | identical on all four atoms, at every recorded step |

**Why two modes.** The unmodified original is *not* bit-reproducible across processes in default mode:
two runs with the same seed already disagree at step 0 (Fe: E₀ = −317.9158 vs −318.7558 Ha; the same
for Kr, Cd and Xe). Some XLA:GPU kernels use atomics, and MCMC accept/reject decisions amplify the last-bit
differences. Disabling autotuning (`--xla_gpu_autotune_level=0`) does not fix this;
`--xla_gpu_deterministic_ops=true` does: in that mode two original runs agree bit for bit (verified on all
four atoms). So bit-wise identity with the original can only be stated in deterministic mode, and that is
where it was checked.
(Records: [bitwise_records.json](validation/bitwise_records.json) → `determinism_study`; the default-mode
self-mismatch was recorded at an earlier batch size of 4096 and is the reason the protocol exists.)

**What this means.**
- In deterministic mode, Faster-FermiNet computes exactly the same numbers as the original, step for step.
- In default mode, neither the original nor Faster-FermiNet is reproducible run to run. The changes do not
  touch the nondeterministic XLA ops; they replace a cuBLAS kernel with one that reproduces its
  arithmetic, which is deterministic in both modes. The kernel probe compares directly against cuBLAS
  without the deterministic flag.
- Speed was measured only in default mode. No speed claim is made for deterministic mode.

## Timing gate

Baseline A → Faster-FermiNet → baseline B, as independent processes on the same GPU (one atom per GPU:
Fe, Kr, Cd, Xe on four H20s). Each process runs the real `train.train(cfg)`. Step 0 (compilation) and two
warm-up steps are excluded; the median of three 5-step windows is the step time.
`speedup = mean(A, B) / candidate`. The run passes if `speedup > max(1.02, 1 + 2·|A−B|/mean)` and every
run's window range is below 10 %.

| Atom | Baseline A windows (s) | Faster-FermiNet windows (s) | Baseline B windows (s) | A/B gap | max window range | Speedup |
|---|---|---|---|---:|---:|---:|
| Fe | 0.5384 / 0.5389 / 0.5390 | 0.4319 / 0.4319 / 0.4326 | 0.5378 / 0.5376 / 0.5377 | 0.22 % | 0.15 % | **1.2463×** |
| Kr | 1.1888 / 1.1903 / 1.1962 | 0.9549 / 0.9569 / 0.9575 | 1.1900 / 1.1899 / 1.1907 | 0.03 % | 0.62 % | **1.2438×** |
| Cd | 2.3869 / 2.3854 / 2.3888 | 1.8827 / 1.8900 / 1.8952 | 2.3853 / 2.3870 / 2.3942 | 0.00 % | 0.66 % | **1.2630×** |
| Xe | 3.7136 / 3.7099 / 3.7092 | 2.9504 / 2.9564 / 2.9641 | 3.7098 / 3.7102 / 3.7111 | 0.01 % | 0.46 % | **1.2549×** |

All thresholds are 1.020 (the baselines agree to within 0.22 %). The previous best configuration (round 7) passed
the same gate at 1.244 / 1.225 / 1.243 / 1.224×. Raw step times: [h20.json](validation/h20.json).

## Bit-wise gate

Reference and Faster-FermiNet, both in deterministic mode, same seed, same configuration. The final state
digests (first 16 hex digits of SHA-256) are identical for both trees:

| Atom | Steps | Final energy (Ha) | params | KFAC state | walkers |
|---|---:|---:|---|---|---|
| Fe | 8 | −326.9190979 | `93aa538019d02c14` | `cda894965534eafe` | `5f982e04fdea7bad` |
| Kr | 8 | −637.7427979 | `ccdc2638768d6c05` | `31eefd99d4896bb2` | `1902adcaab89ee65` |
| Cd | 8 | −1140.6478271 | `07799a0875d870ed` | `36f5adef333ac721` | `caf0b63ffcb8e475` |
| Xe | 8 | −1450.9968262 | `26434fa373a7f6e6` | `26851f56163a61bc` | `a641daa2c8d604aa` |
| Fe | 110 | −359.953125 | `df15958a27eedbbb` | `136b3e7ac8543569` | `39994976ff4c13e5` |
| Kr | 110 | −688.1181641 | `08257f624a0930d9` | `e7ce04112c068843` | `b7bca32b2e33dabf` |
| Cd | 110 | −1195.7983398 | `a572a23497f760b7` | `5fa8762b32ab54b8` | `fbc1de8ecff6a637` |
| Xe | 110 | −1481.3858643 | `87473c8f325e032a` | `b9a0ee9533be4023` | `59df1b2d8bdf0d01` |

Identity is checked at **every** step, not only the last (all 8 and all 110 recorded steps match).
Full digests and per-step energy bits: [bitwise_records.json](validation/bitwise_records.json).

**Kernel contract probe.** `bench/trsm_check.py` compares the Pallas solve with the real cuBLAS path in one
process (override disabled for the reference): n ∈ {9, 16, 26, 36, 48, 54, 64}, all four `lu_solve`
variants, inputs at three scales — 0 mismatching elements.

## How the gate was protected

- The bench drives the unmodified `train.train(cfg)` and only observes it (a writer hook for timestamps
  and energies, a `checkpoint.save` hook for state digests in check mode). Timing and bit-wise runs use the
  same code path; only the XLA flag differs, as described above.
- Every GPU process runs with `JAX_PLATFORMS=cuda`; a run that would fall back to the CPU fails instead.
  One diagnostic once silently fell back to the CPU after a CUDA initialisation error; that record
  was discarded and repeated on the GPU, and the runner now retries such failures.
- Failed gates were kept as failures. A row-major first version of the kernel was 1.14–1.20× faster but not
  bit-wise identical, so it was rejected. A 4× loop unroll passed on Kr/Cd/Xe but not on Fe, so it was
  rejected. See [history.md](history.md).
