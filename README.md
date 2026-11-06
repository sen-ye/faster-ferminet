# Faster-FermiNet

[English](README.md) | [简体中文](README.zh-CN.md)

**1.24–1.26× faster FermiNet training steps for Fe, Kr, Cd and Xe on NVIDIA H20, bit-for-bit identical to the original, optimized by AI agents.**

![Faster-FermiNet: time per training step, original vs optimized](docs/performance-views/benchmark.svg)

| Atom (electrons) | Original: time / step (s) ↓ | Faster-FermiNet: time / step (s) ↓ | Speedup ↑ | Bit-for-bit identical |
|---|---:|---:|---:|:---:|
| Fe (26) | 0.5383 | **0.4319** | **1.246×** | ✓ |
| Kr (36) | 1.1901 | **0.9569** | **1.244×** | ✓ |
| Cd (48) | 2.3870 | **1.8900** | **1.263×** | ✓ |
| Xe (54) | 3.7101 | **2.9564** | **1.255×** | ✓ |

Measured with the formal check on **NVIDIA H20 GPUs** (Hopper, 96 GB), one atom per GPU, **1024 walkers per
GPU**. The configuration is the upstream all-electron atom setup (KFAC, 10 MCMC steps per iteration,
FermiNet 4×(256, 32), 16 full determinants, default Laplacian) with `ano-rcc` pretraining; Fe uses spin
polarisation 4. Precision, hyper-parameters and random streams are unchanged.

Timing uses XLA's **default** flags: original A → Faster-FermiNet → original B, run as independent processes
on the same GPU. The exact ratios are 1.2463 / 1.2438 / 1.2630 / 1.2549×. Bit-for-bit identity is verified
in XLA's **deterministic** mode, because the original itself is not bit-reproducible in default mode. See
[timing vs bit-wise](#timing-vs-bit-wise-verification).

## Quick Start

```bash
pip install -r requirements.txt && pip install --no-deps -e .
ATOM=Xe bash scripts/run_atom.sh          # upstream entry point; optimizations are on by default
```

The optimizations are enabled when `ferminet` is imported; [scripts/configs/faster.env](scripts/configs/faster.env)
lists the measured switches, and `source scripts/configs/upstream.env` turns them off. To reproduce the table
on four GPUs:

```bash
bash scripts/get_upstream.sh              # original FermiNet @ e9d5a604 -> third_party/ferminet-upstream
GPUS=0,1,2,3 bash scripts/run_benchmark.sh                     # ABA timing + bit-wise check + verdict
GPUS=0,1,2,3 LONG_ITERS=110 PROBE=1 bash scripts/run_benchmark.sh   # + 110-step check + kernel probe
```

Tested with **JAX 0.7.2, cuBLAS 12.9.1.4, Python 3.11, driver 535.261.03 on H20**. On another GPU or
CUDA/JAX stack, run `bash scripts/check_kernel.sh` first ([Scope](#scope)). Details:
[reproduction](docs/reproduction.md).

## Timing vs bit-wise verification

| | Speed claim | Identity claim |
|---|---|---|
| XLA flags | default (as upstream runs) | `--xla_gpu_deterministic_ops=true` for **both** trees |
| Compared | wall time per training step | every step: energy bits, `pmove`, SHA-256 of all walkers, parameters, KFAC state, MCMC width |
| Result | 1.244–1.263× | identical on all four atoms over 8 and 110 steps |

In default mode, two runs of the **unmodified original** with the same seed already differ at step 0
(Fe: E₀ −317.9158 vs −318.7558 Ha; all four atoms; recorded during formulation at batch 4096). XLA uses atomics in some kernels and MCMC amplifies the last bits.
Disabling autotuning does not fix this; `--xla_gpu_deterministic_ops=true` does. So bit-wise identity
with the original is meaningful, and was checked, only in deterministic mode. Speed is claimed only for
default mode; deterministic-mode speed was not measured. The replaced kernel is deterministic in both modes
and matches cuBLAS bit for bit without the flag. See [judge and evidence](docs/judge.md).

## Agent Optimization History

Eight rounds and 16 attempts. Rejected attempts include two that were faster but not bit-wise identical.

![Optimization history: 8 rounds, 16 attempts](docs/performance-views/history.svg)

## What changed

- **Bit-exact triangular solve.** In the original, batched cuBLAS `trsm` inside the Laplacian took 28–36 % of
  each step at ≈ 1 TFLOP/s. Its exact operation order was reverse-engineered (aligned 8-row blocks, block
  partial sums from +0 then subtract, in-block FMA chain, IEEE division) and re-implemented as a
  Pallas/Triton kernel with inline PTX rounding. It is installed as the CUDA lowering of
  `triangular_solve_p` for float32, batched, square systems with 9 ≤ n ≤ 64; every other solve keeps cuBLAS.
- **Layout preserved.** The kernel reads and writes cuBLAS's column-major layout, so XLA compiles every
  neighbouring fusion exactly as before. A row-major version was faster but changed the bits.
- **Laplacian loop unrolled ×2.** Same accumulation order, two independent Hessian-vector products per trip.
  Unroll 4 broke bit-identity on Fe and was rejected.

![Where the computation changes](docs/change-views/module-map.svg)

Three files differ from upstream: `ferminet/pallas_trsm.py` (new), `ferminet/__init__.py`,
`ferminet/hamiltonian.py`. Switches: [reproduction § Switches](docs/reproduction.md#switches). Method:
[bit-exact trsm](docs/trsm_bitexact.md).

## Validation

**Bit-wise.** Original and Faster-FermiNet in deterministic mode, same seed and configuration. Energy bits,
`pmove` and SHA-256 digests of every leaf of walkers, parameters, KFAC optimizer state and MCMC width are
identical at **every step**: 8-step and **110-step** runs (crossing the MCMC width adaptation at step 100)
on Fe, Kr, Cd and Xe. Before any candidate, the original reproduced itself in this mode. A kernel probe
against the real cuBLAS path (n 9–64, four variants, three input scales) has 0 mismatches. Digests:
[bitwise_records.json](docs/validation/bitwise_records.json).

**Timing.** Original A → Faster-FermiNet → original B, independent processes on the same GPU. The first step
(compilation) and two warm-up steps are excluded; the median of three 5-step windows is used.

| Atom | Original A / B (s) | Faster-FermiNet (s) | A/B gap | Gate threshold | Speedup |
|---|---|---:|---:|---:|---:|
| Fe | 0.5389 / 0.5377 | 0.4319 | 0.22 % | 1.020 | **1.2463×** |
| Kr | 1.1903 / 1.1900 | 0.9569 | 0.03 % | 1.020 | **1.2438×** |
| Cd | 2.3869 / 2.3870 | 1.8900 | 0.00 % | 1.020 | **1.2630×** |
| Xe | 3.7099 / 3.7102 | 2.9564 | 0.01 % | 1.020 | **1.2549×** |

Window ranges are ≤ 0.66 %. Raw records: [h20.json](docs/validation/h20.json); protocol: [judge](docs/judge.md).

## Scope

- **Hardware/software.** The kernel reproduces the pinned cuBLAS build on H20 (sm_90). Other GPUs or cuBLAS
  versions may use another algorithm; run `scripts/check_kernel.sh` and set `FERMINET_PALLAS_TRSM=0` if it
  reports mismatches.
- **Workload.** Measured for the four atoms, single-device runs with 1024 walkers per GPU, real ground-state
  `laplacian='default'`. Other paths (excited states, PsiFormer, periodic systems, complex wavefunctions,
  multi-device `pmap`) were not benchmarked. The kernel guard is shape-generic and falls back to cuBLAS
  outside its validated range.
- **Bench shortcuts.** The bench shortens pretraining (20 iterations) and burn-in (10 steps) to save start-up
  time; the timed training step is unchanged. Start-up (Hartree–Fock, 30–90 s) and compilation (20–35 s)
  are excluded from step time.
- **What was not measured.** Full training to convergence and time-to-accuracy. With identical arithmetic
  they would follow the original step for step in deterministic mode.

## Why it stopped at 1.25×

The owner set a budget of 8 rounds. In the final profile, the network GEMMs already run near the H20's TF32
peak, and the memory-bound XLA fusions cannot be restructured without changing FMA contraction and
therefore the bits. The remaining lever is the solve kernel: about 1.5–1.7 ms at n = 54 against ≈ 0.25 ms
at memory speed, bound by load latency. See [profile and stopping point](docs/profile.md) and the
[full history](docs/history.md).

## Repository layout

| Path | Content |
|---|---|
| `ferminet/` | FermiNet with the optimization (upstream `e9d5a604` + 3 files) |
| `bench/` | judge and tools: `fnbench.py`, `compare.py`, `gate.py`, `launch.py`, `make_jobs.py`, kernel/profiling tools |
| `scripts/` | `run_benchmark.sh`, `check_bitwise.sh`, `check_kernel.sh`, `run_atom.sh`, `get_upstream.sh`, `configs/` |
| `docs/` | [judge](docs/judge.md), [reproduction](docs/reproduction.md), [history](docs/history.md), [bit-exact trsm](docs/trsm_bitexact.md), [profile](docs/profile.md), `validation/` |
| `task/formulation.json` | frozen workload, baseline, gate rules and evaluator hashes (ResearchBoost-Agent style) |
| `tests/` | kernel contract test (GPU), emulator tests (CPU) |

## Upstream FermiNet documentation

Based on [google-deepmind/ferminet](https://github.com/google-deepmind/ferminet) at `e9d5a604`.
[Apache-2.0 license](LICENSE). The original README is kept in [README.upstream.md](README.upstream.md).
The optimization process followed [ResearchBoost-Agent](https://github.com/DAGroup-PKU/ResearchBoost-Agent).
