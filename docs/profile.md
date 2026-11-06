# Profile and why it stopped at 1.25×

## Original FermiNet (batch 1024, H20)

JAX profiler traces of two steady steps per atom:

| | Fe | Xe |
|---|---:|---:|
| GPU busy / wall | ≈ 99 % | ≈ 99 % |
| KFAC step (`pmap__step`) | 95.4 % | 97.3 % |
| MCMC (`pmap_mcmc_step`) | 4.4 % | 2.6 % |
| batched cuBLAS `trsm` (4 per Laplacian direction) | 28 % | 36 % |
| fusions inside command buffers | ≈ 37 % | ≈ 33 % |

Nearly the whole KFAC step is the Laplacian `fori_loop`: 78 (Fe) to 162 (Xe) sequential Hessian-vector
products.

## After the trsm replacement (round 3 profile, Xe, two steps, command buffers disabled for attribution)

| Group | GPU time | Share |
|---|---:|---:|
| Pallas triangular solves | 1729 ms | 27 % |
| two-electron stream: elementwise/concat fusions + Triton GEMMs on `[1024, 54, 54, 32]` | ≈ 1.6 s | ≈ 25 % |
| cuBLAS GEMMs (TF32 network layers; HIGHEST batched 54×54 matvec) | ≈ 1.05 s | ≈ 16.5 % |
| mean over electrons of two-electron tangents (`kLoop` reduce fusion) | 341 ms | 5.4 % |
| envelope reduction | 240 ms | 3.8 % |
| permutation gather + loop-invariant sort in `lu_solve` | ≈ 96 ms | 1.5 % |
| rest | | ≈ 21 % |

Nsight Compute on the solve kernel at production shape (n = 54, 16384 matrices): 72 % of stall samples
are `long_scoreboard`, L1 hit rate 22 %, L2 hit rate 56 %, achieved occupancy 36 %. The kernel is bound by
load latency, which the final version (V5) partly addresses by issuing a block's loads together.

## Why it stopped here

- **Budget.** The owner fixed 8 RBA rounds; 16 attempts were made within them.
- **GEMMs are already near peak.** The per-direction network GEMMs run at ≈ 68 TFLOP/s TF32 on a
  ≈ 74 TFLOP/s part. Changing their precision or library is excluded by the bit-wise contract.
- **XLA fusions are frozen by the bit-wise contract.** The two-electron-stream kernels are memory-bound
  (≈ 2 TB/s of ≈ 4 TB/s), but moving a fusion boundary changes FMA contraction inside the generated kernel
  and therefore the bits. Loop unrolling changed the bits for one atom at factor 4 (see
  [history.md](history.md)).
- **The solve kernel is the remaining lever.** It moves ≈ 0.76 GB per call (≈ 0.25 ms at memory speed) but
  takes 1.5–1.7 ms at n = 54. Shared-memory staging of solved rows would remove the L2 round trips; Pallas'
  Triton backend does not expose explicit shared memory, so this would need another kernel backend.

### Directions tried and not adopted

| Direction | Outcome |
|---|---|
| row-major kernel I/O | fast, but end-to-end bits changed (layout effect) |
| static unroll of kernel inner loops alone; index-tensor loads; 16/32-column chunks; tile orientation alone | bit-wise ✓, no gain or slower |
| Laplacian unroll 3, 4, 6 | 4 not bit-wise on Fe; 3 and 6 not better than 2 overall |
| XLA while-loop double buffering, latency-hiding scheduler | bit-wise ✓, within ±0.5 % |

### Not attempted (bit-wise risk or small share)

Batching several Laplacian directions with `vmap` changes GEMM shapes and therefore cuBLAS/Triton
algorithm choices. Hoisting the loop-invariant sort and folding the permutation gather into the kernel
would recover at most ≈ 1.5 %, and would require replacing JAX's `lu_solve` rather than only a lowering.
