# Bit-exact batched triangular solve

## Where the solves come from

FermiNet's kinetic energy uses `jax.linearize` of ∇ log|ψ| and a `fori_loop` over the 3N coordinates
(one Hessian-vector product each). log|ψ| contains `slogdet` of 16 N×N orbital matrices per walker. JAX
differentiates `slogdet` through `solve` → `custom_linear_solve` → `lu_solve`. So every Hessian-vector
product performs four batched triangular solves on 1024 × 16 matrices of size N×N, with N×N right-hand
sides: unit-lower and upper for `trans=0`, upper-transposed and unit-lower-transposed for `trans=1`.
XLA:GPU lowers each one to cuBLAS `trsmBatched` (`batch_trsm_left_kernel`). It copies the right-hand side
into the output buffer and runs at ≈ 1 TFLOP/s for 26 ≤ N ≤ 54, which made it 28–36 % of the step.

## Recovering cuBLAS's operation order

1. `bench/trsm_dump.py` records cuBLAS inputs and outputs for n ∈ {2…64}, m ∈ {1, 2, n} and all four
   variants, with LU factors of random matrices as A.
2. `bench/trsm_emul.py` replays candidate operation orders on the CPU with exact float32 FMA. The FMA is
   emulated as a float64 product plus a TwoSum correction, and was checked against rational arithmetic.
3. Findings:
   - n ≤ 5, and single right-hand sides (m = 1, small n): plain sequential substitution
     `s = fma(-a_ij, x_j, s)` in solve order, then IEEE division by the diagonal. This matched exactly.
   - square right-hand sides (the FermiNet case, n = m from 9 to 64): rows form **aligned blocks of 8**
     (0–7, 8–15, …). For every previously solved block P, in solve order, a row receives
     `acc = Σ_{j∈P, ascending} fma(a_ij, x_j, acc)` with `acc = +0`, then `s = s − acc`. Inside its own
     block it uses the sequential FMA chain in solve order, and non-unit diagonals use IEEE division
     (`bs8` in the emulator). This gave **0 mismatches** for n ∈ {9, 16, 17, 26, 32, 33, 36, 48, 54, 64} and
     all four variants.
   - cuBLAS picks the algorithm by shape (e.g. n = 26 uses the sequential path for m = 1 and the blocked
     path for m = 2). The replacement is therefore only installed for the validated case:
     float32, left side, batched, square right-hand side, 9 ≤ n ≤ 64.

## Keeping XLA's layout

The first kernel (row-major input/output) matched cuBLAS exactly, yet end-to-end energies changed from the
first local energy onward. An in-process off/on comparison (`bench/diag_le.py`) showed log ψ and ∇ log ψ
bit-identical but the kinetic energy different. A structural HLO diff (`bench/hlo_sig.py`) found the
cause. The cuBLAS custom call forces column-major operands and results (`{2,3,1,0}`); a row-major Triton
call moved those layouts, so XLA fused the neighbours differently (one `kLoop` fusion became a `kInput`
reduction fusion), and a reduction order changed.

The kernel therefore receives Aᵀ and Bᵀ and writes Xᵀ, i.e. the column-major storage that cuBLAS used,
behind logical transposes that XLA turns into bitcasts. Producers and consumers of the solve compile
exactly as before. With this the kinetic and local energies, and then the full training state, became
identical.

## Kernel design (final, `ferminet/pallas_trsm.py`)

- One Triton program (1 warp) per matrix; installed as the CUDA lowering rule of
  `lax.linalg.triangular_solve_p`, so the jaxpr is unchanged. Unsupported cases call the original lowering.
- Working tile `[column k, row t of the current 8-row block]`. B and Xᵀ tiles map directly onto the
  column-major storage, x_j arrives as a per-column vector, and coefficients as a (1, 8) vector.
- Previous-block updates are statically unrolled: all 8 coefficient and x loads of a block are independent
  and can be in flight together. Terms past n are skipped with a select, never as an FMA with a zero
  coefficient (that could flip the sign of a zero).
- In-block solve is register-resident: row t is extracted with a sum over t filled with −0.0, which is exact
  and keeps the sign of zero, because Pallas lowers `reduce_sum` to a `tt.reduce` without an initial value.
  Later rows then receive `fma(−a, x_t, ·)` in solve order.
- Solved rows are staged in a row-major scratch output for later blocks; one barrier per block.
- Every rounding step is inline PTX: `fma.rn.f32`, `sub.rn.f32`, `div.rn.f32`. The compiler can neither
  contract nor reorder them.

Kernel time vs the cuBLAS path at production batch (1024 × 16 matrices, both including XLA's surrounding
copies in a standalone call):

| n | cuBLAS (ms) | Pallas (ms), L / U / Uᵀ / Lᵀ |
|---:|---:|---|
| 26 | 0.54–0.62 | 0.31 / 0.36 / 0.33 / 0.32 |
| 36 | 1.18–1.30 | 0.85 / 0.89 / 0.87 / 0.86 |
| 48 | 1.64–1.77 | 0.99 / 1.01 / 1.06 / 0.89 |
| 54 | 2.35–2.56 | 1.50 / 1.66 / 1.69 / 1.61 |

## Contract and how to re-check it

The arithmetic contract is "identical to this cuBLAS build's `trsmBatched` on this GPU". Another cuBLAS
version or GPU may choose a different algorithm. Run `bash scripts/check_kernel.sh` (or
`pytest tests/test_pallas_trsm.py`): it compares the kernel with the real cuBLAS path bit for bit. If it
reports mismatches, set `FERMINET_PALLAS_TRSM=0`.
