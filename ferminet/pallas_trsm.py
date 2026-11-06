# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Bit-exact Pallas replacement for XLA:GPU batched triangular solves.

XLA lowers batched `lax.linalg.triangular_solve` to cuBLAS `trsmBatched`
(`batch_trsm_left_kernel`). For small square systems this kernel is slow
(~1 TFLOP/s on H20) and XLA additionally transposes the operands to
column-major and copies the right-hand side into the output buffer.

This module reproduces the arithmetic of that cuBLAS kernel *bit for bit* for
the validated case (float32, left side, batched, 9 <= n == m <= 64), measured
offline against cuBLAS for n in {9, 16, 17, 26, 32, 33, 36, 48, 54, 64} and all
four (lower, transpose_a) variants:

  * rows of op(A) are grouped in aligned blocks of 8 (rows 0-7, 8-15, ...);
  * a row i first receives, for every previously solved block P (in solve
    order), acc = sum_{j in P, ascending j} fma(op(A)[i, j], x_j, acc) with
    acc starting at +0, followed by s = s - acc;
  * then, inside its own block, the sequential chain s = fma(-op(A)[i, j], x_j, s)
    in solve order (ascending for forward, descending for backward
    substitution);
  * non-unit diagonals use IEEE division s / op(A)[i, i].

Every rounding step is pinned with inline PTX (fma.rn / sub.rn / div.rn), so
the compiler cannot contract or reorder anything.

The replacement is installed as a CUDA lowering rule for
`triangular_solve_p`; the jaxpr is unchanged and every other operation lowers
exactly as before. Unsupported shapes/dtypes fall back to the original
lowering. Switch: FERMINET_PALLAS_TRSM=0 disables it.
"""

import functools
import math
import os

import jax
from jax import lax
import jax.numpy as jnp
from jax._src.interpreters import mlir
from jax._src.lax import linalg as lax_linalg
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

_BLOCK = 8
_MIN_N = 9
_MAX_N = 64
_NUM_WARPS = int(os.environ.get('FERMINET_PALLAS_TRSM_WARPS', '1'))
_COL_CHUNK = int(os.environ.get('FERMINET_PALLAS_TRSM_COLS', '64'))

STATS = {'pallas': 0, 'fallback': 0}
ENABLED = True  # checked at lowering time (diagnostics may toggle it)


def _asm(op, *xs):
  regs = ', '.join(f'${i + 1}' for i in range(len(xs)))
  return plgpu.elementwise_inline_asm(
      f'{op} $0, {regs};',
      args=list(xs),
      constraints='=r,' + ','.join(['r'] * len(xs)),
      pack=1,
      result_shape_dtypes=[jax.ShapeDtypeStruct(xs[0].shape, jnp.float32)],
  )[0]


@functools.lru_cache(maxsize=None)
def _make_solve(n, lower, trans, unit, num_warps=1, col_chunk=32):
  """Returns f(aT, bT) -> xT solving op(A) X = B for square n x n systems.

  Storage convention matches the cuBLAS custom call XLA would otherwise emit
  (column-major matrices): the kernel receives A^T and B^T as row-major
  (batch, n, n) arrays and writes X^T. Rows are processed in the cuBLAS blocks
  of 8: the block's B rows are read as a transposed (n x 8) tile, updated with
  the GEMM-style partial sums of all previously solved blocks (one x_j row load
  serves all 8 rows), solved sequentially inside the block, staged row-major
  in a scratch output (for later blocks) and written back as an X^T tile.
  """
  npad = min(pl.next_power_of_2(n), col_chunk)  # columns handled per program
  nchunks = (n + npad - 1) // npad
  forward = lower != trans  # op(A) lower triangular -> forward substitution
  nblk = (n + _BLOCK - 1) // _BLOCK
  b8 = _BLOCK

  def kernel(a_ref, b_ref, o_ref, s_ref):
    # a_ref[0, c, r] = A[r, c]; b_ref[0, k, i] = B[i, k]; o_ref: X^T;
    # s_ref: row-major scratch of solved X rows.
    # Working tile orientation [k, t] (columns x the 8 rows of the block):
    # B / X^T tiles map directly onto the column-major storage, x_j is a
    # per-column vector and the in-block row extraction reduces over t.
    c0 = pl.program_id(1) * npad
    kk = c0 + lax.broadcasted_iota(jnp.int32, (npad, b8), 0)
    tt = lax.broadcasted_iota(jnp.int32, (npad, b8), 1)
    kcol = jnp.minimum(c0 + lax.broadcasted_iota(jnp.int32, (npad, 1), 0), n - 1)
    trow = lax.broadcasted_iota(jnp.int32, (1, b8), 1)

    def coef_row(r0, j):  # op(A)[r0 + t, j] for t < 8, as (1, 8)
      rr = jnp.minimum(r0 + trow, n - 1)
      jj = jnp.broadcast_to(j, (1, b8))
      if trans:  # A[j, r] = aT[r, j]
        return plgpu.load(a_ref.at[0, rr, jj])
      return plgpu.load(a_ref.at[0, jj, rr])  # A[r, j] = aT[j, r]

    def coef(i):  # op(A)[i, i] as (1, 1)
      ii = jnp.broadcast_to(i, (1, 1))
      return plgpu.load(a_ref.at[0, ii, ii])

    def xcol(j):  # X[j, k] over this program's columns, as (npad, 1)
      return plgpu.load(s_ref.at[0, jnp.broadcast_to(j, (npad, 1)), kcol])

    def do_block(bb, carry):
      b = bb if forward else nblk - 1 - bb
      r0 = b * b8
      m_kt = (kk < n) & ((r0 + tt) < n)
      tile = plgpu.load(b_ref.at[0, pl.ds(c0, npad), pl.ds(r0, b8)],
                        mask=m_kt, other=0.0)  # tile[k, t] = B[r0 + t, k]

      def prev_block(pp, tile):
        p = pp if forward else nblk - 1 - pp
        j0 = p * b8
        j1 = jnp.minimum(j0 + b8, n)

        # Statically unrolled: all 8 coefficient / x loads are independent and
        # can be in flight together; terms past n skipped exactly by select.
        cs, xs = [], []
        for q in range(b8):
          j = jnp.minimum(j0 + q, n - 1)
          cs.append(coef_row(r0, j))  # (1, 8)
          xs.append(xcol(j))  # (npad, 1)
        acc = jnp.zeros((npad, b8), jnp.float32)
        for q in range(b8):
          c = jnp.broadcast_to(cs[q], (npad, b8))
          x = jnp.broadcast_to(xs[q], (npad, b8))
          acc = jnp.where(j0 + q < j1, _asm('fma.rn.f32', c, x, acc), acc)
        return _asm('sub.rn.f32', tile, acc)

      nprev = b if forward else nblk - 1 - b
      tile = lax.fori_loop(0, nprev, prev_block, tile)

      # In-block solve (right-looking, register resident). Extraction of row t
      # by a -0.0-filled sum over t (exact, keeps the sign of zero).
      nrow = jnp.minimum(b8, n - r0)

      def own_step(q, tile):
        t = q if forward else nrow - 1 - q
        sel = tt == t
        d = jnp.sum(jnp.where(sel, tile, -0.0), axis=1, keepdims=True)
        db = jnp.broadcast_to(d, (npad, b8))
        if not unit:
          db = _asm('div.rn.f32', db,
                    jnp.broadcast_to(coef(r0 + t), (npad, b8)))
        tile = jnp.where(sel, db, tile)
        later = (tt > t) if forward else (tt < t)
        c = jnp.broadcast_to(coef_row(r0, r0 + t), (npad, b8))
        upd = _asm('fma.rn.f32', -c, db, tile)
        return jnp.where(later, upd, tile)

      tile = lax.fori_loop(0, nrow, own_step, tile)
      plgpu.store(o_ref.at[0, pl.ds(c0, npad), pl.ds(r0, b8)], tile,
                  mask=m_kt)
      rows = jnp.minimum(r0 + tt, n - 1)
      plgpu.store(s_ref.at[0, rows, jnp.minimum(kk, n - 1)], tile, mask=m_kt)
      plgpu.debug_barrier()  # solved rows visible to later blocks
      return carry

    lax.fori_loop(0, nblk, do_block, 0)

  def solve(at, bt):
    batch = at.shape[0]
    return pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((batch, n, n), jnp.float32),
                   jax.ShapeDtypeStruct((batch, n, n), jnp.float32)),
        grid=(batch, nchunks),
        in_specs=[pl.BlockSpec((1, n, n), lambda g, c: (g, 0, 0)),
                  pl.BlockSpec((1, n, n), lambda g, c: (g, 0, 0))],
        out_specs=(pl.BlockSpec((1, n, n), lambda g, c: (g, 0, 0)),
                   pl.BlockSpec((1, n, n), lambda g, c: (g, 0, 0))),
        compiler_params=plgpu.CompilerParams(num_warps=num_warps,
                                             num_stages=1),
        name=f'ferminet_trsm_bs8_n{n}_{"L" if lower else "U"}'
             f'{"t" if trans else ""}',
    )(at, bt)[0]

  return solve


def supported(a_aval, b_aval, left_side, conjugate_a):
  del conjugate_a  # irrelevant for real dtypes
  if not left_side:
    return False
  if a_aval.dtype != jnp.float32 or b_aval.dtype != jnp.float32:
    return False
  if a_aval.ndim < 3 or b_aval.ndim != a_aval.ndim:
    return False
  if a_aval.shape[:-2] != b_aval.shape[:-2]:
    return False
  n = a_aval.shape[-1]
  if a_aval.shape[-2] != n or b_aval.shape[-2] != n or b_aval.shape[-1] != n:
    return False
  if not _MIN_N <= n <= _MAX_N:
    return False
  return math.prod(a_aval.shape[:-2]) >= 2  # cuBLAS batched path only


def _pallas_triangular_solve(a, b, *, left_side, lower, transpose_a,
                             conjugate_a, unit_diagonal):
  del left_side, conjugate_a
  n = a.shape[-1]
  batch = math.prod(a.shape[:-2])
  solve = _make_solve(n, bool(lower), bool(transpose_a), bool(unit_diagonal),
                      num_warps=_NUM_WARPS, col_chunk=_COL_CHUNK)
  # Logical transposes: XLA turns them into bitcasts by keeping A, B and X in
  # the column-major layout the cuBLAS custom call used, so producers and
  # consumers of the solve are compiled exactly as before.
  at = jnp.swapaxes(a, -1, -2).reshape(batch, n, n)
  bt = jnp.swapaxes(b, -1, -2).reshape(batch, n, n)
  xt = solve(at, bt)
  return jnp.swapaxes(xt.reshape(b.shape), -1, -2)


_INSTALLED = False


def install():
  """Installs the lowering override for CUDA (idempotent)."""
  global _INSTALLED
  if _INSTALLED:
    return
  prim = lax_linalg.triangular_solve_p
  orig = (mlir._platform_specific_lowerings['cuda'].get(prim)  # pylint: disable=protected-access
          or mlir._lowerings[prim])  # pylint: disable=protected-access
  orig_rule = orig.rule
  pallas_rule = mlir.lower_fun(_pallas_triangular_solve, multiple_results=False)

  def rule(ctx, a, b, **params):
    a_aval, b_aval = ctx.avals_in
    if ENABLED and supported(a_aval, b_aval, params['left_side'],
                             params['conjugate_a']):
      STATS['pallas'] += 1
      return pallas_rule(ctx, a, b, **params)
    STATS['fallback'] += 1
    return orig_rule(ctx, a, b, **params)

  mlir.register_lowering(prim, rule, platform='cuda')
  _INSTALLED = True


def maybe_install():
  if os.environ.get('FERMINET_PALLAS_TRSM', '1') != '0':
    install()
