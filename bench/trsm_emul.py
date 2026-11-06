"""Offline (CPU) emulation of float32 triangular-solve orderings vs dumped cuBLAS output.

usage: python bench/trsm_emul.py <dump.npz from trsm_dump.py> [sizes] [hypotheses] [nbatch]
Exact float32 FMA is emulated with a float64 product plus a TwoSum correction (validated
against rational arithmetic). The hypothesis `bs8` reproduces cuBLAS batched trsm exactly
for square right-hand sides (see docs/trsm_bitexact.md).
"""
import itertools
import sys

import numpy as np

f32, f64 = np.float32, np.float64


def fma(a, b, c):
  a, b, c = (np.asarray(x, f32).astype(f64) for x in (a, b, c))
  p = a * b  # exact
  s = p + c
  bb = s - p
  e = (p - (s - bb)) + (c - bb)  # TwoSum error, exact
  return _round_exact(s, e)


def _round_exact(s, e):
  r = s.astype(f32)
  rd = r.astype(f64)
  # exact value v = s + e. If e == 0, r is correct. Otherwise r may be wrong only
  # if s is exactly representable-midpoint or s itself is a float32 boundary
  # case; handle generically by checking which float32 neighbour is closest to v.
  cand = np.stack([np.nextafter(r, -np.inf, dtype=f32), r, np.nextafter(r, np.inf, dtype=f32)])
  cd = cand.astype(f64)
  # distance |cd - (s+e)| computed as |(cd - s) - e| (cd - s exact for nearby)
  dist = np.abs((cd - s) - e)
  best = np.argmin(dist, axis=0)
  out = np.take_along_axis(cand, best[None], 0)[0]
  # ties: choose even mantissa
  d_sorted = np.sort(dist, axis=0)
  tie = d_sorted[0] == d_sorted[1]
  if np.any(tie):
    ev = (cand.view(np.uint32) & 1) == 0
    pick = np.where(ev[0] & (dist[0] == d_sorted[0]), 0,
                    np.where(ev[1] & (dist[1] == d_sorted[0]), 1, 2))
    out = np.where(tie, np.take_along_axis(cand, pick[None], 0)[0], out)
  return out


def mul(a, b):
  return (np.asarray(a, f32).astype(f64) * np.asarray(b, f32).astype(f64)).astype(f32)


def add(a, b):
  return (np.asarray(a, f32).astype(f64) + np.asarray(b, f32).astype(f64)).astype(f32)


def sub(a, b):
  return (np.asarray(a, f32).astype(f64) - np.asarray(b, f32).astype(f64)).astype(f32)


def div(a, b):
  # float32 division correctly rounded: compute in f64 (53 bits) then round:
  # double rounding for division is safe (p >= 2*24+2).
  return (np.asarray(a, f32).astype(f64) / np.asarray(b, f32).astype(f64)).astype(f32)


def rcp(b):
  return (1.0 / np.asarray(b, f32).astype(f64)).astype(f32)


def op_matrix(a, v):
  """Return op(A)[..., i, j] as a dense lower(forward) or upper(backward) system."""
  lower, unit, trans = {'L': (1, 1, 0), 'U': (0, 0, 0), 'Ut': (0, 0, 1), 'Lt': (1, 1, 1)}[v]
  A = np.swapaxes(a, -1, -2) if trans else a
  forward = bool(lower) != bool(trans)
  return A, forward, bool(unit)


def solve(a, b, v, hyp):
  """a (B,n,n), b (B,n,m) -> x per hypothesis."""
  A, forward, unit = op_matrix(a, v)
  n = a.shape[-1]
  x = np.zeros_like(b)
  order = list(range(n)) if forward else list(range(n - 1, -1, -1))
  kind = hyp['kind']
  for pos, i in enumerate(order):
    prev = order[:pos]
    if hyp.get('rev'):
      prev = prev[::-1]
    bi = b[:, i, :]
    coef = lambda j: A[:, i, j][:, None]
    if kind == 'seq_fma':
      s = bi
      for j in prev:
        s = fma(-coef(j), x[:, j, :], s)
    elif kind == 'seq_nofma':
      s = bi
      for j in prev:
        s = sub(s, mul(coef(j), x[:, j, :]))
    elif kind == 'acc_fma':  # acc = sum a x (fma chain from 0); s = b - acc
      acc = np.zeros_like(bi)
      for j in prev:
        acc = fma(coef(j), x[:, j, :], acc)
      s = sub(bi, acc)
    elif kind == 'acc_nofma':
      acc = np.zeros_like(bi)
      for j in prev:
        acc = add(acc, mul(coef(j), x[:, j, :]))
      s = sub(bi, acc)
    elif kind == 'blk':  # blocks of K: acc within block (fma from 0), s -= acc
      K = hyp['K']
      s = bi
      for c in range(0, len(prev), K):
        acc = np.zeros_like(bi)
        for j in prev[c:c + K]:
          acc = fma(coef(j), x[:, j, :], acc)
        s = sub(s, acc)
    elif kind == 'blkseq':  # aligned row blocks of K; earlier blocks via GEMM-acc
      K = hyp['K']
      bi_blk = i // K
      s = bi
      blocks = sorted({j // K for j in prev if j // K != bi_blk},
                      reverse=not forward)
      if hyp.get('blkrev'):
        blocks = blocks[::-1]
      for blk in blocks:
        js = [j for j in prev if j // K == blk]
        js = sorted(js, reverse=hyp.get('accrev', False))
        if hyp.get('upd') == 'seq':
          for j in js:
            s = fma(-coef(j), x[:, j, :], s)
          continue
        acc = np.zeros_like(bi)
        for j in js:
          acc = fma(coef(j), x[:, j, :], acc)
        if hyp.get('upd') == 'fma_neg1':
          s = fma(-np.ones_like(acc), acc, s)
        else:
          s = sub(s, acc)
      for j in [j for j in prev if j // K == bi_blk]:
        s = fma(-coef(j), x[:, j, :], s)
    elif kind == 'multi':  # K interleaved accumulators, combined in order
      K = hyp['K']
      accs = [np.zeros_like(bi) for _ in range(K)]
      for t, j in enumerate(prev):
        accs[t % K] = fma(coef(j), x[:, j, :], accs[t % K])
      tot = accs[0]
      for k in range(1, K):
        tot = add(tot, accs[k])
      s = sub(bi, tot)
    else:
      raise ValueError(kind)
    if not unit:
      d = A[:, i, i][:, None]
      dv = hyp.get('div', 'rn')
      if dv == 'rn':
        s = div(s, d)
      elif dv == 'rcp':
        s = mul(s, rcp(d))
    x[:, i, :] = s
  return x


HYPS = {
    'seq_fma': dict(kind='seq_fma'),
    'seq_fma_rcp': dict(kind='seq_fma', div='rcp'),
    'seq_fma_rev': dict(kind='seq_fma', rev=True),
    'seq_nofma': dict(kind='seq_nofma'),
    'acc_fma': dict(kind='acc_fma'),
    'acc_fma_rcp': dict(kind='acc_fma', div='rcp'),
    'acc_fma_rev': dict(kind='acc_fma', rev=True),
    'acc_nofma': dict(kind='acc_nofma'),
}
for K in (4, 8, 16):
  HYPS[f'bs{K}'] = dict(kind='blkseq', K=K)
  HYPS[f'bs{K}r'] = dict(kind='blkseq', K=K, accrev=True)
  HYPS[f'bs{K}br'] = dict(kind='blkseq', K=K, blkrev=True)
for K in (2, 4, 8, 16, 32):
  HYPS[f'blk{K}'] = dict(kind='blk', K=K)
  HYPS[f'multi{K}'] = dict(kind='multi', K=K)


def main():
  d = np.load(sys.argv[1])
  ns = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [2, 3, 4, 5, 8]
  hyps = sys.argv[3].split(',') if len(sys.argv) > 3 else list(HYPS)
  nb = int(sys.argv[4]) if len(sys.argv) > 4 else 64
  for n in ns:
    for m in sorted({1, 2, n}):
      if f'a_{n}_{m}' not in d:
        continue
      a = d[f'a_{n}_{m}'][:nb]; b = d[f'b_{n}_{m}'][:nb]
      for v in ('L', 'U', 'Ut', 'Lt'):
        ref = d[f'x_{v}_{n}_{m}'][:nb]
        line = []
        for h in hyps:
          x = solve(a, b, v, HYPS[h])
          mism = int((x.view(np.uint32) != ref.view(np.uint32)).sum())
          line.append(f'{h}={mism}')
        print(f'n={n} m={m} {v} (tot {ref.size}): ' + ' '.join(line), flush=True)


if __name__ == '__main__':
  main()
