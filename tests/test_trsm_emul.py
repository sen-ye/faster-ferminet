"""CPU tests of the offline emulator used to recover cuBLAS's operation order."""
from fractions import Fraction
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bench'))
import trsm_emul  # pylint: disable=g-import-not-at-top


def _exact_fma32(a, b, c):
  ex = Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c))
  r = np.float32(float(ex))
  cands = [np.nextafter(r, -np.inf, dtype=np.float32), r, np.nextafter(r, np.inf, dtype=np.float32)]
  dist = [abs(Fraction(float(x)) - ex) for x in cands]
  best = [x for x, d in zip(cands, dist) if d == min(dist)]
  if len(best) > 1:
    best = [x for x in best if (x.view(np.uint32) & 1) == 0]
  return best[0]


def test_fma_emulation_is_exact():
  rng = np.random.default_rng(0)
  a, b, c = (rng.standard_normal(2000).astype(np.float32) for _ in range(3))
  got = trsm_emul.fma(a, b, c)
  want = np.array([_exact_fma32(x, y, z) for x, y, z in zip(a, b, c)], np.float32)
  assert np.array_equal(got.view(np.uint32), want.view(np.uint32))


def test_bs8_equals_sequential_within_one_block():
  rng = np.random.default_rng(1)
  for n in (3, 5, 8):
    x = rng.standard_normal((4, n, n)).astype(np.float32)
    a = np.tril(x, -1) + np.triu(x) + np.eye(n, dtype=np.float32) * 4
    b = rng.standard_normal((4, n, n)).astype(np.float32)
    for v in ('L', 'U', 'Ut', 'Lt'):
      s = trsm_emul.solve(a, b, v, trsm_emul.HYPS['seq_fma'])
      k = trsm_emul.solve(a, b, v, trsm_emul.HYPS['bs8'])
      assert np.array_equal(s.view(np.uint32), k.view(np.uint32))
