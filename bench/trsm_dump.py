"""Dump XLA/cuBLAS batched triangular_solve inputs+outputs for offline analysis."""
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

VARIANTS = {
    'L': dict(lower=True, unit=True, trans=False),
    'U': dict(lower=False, unit=False, trans=False),
    'Ut': dict(lower=False, unit=False, trans=True),
    'Lt': dict(lower=True, unit=True, trans=True),
}


def main():
  out = sys.argv[1]
  res = {}
  for n in (2, 3, 4, 5, 8, 9, 16, 17, 26, 32, 33, 36, 48, 54, 64):
    for m in sorted({1, 2, n}):
      batch = 512
      k1, k2 = jax.random.split(jax.random.PRNGKey(100 * n + m))
      x = jax.random.normal(k1, (batch, n, n), jnp.float32)
      lu, _, _ = lax.linalg.lu(x)
      b = jax.random.normal(k2, (batch, n, m), jnp.float32)
      res[f'a_{n}_{m}'] = np.asarray(lu)
      res[f'b_{n}_{m}'] = np.asarray(b)
      for v, p in VARIANTS.items():
        f = jax.jit(lambda a, b, p=p: lax.linalg.triangular_solve(
            a, b, left_side=True, lower=p['lower'], transpose_a=p['trans'],
            unit_diagonal=p['unit']))
        res[f'x_{v}_{n}_{m}'] = np.asarray(f(lu, b))
  np.savez_compressed(out, **res)
  print('saved', out, len(res))


if __name__ == '__main__':
  main()
