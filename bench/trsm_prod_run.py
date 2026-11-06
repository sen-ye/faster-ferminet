"""Kernel-only runner at production shape (for ncu): n, batch (1024,16), all 4 variants x 3 reps."""
import os, sys
tree = sys.argv[1]; n = int(sys.argv[2]); sys.path.insert(0, tree)
os.environ['FERMINET_PALLAS_TRSM'] = '0'
import jax, jax.numpy as jnp
from jax import lax
from ferminet import pallas_trsm as pt
k1, k2 = jax.random.split(jax.random.PRNGKey(0))
lu, _, _ = lax.linalg.lu(jax.random.normal(k1, (1024, 16, n, n)))
b = jax.random.normal(k2, (1024, 16, n, n))
for lo, un, tr in ((True, True, False), (False, False, False), (False, False, True), (True, True, True)):
  f = jax.jit(lambda a, b: pt._pallas_triangular_solve(a, b, left_side=True, lower=lo, transpose_a=tr, conjugate_a=False, unit_diagonal=un))
  for _ in range(3):
    f(lu, b).block_until_ready()
print('done')
