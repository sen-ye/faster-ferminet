"""Bitwise probe sweep + timing: tree's ferminet.pallas_trsm vs XLA/cuBLAS."""
import sys, time
tree = sys.argv[1]; sys.path.insert(0, tree)
import os; os.environ['FERMINET_PALLAS_TRSM'] = '0'  # reference must stay on cuBLAS
import jax, jax.numpy as jnp, numpy as np
if jax.default_backend() != 'gpu': raise SystemExit('BACKEND_NOT_GPU: ' + jax.default_backend())
print('backend', jax.default_backend(), jax.devices()[0].device_kind, flush=True)
from jax import lax
from ferminet import pallas_trsm as pt
assert not pt._INSTALLED, 'override must not be installed in the probe process'

V = {'L': (True, True, False), 'U': (False, False, False), 'Ut': (False, False, True), 'Lt': (True, True, True)}

def ref(a, b, v):
  lo, un, tr = V[v]
  return lax.linalg.triangular_solve(a, b, left_side=True, lower=lo, transpose_a=tr, unit_diagonal=un)

def mine(a, b, v):
  lo, un, tr = V[v]
  return pt._pallas_triangular_solve(a, b, left_side=True, lower=lo, transpose_a=tr, conjugate_a=False, unit_diagonal=un)

def inputs(key, shape_b, n, scale):
  k1, k2, k3 = jax.random.split(key, 3)
  x = jax.random.normal(k1, shape_b + (n, n)) * scale
  lu, _, _ = lax.linalg.lu(x)
  b = jax.random.normal(k2, shape_b + (n, n)) * jnp.exp(jax.random.normal(k3, shape_b + (1, n)))
  return lu, b

ok = True
for n in [int(s) for s in sys.argv[2].split(',')]:
  for scale in (1.0, 1e-3, 1e3):
    a, b = inputs(jax.random.PRNGKey(n * 7 + int(scale > 1)), (64, 16), n, scale)
    line = []
    for v in V:
      r = jax.jit(lambda a, b: ref(a, b, v))(a, b); o = jax.jit(lambda a, b: mine(a, b, v))(a, b)
      mm = int(jnp.sum(lax.bitcast_convert_type(r, jnp.int32) != lax.bitcast_convert_type(o, jnp.int32)))
      ok &= mm == 0
      line.append(f'{v}:{mm}')
    print(f'n={n} scale={scale:g} mismatches ' + ' '.join(line), flush=True)
print('PROBE', 'ALL_BITWISE' if ok else 'MISMATCH', flush=True)

def bench(f, *args, reps=20):
  f = jax.jit(f); r = f(*args); r.block_until_ready()
  t = time.perf_counter()
  for _ in range(reps): r = f(*args)
  r.block_until_ready(); return (time.perf_counter() - t) / reps * 1e3

for n in (26, 36, 48, 54):
  a, b = inputs(jax.random.PRNGKey(1), (1024, 16), n, 1.0)
  print(f'TIME n={n} ' + ' '.join(f'{v}: cublas {bench(lambda a, b: ref(a, b, v), a, b):.3f}ms pallas {bench(lambda a, b: mine(a, b, v), a, b):.3f}ms' for v in V), flush=True)
