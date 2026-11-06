"""Kernel contract: the Pallas solve equals XLA's cuBLAS trsmBatched bit for bit (GPU only).

Run on every new GPU / cuBLAS / JAX stack:  pytest -q tests/test_pallas_trsm.py
The reference is compiled with the lowering override disabled (pallas_trsm.ENABLED = False), so
it really runs cuBLAS even if the override is installed in this process.
"""
import itertools

import jax
from jax import lax
import jax.numpy as jnp
import pytest

from ferminet import pallas_trsm as pt

needs_gpu = pytest.mark.skipif(jax.default_backend() != 'gpu', reason='needs a CUDA GPU')

VARIANTS = {'L': (True, True, False), 'U': (False, False, False),
            'Ut': (False, False, True), 'Lt': (True, True, True)}


def _inputs(n, scale, batch=(32, 16)):
  k1, k2, k3 = jax.random.split(jax.random.PRNGKey(n * 7919 + int(scale * 10)), 3)
  lu, _, _ = lax.linalg.lu(jax.random.normal(k1, batch + (n, n)) * scale)
  b = jax.random.normal(k2, batch + (n, n)) * jnp.exp(jax.random.normal(k3, batch + (1, n)))
  return lu, b


@needs_gpu
@pytest.mark.parametrize('n,variant,scale', list(itertools.product(
    [9, 16, 17, 26, 32, 33, 36, 48, 54, 64], list(VARIANTS), [1.0, 1e-3, 1e3])))
def test_bitwise_vs_cublas(n, variant, scale):
  lower, unit, trans = VARIANTS[variant]
  a, b = _inputs(n, scale)
  kw = dict(left_side=True, lower=lower, transpose_a=trans, unit_diagonal=unit)
  enabled = pt.ENABLED
  try:
    pt.ENABLED = False  # lowering-time switch: compile the reference on cuBLAS
    ref = jax.jit(lambda a, b: lax.linalg.triangular_solve(a, b, **kw))(a, b)
  finally:
    pt.ENABLED = enabled
  out = jax.jit(lambda a, b: pt._pallas_triangular_solve(a, b, conjugate_a=False, **kw))(a, b)  # pylint: disable=protected-access
  ri = lax.bitcast_convert_type(ref, jnp.int32)
  oi = lax.bitcast_convert_type(out, jnp.int32)
  assert int(jnp.sum(ri != oi)) == 0


def test_guard():
  f32 = jax.ShapeDtypeStruct
  assert pt.supported(f32((4, 26, 26), jnp.float32), f32((4, 26, 26), jnp.float32), True, False)
  assert not pt.supported(f32((26, 26), jnp.float32), f32((26, 26), jnp.float32), True, False)   # unbatched
  assert not pt.supported(f32((4, 8, 8), jnp.float32), f32((4, 8, 8), jnp.float32), True, False)  # n < 9
  assert not pt.supported(f32((4, 26, 26), jnp.float32), f32((4, 26, 1), jnp.float32), True, False)  # m != n
  assert not pt.supported(f32((4, 26, 26), jnp.float64), f32((4, 26, 26), jnp.float64), True, False)
  assert not pt.supported(f32((4, 26, 26), jnp.float32), f32((4, 26, 26), jnp.float32), False, False)
