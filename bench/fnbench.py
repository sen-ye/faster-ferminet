"""FermiNet bench: one atom, one GPU, one process (frozen measurement logic).

Drives the tree's own ``train.train(cfg)`` unchanged; only observes it:
  * a writer_manager records a host timestamp + energy bits at every step
    (the training loop already synchronises each step via np.asarray(loss));
  * in --mode check, cfg.log.save_frequency=0 makes train() call
    checkpoint.save every step; it is replaced by a hook that hashes every leaf
    of (data, params, opt_state, mcmc_width) so runs can be compared bit for bit.

Usage:
  JAX_PLATFORMS=cuda python -I bench/fnbench.py --tree <tree> --atom Fe --iters 18 --mode time --out <dir>

<tree> is a directory that contains a `ferminet/` package (this repository for the
optimized code, an upstream checkout for the original). The measurement logic is the
one used for all recorded results; the published copy only omits machine-identifying
fingerprint fields (host name, command line, absolute paths, GPU indices).
"""

import argparse
import hashlib
import json
import os
import sys
import time


SPIN = {'Fe': 4, 'Kr': 0, 'Cd': 0, 'Xe': 0}


def tree_digest(tree_dir):
  h = hashlib.sha256()
  root = os.path.join(tree_dir, 'ferminet')
  for dirpath, dirnames, filenames in sorted(os.walk(root)):
    dirnames.sort()
    for fn in sorted(filenames):
      if fn.endswith('.py'):
        p = os.path.join(dirpath, fn)
        h.update(os.path.relpath(p, tree_dir).encode())
        with open(p, 'rb') as f:
          h.update(f.read())
  return h.hexdigest()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--tree', required=True)
  ap.add_argument('--atom', required=True, choices=sorted(SPIN))
  ap.add_argument('--iters', type=int, default=20)
  ap.add_argument('--pretrain-iters', type=int, default=20)
  ap.add_argument('--burn-in', type=int, default=10)
  ap.add_argument('--batch', type=int, default=1024)
  ap.add_argument('--mode', choices=['time', 'check'], default='time')
  ap.add_argument('--out', required=True)
  ap.add_argument('--profile', default='',
                  help='a:b -> jax.profiler trace from step a to step b')
  args = ap.parse_args()

  tree = os.path.abspath(args.tree)
  sys.path.insert(0, tree)
  os.makedirs(args.out, exist_ok=True)

  # Environment fix: some container runtimes hide /proc/<pid> (PID namespaces)
  # while /proc/self stays visible. pyscf only uses this for memory heuristics;
  # the fix is applied identically to every tree.
  import pyscf.lib.misc as _pyscf_misc

  def _current_memory():
    with open('/proc/self/statm') as f:
      vms, rss = [int(x) * 4096 / 1e6 for x in f.readline().split()[:2]]
    return rss, vms

  _pyscf_misc.current_memory = _current_memory
  import pyscf.lib as _pyscf_lib
  _pyscf_lib.current_memory = _current_memory

  import jax
  import numpy as np
  import ferminet
  from ferminet import base_config
  from ferminet import checkpoint
  from ferminet import train
  from ferminet.configs import atom as atom_cfg

  if jax.default_backend() != 'gpu':
    raise SystemExit('BACKEND_NOT_GPU: ' + jax.default_backend())
  assert os.path.dirname(os.path.abspath(ferminet.__file__)) == os.path.join(
      tree, 'ferminet'), ferminet.__file__

  cfg = atom_cfg.get_config()
  cfg.system.atom = args.atom
  cfg.system.spin_polarisation = SPIN[args.atom]
  cfg.pretrain.basis = 'ano-rcc'
  cfg.pretrain.iterations = args.pretrain_iters
  cfg.mcmc.burn_in = args.burn_in
  cfg.batch_size = args.batch
  cfg.optim.iterations = args.iters
  cfg.debug.deterministic = True
  cfg.log.save_path = os.path.join(args.out, 'ckpt')
  cfg.log.save_frequency = 0.0 if args.mode == 'check' else 1.0e9
  cfg = base_config.resolve(cfg)

  rec = {
      'atom': args.atom,
      'electrons': list(cfg.system.electrons),
      'mode': args.mode,
      'iters': args.iters,
      'pretrain_iters': args.pretrain_iters,
      'burn_in': args.burn_in,
      'batch': args.batch,
      'tree_sha256': tree_digest(tree),
      'fingerprint': {
          'jax': jax.__version__,
          'devices': [str(d) for d in jax.devices()],
          'device_kind': jax.devices()[0].device_kind,
          'XLA_FLAGS': os.environ.get('XLA_FLAGS', ''),
          'switches': {k: v for k, v in os.environ.items()
                       if k.startswith('FERMINET_')},
      },
      'steps': [],
      'state': [],
  }

  t_start = time.perf_counter()
  marks = {'t0': None}
  prof = tuple(int(x) for x in args.profile.split(':')) if args.profile else ()
  rec['profile'] = list(prof)

  class Writer:

    def __enter__(self):
      marks['t0'] = time.perf_counter()
      return self

    def __exit__(self, *a):
      return False

    def write(self, t, **kw):
      now = time.perf_counter()
      if prof and t == prof[0]:
        jax.profiler.start_trace(os.path.join(args.out, 'trace'),
                                 create_perfetto_trace=True)
      if prof and t == prof[1]:
        jax.profiler.stop_trace()
      e = np.float64(np.asarray(kw['energy']))
      rec['steps'].append({
          't': int(t),
          'wall': now,
          'energy': float(e),
          'energy_bits': np.asarray(kw['energy']).tobytes().hex(),
          'pmove': float(np.asarray(kw['pmove'])),
      })

    def flush(self):
      pass

  def hash_state(save_path, t, data, params, opt_state, mcmc_width,
                 density_state=None):
    del save_path, density_state
    out = {}
    for name, obj in (('data', data), ('params', params),
                      ('opt_state', opt_state), ('mcmc_width', mcmc_width)):
      leaves = jax.tree_util.tree_leaves_with_path(obj)
      h = hashlib.sha256()
      for path, leaf in leaves:
        a = np.asarray(leaf)
        h.update(jax.tree_util.keystr(path).encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
      out[name] = h.hexdigest()
      out[name + '_nleaves'] = len(leaves)
    rec['state'].append({'t': int(t), **out})
    return ''

  if args.mode == 'check':
    checkpoint.save = hash_state

  train.train(cfg, writer_manager=Writer())
  t_end = time.perf_counter()

  walls = [s['wall'] for s in rec['steps']]
  rec['startup_s'] = marks['t0'] - t_start if marks['t0'] else None
  rec['first_step_s'] = walls[0] - marks['t0'] if walls else None
  rec['step_s'] = [b - a for a, b in zip(walls[:-1], walls[1:])]
  rec['total_s'] = t_end - t_start
  for s in rec['steps']:
    s['wall'] -= t_start
  with open(os.path.join(args.out, 'result.json'), 'w') as f:
    json.dump(rec, f, indent=1)
  print('BENCH_DONE', args.atom, json.dumps({
      'startup_s': rec['startup_s'], 'first_step_s': rec['first_step_s'],
      'step_s': rec['step_s']}))


if __name__ == '__main__':
  main()
