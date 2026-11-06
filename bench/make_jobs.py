"""Write a jobs.json for bench/launch.py.

usage: python bench/make_jobs.py --plan {gate,bitwise,time} --reference <tree> --candidate <tree>
                                 --out <dir> [--gpus 0,1,2,3] [--atoms Fe,Kr,Cd,Xe]
                                 [--long-iters 110] [--probe]

gate    : per atom, pinned to one GPU: reference A -> candidate -> reference B timing runs
          (default XLA flags), plus deterministic-mode records of reference and candidate
          (shared queue) and, optionally, long deterministic records and the trsm probe.
bitwise : only the deterministic-mode records (+ optional long records).
time    : only candidate timing runs (pinned).
Atoms are assigned to GPUs in order (Fe->first GPU, ...), wrapping if fewer GPUs.
"""
import argparse
import json
import os

DET = '--xla_gpu_deterministic_ops=true'


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--plan', choices=['gate', 'bitwise', 'time'], default='gate')
  ap.add_argument('--reference', required=False, default='')
  ap.add_argument('--candidate', required=True)
  ap.add_argument('--out', required=True)
  ap.add_argument('--gpus', default=os.environ.get('GPUS', '0,1,2,3'))
  ap.add_argument('--atoms', default='Fe,Kr,Cd,Xe')
  ap.add_argument('--iters', type=int, default=18)
  ap.add_argument('--check-iters', type=int, default=8)
  ap.add_argument('--long-iters', type=int, default=0)
  ap.add_argument('--probe', action='store_true')
  a = ap.parse_args()
  out = os.path.abspath(a.out)
  ref = os.path.abspath(a.reference) if a.reference else ''
  cand = os.path.abspath(a.candidate)
  gpus = [int(g) for g in a.gpus.split(',')]
  atoms = a.atoms.split(',')
  if a.plan in ('gate', 'bitwise') and not ref:
    ap.error('--reference is required for plans gate and bitwise')
  det_env = {'XLA_FLAGS': (os.environ.get('XLA_FLAGS', '') + ' ' + DET).strip()}
  jobs = []
  for i, atom in enumerate(atoms):
    g = gpus[i % len(gpus)]
    if a.plan == 'gate':
      jobs += [dict(gpu=g, atom=atom, tree=ref, mode='time', iters=a.iters, out=f'{out}/refA/{atom}'),
               dict(gpu=g, atom=atom, tree=cand, mode='time', iters=a.iters, out=f'{out}/cand/{atom}'),
               dict(gpu=g, atom=atom, tree=ref, mode='time', iters=a.iters, out=f'{out}/refB/{atom}')]
    elif a.plan == 'time':
      jobs.append(dict(gpu=g, atom=atom, tree=cand, mode='time', iters=a.iters, out=f'{out}/cand/{atom}'))
  if a.probe:
    jobs.append(dict(cmd=['bench/trsm_check.py', cand, '9,16,26,36,48,54,64'], out=f'{out}/trsm_check'))
  if a.plan in ('gate', 'bitwise'):
    for atom in reversed(atoms):  # largest first in the shared queue
      jobs += [dict(atom=atom, tree=ref, mode='check', iters=a.check_iters, out=f'{out}/ref_det/{atom}', env=det_env),
               dict(atom=atom, tree=cand, mode='check', iters=a.check_iters, out=f'{out}/cand_det/{atom}', env=det_env)]
    if a.long_iters:
      for atom in reversed(atoms):
        jobs += [dict(atom=atom, tree=ref, mode='check', iters=a.long_iters, out=f'{out}/long/ref/{atom}', env=det_env),
                 dict(atom=atom, tree=cand, mode='check', iters=a.long_iters, out=f'{out}/long/cand/{atom}', env=det_env)]
  os.makedirs(out, exist_ok=True)
  with open(f'{out}/jobs.json', 'w') as f:
    json.dump(jobs, f, indent=1)
  print(f'{len(jobs)} jobs -> {out}/jobs.json')


if __name__ == '__main__':
  main()
