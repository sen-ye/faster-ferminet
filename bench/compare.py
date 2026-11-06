"""Bitwise comparison of two fnbench check runs.

usage: python compare.py <runA_dir> <runB_dir> [atoms...]
Each run dir contains <atom>/result.json written by fnbench.py --mode check.
Compares, step by step: energy bits, pmove, and sha256 of every leaf of
data / params / opt_state / mcmc_width. Exit 0 iff all atoms are identical.
"""
import json
import os
import sys


def load(d, a):
  with open(os.path.join(d, a, 'result.json')) as f:
    return json.load(f)


def main():
  da, db = sys.argv[1], sys.argv[2]
  atoms = sys.argv[3:] or ['Fe', 'Kr', 'Cd', 'Xe']
  ok_all = True
  out = {}
  for a in atoms:
    try:
      ra, rb = load(da, a), load(db, a)
    except FileNotFoundError as e:
      print(f'{a}: MISSING {e}')
      ok_all = False
      out[a] = 'MISSING'
      continue
    msgs = []
    assert ra['mode'] == rb['mode'] == 'check', 'both runs must be check mode'
    for k in ('iters', 'pretrain_iters', 'burn_in', 'batch', 'electrons'):
      if ra[k] != rb[k]:
        msgs.append(f'config {k}: {ra[k]} vs {rb[k]}')
    if len(ra['steps']) != len(rb['steps']) or len(ra['state']) != len(rb['state']):
      msgs.append(f"step count {len(ra['steps'])}/{len(ra['state'])} vs "
                  f"{len(rb['steps'])}/{len(rb['state'])}")
    first_bad = None
    for sa, sb in zip(ra['steps'], rb['steps']):
      if sa['energy_bits'] != sb['energy_bits'] or sa['pmove'] != sb['pmove']:
        first_bad = first_bad or ('energy/pmove', sa['t'], sa['energy'], sb['energy'])
    for sa, sb in zip(ra['state'], rb['state']):
      for k in ('data', 'params', 'opt_state', 'mcmc_width'):
        if sa[k] != sb[k] or sa[k + '_nleaves'] != sb[k + '_nleaves']:
          if first_bad is None or first_bad[1] > sa['t']:
            first_bad = (k, sa['t'])
          break
    if first_bad:
      msgs.append(f'first mismatch: {first_bad}')
    n = len(ra['state'])
    status = 'BITWISE_IDENTICAL' if not msgs else 'DIFFERENT'
    ok_all &= not msgs
    out[a] = status
    ea = ra['steps'][-1]['energy'] if ra['steps'] else None
    print(f"{a}: {status} over {n} steps (energy, pmove, data, params, opt_state, "
          f"mcmc_width each step); final E={ea}; " + '; '.join(msgs))
  print('ALL_BITWISE_IDENTICAL' if ok_all else 'NOT_IDENTICAL')
  return 0 if ok_all else 1


if __name__ == '__main__':
  sys.exit(main())
