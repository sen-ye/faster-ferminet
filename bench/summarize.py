"""Per-atom step-time summary for fnbench runs.

usage: python summarize.py [--warmup K] [--window W] [--base DIR[,DIR...]] DIR [DIR...]
Steady-state = step durations after the first K (compile + warmup) steps,
split into windows of W steps; reports each window's mean, the median of the
window means, and the (max-min)/median window range. With --base, reports the
speedup of each DIR against the mean of the base runs' medians.
"""
import argparse
import json
import os
import statistics

ATOMS = ['Fe', 'Kr', 'Cd', 'Xe']


def stats(d, a, warmup, window):
  p = os.path.join(d, a, 'result.json')
  if not os.path.exists(p):
    return None
  r = json.load(open(p))
  s = r['step_s'][warmup:]
  wins = [s[i:i + window] for i in range(0, len(s) - window + 1, window)]
  means = [sum(w) / len(w) for w in wins] or [float('nan')]
  med = statistics.median(means)
  rng = (max(means) - min(means)) / med if med == med and med > 0 else float('nan')
  return {'median': med, 'windows': means, 'range': rng,
          'first_step_s': r['first_step_s'], 'startup_s': r['startup_s'],
          'n': len(s)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--warmup', type=int, default=2)
  ap.add_argument('--window', type=int, default=5)
  ap.add_argument('--base', default='')
  ap.add_argument('--json', default='')
  ap.add_argument('dirs', nargs='+')
  a = ap.parse_args()
  base = {}
  if a.base:
    for at in ATOMS:
      vals = [stats(d, at, a.warmup, a.window) for d in a.base.split(',')]
      vals = [v['median'] for v in vals if v]
      if vals:
        base[at] = sum(vals) / len(vals)
  out = {}
  for d in a.dirs:
    print(f'== {d}')
    out[d] = {}
    for at in ATOMS:
      s = stats(d, at, a.warmup, a.window)
      if not s:
        print(f'  {at}: missing')
        continue
      sp = base[at] / s['median'] if at in base else None
      out[d][at] = dict(s, speedup=sp, base=base.get(at))
      w = ' '.join(f'{x:.3f}' for x in s['windows'])
      sps = f' speedup={sp:.3f}x (base {base[at]:.3f}s)' if sp else ''
      print(f"  {at}: median={s['median']:.4f}s windows=[{w}] range={100*s['range']:.2f}% "
            f"n={s['n']} first={s['first_step_s']:.1f}s startup={s['startup_s']:.1f}s{sps}")
  if a.json:
    json.dump(out, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
  main()
