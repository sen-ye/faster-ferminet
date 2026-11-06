"""Formal gate: ABA timing verdict + bit-wise verdict -> <run>/final.json and a table.

usage: python bench/gate.py <run_dir> [reference_det_dir, default <run_dir>/ref_det]

Expects (written by make_jobs.py --plan gate + launch.py):
  <run_dir>/{refA,cand,refB}/<atom>/result.json        timing, default XLA flags
  <run_dir>/cand_det/<atom>/result.json                deterministic-mode record
  <run_dir>/long/{ref,cand}/<atom>/result.json         optional long records
Timing: per run, median over windows of 5 steady steps after 2 warm-up steps (step 0 =
compile is excluded). speedup = mean(refA, refB) / cand. Benefit passes if
speedup > max(1.02, 1 + 2 * |refA - refB| / mean) and every run's window range < 10%.
Bit-wise: bench/compare.py (energy bits, pmove and sha256 of every leaf of walkers,
parameters, optimizer state and MCMC width at every step).
"""
import json
import os
import statistics
import subprocess
import sys

ATOMS = ['Fe', 'Kr', 'Cd', 'Xe']
HERE = os.path.dirname(os.path.abspath(__file__))


def med(d, a, warmup=2, window=5):
  with open(os.path.join(d, a, 'result.json')) as f:
    r = json.load(f)
  s = r['step_s'][warmup:]
  wins = [sum(s[i:i + window]) / window for i in range(0, len(s) - window + 1, window)]
  return statistics.median(wins), wins, r


def bitwise(da, db, atom):
  out = subprocess.run([sys.executable, '-I', os.path.join(HERE, 'compare.py'), da, db, atom],
                       capture_output=True, text=True, check=False).stdout
  return 'BITWISE_IDENTICAL' in out, (out.strip().splitlines() or ['missing'])[0]


def main():
  run = sys.argv[1]
  refdet = sys.argv[2] if len(sys.argv) > 2 else os.path.join(run, 'ref_det')
  res = {}
  for a in ATOMS:
    if not os.path.exists(os.path.join(run, 'cand', a, 'result.json')):
      continue
    ra, wa, _ = med(f'{run}/refA', a)
    c, wc, rc = med(f'{run}/cand', a)
    rb, wb, _ = med(f'{run}/refB', a)
    ref = (ra + rb) / 2
    gap = abs(ra - rb) / ref
    thr = max(1.02, 1 + 2 * gap)
    sp = ref / c
    win_rng = max((max(w) - min(w)) / statistics.median(w) for w in (wa, wc, wb))
    bit_ok, bit_line = bitwise(refdet, f'{run}/cand_det', a)
    long_ok, long_line = None, 'not run'
    if os.path.exists(f'{run}/long/cand/{a}/result.json'):
      long_ok, long_line = bitwise(f'{run}/long/ref', f'{run}/long/cand', a)
    res[a] = dict(electrons=rc['electrons'], ref_step_s=ref, refA=ra, refB=rb, cand_step_s=c,
                  speedup=sp, threshold=thr, ref_gap=gap, max_window_range=win_rng,
                  benefit='PASS' if sp > thr and win_rng < 0.10 else 'FAIL',
                  bitwise_8step='PASS' if bit_ok else 'FAIL', bitwise_8step_detail=bit_line,
                  bitwise_long='PASS' if long_ok else ('FAIL' if long_ok is False else 'n/a'),
                  bitwise_long_detail=long_line,
                  verdict='PASS' if (sp > thr and win_rng < 0.10 and bit_ok
                                     and long_ok is not False) else 'FAIL')
  with open(f'{run}/final.json', 'w') as f:
    json.dump(res, f, indent=1)
  print('| atom | e- | ref step (s) | cand step (s) | speedup | gate thr | bitwise 8-step | bitwise long | verdict |')
  print('|---|---|---|---|---|---|---|---|---|')
  for a, x in res.items():
    print(f"| {a} | {sum(x['electrons'])} | {x['ref_step_s']:.4f} (A {x['refA']:.4f} / B {x['refB']:.4f}) | "
          f"{x['cand_step_s']:.4f} | **{x['speedup']:.3f}x** | {x['threshold']:.3f} | {x['bitwise_8step']} | "
          f"{x['bitwise_long']} | {x['verdict']} |")


if __name__ == '__main__':
  main()
