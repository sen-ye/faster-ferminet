"""Render the README figures from docs/validation/*.json.

usage: python -m pip install matplotlib; python docs/tools/make_figures.py
Visual layout follows the Faster-* case studies (benchmark bars + headline ratio;
history line + milestone cards + footnote; module map).
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.lines  # pylint: disable=g-import-not-at-top
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
from matplotlib.patches import FancyBboxPatch  # pylint: disable=g-import-not-at-top

DOCS = Path(__file__).resolve().parents[1]
ATOMS = ['Fe', 'Kr', 'Cd', 'Xe']
RED, PINK, INK, MUTED, LIGHT, GRID, GREY = '#8b0000', '#d8bcc1', '#292626', '#7d7174', '#f7f2f1', '#ebe3e3', '#a89c9e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'svg.fonttype': 'path'})


def gmean(xs):
  return math.exp(sum(math.log(x) for x in xs) / len(xs))


def load():
  h20 = json.loads((DOCS / 'validation' / 'h20.json').read_text())
  att = {a['attempt']: a for a in json.loads((DOCS / 'validation' / 'attempts.json').read_text())['attempts']}
  return h20, att


def benchmark(h20):
  rows = h20['rows']
  final = gmean([rows[a]['speedup'] for a in ATOMS])
  fig = plt.figure(figsize=(14, 7), facecolor='white')
  fig.text(.06, .90, 'Faster-FermiNet, optimized with an AI agent', color=RED, size=26, weight='bold')
  fig.text(.06, .83, 'Fe · Kr · Cd · Xe  ·  1 NVIDIA H20 per atom  ·  1024 walkers per GPU  ·  bit-for-bit identical',
           color=MUTED, size=14)
  ax = fig.add_axes([.17, .20, .50, .54])
  y = 0
  ticks, labels = [], []
  for a in ATOMS:
    r = rows[a]
    for lab, val, col in (('Baseline', 1.0, PINK), ('Faster', 1 / r['speedup'], RED)):
      sec = r['ref_step_s'] if lab == 'Baseline' else r['cand_step_s']
      ax.barh(y, val, height=.62, color=col)
      ax.text(val + .015, y, f'{sec:.3f} s', va='center', size=12, color=INK, weight='bold')
      ticks.append(y)
      labels.append(f'{a} {lab.lower()}' if lab == 'Baseline' else f'{a} Faster-FermiNet')
      y -= .8
    y -= .55
  ax.set_yticks(ticks)
  ax.set_yticklabels(labels, color=MUTED, size=12)
  ax.set_xlim(0, 1.25)
  ax.set_xticks([0, .25, .5, .75, 1.0])
  ax.set_xticklabels(['0', '0.25', '0.50', '0.75', '1.00'])
  ax.set_xlabel('Time per training step, relative to baseline ↓', color=MUTED, size=14, labelpad=12)
  ax.grid(axis='x', color=GRID)
  ax.set_axisbelow(True)
  for s in ax.spines.values():
    s.set_visible(False)
  ax.tick_params(length=0, labelcolor=MUTED, pad=8)
  fig.text(.84, .55, f'{final:.2f}×', color=RED, size=60, weight='bold', ha='center')
  fig.text(.84, .47, 'step speedup (geo-mean)', color=MUTED, size=17, ha='center')
  fig.text(.84, .42, ' / '.join(f'{rows[a]["speedup"]:.3f}×' for a in ATOMS), color=MUTED, size=12.5, ha='center')
  fig.text(.84, .38, 'formal check: PASS', color=MUTED, size=13, ha='center')
  fig.text(.06, .07, 'Baseline → candidate → baseline in independent processes, default XLA flags; baseline = mean of the '
           'two baseline runs.', color=MUTED, size=12)
  fig.text(.06, .035, 'Bit-for-bit identity verified separately in deterministic XLA mode (8 and 110 training steps).',
           color=MUTED, size=12)
  fig.savefig(DOCS / 'performance-views' / 'benchmark.svg', facecolor='white')
  plt.close(fig)


def history(h20, att):
  rows = h20['rows']
  final = gmean([rows[a]['speedup'] for a in ATOMS])
  r7 = gmean([h20['previous_best_round7'][a]['speedup'] for a in ATOMS])
  g = lambda k: gmean([att[k]['speedup_vs_frozen_baseline'][a] for a in ATOMS])
  # (label, value, kind): kind = screen (open), gate (filled), rejected (grey)
  pts = [('R2a', g('2a'), 'rejected'), ('R2c', g('2c'), 'screen'), ('R2d', g('2d'), 'screen'),
         ('R4a', g('4a'), 'screen'), ('R4c', g('4c'), 'rejected'), ('R7', r7, 'gate'), ('R8', final, 'gate')]
  fig = plt.figure(figsize=(14, 13.5), facecolor='white')
  fig.text(.09, .955, 'Faster-FermiNet', color=RED, size=17, weight='bold')
  fig.text(.09, .918, 'Agent Optimization History', color=INK, size=31, weight='bold')
  fig.text(.93, .930, f'{final:.2f}×', color=RED, size=48, weight='bold', ha='right')
  fig.text(.93, .906, 'step speedup (geo-mean)', color=MUTED, size=14, ha='right')
  fig.text(.09, .872, 'Cumulative speedup · Fe, Kr, Cd, Xe · 1 H20 per atom · 1024 walkers · 8 rounds, 16 attempts, '
           '4 adopted', color=MUTED, size=14.2)
  ax = fig.add_axes([.10, .505, .85, .27])
  xs = list(range(len(pts)))
  line = [(x, v) for x, (_, v, k) in zip(xs, pts) if k != 'rejected']
  ax.plot([p[0] for p in line], [p[1] for p in line], color=RED, lw=3, zorder=2)
  for i, (x, (lab, v, kind)) in enumerate(zip(xs, pts)):
    if kind == 'gate':
      ax.plot(x, v, 'o', color=RED, markersize=11, zorder=3)
    elif kind == 'screen':
      ax.plot(x, v, 'o', markerfacecolor='white', markeredgecolor=RED, markeredgewidth=2.5, markersize=11, zorder=3)
    else:
      ax.plot(x, v, 'o', markerfacecolor='white', markeredgecolor=GREY, markeredgewidth=2.5, markersize=11, zorder=3)
    up = i % 2 == 0
    col = RED if lab == 'R8' else (GREY if kind == 'rejected' else INK)
    ax.annotate(f'{v:.2f}×', (x, v), xytext=(0, 16 if up else -26), textcoords='offset points', ha='center',
                size=15, weight='bold', color=col)
  ax.set_xticks(xs)
  ax.set_xticklabels([p[0] for p in pts], color=MUTED, size=14)
  ax.set_ylim(.98, 1.32)
  ax.set_yticks([1.0, 1.1, 1.2, 1.3])
  ax.set_yticklabels(['1.0×', '1.1×', '1.2×', '1.3×'], color=MUTED, size=14)
  ax.set_xlim(-.5, len(pts) - .5)
  ax.grid(axis='y', color=GRID)
  ax.set_axisbelow(True)
  for s in ax.spines.values():
    s.set_visible(False)
  ax.tick_params(length=0, pad=10)
  for y, (face, edge, lab) in zip((.838, .815, .792), (
      (RED, RED, 'Formal check: ABA timing + bit-wise 8 & 110 steps'),
      ('white', RED, 'Screening: vs frozen baseline + bit-wise 8 steps'),
      ('white', GREY, 'Faster but not bit-for-bit identical (rejected)'))):
    fig.lines.append(matplotlib.lines.Line2D([.105], [y], marker='o', markerfacecolor=face, markeredgecolor=edge,
                                             markeredgewidth=2.2, markersize=9, transform=fig.transFigure))
    fig.text(.12, y - .005, lab, size=12.5, color=INK)
  cards = [('R1', 'Profile: cuBLAS trsm = 28–36% of step', False),
           ('R2a', 'Row-major Pallas trsm: fast, bits changed', True),
           ('R2c', 'Bit-exact Pallas trsm, cuBLAS layout kept', False),
           ('R2d', 'Register-resident in-block solve', False),
           ('R4a', 'Laplacian loop unroll ×2', False),
           ('R4c', 'Unroll ×4: Fe not bit-wise (rejected)', True),
           ('R7', 'Formal check; XLA scheduling flags: no gain', False),
           ('R8', 'Kernel V5: independent block loads', False)]
  for i, (n, title, rej) in enumerate(cards):
    col, row = i % 2, i // 2
    xx, yy = .085 + col * .45, .375 - row * .075
    fig.patches.append(FancyBboxPatch((xx, yy), .43, .058, boxstyle='round,pad=0.004', transform=fig.transFigure,
                                      facecolor=LIGHT, edgecolor='none'))
    fig.text(xx + .015, yy + .022, n, size=17, color=GREY if rej else RED, weight='bold')
    fig.text(xx + .085, yy + .022, title, size=13.6, color=INK, weight='bold')
  notes = ['Values: geometric mean of the Fe, Kr, Cd and Xe speedups. R7 and R8 are formal checks (original A → candidate →',
           'original B); other points are screening runs against the frozen baseline. Cumulative configurations, not isolated',
           'ablations. Timing uses default XLA flags; bit-wise identity is checked in deterministic XLA mode, because the',
           'original is not bit-reproducible in default mode. All 16 attempts are listed in docs/history.md.']
  for k, ln in enumerate(notes):
    fig.text(.09, .072 - k * .019, ln, size=12, color=MUTED)
  fig.savefig(DOCS / 'performance-views' / 'history.svg', facecolor='white')
  plt.close(fig)


def module_map():
  fig = plt.figure(figsize=(14, 6.2), facecolor='white')
  fig.text(.05, .91, 'Where Faster-FermiNet changes the computation', color=RED, size=22, weight='bold')
  fig.text(.05, .845, 'Highlighted: changed. Everything else lowers exactly as in the original.', color=MUTED, size=13)
  top = [('KFAC training step', 'loss, gradient, curvature', False),
         ('Local energy', 'linearize(∇ log|ψ|)', False),
         ('Hessian-diagonal loop', 'fori_loop over 3N, unroll=2', True),
         ('slogdet JVP / transpose', 'custom_linear_solve → lu_solve', False),
         ('triangular_solve_p', 'CUDA lowering rule', True)]
  w, gap, y0 = .172, .018, .55
  for i, (t, s, ch) in enumerate(top):
    x = .05 + i * (w + gap)
    fig.patches.append(FancyBboxPatch((x, y0), w, .2, boxstyle='round,pad=0.006', transform=fig.transFigure,
                                      facecolor='#f3dede' if ch else LIGHT, edgecolor=RED if ch else 'none', lw=1.5))
    fig.text(x + w / 2, y0 + .125, t, ha='center', size=12.5, weight='bold', color=RED if ch else INK)
    fig.text(x + w / 2, y0 + .06, s, ha='center', size=10.5, color=MUTED)
    if i < len(top) - 1:
      fig.text(x + w + gap / 2, y0 + .09, '→', ha='center', size=16, color=MUTED)
  bottom = [(.05, .37, 'Loop unroll ×2', ['same accumulation order val + h(0) + h(1) + …;',
                                          'two independent HVPs per loop trip'], True),
            (.44, .25, 'All other solves', ['e.g. KFAC factor inverses:', 'original cuBLAS lowering'], False),
            (.71, .24, 'f32, batched, 9 ≤ n = m ≤ 64', ['Pallas/Triton kernel = cuBLAS bits;',
                                                         'column-major I/O, inline PTX rounding'], True)]
  for x, w, t, lines, ch in bottom:
    fig.patches.append(FancyBboxPatch((x, .1), w, .27, boxstyle='round,pad=0.006', transform=fig.transFigure,
                                      facecolor='#f3dede' if ch else LIGHT, edgecolor=RED if ch else 'none', lw=1.5))
    fig.text(x + .015, .3, t, size=13, weight='bold', color=RED if ch else INK)
    for k, ln in enumerate(lines):
      fig.text(x + .015, .22 - k * .055, ln, size=11, color=INK)
  fig.savefig(DOCS / 'change-views' / 'module-map.svg', facecolor='white')
  plt.close(fig)


def main():
  h20, att = load()
  benchmark(h20)
  history(h20, att)
  module_map()
  print('figures written')


if __name__ == '__main__':
  main()
