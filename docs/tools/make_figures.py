"""Render the README figures from docs/validation/*.json (no third-party dependencies).

usage: python docs/tools/make_figures.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(HERE)
ATOMS = ['Fe', 'Kr', 'Cd', 'Xe']
COL = {'Fe': '#4C78A8', 'Kr': '#F58518', 'Cd': '#54A24B', 'Xe': '#B279A2'}
FONT = 'font-family="Helvetica,Arial,sans-serif"'


def svg(w, h, body):
  return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
          f'<rect width="{w}" height="{h}" fill="white"/>{body}</svg>\n')


def text(x, y, s, size=13, anchor='middle', weight='normal', fill='#222'):
  return (f'<text x="{x:.1f}" y="{y:.1f}" {FONT} font-size="{size}" text-anchor="{anchor}" '
          f'font-weight="{weight}" fill="{fill}">{s}</text>')


def benchmark(h20):
  w, h = 880, 420
  x0, y0, pw, ph = 70, 110, 780, 240
  b = [text(w / 2, 30, 'Faster-FermiNet: time per training step (original = 1.00, lower is better)', 16,
            weight='bold'),
       text(w / 2, 52, 'NVIDIA H20, 1024 walkers per GPU, default XLA flags; bit-for-bit identical results', 12,
            fill='#555')]
  for i in range(0, 6):
    v = i * 0.2
    y = y0 + ph - v / 1.0 * ph
    b.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + pw}" y2="{y:.1f}" stroke="#ddd"/>')
    b.append(text(x0 - 8, y + 4, f'{v:.1f}', 11, 'end'))
  gw = pw / len(ATOMS)
  for k, a in enumerate(ATOMS):
    r = h20['rows'][a]
    gx = x0 + k * gw
    for j, (lab, val, sec, fill) in enumerate(
        (('original', 1.0, r['ref_step_s'], '#BBBBBB'),
         ('Faster-FermiNet', 1 / r['speedup'], r['cand_step_s'], COL[a]))):
      bx = gx + gw * 0.18 + j * gw * 0.34
      bh = val * ph
      b.append(f'<rect x="{bx:.1f}" y="{y0 + ph - bh:.1f}" width="{gw * 0.3:.1f}" height="{bh:.1f}" fill="{fill}"/>')
      b.append(text(bx + gw * 0.15, y0 + ph - bh - 6, f'{sec:.3f} s', 11))
    b.append(text(gx + gw / 2, y0 + ph + 20, f'{a} ({sum(r["electrons"])} e-)', 13, weight='bold'))
    b.append(text(gx + gw / 2, y0 + ph + 38, f'{r["speedup"]:.3f}x', 13, fill=COL[a], weight='bold'))
  b.append(f'<rect x="{w / 2 - 190}" y="66" width="12" height="12" fill="#BBBBBB"/>')
  b.append(text(w / 2 - 172, 76, 'original FermiNet', 11, 'start'))
  b.append(f'<rect x="{w / 2 - 40}" y="66" width="12" height="12" fill="#4C78A8"/>')
  b.append(text(w / 2 - 22, 76, 'Faster-FermiNet (per-atom colour)', 11, 'start'))
  return svg(w, h, ''.join(b))


def history(att, h20):
  rows = att['attempts']
  w, h = 980, 440
  x0, y0, pw, ph = 70, 70, 860, 280
  lo, hi = 0.80, 1.30
  b = [text(w / 2, 30, 'Agent optimization history: 8 rounds, 16 attempts (speedup vs frozen baseline)', 16,
            weight='bold'),
       text(w / 2, 52, 'filled = bit-wise identical on all four atoms; x = bit-wise mismatch (rejected); '
            'shaded = adopted into the final configuration', 12, fill='#555')]
  for i in range(0, 6):
    v = lo + i * 0.1
    y = y0 + ph - (v - lo) / (hi - lo) * ph
    b.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + pw}" y2="{y:.1f}" stroke="#ddd"/>')
    b.append(text(x0 - 8, y + 4, f'{v:.2f}x', 11, 'end'))
  y1 = y0 + ph - (1.0 - lo) / (hi - lo) * ph
  b.append(f'<line x1="{x0}" y1="{y1:.1f}" x2="{x0 + pw}" y2="{y1:.1f}" stroke="#888" stroke-dasharray="4,3"/>')
  adopted = {'2c', '2d', '4a', '8a'}
  n = len(rows)
  dx = pw / n
  prev_round = None
  for i, r in enumerate(rows):
    cx = x0 + (i + 0.5) * dx
    if r['attempt'] in adopted:
      b.append(f'<rect x="{cx - dx / 2:.1f}" y="{y0}" width="{dx:.1f}" height="{ph}" fill="#FFF3C4"/>')
    if r['round'] != prev_round:
      b.append(f'<line x1="{cx - dx / 2:.1f}" y1="{y0}" x2="{cx - dx / 2:.1f}" y2="{y0 + ph}" stroke="#ccc"/>')
      b.append(text(cx - dx / 2 + 4, y0 + ph + 52, f'R{r["round"]}', 11, 'start', fill='#555'))
      prev_round = r['round']
    b.append(text(cx, y0 + ph + 18, r['attempt'], 11))
    for k, a in enumerate(ATOMS):
      v = max(lo, min(hi, r['speedup_vs_frozen_baseline'][a]))
      y = y0 + ph - (v - lo) / (hi - lo) * ph
      px = cx + (k - 1.5) * 5
      if r['bitwise_8step'][a]:
        b.append(f'<circle cx="{px:.1f}" cy="{y:.1f}" r="4" fill="{COL[a]}"/>')
      else:
        b.append(f'<path d="M{px - 4:.1f},{y - 4:.1f} L{px + 4:.1f},{y + 4:.1f} M{px - 4:.1f},{y + 4:.1f} '
                 f'L{px + 4:.1f},{y - 4:.1f}" stroke="{COL[a]}" stroke-width="2"/>')
  for k, a in enumerate(ATOMS):
    b.append(f'<circle cx="{x0 + 20 + k * 70}" cy="{y0 + ph + 75}" r="5" fill="{COL[a]}"/>')
    b.append(text(x0 + 30 + k * 70, y0 + ph + 79, a, 12, 'start'))
  fin = h20['rows']
  b.append(text(x0 + 330, y0 + ph + 79, 'final formal check (ABA): ' + ', '.join(
      f'{a} {fin[a]["speedup"]:.3f}x' for a in ATOMS), 12, 'start'))
  return svg(w, h, ''.join(b))


def module_map():
  w, h = 980, 330
  boxes = [
      (20, 'KFAC training step', 'pmap(_step): loss, gradient, curvature'),
      (190, 'Local energy', 'kinetic term via jax.linearize(grad log|psi|)'),
      (360, 'Hessian-diagonal loop', 'fori_loop over 3N directions, unroll=2 *'),
      (530, 'slogdet JVP / transpose', 'custom_linear_solve -> lu_solve'),
      (700, 'triangular_solve_p', 'CUDA lowering rule *'),
  ]
  b = [text(w / 2, 28, 'Where Faster-FermiNet changes the computation (highlighted = changed)', 16, weight='bold')]
  for x, t, s in boxes:
    star = '*' in s
    b.append(f'<rect x="{x}" y="60" width="155" height="80" rx="8" fill="{"#FFF3C4" if star else "#F2F2F2"}" '
             f'stroke="#999"/>')
    b.append(text(x + 77, 92, t, 13, weight='bold'))
    words = s.replace(' *', '').split(' ')
    lines, cur = [], ''
    for wd in words:
      if len(cur) + len(wd) > 24:
        lines.append(cur)
        cur = wd
      else:
        cur = (cur + ' ' + wd).strip()
    lines.append(cur)
    for li, ln in enumerate(lines[:3]):
      b.append(text(x + 77, 110 + li * 14, ln, 10.5, fill='#444'))
    if x < 700:
      b.append(f'<path d="M{x + 157},100 L{x + 168},100" stroke="#666" stroke-width="2" marker-end="url(#a)"/>')
  b.insert(0, '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">'
              '<path d="M0,0 L8,4 L0,8 z" fill="#666"/></marker></defs>')
  bottom = [
      (20, 270, 'Hessian-diagonal loop', ['fori_loop(..., unroll=2): same', 'accumulation order val + h(0) + h(1)',
                                           '+ ...; two independent HVPs per', 'loop trip for XLA to schedule'], True),
      (320, 270, 'all other solves', ['e.g. KFAC factor inverses and', 'unbatched / non-square systems:',
                                      'original cuBLAS lowering,', 'unchanged'], False),
      (620, 340, 'f32, left side, batched, 9 &lt;= n = m &lt;= 64', [
          'Pallas/Triton kernel reproducing cuBLAS trsmBatched', 'bit for bit: 8-row blocks, block partial sums,',
          'in-block FMA chain, IEEE division (inline PTX);', 'column-major I/O, so neighbouring XLA fusions',
          'compile exactly as with cuBLAS'], True)]
  for x, bw, t, lines, star in bottom:
    b.append(f'<rect x="{x}" y="185" width="{bw}" height="125" rx="8" fill="{"#FFF3C4" if star else "#F2F2F2"}" '
             f'stroke="#999"/>')
    b.append(text(x + bw / 2, 207, t, 12, weight='bold'))
    for i, ln in enumerate(lines):
      b.append(text(x + bw / 2, 227 + i * 16, ln, 11, fill='#333'))
  b.append('<path d="M437,142 L200,182" stroke="#666" stroke-width="2" marker-end="url(#a)"/>')
  b.append('<path d="M777,142 L790,182" stroke="#666" stroke-width="2" marker-end="url(#a)"/>')
  b.append('<path d="M777,142 L470,182" stroke="#666" stroke-width="2" marker-end="url(#a)"/>')
  return svg(w, h, ''.join(b))


def main():
  with open(os.path.join(DOCS, 'validation', 'h20.json')) as f:
    h20 = json.load(f)
  with open(os.path.join(DOCS, 'validation', 'attempts.json')) as f:
    att = json.load(f)
  os.makedirs(os.path.join(DOCS, 'performance-views'), exist_ok=True)
  os.makedirs(os.path.join(DOCS, 'change-views'), exist_ok=True)
  with open(os.path.join(DOCS, 'performance-views', 'benchmark.svg'), 'w') as f:
    f.write(benchmark(h20))
  with open(os.path.join(DOCS, 'performance-views', 'history.svg'), 'w') as f:
    f.write(history(att, h20))
  with open(os.path.join(DOCS, 'change-views', 'module-map.svg'), 'w') as f:
    f.write(module_map())
  print('figures written')


if __name__ == '__main__':
  main()
