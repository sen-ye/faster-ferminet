"""Map top GPU ops of a trace to their HLO fusion + JAX source (op_name metadata).

usage: python fusion_map.py <trace_dir> <hlo_after_optimizations.txt> [topN]
"""
import collections
import glob
import gzip
import json
import re
import sys


def main():
  tdir, hlo = sys.argv[1], sys.argv[2]
  top = int(sys.argv[3]) if len(sys.argv) > 3 else 40
  path = sorted(glob.glob(tdir + '/**/*perfetto_trace.json.gz', recursive=True))[-1]
  ev = json.load(gzip.open(path))
  ev = ev['traceEvents'] if isinstance(ev, dict) else ev
  pname = {e['pid']: e['args']['name'] for e in ev
           if e.get('ph') == 'M' and e.get('name') == 'process_name'}
  gpu = [e for e in ev if e.get('ph') == 'X' and '/device:GPU' in str(pname.get(e['pid'], ''))]
  ops = collections.Counter(); cnt = collections.Counter()
  for e in gpu:
    a = e.get('args', {})
    if a.get('hlo_module') != 'pmap__step':
      continue
    o = a.get('hlo_op', e['name'])
    ops[o] += e['dur']; cnt[o] += 1
  tot = sum(ops.values())
  # instruction name -> (op line, called computation)
  text = open(hlo).read()
  inst = {}
  for line in text.splitlines():
    m = re.match(r'\s+(?:ROOT )?%([\w.\-]+) = (.*)', line)
    if m:
      inst[m.group(1)] = m.group(2)
  comps = {}
  cur = None
  for line in text.splitlines():
    m = re.match(r'^(?:ENTRY )?%?([\w.\-]+) .*\{\s*$', line)
    if m and not line.startswith(' '):
      cur = m.group(1); comps[cur] = []
    elif cur and line.startswith('  '):
      comps[cur].append(line)
  print(f'pmap__step GPU total {tot/1e3:.1f} ms')
  for o, d in ops.most_common(top):
    rhs = inst.get(o, '')
    shape = rhs.split(' ')[0][:60]
    kind = re.search(r'kind=(\w+)', rhs)
    called = re.search(r'calls=%?([\w.\-]+)', rhs)
    names = collections.Counter()
    srcs = collections.Counter()
    body = comps.get(called.group(1), []) if called else [rhs]
    for l in body + [rhs]:
      for nm in re.findall(r'op_name="([^"]+)"', l):
        names[re.sub(r'\[.*?\]', '', nm)[-110:]] += 1
      for sf in re.findall(r'source_file="([^"]+)" source_line=(\d+)', l):
        srcs[sf[0].split('/')[-1] + ':' + sf[1]] += 1
    nm = names.most_common(1)[0][0] if names else ''
    sr = ','.join(f'{k}' for k, _ in srcs.most_common(3))
    print(f'{d/1e3:9.2f}ms {100*d/tot:5.1f}% n={cnt[o]:5d} {o[:28]:28} {kind.group(1) if kind else "":7} {shape:40} | {sr[:70]} | {nm}')


if __name__ == '__main__':
  main()
