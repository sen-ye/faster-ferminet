"""Structural diff of two optimized HLO texts: per-computation signature multisets."""
import collections, re, sys
def parse(path):
  comps = {}; cur = None
  for line in open(path):
    m = re.match(r'^(?:ENTRY )?%?([\w.\-]+) .*\{\s*$', line)
    if m and not line.startswith(' '):
      cur = m.group(1); comps[cur] = []; continue
    if cur and line.startswith('  ') and '=' in line:
      l = re.sub(r', metadata=\{[^}]*\}', '', line)
      l = re.sub(r'backend_config=\{.*', '', l)
      rhs = l.split('=', 1)[1].strip()
      # signature: result shape/layout + opcode + operand shapes stripped of names
      sig = re.sub(r'%[\w.\-]+', '%', rhs)
      sig = re.sub(r'(calls|to_apply|condition|body)=%?[\w.\-]+', r'\1=C', sig)
      comps[cur].append(sig.strip())
  return comps
a, b = parse(sys.argv[1]), parse(sys.argv[2])
ca = collections.Counter(s for v in a.values() for s in v); cb = collections.Counter(s for v in b.values() for s in v)
print('only in A:'); [print(f'  {n}x {s[:260]}') for s, n in (ca - cb).most_common(60)]
print('only in B:'); [print(f'  {n}x {s[:260]}') for s, n in (cb - ca).most_common(60)]
