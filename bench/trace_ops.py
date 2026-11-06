"""Group GPU kernel time by hlo_module and hlo_op (count, ms)."""
import collections, glob, gzip, json, sys
path = sorted(glob.glob(sys.argv[1] + '/**/*perfetto_trace.json.gz', recursive=True))[-1]
ev = json.load(gzip.open(path)); ev = ev['traceEvents'] if isinstance(ev, dict) else ev
pname = {e['pid']: e['args']['name'] for e in ev if e.get('ph') == 'M' and e.get('name') == 'process_name'}
gpu = [e for e in ev if e.get('ph') == 'X' and '/device:GPU' in str(pname.get(e['pid'], ''))]
mods = collections.Counter(); modn = collections.Counter()
ops = collections.Counter(); opn = collections.Counter()
for e in gpu:
  a = e.get('args', {}); m = a.get('hlo_module', '?'); o = a.get('hlo_op', e['name'])
  mods[m] += e['dur']; modn[m] += 1; ops[(m, o)] += e['dur']; opn[(m, o)] += 1
tot = sum(mods.values())
print('total GPU ms', tot / 1e3)
for m, d in mods.most_common(): print(f'{d/1e3:10.1f}ms {100*d/tot:5.1f}% n={modn[m]:6d} {m}')
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
print('--- top ops')
for (m, o), d in ops.most_common(N): print(f'{d/1e3:9.2f}ms {100*d/tot:5.1f}% n={opn[(m,o)]:5d} {m[:30]:30} {o[:80]}')
cnt = collections.Counter(opn[(m, o)] for (m, o) in ops)
print('--- histogram of per-op launch counts (count: #ops, ms)')
cms = collections.Counter()
for k, d in ops.items(): cms[opn[k]] += d
for c, k in sorted(cnt.items(), key=lambda x: -cms[x[0]])[:15]: print(f'  n={c:6d}: {k:5d} ops, {cms[c]/1e3:9.1f} ms')
