"""Run a list of bench jobs on several GPUs (one process per GPU at a time).

usage: python bench/launch.py <jobs.json> [gpu_ids, default $GPUS or 0,1,2,3]

Job entries (written by bench/make_jobs.py):
  {"gpu": 0, "atom": "Fe", "tree": "...", "mode": "time", "iters": 18, "out": "...", "env": {...}}
  {"cmd": ["bench/trsm_check.py", "<tree>", "9,16"], "out": "...", "env": {...}}
Jobs with "gpu" are pinned (timing runs: each atom stays on one GPU); jobs without it go
to a shared queue served by whichever GPU becomes free. Every process runs with
JAX_PLATFORMS=cuda so JAX fails instead of silently falling back to the CPU; CUDA
initialisation failures are retried (logs kept as *.initfailN).
"""
import collections
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.environ.get('PYTHON', sys.executable)
INIT_ERR = ('cuInit', 'CUDA_ERROR_OPERATING_SYSTEM', 'CUDA_ERROR_NO_DEVICE',
            'Unable to initialize backend', 'BACKEND_NOT_GPU')


def run_one(j, gpu):
  rc, t0 = 1, time.time()
  for attempt in range(4):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS='cuda')
    env.update(j.get('env', {}))
    os.makedirs(os.path.dirname(os.path.abspath(j['out'])), exist_ok=True)
    if 'cmd' in j:
      script = j['cmd'][0]
      if not os.path.isabs(script):
        script = os.path.join(ROOT, script)
      cmd = [PYTHON, '-I', script] + j['cmd'][1:]
    else:
      cmd = [PYTHON, '-I', os.path.join(ROOT, 'bench', 'fnbench.py'), '--tree', j['tree'],
             '--atom', j['atom'], '--mode', j['mode'], '--iters', str(j['iters']),
             '--out', j['out']] + j.get('args', [])
    t0 = time.time()
    logp = j['out'].rstrip('/') + '.log'
    with open(logp, 'w') as log:
      log.write(json.dumps(dict(j, gpu_used=gpu, attempt=attempt)) + '\n')
      log.flush()
      rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    with open(logp) as f:
      txt = f.read()
    if rc != 0 and any(e in txt for e in INIT_ERR):
      os.rename(logp, logp + f'.initfail{attempt}')
      print(f'[gpu{gpu}] CUDA init failure, retry {attempt + 1}: {j["out"]}', flush=True)
      time.sleep(10 * (attempt + 1))
      continue
    break
  return rc, t0


def run_queue(gpu, pinned, shared, lock, results):
  while True:
    if pinned:
      j = pinned.pop(0)
    else:
      with lock:
        if not shared:
          return
        j = shared.pop(0)
    rc, t0 = run_one(j, gpu)
    results.append((gpu, j['out'], rc))
    print(f"[gpu{gpu}] rc={rc} {time.time() - t0:.0f}s {j['out']}", flush=True)


def main():
  with open(sys.argv[1]) as f:
    jobs = json.load(f)
  spec = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('GPUS', '0,1,2,3')
  gpus = [int(g) for g in spec.split(',')]
  pinned = collections.OrderedDict((g, []) for g in gpus)
  shared = []
  for j in jobs:
    (pinned[j['gpu']] if j.get('gpu') is not None else shared).append(j)
  lock = threading.Lock()
  results = []
  threads = [threading.Thread(target=run_queue, args=(g, pinned[g], shared, lock, results))
             for g in gpus]
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  bad = [r for r in results if r[2] != 0]
  print('LAUNCH_DONE', len(results), 'jobs', len(bad), 'failed')
  sys.exit(1 if bad else 0)


if __name__ == '__main__':
  main()
