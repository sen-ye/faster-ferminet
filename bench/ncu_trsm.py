"""Nsight Compute on the Pallas trsm kernel at production shape (diagnostic only).

usage: NCU=/path/to/ncu python bench/ncu_trsm.py <tree> <out.txt> [n]
Requires permission to read GPU performance counters.
"""
import os
import subprocess
import sys

tree, out = sys.argv[1], sys.argv[2]
n = sys.argv[3] if len(sys.argv) > 3 else '54'
ncu = os.environ.get('NCU', 'ncu')
runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trsm_prod_run.py')
metrics = ','.join([
    'smsp__pcsamp_warps_issue_stalled_long_scoreboard',
    'smsp__pcsamp_warps_issue_stalled_barrier',
    'smsp__pcsamp_warps_issue_stalled_short_scoreboard',
    'smsp__pcsamp_warps_issue_stalled_mio_throttle',
    'smsp__pcsamp_warps_issue_stalled_wait',
    'smsp__pcsamp_warps_issue_stalled_math_pipe_throttle',
    'smsp__pcsamp_warps_issue_stalled_lg_throttle'])
cmd = [ncu, '--target-processes', 'all', '--kernel-name', f'regex:ferminet_trsm_bs8_n{n}_U_',
       '--launch-skip', '1', '--launch-count', '1',
       '--section', 'SpeedOfLight', '--section', 'Occupancy', '--section', 'LaunchStats',
       '--section', 'WarpStateStats', '--section', 'SchedulerStats',
       '--section', 'MemoryWorkloadAnalysis', '--metrics', metrics,
       sys.executable, '-I', runner, tree, n]
r = subprocess.run(cmd, capture_output=True, text=True, check=False)
with open(out, 'w') as f:
  f.write(r.stdout + '\n' + r.stderr)
print(r.stdout[-8000:], r.stderr[-1500:], flush=True)
