# AGENTS.md

Guidance for humans and coding agents who continue optimizing this repository. It follows the
[ResearchBoost-Agent](https://github.com/DAGroup-PKU/ResearchBoost-Agent) method: a frozen judge, one
hotspot class per round, and every round logged.

## Layout

| Path | Content |
|---|---|
| `ferminet/` | the FermiNet package (upstream `e9d5a604` + the optimization; see `docs/history.md`) |
| `ferminet/pallas_trsm.py` | bit-exact Pallas replacement of batched cuBLAS triangular solves |
| `bench/` | the judge: `fnbench.py` (time/check), `compare.py` (bit-wise), `gate.py` (verdict), `launch.py` / `make_jobs.py` (parallel runs), kernel and profiling tools |
| `scripts/` | portable entry points; `scripts/configs/*.env` switch sets |
| `task/formulation.json` | frozen workload, baseline, gate rules and evaluator hashes |
| `docs/` | method, evidence, history and validation records |
| `tests/` | kernel contract tests (GPU) and emulator tests (CPU) |

## Rules

- **The judge is frozen.** Do not edit `bench/fnbench.py`, `bench/compare.py` or `bench/gate.py` to make a
  candidate pass. Check their hashes against `task/formulation.json` before reporting a result.
- **Bit-wise means bit-wise.** Every candidate must be identical to the original in deterministic mode
  (`XLA_FLAGS=--xla_gpu_deterministic_ops=true`) on Fe, Kr, Cd and Xe at every recorded step. "Same math"
  is not enough; XLA layout and fusion changes alter bits.
- **One configuration for all atoms.** No branches on element, electron count or atom-specific tuning.
  Kernel guards may depend only on validated generic properties such as dtype or matrix size range.
- **Same code path.** Timing and checks run the same code; only the XLA determinism flag differs between
  them, as documented in `docs/judge.md`.
- **GPU only.** Run GPU processes with `JAX_PLATFORMS=cuda`; never accept a record from a CPU fallback.
- **Re-validate the kernel on new stacks.** Run `scripts/check_kernel.sh` on any new GPU, cuBLAS or JAX
  version; on mismatch, disable it with `FERMINET_PALLAS_TRSM=0`.

## Per round

1. Profile the current best (`fnbench.py --profile`, `bench/fusion_map.py`).
2. Change one hotspot class behind a switch.
3. Time it (`scripts/run_benchmark.sh` or `bench/make_jobs.py --plan time`) and check it
   (`scripts/check_bitwise.sh`).
4. Log the profile finding, change, timing and check result, including failures.
