# bench — judge and tools

| File | Role |
|---|---|
| `fnbench.py` | one atom on one GPU: drives the unmodified `train.train(cfg)`; `--mode time` records step wall times, `--mode check` also records state digests every step |
| `compare.py` | bit-wise verdict between two `check` runs (energy bits, pmove, walkers, params, KFAC state, MCMC width) |
| `summarize.py` | step-time windows / medians / speedup vs a base run |
| `gate.py` | formal verdict: ABA timing threshold + bit-wise (8-step and optional long) |
| `make_jobs.py`, `launch.py` | parallel runs on several GPUs (timing pinned per GPU, checks in a shared queue, `JAX_PLATFORMS=cuda`, CUDA-init retry) |
| `trsm_check.py` | kernel contract probe: Pallas solve vs real cuBLAS, bit for bit, plus timings |
| `trsm_dump.py`, `trsm_emul.py` | recover cuBLAS's operation order (dump on GPU, exact-FMA emulation on CPU) |
| `diag_le.py` | in-process override off/on comparison of log ψ, ∇ log ψ, kinetic and local energy |
| `trace_ops.py`, `fusion_map.py`, `hlo_sig.py` | profiler-trace and HLO analysis |
| `trsm_prod_run.py`, `ncu_trsm.py` | Nsight Compute of the kernel at production shape |

`fnbench.py`, `compare.py` and `gate.py` form the frozen judge; their hashes are in `../task/formulation.json`.
