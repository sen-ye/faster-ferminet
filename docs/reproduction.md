# Reproduction

## Environment

Tested stack (all packages public on PyPI; see [requirements.txt](../requirements.txt)):

| Component | Version |
|---|---|
| GPU | NVIDIA H20 (Hopper, sm_90), 96 GB HBM3, 500 W board limit; one GPU per run |
| Driver | 535.261.03 |
| Python | 3.11.2 |
| JAX / jaxlib / jax-cuda12-plugin | 0.7.2 |
| cuBLAS / cuDNN / cuSOLVER | nvidia-cublas-cu12 12.9.1.4 / nvidia-cudnn-cu12 9.24.0.43 / nvidia-cusolver-cu12 11.7.5.82 |
| kfac-jax / optax / folx | 0.0.8 / 0.2.6 / 0.2.31 |
| pyscf | 2.12.1 |

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .     # this repository's ferminet package
```

`setup.py` is the upstream one (it lists `folx` and `kfac_jax` from git); `--no-deps` keeps the pinned
versions above. The bit-exact kernel reproduces the arithmetic of the pinned cuBLAS build on H20. On
other GPUs or cuBLAS versions, run `bash scripts/check_kernel.sh` first (see [Scope](../README.md#scope)).

## Workload

`ferminet/configs/atom.py` with upstream defaults (KFAC, 10 MCMC steps per iteration, FermiNet with hidden
dims (256, 32)×4, 16 full determinants, `laplacian='default'`), plus:

| Setting | Value |
|---|---|
| `cfg.batch_size` | 1024 (walkers per GPU; one device per run) |
| `cfg.pretrain.basis` | `ano-rcc` (all-electron, no ECP) |
| `cfg.system.spin_polarisation` | Fe 4 (15 α / 11 β); Kr, Cd, Xe 0 |
| `cfg.debug.deterministic` | True (seed 23) |
| bench only | `pretrain.iterations=20`, `mcmc.burn_in=10` (one-off phases; the timed step is unchanged) |

Training an atom with this configuration:

```bash
ATOM=Xe bash scripts/run_atom.sh               # upstream entry point: python -m ferminet.main
ATOM=Fe ITERATIONS=100000 SAVE_PATH=out/fe bash scripts/run_atom.sh
```

## Benchmark and checks

```bash
bash scripts/get_upstream.sh                   # original FermiNet @ e9d5a604 -> third_party/ferminet-upstream
GPUS=0,1,2,3 bash scripts/run_benchmark.sh     # ABA timing + deterministic bit-wise check + gate verdict
GPUS=0,1,2,3 LONG_ITERS=110 PROBE=1 bash scripts/run_benchmark.sh   # + 110-step check + kernel probe
GPUS=0,1,2,3 bash scripts/check_bitwise.sh     # bit-wise only
bash scripts/check_kernel.sh                   # kernel vs cuBLAS on one GPU
```

`run_benchmark.sh` writes `runs/<name>/{refA,cand,refB,ref_det,cand_det}/<atom>/result.json`,
`final.json` and `final_table.md`. Fe, Kr, Cd and Xe run on the four GPUs in parallel; timing runs stay on
their GPU (A → candidate → B) and the deterministic records fill whichever GPU is free.

Approximate wall time on 4 × H20: the formal benchmark takes about 15 minutes. Each process spends 30–90 s
in startup (pyscf Hartree–Fock with ano-rcc) and 20–35 s compiling. The 110-step check adds about 10 minutes.

Single runs, if you prefer to drive the bench yourself:

```bash
export JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0
python -I bench/fnbench.py --tree . --atom Xe --mode time --iters 18 --out runs/x/cand/Xe
python -I bench/fnbench.py --tree third_party/ferminet-upstream --atom Xe --mode time --iters 18 --out runs/x/ref/Xe
XLA_FLAGS=--xla_gpu_deterministic_ops=true python -I bench/fnbench.py --tree . --atom Xe --mode check --iters 8 --out runs/x/cand_det/Xe
python -I bench/compare.py runs/x/ref_det runs/x/cand_det Xe
python -I bench/summarize.py --base runs/x/ref runs/x/cand
```

## Switches

Both optimizations are on by default; [scripts/configs/faster.env](../scripts/configs/faster.env) states
the measured values, [scripts/configs/upstream.env](../scripts/configs/upstream.env) turns them off.

| Variable | Default | Effect |
|---|---|---|
| `FERMINET_PALLAS_TRSM` | `1` | Lower validated batched solves to the bit-exact Pallas kernel; `0` = original cuBLAS lowering |
| `FERMINET_PALLAS_TRSM_WARPS` | `1` | Warps per Triton program (measured value) |
| `FERMINET_PALLAS_TRSM_COLS` | `64` | Columns per program; 64 = one program per matrix (16 and 32 were slower) |
| `FERMINET_LAPLACIAN_UNROLL` | `2` | Unroll of the Hessian-diagonal `fori_loop`; `1` = upstream form |

The kernel is used only for float32, left-side, batched solves with a square right-hand side and
9 ≤ n ≤ 64. Every other `triangular_solve`, e.g. the KFAC factor inverses, keeps the original cuBLAS
lowering.

## Diagnostics used during optimization

| Tool | Purpose |
|---|---|
| `fnbench.py --profile a:b` | JAX profiler (perfetto) trace of steps a+1..b |
| `bench/trace_ops.py`, `bench/fusion_map.py` | GPU time per HLO op; map ops to JAX source via HLO `op_name` metadata (dump HLO with `--xla_dump_to`) |
| `bench/hlo_sig.py` | structural diff of two optimized HLO modules (found the layout effect, see [trsm_bitexact.md](trsm_bitexact.md)) |
| `bench/diag_le.py` | in-process off/on comparison of log ψ, ∇ log ψ, kinetic and local energy |
| `bench/trsm_dump.py`, `bench/trsm_emul.py` | dump cuBLAS solve inputs/outputs; emulate candidate operation orders on the CPU |
| `bench/ncu_trsm.py` | Nsight Compute on the kernel at production shape |
