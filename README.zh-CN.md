# Faster-FermiNet

[English](README.md) | [简体中文](README.zh-CN.md)

本项目对 [FermiNet](https://github.com/google-deepmind/ferminet) 训练做基础设施层面的加速，优化流程遵循 [ResearchBoost-Agent](https://github.com/DAGroup-PKU/ResearchBoost-Agent)：先冻结任务定义和评判器，之后每轮依次做 profile → 针对一类热点 → 计时 → 逐 bit 检查 → 记录。仓库包含锁定的优化后源码、可移植的复现入口和本次实验记录的结果。

**状态：已完成 8 轮 ResearchBoost-Agent 优化（共 16 次尝试）；最终候选在四种原子上均通过冻结的计时与逐 bit 检查。**

**AI agent 优化的 FermiNet：在 NVIDIA H20 上，Fe、Kr、Cd、Xe 的训练步加速 1.25×，结果与原始实现逐 bit 一致。**

![Faster-FermiNet：NVIDIA H20 上训练步加速 1.25 倍，结果逐 bit 一致](docs/performance-views/benchmark.svg)

| 实现 | Fe (26 e⁻) | Kr (36 e⁻) | Cd (48 e⁻) | Xe (54 e⁻) |
|---|---:|---:|---:|---:|
| 基线，每步耗时（秒）↓ | 0.5383 | 1.1901 | 2.3870 | 3.7101 |
| **Faster-FermiNet**，每步耗时（秒）↓ | **0.4319** | **0.9569** | **1.8900** | **2.9564** |
| 加速比 ↑ | **1.246×** | **1.244×** | **1.263×** | **1.255×** |

测量方式为完整检查。硬件为 **NVIDIA H20**（Hopper，96 GB），每个原子独占一张卡，**每卡 1024 个 walker**。
配置为上游的全电子原子设置：KFAC、每次迭代 10 步 MCMC、FermiNet 4×(256, 32)、16 个 full determinant、默认 Laplacian；
预训练使用 `ano-rcc` 基组，Fe 的自旋极化为 4。精度、超参数和随机数流均保持不变。

计时采用 XLA **默认**设置，在同一张卡上按原始 A → Faster-FermiNet → 原始 B 的顺序独立启动进程。
精确加速比为 1.2463 / 1.2438 / 1.2630 / 1.2549×，几何平均 1.252×。逐 bit 一致性另在 XLA **确定性**模式下验证，
因为原始实现在默认模式下本身就不能逐 bit 复现（见[正确性验证](#正确性验证)）。

## 快速开始

按[锁定的环境](docs/reproduction.md#environment)安装依赖。导入 `ferminet` 时优化即生效。
[scripts/configs/faster.env](scripts/configs/faster.env) 列出了实测所用的开关，`scripts/configs/upstream.env` 可关闭全部优化。
用实测配置训练一个原子：

```bash
pip install -r requirements.txt && pip install --no-deps -e .
ATOM=Xe bash scripts/run_atom.sh          # 上游入口（python -m ferminet.main）
```

在四张卡上复现上表。脚本依次运行原始 A → Faster-FermiNet → 原始 B，再生成逐 bit 记录并给出判定：

```bash
bash scripts/get_upstream.sh              # 原始 FermiNet @ e9d5a604 -> third_party/ferminet-upstream
GPUS=0,1,2,3 bash scripts/run_benchmark.sh
GPUS=0,1,2,3 LONG_ITERS=110 PROBE=1 bash scripts/run_benchmark.sh   # 另加 110 步检查与内核比对
```

测试环境为 **H20 上的 JAX 0.7.2、cuBLAS 12.9.1.4、Python 3.11、驱动 535.261.03**。
这些可移植启动脚本是对实测流程的重新封装，本身尚未端到端重跑。详见[复现说明](docs/reproduction.md)。

## Agent 优化过程

优化遵循 [ResearchBoost-Agent](https://github.com/DAGroup-PKU/ResearchBoost-Agent) 的循环，共 8 轮、16 次尝试，最终采纳 4 项改动。
图中各点为累积配置：实心点为正式检查，空心点为筛选运行，灰色点表示虽然更快、但未通过逐 bit 检查而被否决的尝试。

![Faster-FermiNet 优化过程：8 轮、16 次尝试、1.25 倍](docs/performance-views/history.svg)

全部 16 次尝试（包括没有收益的）见[优化过程记录](docs/history.md)。

## 主要改动

- **逐 bit 一致的三角求解。** 原始实现中，Laplacian 内部的批量 cuBLAS `trsm` 占每步 28–36% 的时间，效率约 1 TFLOP/s。
  我们反推出它的精确运算顺序：按 8 行对齐分块；块的部分和从 +0 开始累加后再相减；块内为 FMA 链；采用 IEEE 除法。
  然后用 Pallas/Triton kernel 重新实现，以 inline PTX 固定每一次舍入，作为 `triangular_solve_p` 的 CUDA lowering 安装。
  它只用于 float32、批量、方阵且 9 ≤ n ≤ 64 的求解，其余求解仍走 cuBLAS。
- **保持原有数据布局。** 该 kernel 按 cuBLAS 的列主序读写，XLA 对所有相邻融合的编译因此与原来完全相同。
  行主序版本更快，但改变了数值 bits。
- **Laplacian 循环展开 ×2。** 累加顺序不变，每次循环处理两个相互独立的 Hessian-向量积。展开 4 倍时 Fe 不再逐 bit 一致，因此被否决。

![优化模块图：计算中被改动的位置](docs/change-views/module-map.svg)

相对上游只改动三个文件：新增 `ferminet/pallas_trsm.py`，修改 `ferminet/__init__.py` 和 `ferminet/hamiltonian.py`。
参见[开关表](docs/reproduction.md#switches)和[逐 bit trsm 方法](docs/trsm_bitexact.md)。

## 正确性验证

**计时与逐 bit 检查使用不同的 XLA 模式。** 计时与上游一致，采用 XLA 默认设置。在该模式下，未经修改的原始实现用同一 seed 跑两次，
第 0 步能量就已不同（Fe：E₀ −317.9158 对 −318.7558 Ha，在 batch 4096 的任务定义阶段记录）。
原因是 XLA 的部分 kernel 使用原子操作，MCMC 又会放大末位差异。`--xla_gpu_deterministic_ops=true` 能消除这一差异，关闭 autotuning 则不能。
因此逐 bit 一致性只在确定性模式下陈述并验证，速度结论只针对默认模式。

**逐 bit 检查。** 原始实现与 Faster-FermiNet 都设 `--xla_gpu_deterministic_ops=true`，使用相同 seed 和配置。
在 Fe、Kr、Cd、Xe 的 8 步与 **110 步**运行中，**每一步**的能量 bits、`pmove`，以及 walker、参数、KFAC 优化器状态、MCMC 步长所有张量的 SHA-256 摘要全部相同。
110 步运行跨过了第 100 步的 MCMC 步长调整。在评估任何候选之前，已确认原始实现在该模式下能复现自身。

**内核检查。** 在不加确定性 flag 的情况下，Pallas 求解与真实 cuBLAS 路径逐 bit 相同（n 为 9–64，四种变体，三种输入尺度）。
摘要见 [bitwise_records.json](docs/validation/bitwise_records.json)。

**计时。** 原始 A → Faster-FermiNet → 原始 B 在同一张卡上独立启动进程。每次 18 个迭代：排除第 0 步（编译）和两步预热，取三个 5 步窗口的中位数。
判定要求加速比大于 max(1.02, 1 + 2 × A/B 差)，且窗口极差小于 10%。

| 原子 | 原始 A / B（秒/步）↓ | Faster-FermiNet（秒/步）↓ | A/B 差 | 加速比 ↑ |
|---|---|---:|---:|---:|
| Fe | 0.5389 / 0.5377 | 0.4319 | 0.22% | **1.2463×** |
| Kr | 1.1903 / 1.1900 | 0.9569 | 0.03% | **1.2438×** |
| Cd | 2.3869 / 2.3870 | 1.8900 | 0.00% | **1.2630×** |
| Xe | 3.7099 / 3.7102 | 2.9564 | 0.01% | **1.2549×** |

阈值均为 1.020，窗口极差 ≤ 0.66%。原始记录见 [h20.json](docs/validation/h20.json)，比较规则见[判定说明](docs/judge.md)。

**验证范围。** 该 kernel 复现的是 H20（sm_90）上所锁定 cuBLAS 版本的运算顺序。换用其他 GPU 或 cuBLAS 版本时，请运行
`bash scripts/check_kernel.sh`，若报告不一致，设 `FERMINET_PALLAS_TRSM=0`。只测量了单卡、每卡 1024 个 walker、实数基态 `laplacian='default'` 的运行。
基准把预训练缩为 20 次迭代、burn-in 缩为 10 步，被计时的训练步不受影响；启动和编译不计入每步时间。
没有进行完整训练至收敛，也没有测量确定性模式下的速度。

## 为什么停在 1.25×

任务方设定的预算为 8 轮。在最终 profile 中，网络 GEMM 已接近 H20 的 TF32 峰值；受内存带宽限制的 XLA 融合无法重排，
因为重排会改变 FMA 合并方式，从而改变数值 bits。剩下可动的只有求解 kernel：n = 54 时约 1.5–1.7 ms，
而按内存带宽估算约 0.25 ms，瓶颈在 load 延迟。

详见[为什么停在这里](docs/profile.md)以及尝试过但未采纳的方向。

## 上游 FermiNet 文档

基于 [google-deepmind/ferminet](https://github.com/google-deepmind/ferminet) 的 `e9d5a604`（DeepMind），采用 [Apache-2.0 许可证](LICENSE)。
原始 README 保留在 [README.upstream.md](README.upstream.md)。优化流程遵循 [ResearchBoost-Agent](https://github.com/DAGroup-PKU/ResearchBoost-Agent)；
本仓库是独立的案例研究，不代表官方认证或背书。
