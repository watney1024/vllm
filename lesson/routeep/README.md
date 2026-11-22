# RouteEP：从零手写 MoE 专家并行后端

> 课程大纲：`lesson/MoE专家并行实现.md`（13 课 / 7 Module）。
> 源码锚点：vLLM `main@8c17e46`（`git rev-parse --short HEAD` = `8c17e4692a`）。
> 定位：工程实战——手写一套可接入 vLLM 的 EP 库，覆盖路由、All-to-All、Triton 算子、Runner 集成、CUDA Graph、对齐压测。

## 0. 环境状态

| 环境 | GPU | torch / vllm / Triton / NCCL | 文件系统 | 可执行内容 |
|---|---|---|---|---|
| 本机 `client:1072024` (linux) | ❌ 无 `nvidia-smi` | ❌ 未安装（Python 3.14，无 `.venv`） | 共享本仓库 `/home/watneyzhu/infra/vllm` | 源码走读、纯 Python 逻辑、CPU 参考实现与单测框架 |
| 云主机 `vm` (linux) | ❌ 无 | ❌ 未安装 | 独立，不共享本仓库 | 暂不使用 |
| 其他 GPU 机器（待接入） | ✅ 有（教程目标 4×H100） | 待安装/核对 | 需同步本目录 | NCCL 多 rank、Triton kernel、CUDA Graph、真实模型压测 |

**执行策略**：GPU 代码按可运行标准写完整实现（不写空壳）；所有依赖 CUDA 的入口在无 GPU 时跳过并给出明确报错；CPU 可验证的逻辑（路由、元数据、通信模拟、数值对齐框架）在本机用参考实现跑通。

## 1. 目录约定

```text
lesson/routeep/
├── README.md                  # 本文件：环境台账 + 进度跟踪 + 待跑清单
├── notes/                     # 每课原理调研笔记（含 vLLM 源码锚点）
├── routeep/                   # 手写 EP 库源码
│   ├── config.py              # EP 配置（num_experts/top_k/hidden/EP 拓扑等）
│   ├── router.py             # L2：Router + TopK + 权重重归一化
│   ├── metadata.py           # L3：token 重排、去重、dispatch 索引/位置表
│   ├── comm/                  # L4-5：All-to-All 通信层
│   ├── kernels/               # L6-8：Triton dispatch/combine、grouped GEMM
│   ├── experts/               # 专家 MLP（SwiGLU）
│   ├── moe_runner.py          # L10：自定义 MoE Runner，对接 vLLM 接口
│   └── graph/                 # L11：CUDA Graph 分桶与 replay
└── tests/                     # 单测：CPU 参考对齐 + GPU 待跑用例标注
```

## 2. 阶段进度台账

| Stage | 课时 | 交付物 | 本机状态 | GPU 待跑 |
|---|---|---|---|---|
| 0 | L1 基线调研 | `notes/01-background.md`、本台账 | ✅ 本机完成 | — |
| 1 | L2-3 Router & 调度元数据 | `router.py`、`metadata.py`、`tests/test_router.py`、`tests/test_metadata.py` | 🟡 代码完成待 torch 验证 | — |
| 2 | L4-5 All-to-All | `comm/base.py`、`comm/nccl_a2a.py`、`comm/sim_a2a.py`、`comm/placement.py`、`comm/grouping.py` | 🟡 代码完成待 torch 验证 | 多 rank NCCL 固定/变长 A2A |
| 3 | L6-8 Triton 算子 | `kernels/dispatch_combine.py`、`kernels/grouped_gemm.py`、`experts/mlp.py` | 🟡 代码完成待 torch/CUDA 验证 | kernel 编译、数值对齐、精度/边界 case |
| 4 | L9-10 接入 vLLM | `notes/09-vllm-path.md`、`moe_runner.py`、`local_plan.py` | 🟡 代码完成待 torch/CUDA 验证 | 单轮 prefill/decode 跑通 |
| 5 | L11 Graph & 拓扑 | `graph/bucket_graph.py`、`parallel.py`、拓扑/专家放置配置 | 🟡 代码完成待 CUDA 验证 | Graph warmup/capture/replay、EP=1/2/4 |
| 6 | L12-13 对齐压测 | `alignment.py`、`bench.py`、`examples/smoke_sim.py`、对齐/压测笔记 | 🟡 代码完成待 torch/CUDA 验证 | 逐层 hidden state 对齐、17038 请求稳定性压测 |

状态标记：⬜ 待开始 / 🔵 进行中 / ✅ 本机完成 / 🟡 代码完成待 GPU 验证。

## 3. GPU 机器待跑清单（移交时逐项勾选）

- [ ] 环境：CUDA、torch（匹配 CUDA 版本）、Triton、NCCL、可编辑安装本仓库 vLLM；准备小尺寸 MoE 模型（先随机权重/小模型，再上 Qwen3 MoE）。
- [ ] L4-5：`torchrun` 多 rank 跑固定容量 A2A；变长 A2A 前缀元数据同步；带宽/延迟 profile，对比原生 NCCL all2all。
- [ ] L6-8：Triton dispatch/combine、grouped GEMM + SwiGLU 编译运行；与 eager 参考逐 token 数值对齐；溢出/边界 case。
- [ ] L10：自定义 Runner 单轮 prefill / decode；对照 vLLM 原生 MoE 输出。
- [ ] L11：CUDA Graph warmup/capture/replay；容量分桶；EP=1/2/4、TP×DP 拓扑与专家放置。
- [ ] L12：固定输入逐层对比 hidden state，定位路由/通信/GEMM 差异。
- [ ] L13：OpenAI 兼容服务；大规模压测 0 失败；Prefill/Decode 指标复盘；profiling 定位瓶颈。

## 4. 开发约束

- Python 命令一律走 `uv` + 项目 `.venv`（见仓库 `AGENTS.md`）；不在本机用系统 `python3` / `pip` 装包。
- 算子代码保持"eager 参考实现 ↔ Triton/CUDA 实现"双份，测试先对参考实现对齐。
- 所有 GPU 入口必须在函数内惰性 import `torch.cuda` / triton / dist 组件，保证无 GPU 环境导入本包不崩。
- 不修改 vLLM 源码树内既有文件；对接点通过 `routeep` 包内的适配层完成。
