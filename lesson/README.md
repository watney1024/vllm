# vLLM 源码教程

> 版本锚点：`main@8c17e46`（2026-11）。20 个课时，覆盖部署、调优与源码分析。
> 每章统一结构：本节要解决什么 → 主干讲解（数据结构 / 调用链 / 源码）→ 最小实验 → 常见误区 → 自检 5 问 → 延伸阅读。
> 实验默认需要一张能跑 vLLM 的 NVIDIA GPU（用 `facebook/opt-125m`，约 250 MB）；没有 GPU 时只能读源码，带 `vllm serve` / `LLM` 的步骤不可执行。

| 课时 | 主题 | 源码主线 |
|---|---|---|
| [01](01-quickstart/README.md) | 快速上手与环境搭建 | `LLM`、`vllm serve`、`vllm bench` |
| [02](02-architecture/README.md) | 架构总览与请求生命周期 | `LLMEngine` → `EngineCore` → `ModelRunner` |
| [03](03-engine-ipc/README.md) | EngineCore 与流式执行 | ZMQ、`AsyncLLM`、`OutputProcessor` |
| [04](04-worker-executor/README.md) | Worker / Executor 协作 | `MultiprocExecutor`、`WorkerProc`、RPC |
| [05](05-scheduler/README.md) | 调度器 | `Scheduler.schedule`、`update_from_output` |
| [06](06-kv-cache/README.md) | KV Cache 与 PagedAttention | `BlockPool`、`KVCacheManager` |
| [07](07-model-runner/README.md) | ModelRunner 与模型接入 | `GPUModelRunner`、`ModelRegistry` |
| [08](08-attention-backends/README.md) | Attention 后端与内核 | `AttentionBackend`、`get_attn_backend` |
| [09](09-cuda-graph/README.md) | CUDA Graph | `CUDAGraphMode`、`CudaGraphManager` |
| [10](10-compilation/README.md) | torch.compile 编译栈 | `VllmBackend`、piecewise |
| [11](11-gpu-kernels/README.md) | GPU 算子开发 | Triton / CuTe / TileLang |
| [12](12-quantization/README.md) | 量化原理与实现 | `QuantizationConfig`、FP8/MXFP4 |
| [13](13-data-parallel/README.md) | 数据并行 | `DPCoordinator`、`DPEngineCoreProc` |
| [14](14-tensor-pipeline-parallel/README.md) | 张量并行与流水线并行 | `ColumnParallelLinear`、`PPHandler` |
| [15](15-expert-parallel/README.md) | 专家并行 | `FusedMoEFactory`、All2All |
| [16](16-eplb/README.md) | EPLB 与弹性专家并行 | `EplbState`、`rearrange` |
| [17](17-speculative-decoding/README.md) | 投机解码 | Speculator、RejectionSampler |
| [18](18-pd-disaggregation/README.md) | PD 分离与 KV 传输 | `KVConnectorBase_V1`、NIXL |
| [19](19-profiling/README.md) | 性能分析与瓶颈定位 | Torch Profiler、nsys、metrics |
| [20](20-mrv2-async-dbo/README.md) | 异步调度、DBO 与 MRV2 | `AsyncScheduler`、`UBatchRunner` |

## 目录约定

- `category.md`：课程大纲；
- `0X-*/README.md`：课时正文；
- `02-architecture/figures/`：样章配图（5 张正文图 + 1 张封面，Seedream 5.5 草稿）及 `PROMPTS.md`；
- `check_anchors.py`：校验正文里的 `文件:行号` 锚点是否仍然存在；换分支或 rebase 后先跑 `python lesson/check_anchors.py`；
- 代码块标题（如 `vllm/v1/engine/core.py:714`）是片段起点，换分支后需要重新核对行号。
