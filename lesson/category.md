# 课程目录

基于 vLLM main 分支（2026-11，`main@8c17e46`）修订，共 20 个课时，覆盖部署、调优与源码分析。
v0 引擎已移除，课程中的源码分析均以 `vllm/v1/` 为准。

> 说明：标「待补」的小节目前只有概述或几段说明，正文尚未展开；其余小节可作为主线阅读。

## 课时01 vLLM 快速上手与环境搭建
- 安装与环境配置（uv、预编译 wheel、源码编译）
- 离线推理部署（LLM 类、EngineArgs 与常用参数）
- 在线推理服务部署与调用（`vllm serve`、OpenAI 兼容接口、gRPC）
- 离线与在线推理性能测试（`vllm bench latency/throughput/serve/sweep`）
- 配置系统与调优入口（VllmConfig 及子配置）
- 总结

## 课时02 vLLM 架构总览与请求生命周期
- vLLM v1 分层架构（entrypoints → EngineClient → EngineCore → Executor → Worker → ModelRunner）
- 核心类与进程拓扑（AsyncLLM、EngineCoreProc、Worker、GPUModelRunner）
- 一次 generate 的端到端时序（add_request → schedule → execute_model → sample → output）
- 引擎边界与前端形态（Python API、OpenAI/gRPC 服务、Rust 前端）
- 源码阅读方法与调试入口（日志、metrics、profiler）
- 总结

## 课时03 vLLM 核心组件：EngineCore 与流式执行
- ZMQ 通信模式与 vLLM 的选择（DEALER/ROUTER、PUSH/PULL、PAIR，对比 REQ/REP）
- 客户端实现：InprocClient、SyncMPClient、AsyncMPClient
- EngineCore 接收请求与 EngineCoreOutputs 回流
- AsyncLLM 的请求队列与流式输出（OutputProcessor、Detokenizer）
- DP 客户端：DPAsyncMPClient 与 DPLBAsyncMPClient
- 总结

## 课时04 vLLM 核心组件：Worker / Executor 详解及协作
- WorkerBase 与 GPU Worker 初始化（init_device、determine_available_memory、initialize_from_config、warmup）
- UniProcExecutor、MultiprocExecutor、RayExecutorV2
- WorkerWrapperBase 与 WorkerProc 的 RPC 过程
- Executor 与 Worker 协作：execute_model、sample_tokens、profile_run
- 启动计划与共享内存消息队列（startup_plan、MessageQueue）（待补）
- 总结

## 课时05 vLLM 核心组件：调度器如何实现
- Scheduler 数据结构与 SchedulerOutput
- 请求入队与 waiting 队列调度
- running 队列调度与 chunked prefill
- 前缀缓存命中与 token budget
- 抢占与重算
- AsyncScheduler 与异步调度
- 调度后处理 update_from_output
- 总结

## 课时06 vLLM 核心组件：KV Cache 与 PagedAttention 显存管理
- 传统 KV cache 策略及其缺陷
- 分页显存管理与 BlockPool
- 块分配、释放与哈希前缀缓存
- KVCacheManager 的 get_computed_blocks / allocate_slots / free
- Hybrid KV Cache：Full / SlidingWindow / Mamba / CrossAttention
- KVCacheCoordinator 与显存规划
- KV cache 指标与调优（block_size、gpu_memory_utilization、kv_cache_dtype）（待补）
- 总结

## 课时07 vLLM 核心组件：ModelRunner 与模型接入
- GPUModelRunner 初始化与状态管理（RequestState、InputBatch、InputBuffers）
- initialize_model 与 ModelRegistry
- 模型权重加载流程与自定义 loader
- 让 vLLM 支持一个新模型：注册、权重映射与 dummy run
- prepare_inputs：输入张量与 attention metadata 构建
- execute_model 与 sample_tokens
- 总结

## 课时08 vLLM 核心组件：Attention 后端与内核实现
- AttentionBackend / MetadataBuilder / Impl 三段式设计
- 后端注册与选择（FlashAttention、FlashInfer、Triton、ROCm）
- KV cache layout 与后端约束
- MLA 与稀疏 Attention（indexer、sparse MLA）
- Decode Context Parallel（待补）
- 后端调试与选型（待补）
- 总结

## 课时09 vLLM 高级特性：CUDA Graph 详解
- CUDA Graph 原理与使用约束
- CUDAGraphWrapper 与捕获流程
- CUDAGraphMode：NONE / PIECEWISE / FULL / FULL_AND_PIECEWISE
- 捕获尺寸、显存开销与多模态场景
- 常见问题与调试
- 总结

## 课时10 vLLM 高级特性：torch.compile 与编译栈
- torch.compile 基础（Dynamo / Inductor / FX Graph）
- VllmBackend 与 CompilerManager
- 分段编译（Piecewise Compilation）
- 编译 pass、融合与 custom op
- 优化等级 -O0 ~ -O3 与编译缓存
- 编译调试
- 总结

## 课时11 GPU 算子开发：Triton / CuTe / TileLang
- Attention 算子成为瓶颈的原因
- Triton 核心概念与编译过程
- 第一个 Triton 算子：向量加法与 Softmax
- Matmul 与 Attention 算子实现（待补）
- vLLM 的 Triton 代码组织（vllm/kernels/triton、triton_utils）
- CuTe DSL 与 TileLang 算子（待补）
- vLLM Triton Attention 后端剖析（待补）
- 总结

## 课时12 vLLM 量化：原理与实现
- 模型量化原理与数值格式（定点数、浮点数）
- 浮点量化改进（FP8、MXFP4、NVFP4）
- AWQ / GPTQ 等量化算法（待补）
- vLLM 量化配置与注册机制
- 量化线性层与 fused MoE 的源码（待补）
- 在线量化与 KV cache 量化
- 量化模型部署与精度/性能对比
- 总结

## 课时13 vLLM 分布式推理：数据并行
- DP 概述与适用场景
- DP 逻辑框架与 DPCoordinator
- DPEngineCoreProc 与请求分发
- DP + EP 混合部署
- DP 负载均衡（DPLBAsyncMPClient）
- 总结

## 课时14 vLLM 分布式推理：张量并行与流水线并行
- AllReduce 原理与通信后端
- Transformer 的张量并行切分
- vLLM 中的 TP（parallel_state、communication_op、custom allreduce）
- 流水线并行调度（PPHandler、pp_transport）
- TP/PP 组合与调优
- 总结

## 课时15 vLLM 分布式推理：专家并行
- MoE 概述
- AlltoAll 通信原理
- vLLM EP 架构与推理流程
- Dispatch-Compute-Combine 流程
- fused MoE 内核实现（待补）
- All2All 后端选型（DeepEP、NIXL、FlashInfer、Mori）
- 完整调用链总结

## 课时16 vLLM 分布式推理：EPLB 与弹性专家并行
- EPLB 模块概述
- 专家并行 + EPLB 参数与服务实例
- vLLM EPLB 设计方案
- 重平衡流程（eplb_state、rebalance_execute、async_worker）
- 弹性 EP 与专家替换
- 总结

## 课时17 vLLM 高级特性：投机解码
- Transformer 解码瓶颈
- 推测解码基本框架
- EAGLE、MTP、ngram、dflash、dspark
- Speculator 抽象与 draft 模型
- 拒绝采样与自适应验证
- vLLM 投机采样主流程
- 调优：接受率、草稿长度与开销
- 总结

## 课时18 vLLM PD 分离部署与 KV 传输
- PD 分离架构与收益
- KV Connector 抽象与配置
- NIXL（pull/push）与 Mooncake
- KV offloading（CPU / 文件 / 分层）与 HiSparse
- 端到端流程与异常处理（待补）
- 部署与调优
- 总结

## 课时19 vLLM 性能分析与瓶颈定位
- 性能模型：prefill、decode、通信与调度开销
- Torch Profiler 与 layerwise profile
- NVIDIA Nsight Systems
- Metrics 与日志（Prometheus、per-request metrics）
- `vllm bench` 压测与参数扫描
- 瓶颈定位案例
- 总结

## 课时20 执行引擎演进：异步调度、DBO 与 ModelRunner V2
- 异步调度与流水线深度
- Dual Batch Overlap 与 UBatch
- ModelRunner V2 设计（Persistent Batch 重构）
- 故障容忍、弹性伸缩与权重热更新
- 总结与后续学习路线

## 选修专题
- 多模态推理与 Encoder Cache
- Pooling / Embedding / 重排序模型
- 结构化输出、Reasoning 与 Tool Calling
- 扩散语言模型（dLLM）
- Rust 前端与 gRPC 服务

## 本版主要调整
- 原课时07“架构总览”前移为课时02，先建立全局调用链，再拆组件。
- 原课时06“模型加载”并入课时07“ModelRunner 与模型接入”。
- 原课时10/11“量化”合并为课时12，并补充 FP8 / MXFP4 / NVFP4 与在线量化。
- 新增课时08“Attention 后端与内核实现”；课时14 补充流水线并行；课时20 补充 DBO 与异步调度。
- 原课时19 的 P2pNcclConnector 已不存在，改为 NIXL / Mooncake / offloading / HiSparse 的现状。
- 原课时20 的 ModelRunner V2 当前只有设计文档（`docs/design/model_runner_v2.md`），改为与异步调度、DBO 合并的演进课时。
