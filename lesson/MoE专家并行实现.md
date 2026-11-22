# RouteEP 课程：从零写一个 MoE 专家并行后端（vLLM 实战，共13课）
定位：工程实战课，目标是**手写一套可接入vLLM的EP库**，从原理、通信、Kernel、集成到压测对齐，适合AI Infra。
整体思路：先建立MoE基础认知 → 单机MoE路由与算子 → 分布式All-to-All通信 → Triton算子优化 → 接入vLLM框架 → CUDA Graph、拓扑扩展 → 对齐验证、压测调优、复盘。
> 前置：vLLM基础、CUDA/Triton基础、NCCL基础

## Module 0 课程开篇 & 基线调研（第1课）
01. MoE 专家并行背景、DeepEP 原理与性能基线
    - MoE 并行范式：TP/DP/EP 区别，什么时候选EP
    - DeepEP 架构拆解、它的优势与局限
    - 本项目目标：在4×H100上实现、对齐DeepEP、Prefill/Decode性能指标
    - 环境搭建：vLLM、NCCL、Triton、Qwen3.5 MoE 模型准备

## Module 1 MoE 路由层：Router & Top-K 调度（第2–3课）
02. MoE Router 原理、Top-K 路由实现
    - 路由权重、softmax、TopK选择、路由偏差
    - Token-to-expert 映射、sort/scatter 基础逻辑
03. Token重排、去重、Dispatch元数据构造
    - 连续token重排、专家负载mask
    - 构造Dispatch需要的索引表、位置表（路由元数据）
    - 路由正确性单元测试

## Module 2 分布式通信：All-to-All（EP核心）（第4–5课）
04. NCCL All-to-All 基础：固定容量通信
    - NCCL原语、rank/device拓扑、同步语义
    - 固定大小All-to-All实现、带宽延迟分析
05. 变长All-to-All、TMA直连、Peer通信优化
    - 变长消息分块、前缀元数据同步
    - H100 TMA、Peer direct 通信优化
    - 通信性能profile，和原生NCCL对比

## Module 3 Triton Kernel：Dispatch / Combine & Grouped GEMM（第6–8课）
06. Dispatch、Combine Triton Kernel 手写实现
    - Scatter/Gather 算子，数据拷贝、bank冲突优化
    - 融合Dispatch，减少中间显存读写
07. Grouped GEMM + SwiGLU Triton Kernel
    - 分组GEMM（不同专家独立GEMM）
    - SwiGLU融合，消除多余读写
    - 访存流水线、分块tiling调优
08. Kernel 正确性验证、数值误差排查
    - 和PyTorch原生实现做数值对齐
    - 精度、溢出、边界case测试

## Module 4 集成vLLM：替换MoE Runner（第9–10课）
09. vLLM MoE执行路径源码阅读
    - vLLM原有MoE Runner、MoE层调用链路
    - 输入输出tensor布局、元数据格式
10. 接入RouteEP：自定义MoE Runner替换
    - 封装我们的Router+Dispatch+AlltoAll+Expert+Combine
    - 跑通单轮prefill/decode，基础推理验证

## Module 5 高级优化：CUDA Graph、多拓扑扩展（第11课）
11. CUDA Graph捕获、分桶Graph、Decode加速
    - Graph warmup、capture、replay
    - 不同专家容量分桶策略，规避Graph重捕获
    - 多EP拓扑扩展：EP=1/2/4，TP×DP拓扑、专家放置策略

## Module 6 对齐、压测、性能调优（第12–13课）
12. 端到端对齐：Qwen3 / Qwen3.5 MoE 和 DeepEP 逐Token对齐
    - 输入固定、逐层对比hidden state、输出token
    - 定位路由、通信、GEMM引入的差异bug
13. 服务压测、性能分析、最终验收
    - OpenAI兼容服务封装
    - 大规模压测：17038请求，稳定性验证（0失败）
    - Prefill / Decode 指标复盘：3.1×Decode、23% Prefill提升
    - 瓶颈定位：通信/计算/显存、profiling工具使用

---

## 配套课程总结（放在最后一页）
✅ 学完可以独立完成：
1. 手写MoE路由、TopK、token重排
2. 变长All-to-All通信，理解NCCL/TMA
3. Triton Grouped GEMM + Dispatch/Combine融合算子
4. 在vLLM里替换MoE后端、接入CUDA Graph
5. 设计EP拓扑、做MoE服务稳定性压测、性能对比