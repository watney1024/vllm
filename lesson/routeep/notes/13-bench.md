# L13 服务压测、性能分析与最终验收

> 对应课程 Module 6（第 13 课）。代码：`routeep/bench.py`。

## 1. 本节要解决什么

1. 封装 OpenAI 兼容服务入口；
2. 大规模压测与稳定性验证（教程基线 17038 请求、0 失败）；
3. 采集 Prefill / Decode 指标并复盘；
4. 用 profiling 定位通信/计算/显存瓶颈。

**一句话结论**：压测分三层看——稳定性（失败率）、单步延迟（Prefill/Decode）、吞吐（token/s）；瓶颈定位靠 timeline 区分通信等待、GEMM 计算与显存拷贝，而不是只看端到端数字。

## 2. 服务封装

- 推理入口：Runner 前向 → 采样 → 回包；
- OpenAI 兼容：`/v1/chat/completions`、`/v1/completions` 请求字段映射到内部 prompt/参数；
- 压测客户端：并发发送请求，记录排队、首 token、生成完成时间；
- 本项目先提供 Runner 级前向与指标采集（`bench.py`），vLLM server 方法注册在集成阶段补齐。

## 3. 压测指标

| 指标 | 定义 | 关注点 |
|---|---|---|
| 失败率 | failed / total | 稳定性验收：0 失败 |
| Prefill 延迟 | 长 prompt 首步前向耗时 | 大 batch 吞吐 |
| Decode 步延迟 | 单 token 前向耗时（P50/P99） | 尾延迟 |
| Decode 吞吐 | 1000 / 步均毫秒（token/s） | 解码效率 |
| 生成 token 数 | 累计输出 token | 吞吐归一化 |

教程基线目标（待实测复核）：Decode 约 3.1×、Prefill 约 23% 提升，对照原生 NCCL / DeepEP 后端。

## 4. 稳定性压测流程

```text
固定模型与并行拓扑
  → 批量请求（17038），混合输入长度/输出长度
  → 记录每请求 ok/fail、prefill、decode 步耗时、token 数
  → 失败率必须为 0；统计 P50/P99 与吞吐分布
  → 异常请求留存输入与栈，定位显存/通信/竞态问题
```

`run_stability` 逐条执行，任意异常计入失败；`BenchResult` 聚合完成数、失败数、延迟分布。

## 5. 瓶颈定位

| 瓶颈类型 | 特征 | 工具/动作 |
|---|---|---|
| 通信等待 | GEMM 前有明显空档，NVLink/RDMA 流量高 | timeline 看通信算子与计算重叠；对比 all2all 后端 |
| 计算受限 | SM 占用高，GEMM 时间占比大 | 检查 tiling、tensor core、分组粒度 |
| 显存带宽 | 权重/激活搬运时间高，计算空档 | 融合算子、减少中间张量、布局优化 |
| 显存容量 | OOM、频繁换入换出 | 专家分片、量化、缓冲分桶 |

工具：torch profiler、NCCL 日志、`nsys` timeline；按 Prefill/Decode 分别统计。

## 6. 复盘模板

1. 环境：GPU 型号/数量、拓扑（TP×DP×EP）、模型、dtype；
2. 配置：后端、桶容量、并行度、专家放置策略；
3. 结果：失败率、Prefill/Decode 均值与 P99、吞吐、对比基线；
4. 瓶颈：通信/计算/显存占比与证据；
5. 优化项：已做融合/分桶/重叠，待办与预期收益；
6. 未决问题：边界 case、兼容性、遗留误差。

## 7. 自检

1. 稳定性验收的硬性指标是什么？
2. Prefill 与 Decode 分别该优化什么指标？
3. 如何区分通信瓶颈与计算瓶颈？
4. 压测报告必须记录哪些环境与配置信息？
