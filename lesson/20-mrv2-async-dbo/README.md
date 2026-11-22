# 课时20 执行引擎演进：异步调度、DBO 与 ModelRunner V2

> 版本锚点：`main@8c17e46`。代码块标题是片段起点，省略处用 `...` 标注。
>
> 注意：ModelRunner V2 已经落地为当前默认的 `GPUModelRunner`（`vllm/v1/worker/gpu/model_runner.py`，旧的 `vllm/v1/worker/gpu_model_runner.py` 已删除），但 `docs/design/model_runner_v2.md` 仍标注它尚未 feature-complete。本文把 MRV2 与异步调度、DBO 放在一起讲，区分“已落地”和“仍在演进”。

## 0. 本节要解决什么

1. 异步调度改变了什么；
2. DBO（Dual Batch Overlap）如何把通信藏到计算后面；
3. ModelRunner V2 为什么要重构 persistent batch；
4. 故障容忍 / 弹性伸缩 / 权重热更新各自解决什么问题。

**一句话结论**：这三项优化都在解决同一个矛盾——GPU 越来越快，但 CPU 调度、批处理重组和通信开始成为瓶颈；异步调度把调度提前，DBO 把通信重叠，MRV2 把状态管理和输入构建解耦。

## 1. 异步调度（Async Scheduling）

默认调度是“同步”的：一步 forward 完成、拿到采样结果，再调度下一步。

异步调度允许在上一步 forward 还没结束时，先调度下一步：

```text
同步：  [schedule][forward][sample][update][schedule][forward]...
异步：  [schedule][forward][sample][update]
                 [schedule][forward][sample][update]
```

代价是 token 的“可见性”滞后：调度器必须用占位符表示“已派发但还没拿到结果”的 token。

```python title="vllm/v1/core/sched/async_scheduler.py:18"
def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
    super()._update_after_schedule(scheduler_output)
    ...
    request.num_output_placeholders += (
        self.num_sampled_tokens_per_step + cur_num_spec_tokens
    )
    request.num_output_placeholder_steps += 1
    request.next_decode_eligible_step = self.current_step + self.decode_stagger
```

相关配置：

- `SchedulerConfig.async_scheduling`（`vllm/config/scheduler.py:270`）；
- `async_scheduling_depth`（默认 2）：允许提前多少步；
- `max_concurrent_batches` 会随之增加。

## 2. DBO：Dual Batch Overlap

DBO 的目标：把 MoE 的 all-to-all 通信和前后计算重叠。典型场景是 **DP + EP**。

```text
Comp: |-A0₀-A1₀-||-MLP₁-||-S₁-MLP₀-||-S₀-A0₁-A1₁-|
Comm: |----D₁---||--D₀--||----C₁---||-----C₀-----|
```

（图来自 `docs/design/dbo.md`：S=shared expert，A=attention 投影，D=dispatch，C=combine。）

做法是把一个 batch 切成两个 **ubatch**：

- ubatch 0 做计算时，ubatch 1 做 dispatch；
- ubatch 1 做计算时，ubatch 0 做 combine；
- 两个 ubatch 交替，通信被计算覆盖。

相关代码：

- `vllm/v1/worker/ubatching.py:21` `UBatchContext`：每个 ubatch 的 stream、事件、状态；
- `vllm/v1/worker/gpu/ubatch_utils.py:308` `UBatchRunner`：执行两个 ubatch；
- `vllm/v1/worker/gpu/model_runner.py` 里的 `dispatch_cg_and_sync_dp` / `ubatch_runner`。

配置（`ParallelConfig`）：

```python
enable_dbo: bool = False
dbo_decode_token_threshold: int = 32
dbo_prefill_token_threshold: int = 512
dbo_schedule: DBOSchedule | None = None
```

DBO 的限制：

- 需要 FULL CUDA Graph（`cudagraph_mode=FULL` 或 `FULL_AND_PIECEWISE`）；
- 主要面向 DP + EP 的 MoE；
- 小 batch 下切 ubatch 的收益有限。

## 3. ModelRunner V2

### 3.1 V1 的问题

V1 的 persistent batch 把“常驻状态”和“每步输入”混在一起：

- 常驻张量直接作为模型输入，行顺序被固定；
- 请求加入/退出要重排整行；
- 需要 `CachedRequestState` 备份，因为行可能被覆盖；
- 异步调度下 bookkeeping 更复杂。

### 3.2 MRV2 的方案

`docs/design/model_runner_v2.md` 的核心：

1. 预分配 `max_num_reqs` 行的状态张量；
2. 每个请求在生命周期内占一个**永久行号**；
3. 每步根据请求顺序 **gather** 出输入张量；
4. 抢占视为结束，恢复时重新加入；
5. 去掉 `CachedRequestState`。

```text
V1:  persistent state == model input，加入/退出要搬行
V2:  persistent state + per-step gather，加入/退出只改映射
```

当前 `GPUModelRunner` 走的就是 MRV2 这一套：

- `vllm/v1/worker/gpu/states.py` 的 `RequestState`；
- `vllm/v1/worker/gpu/input_batch.py` 的 `InputBatch` / `InputBuffers`；
- `GPUModelRunner.add_requests` / `update_requests` / `gather_batch_req_state`。

设计文档标注的“尚未 feature-complete”，指的是少数边缘能力（多模态 encoder、部分并行/投机组合）仍在补齐，不是“主路径还在 V1”。读代码时以 `vllm/v1/worker/gpu/` 为准。

## 4. 故障容忍与弹性

### 4.1 故障容忍

`FaultToleranceConfig`（`vllm/config/fault_tolerance.py:13`）和 `EngineCoreSentinel`（`vllm/v1/fault_tolerance/engine_core_sentinel.py:55`）负责：

- 监控 EngineCore / Worker 是否存活；
- 失败时触发回调，返回明确错误而不是挂死；
- 与 KV connector 配合，保住已经传输的 KV output。

### 4.2 弹性伸缩

`vllm/distributed/elastic_ep/` 支持 EP rank 的动态增减：

- `ElasticEPScalingExecutor`：执行扩缩容；
- `ElasticEPScalingState`：状态机；
- `entrypoints/serve/elastic_ep`：服务层接口。

### 4.3 权重热更新

`vllm/distributed/weight_transfer/` 支持把新权重从训练进程传到推理进程（RL / 在线更新场景），包括 NCCL、IPC、ModelExpress 等后端。

### 4.4 sleep mode

`EngineCore.sleep()` / `wake_up()`（`core.py:1103` / `:1156`）可以把模型权重卸载到 CPU 或丢弃，释放 GPU 给其他任务；醒来时重新加载。适合 RL 训练和推理共卡。

## 5. 最小实验：打开异步调度

```bash
vllm serve facebook/opt-125m \
  --max-model-len 512 \
  --async-scheduling
```

对比默认同步调度的 `vllm bench serve` 结果，重点看：

- TPOT 是否下降；
- waiting 队列是否更平滑；
- 输出是否仍然确定（temperature=0 时）。

如果日志里出现 placeholder / stale output 相关警告，通常是 batch 里有请求被 abort 或 preempt，属于异步调度的边界情况。

DBO 需要 DP + EP + MoE，建议用课时15 的 Granite MoE 示例再加：

```bash
vllm serve ibm-granite/granite-3.0-1b-a400m \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --enable-dbo \
  -cc.cudagraph_mode=FULL
```

（具体 flag 以当前 `vllm serve --help` 为准。）

## 6. 常见误区

1. **以为异步调度免费**。它增加了停止条件、投机回滚、KV 释放的复杂度。
2. **以为 DBO 适用于所有模型**。它主要面向 DP + EP 的 MoE，且依赖 FULL CUDA Graph。
3. **以为 MRV2 还是“设计稿”**。它已经是默认路径；但设计文档标注尚未 feature-complete，少数边缘能力仍在补。
4. **把故障容忍当成“自动重启”**。它更多是检测 + 明确失败 + 资源清理。
5. **忽略 sleep mode 的恢复成本**。wake_up 要重新加载权重，延迟不低。

## 7. 自检 5 问

1. 异步调度用哪个字段表示“已派发未完成”的 token？
2. DBO 把哪两个阶段重叠？
3. MRV2 为什么要去掉 `CachedRequestState`？
4. 弹性 EP 和 EPLB 的区别是什么？
5. sleep mode 适合什么场景？

参考答案：

1. `num_output_placeholders` / `num_in_flight_tokens`。
2. 一个 ubatch 的计算与另一个 ubatch 的 dispatch/combine。
3. 因为状态和输入解耦后，行不会被覆盖，不需要备份。
4. 弹性 EP 改 rank 数量；EPLB 改 expert 分布。
5. RL 训练/推理共卡、需要临时释放 GPU 的场景。

## 8. 课程总结

到这里，20 节课走完了 vLLM v1 的主干：

```text
入口 -> EngineCore -> Scheduler -> KV Cache -> ModelRunner -> Attention
     -> CUDA Graph / compile -> kernels -> quantization
     -> DP / TP / PP / EP / EPLB -> spec decode -> PD -> profiling
```

建议的复习方式：

1. 用课时02 的 trace 脚本再跑一次，把每节课的名字标在调用链上；
2. 选一个你关心的优化（prefix caching / CUDA Graph / spec decode / PD），沿源码走一遍；
3. 用 `vllm bench` 做一次 A/B，验证你的理解。

## 9. 延伸阅读

- `docs/design/model_runner_v2.md`；
- `docs/design/dbo.md`；
- `docs/design/optimization_levels.md`；
- `vllm/v1/worker/ubatching.py`；
- `vllm/v1/fault_tolerance/`。
