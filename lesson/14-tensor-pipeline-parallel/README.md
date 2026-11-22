# 课时14 vLLM 分布式推理：张量并行与流水线并行

> 版本锚点：`main@8c17e46`。代码块标题是片段起点，省略处用 `...` 标注。

## 0. 本节要解决什么

1. TP 为什么在 attention/MLP 里切两刀；
2. AllReduce 在 TP 的哪个位置发生；
3. PP 如何把层分到多个 stage，气泡怎么填；
4. TP/PP 组合时进程和通信组怎么排。

**一句话结论**：TP 切“单层内的权重”，每层用 all-reduce 合并；PP 切“层与层之间”，stage 边界只传 hidden states。TP 降单请求延迟，PP 让单卡放不下的模型跨卡。

## 1. TP：列并行 + 行并行

Transformer 的 MLP 是 `Y = X A`。TP 把 A 按列切：

```text
A = [A_1, ..., A_p]
Y_i = X A_i        # 每个 rank 算一部分输出
```

这就是 `ColumnParallelLinear`（`layers/linear.py:447`）：

```python title="vllm/model_executor/layers/linear.py:447"
class ColumnParallelLinear(LinearBase):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """
```

紧接着的 `RowParallelLinear`（`linear.py:1722`）按行切：

```text
A = [A_1; ...; A_p]
Y = X_1 A_1 + ... + X_p A_p   # 需要 all-reduce
```

所以一个 MLP 的标准 TP 结构是：

```text
X -> ColumnParallel (gate/up) -> activation -> RowParallel (down) -> all-reduce
```

attention 类似：Q/K/V 按 head 切（`QKVParallelLinear`，`linear.py:1093`），输出投影是 RowParallel，最后 all-reduce。

## 2. 通信原语

`communication_op.py` 提供 TP 常用集合通信：

```python title="vllm/distributed/communication_op.py:12"
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)
```

vLLM 会根据硬件选择实现：

| 实现 | 场景 |
|---|---|
| NCCL | 通用 |
| `CustomAllreduce`（`custom_all_reduce.py:120`） | 单节点 NVLink，低延迟 |
| `QuickAllReduce`（`quick_all_reduce.py:46`） | ROCm |
| FlashInfer AllReduce | 特定平台融合 |

自定义 all-reduce 只适合小张量、单节点；跨节点或大张量还是走 NCCL。

## 3. 进程组

`initialize_model_parallel`（`parallel_state.py:2131`）创建 TP/PP/DP/EP/DCP/PCP 组。文档里的例子：

```text
8 张卡，TP=2，PP=4：
  TP 组：[g0,g1], [g2,g3], [g4,g5], [g6,g7]
  PP 组：[g0,g2,g4,g6], [g1,g3,g5,g7]
```

所以 `rank` 是一个多维坐标：`(dp_rank, pp_rank, tp_rank)`。模型层只感知 TP 组，PP 只影响 stage 划分。

## 4. PP：按层切 stage

PP 把模型的 layer 0..N 分成 `pp_size` 段：

```text
stage 0: embedding + layers 0..k
stage 1: layers k+1..2k
...
stage p-1: last layers + lm_head
```

每个 stage 只持有自己那部分权重。stage 之间传 `IntermediateTensors`（hidden states + 必要的残差）。

### 4.1 气泡与 batch queue

如果串行执行：

```text
stage0: [batch0]
stage1:          [batch0]
stage0:                   [batch1]
```

GPU 会有大量空闲。vLLM 用 `max_concurrent_batches` 个并发 batch 填 pipeline：

```python title="vllm/config/vllm.py:649"
@property
def max_concurrent_batches(self) -> int:
    # PP requires PP-size concurrent batches to fill the pipeline.
    # Async scheduling requires 2 concurrent batches to overlap.
    pp_size = self.parallel_config.pipeline_parallel_size
    if self.scheduler_config.async_scheduling:
        return pp_size + self.scheduler_config.async_scheduling_depth - 1
    return pp_size
```

EngineCore 在 `batch_queue` 里同时持有多个 `SchedulerOutput`，`step_with_batch_queue` 负责调度和执行重叠。

### 4.2 PPHandler

`PPHandler`（`pp_utils.py:57`）负责 stage 边界的通信：

- 非最后 rank：`send_positions` / `irecv_tensor_dict` 接收上一 stage 的 hidden states；
- 最后 rank：采样后把 token 广播回所有 stage；
- 用独立的 broadcast stream，避免和 hidden-state p2p 抢同一条 NCCL 通信。

```python title="vllm/v1/worker/gpu/pp_utils.py:57"
class PPHandler:
    """Runs the PP sampled-token broadcast/recv on a side stream so the
    default stream isn't gated by the matching peer call. ...
    """
```

PP 的传输默认走 NCCL；也可以用 `VLLM_PP_TRANSPORT=copy_engine` 切换（`vllm/distributed/pp_transport/`，ROCm 另有 `rocm_sdma` 实现）。

## 5. 部署形态

### 5.1 TP

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 2
```

### 5.2 PP

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --pipeline-parallel-size 2
```

### 5.3 TP + PP

```bash
vllm serve <model> --tensor-parallel-size 2 --pipeline-parallel-size 2
```

总卡数 = `TP × PP × DP`。例如 TP=2、PP=2、DP=2 需要 8 张卡。

## 6. 最小实验：TP=2

```bash
vllm serve facebook/opt-125m \
  --tensor-parallel-size 2 \
  --max-model-len 512 \
  --enforce-eager
```

观察：

- 日志里每个 rank 只加载一半的权重；
- `nvidia-smi` 两张卡都有显存占用；
- `vllm bench serve` 的 TPOT 可能比单卡低，但吞吐不一定翻倍（通信有开销）。

如果启动报 `head_size` / `num_heads` 不能被 TP 整除，换一个 head 数更友好的模型，或减小 TP。

## 7. 常见误区

1. **以为 TP 一定比单卡快**。小模型 + 高通信开销时，TP 可能更慢。
2. **以为 TP 会减少总显存**。权重和 KV 是分摊了，但每张卡还有 activation、通信 buffer。
3. **忽略 head 数约束**。Q head 数必须能被 TP 整除，否则要 padding。
4. **以为 PP 没有气泡**。batch queue 只能缓解，不能消除；PP 越大，气泡越明显。
5. **TP 和 PP 的通信组搞混**。TP 组内是 all-reduce，PP 组内是 send/recv。

## 8. 自检 5 问

1. ColumnParallel 和 RowParallel 分别切哪个维度？
2. TP 的 all-reduce 通常发生在哪一层之后？
3. PP 的 stage 之间传什么？
4. `max_concurrent_batches` 为什么至少等于 `pp_size`？
5. TP 和 PP 分别适合什么场景？

参考答案：

1. Column 切输出维（A 的列），Row 切输入维（A 的行）。
2. RowParallel 的输出投影之后（以及 attention 的 out_proj 之后）。
3. `IntermediateTensors`（hidden states + 残差等）。
4. 每个 stage 要同时处理不同 batch，才能填满 pipeline。
5. TP 适合单卡放不下、要低延迟；PP 适合超深层模型、跨节点。

## 9. 延伸阅读

- `vllm/distributed/parallel_state.py`；
- `vllm/model_executor/layers/linear.py`；
- `vllm/v1/worker/gpu/pp_utils.py`；
- 下一课：课时15 进入专家并行。
