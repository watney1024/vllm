# L11 CUDA Graph 捕获、分桶与多 EP 拓扑扩展

> 对应课程 Module 5（第 11 课）。代码：`routeep/graph/bucket_graph.py`、`routeep/parallel.py`。

## 1. 本节要解决什么

1. decode 阶段为什么需要 CUDA Graph，Graph 捕获/回放的生命周期；
2. 按专家容量/token 数分桶，规避动态 shape 触发重复捕获；
3. EP=1/2/4、TP×DP 拓扑与专家放置策略；
4. 多拓扑下代码与配置如何组织。

**一句话结论**：CUDA Graph 把前向算子序列录制成一张图，回放时省去逐 kernel 启动开销；decode 每步 shape 抖动，用容量分桶把相近 shape 归入同一静态图，超桶再升级，避免每步重捕获。

## 2. Graph 生命周期

```text
warmup（eager 跑通，初始化内存/缓存）
  → capture（torch.cuda.graph 内执行一次前向，记录算子与依赖）
  → replay（静态缓冲拷入新输入，graph.replay()）
```

要点：

- 捕获前必须完成预热，避免懒初始化、autotune 分支被录进图；
- 输入/输出走静态缓冲，回放只拷数据，不重新分配；
- 图内不能有依赖动态 shape 的分支；shape 变化必须换桶或重捕获。

## 3. 分桶策略

- 桶容量上界列表（如 `[1, 8, 32, 128, 512]`）；
- 当前 token 数落入第一个容量 ≥ 它的桶；超出最后一桶则留在最后一桶；
- 每个桶独立静态缓冲与 Graph；
- 桶边界按 decode 常见 batch/序列长度分布设定，桶太少回放利用率低，太多捕获成本高。

本项目 `BucketedGraphManager`：`capture` 预热后录图，`replay` 拷入静态缓冲回放，`run` 在无 CUDA 或未捕获时回退 eager。

## 4. 多拓扑扩展

### 4.1 并行维度

| 维度 | 含义 | 本项目配置 |
|---|---|---|
| TP | 专家权重/隐状态维切分 | `ParallelTopology.tp_size` |
| DP | token 批次并行 | `dp_size`，`world = TP × DP` |
| EP | 专家跨 rank 分布 | `ep_size ≤ dp_size`，专家数整除 |

### 4.2 专家放置

- `linear`：连续分片，邻接专家同 rank；
- `round_robin`：专家 e → rank `e % EP`，热点打散；
- 映射表 `expert_map`：全局 id → 本地下标，非本地为 -1。

### 4.3 拓扑校验

`ParallelTopology.validate(num_experts)` 校验：并行度 ≥1、`ep ≤ dp`、专家数整除 EP；EP 组 DP rank 列表由 `ep_group_dp_ranks` 给出。

## 5. 待 GPU 验证项

- 各桶 warmup/capture/replay 数值与 eager 一致；
- decode 多步连续回放，静态缓冲无串值；
- EP=1/2/4 下专家分布与映射正确；
- Graph 回放相比 eager 的 decode 步时收益；
- TP×DP×EP 组合下通信组划分与 rank 映射。

## 6. 常见误区

1. 捕获时包含懒初始化，回放行为不确定；
2. 直接用动态 shape 回放已捕获的图，越界或串数据；
3. 桶划分过粗，静态缓冲浪费；过细则图数量膨胀；
4. 误以为 EP 组跨 TP：EP 在 DP 维内划分。

## 7. 自检

1. Graph 为什么要先 warmup 再 capture？
2. 分桶解决的核心问题是什么？
3. `ep_size` 与 `dp_size` 的约束关系？
4. linear 与 round_robin 放置分别适合什么负载？
