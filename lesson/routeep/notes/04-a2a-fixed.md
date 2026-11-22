# L4 NCCL All-to-All 基础：固定容量通信

> 对应课程 Module 2（第 4 课）。代码：`routeep/comm/nccl_a2a.py`、`routeep/comm/placement.py`。

## 1. 本节要解决什么

1. EP 通信为什么是 all-to-all，而不是 all-reduce / all-gather；
2. NCCL/torch.distributed 的原语、rank/设备拓扑与同步语义；
3. 固定容量（各 rank 发送条目数相同）A2A 的实现；
4. 如何从带宽与延迟两个维度分析通信成本。

**一句话结论**：EP 下每个 token 要去持有目标专家的 rank，每个 rank 也会收到其他 rank 发来的 token，天然是两两交换的 all-to-all；固定容量版本假设各方向消息条数一致，可直接按维度均分后 `all_to_all`。

## 2. 为什么是 All-to-All

| 通信原语 | 语义 | TP/EP 适用性 |
|---|---|---|
| all-reduce | 各 rank 张量求和后广播回所有 rank | TP 每层激活聚合 |
| all-gather | 把各 rank 分片拼成全量 | 收集全量 hidden/专家权重 |
| reduce-scatter | 求和后按维度切分回各 rank | TP 分片还原 |
| **all-to-all** | 各 rank 向每个其他 rank 发一份不同分片 | **EP：token 按专家归属跨 rank 交换** |

dispatch 阶段：rank r 把属于其他 rank 专家的 token 发出去；combine 阶段反向把输出送回。两个方向都是全连接交换。

## 3. 分布式基础概念

- **rank**：通信组内进程编号 `0..world_size-1`；**ep_rank** 是 EP 组内编号；
- **device / group**：NCCL 组走 CUDA tensor，计数/元数据常走 CPU（Gloo）组；vLLM `All2AllManagerBase` 同时持有 `cpu_group` 与 `tcp_store_group`；
- **同步语义**：集合通信是阻塞对齐点，返回时本 rank 的发送缓冲可复用、接收数据已就绪；异步算子需要额外流同步；
- **拓扑**：节点内走 NVLink，跨节点走 RDMA（IB/RoCE）；H100 节点内 NVLink 带宽远高于跨节点 RDMA，消息布局要尽量提高节点内占比。

## 4. 固定容量 A2A 实现

前提：每个 rank 发往每个方向的条目数相同（或已 padding 到相同容量）。

```text
tensor (M, H)，M = world_size × C
  ├─ chunk_r = tensor[r·C : (r+1)·C]          # 发往 rank r 的分片
  ├─ all_to_all(recv_chunks, send_chunks)
  └─ cat(recv_chunks, dim=0) → (M, H)
```

对应 torch.distributed：

```python
chunks = list(tensor.chunk(world_size, dim=0))
recv = [torch.empty_like(c) for c in chunks]
dist.all_to_all(recv, chunks, group=group)
output = torch.cat(recv, dim=0)
```

本项目 `NcclAll2All` 在 `ep_size == 1` 时短路为恒等；多 rank 走变长路径（L5），固定容量作为变长的特例基线。

## 5. 专家放置与 rank 归属

通信分组前必须先确定全局专家在哪些 rank 上（`placement.py`）：

| 策略 | 规则 | 特点 |
|---|---|---|
| `linear` | 连续分片：rank r 持有一段连续专家 | 邻接专家同 rank，组内局部性好 |
| `round_robin` | 专家 e → rank `e % EP` | 热点打散，负载更均匀 |

映射表 `expert_map (E,)`：全局 id → 本地下标，非本地专家为 -1（对齐 vLLM `expert_map_manager.py:22`）。

## 6. 带宽 / 延迟分析口径

- **延迟（latency）**：小消息下由通信启动开销 + 往返决定，decode 阶段 token 少，对尾延迟敏感；
- **带宽（bandwidth）**：大消息下由链路吞吐决定，prefill 阶段 token 多，看每 token 通信量与有效带宽；
- **每 token 通信量**：dispatch 发送 + combine 返回，双向都要算；隐状态维 H、dtype 决定单条消息字节数；
- **profile 要点**：区分节点内 NVLink 与跨节点 RDMA 占比；对比原生 NCCL `all_to_all` 与自研/DeepEP 后端的吞吐与尾延迟。

## 7. 常见误区

1. **把 EP 通信当成 all-reduce**：EP 交换的是 token 分片，不是求和全量。
2. **计数张量走 NCCL 组传 CPU 标量**：元数据计数应走 CPU 组，避免设备同步开销与类型约束问题。
3. **忽略阻塞语义**：集合通信返回前复用发送缓冲会数据竞争。
4. **放置策略与热点分布不匹配**：连续分片下热门专家集中在同一 rank 会形成长尾。

## 8. 自检 5 问

1. all-to-all 与 all-reduce 的语义差别是什么？EP 为什么选前者？
2. ep_rank 和 world_size 的关系是什么？
3. 固定容量 A2A 对各方向消息条数有什么假设？
4. linear 与 round_robin 放置各自的优缺点？
5. 分析通信成本时，延迟和带宽分别对应什么负载场景？

<details>
<summary>参考答案</summary>

1. all-reduce 求和后广播全量；all-to-all 两两交换不同分片。EP 中 token 按专家归属发往不同 rank，需要全连接交换。
2. world_size 是通信组总进程数；ep_rank 是进程在 EP 组内的编号，`0 <= ep_rank < ep_size`。
3. 各方向条目数相同（或已 padding 到同一容量 C），可按维度均分。
4. linear 局部性好但热点可能集中；round_robin 打散热点、负载均匀，但邻接专家分散到不同 rank。
5. 延迟对应 decode 小消息高频场景；带宽对应 prefill 大 batch 吞吐场景。
</details>
