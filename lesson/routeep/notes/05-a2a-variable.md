# L5 变长 All-to-All、TMA 直连与 Peer 通信优化

> 对应课程 Module 2（第 5 课）。代码：`routeep/comm/nccl_a2a.py`（`_all_to_all_variable`）。

## 1. 本节要解决什么

1. 为什么实际 EP 通信是变长的，变长带来什么对齐问题；
2. 变长消息如何分块、前缀计数元数据如何同步；
3. H100 TMA、peer direct 通信的优化方向；
4. 通信性能 profile 方法，以及与原生 NCCL 的对比口径。

**一句话结论**：每轮各 rank 命中的专家不同，收发条数是变长的；先用一轮计数交换得到各方向长度，padding 到最大容量后走固定 A2A，接收侧再按计数裁掉 padding——本项目 `_all_to_all_variable` 即按此实现。

## 2. 变长问题的来源

- 路由结果每步变化：热门专家命中多、冷门专家命中少；
- 不同 rank 的本地 token 数不同（DP/SP 下尤其明显）；
- 因此 dispatch/combine 各方向消息条数 `send_counts[r]` 不固定，不能直接均分张量。

## 3. 变长实现流程

```text
1. 本地按目标 rank 分组，得到 send_counts[r]（L4 grouping.py）
2. all_gather 计数张量：各 rank 交换自己的 send_counts
3. 本 rank 从 rank s 收到的条数 = gathered[s][ep_rank] = recv_counts[s]
4. max_count = max(所有发送/接收计数)
5. 每个发送分片 padding 到 max_count（零填充尾部）
6. all_to_all 交换 padding 后的分片
7. 接收侧按 recv_counts[s] 裁掉 padding，按源 rank 顺序拼接
```

计数交换与数据交换分离：计数小、走 CPU 组；数据张量走设备组。多张张量（hidden、token_id、slot、expert_id、weight、src_rank）共享同一套计数，一起交换。

### 3.1 元数据随附

dispatch 每条目必须带上溯源信息，combine 才能还原：

| 字段 | 用途 |
|---|---|
| `token_id` | 源 rank 上的原 token 下标 |
| `slot_id` | 路由槽位号 |
| `expert_id` | 全局专家 id（接收侧查本地权重） |
| `weight` | combine 加权 |
| `src_rank` | combine 反向发送的目标 rank |

## 4. 高级优化方向（H100）

### 4.1 TMA（Tensor Memory Accelerator）

- 硬件异步批量张量搬移，减少线程组参与地址计算与逐元素拷贝；
- 适合 dispatch/combine 的大块连续读写；本项目 kernel 阶段预留 TMA 加载/存储接口，实际 tile 编排结合 grouped GEMM 一起调优。

### 4.2 Peer Direct / NVLink P2P

- 同节点内 rank 间可直接映射显存（peer access），绕过部分缓存层次；
- 减少一次中转拷贝，降低小消息延迟；
- 跨节点仍走 RDMA，需区分节点内/跨节点缓冲与 QP 配置（对照 vLLM `DeepEPHTAll2AllManager._make_all2all_kwargs`：`num_nvl_bytes` / `num_rdma_bytes` / `num_qps_per_rank`）。

### 4.3 通信-计算重叠

- dispatch 打包与专家 GEMM 流水化：先发高优先级条目，计算与剩余发送重叠；
- combine 输出不等全部到齐即可逐步累加；
- 缓冲按容量分桶复用，避免每轮重新分配。

## 5. Profile 与对比方案

| 指标 | 测量方法 | 对比基线 |
|---|---|---|
| 每方向收发条数 | 记录 `send_counts` / `recv_counts` 分布 | — |
| 通信吞吐 | 总字节 / 通信耗时，分节点内/跨节点 | 原生 NCCL `all_to_all`、DeepEP HT/LL |
| 延迟尾部分布 | decode 小消息逐轮计时，看 P50/P99 | 原生 NCCL、DeepEP LL |
| 重叠收益 | 对比通信-计算串行与流水化的端到端步时 | — |

工具：torch profiler（通信算子时间线）、NCCL debug 日志、`nsys` 看通信/计算流重叠；压测脚本在 Stage 6 落地。

## 6. 常见误区

1. **只交换 hidden 不交换溯源元数据**：combine 无法把输出送回原 token。
2. **padding 参与后续 GEMM**：必须按计数裁剪，padding 条目不能进入专家 batch。
3. **计数与数据走同一组却混用设备/CPU 张量**：计数走 CPU 组，数据走设备组。
4. **假设各方向等长**：变长场景直接均分张量会静默错位。

## 7. 自检 5 问

1. 变长通信为什么不能直接按维度均分？
2. `recv_counts[s]` 是怎么从 all_gather 结果得到的？
3. padding 填什么？接收侧如何去掉？
4. TMA 和 peer direct 分别优化什么环节？
5. decode 与 prefill 场景分别该看延迟还是带宽指标？

<details>
<summary>参考答案</summary>

1. 各 rank 路由结果不同，收发条数不固定，均分需要等长分片。
2. `gathered[s]` 是 rank s 的发送计数数组；本 rank 从 s 收到的条数是其第 `ep_rank` 项。
3. 尾部零填充；按 `recv_counts[s]` 截取每个接收分片的有效长度。
4. TMA 优化硬件批量张量搬移、减少线程组开销；peer direct 优化同节点显存直连、降低中转拷贝。
5. decode 看尾延迟，prefill 看有效带宽与吞吐。
</details>
