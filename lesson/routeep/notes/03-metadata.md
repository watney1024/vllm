# L3 Token 重排、去重与 Dispatch 元数据

> 对应课程 Module 1（第 3 课）。代码：`routeep/metadata.py`；测试：`tests/test_metadata.py`。

## 1. 本节要解决什么

1. 把 `(T, K)` 的路由结果展开成扁平条目表，并记录每条目来自哪个 token、哪个槽位；
2. 过滤 padding 无效槽位，合并同一 token 重复命中同一专家的条目；
3. 按专家重排 token，构造每个专家的条目数与前缀偏移（dispatch 索引表/位置表）；
4. 提供 combine 反向还原：按原 token 顺序把专家输出加权累加回去。

**一句话结论**：元数据层是 dispatch 与 combine 之间的"账本"——dispatch 前把 token 按专家归类并记住来路，combine 时靠同一张表把结果送回原位置；表错一位，输出就串号。

## 2. 扁平路由表

Router 输出 `topk_ids (T, K)`、`topk_weights (T, K)`，展开为每条目：

| 字段 | 形状 | 含义 |
|---|---|---|
| `token_id` | `(M,)` | 原 token 下标 `0..T-1` |
| `slot_id` | `(M,)` | 路由槽位号 `0..K-1` |
| `expert_id` | `(M,)` | 全局专家 id；无效槽位为 -1 |
| `weight` | `(M,)` | 对应路由权重 |

展开顺序：token 主序、槽位次序（`arange(T).repeat_interleave(K)`）。

## 3. 过滤与去重

### 3.1 无效槽位过滤

`expert_id < 0` 的条目（padding / Graph 补位）直接丢弃，不进入 dispatch。

### 3.2 按 (token, expert) 去重

正常 TopK 不会让同一 token 重复命中同一专家，但分组 TopK、自定义路由或偏差扰动下可能出现。规则：

- 复合 key = `token_id * E + expert_id`；
- 相同 key 的条目合并，**权重相加**为一条；
- 保留首个出现的槽位号作为代表。

## 4. 按专家排序与计数表

去重后按 `(expert_id, token_id)` 稳定排序，得到 dispatch 顺序：

- `expert_counts[e]`：专家 `e` 的条目数；
- `expert_start_loc[e]`：专家 `e` 条目在排序表中的起始偏移（前缀和，长度 `E+1`）；
- 专家 `e` 的条目区间：`[start_loc[e], start_loc[e+1])`。

这张表就是后续 grouped GEMM 的分组依据：每个专家一段连续 token，段长即 `expert_counts[e]`。

## 5. 正反向操作

### 5.1 dispatch 侧：permute

```text
permuted[i] = hidden[ sorted_token_ids[i] ]     # (M, H)
```

### 5.2 combine 侧：index_add 还原

```text
out[t] = Σ_{i: sorted_token_ids[i] = t}  weight[i] · out_i
```

用 `index_add_` 按原 token 下标 scatter-add，天然支持一个 token 对应 K 个专家条目。

## 6. 边界 case（测试覆盖）

| 场景 | 预期 |
|---|---|
| 全部槽位无效 | `M=0`，counts 全 0，start_loc 全 0 |
| 重复命中同一专家 | 合并一条，权重求和 |
| 某专家无 token | 区间为空，count=0 |
| 往返一致性 | 恒等专家下 combine(permute(hidden)) ≈ hidden |

## 7. 常见误区

1. **排序不稳定**：同专家内条目顺序不确定会导致 GEMM 结果顺序与权重表错位，必须稳定排序。
2. **去重时权重不取和**：重复条目应权重聚合，否则同一 token 被重复计入专家 batch。
3. **combine 用 scatter 覆盖而非累加**：同一 token 的 K 个专家输出必须累加，直接赋值会互相覆盖。
4. **偏移表长度少 1**：`start_loc` 长度应为 `E+1`，最后一段靠尾偏移界定。

## 8. 自检 5 问

1. 扁平表四个字段分别记录什么？
2. 去重的复合 key 怎么构造？权重如何处理？
3. `expert_counts` 与 `expert_start_loc` 是什么关系？
4. combine 为什么用 `index_add_` 而不是普通索引赋值？
5. 全部 token 都被路由到同一专家时，排序表长什么样？

<details>
<summary>参考答案</summary>

1. 原 token 下标、槽位号、全局专家 id、路由权重。
2. `key = token_id * E + expert_id`；同 key 权重相加，保留首个槽位。
3. `start_loc[0]=0`，`start_loc[e+1] = start_loc[e] + counts[e]`，前缀和关系。
4. 一个 token 对应多个专家条目，需要按位置累加；赋值会覆盖之前的结果。
5. 排序表前 `M=T·K'` 条都属于该专家，`counts` 仅该专家非零，其余区间为空。
</details>
