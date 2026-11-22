# L12 端到端对齐：逐 Token / 逐层 Hidden State 对比

> 对应课程 Module 6（第 12 课）。代码：`routeep/alignment.py`。

## 1. 本节要解决什么

1. 固定输入下，逐层对比 RouteEP 与参考实现的 hidden state；
2. 定位差异来自路由、通信还是 GEMM；
3. 建立可重复的对齐流程与误差口径；
4. 与 vLLM 原生 MoE / DeepEP 逐 token 对齐。

**一句话结论**：对齐要"分层层层剥皮"——先对齐 router 输出，再对齐 dispatch 表，再对齐专家段输出，最后对齐 combine 结果；哪一层开始出现误差，bug 就在上一段链路。

## 2. 对齐流程

```text
固定随机种子 + 固定输入 hidden
  ├─ 第 1 层：对比 router logits / topk_ids / topk_weights
  ├─ 第 2 层：对比 dispatch 元数据（counts、start_loc、溯源字段）
  ├─ 第 3 层：对比各专家段 GEMM 输入/输出
  ├─ 第 4 层：对比 combine 后 routed hidden
  └─ 最终：对比输出 token（argmax / 采样一致）
```

参考基线优先级：eager 逐专家循环 → vLLM 原生 MoE → DeepEP。先过 eager，再上框架级对比。

## 3. 误差指标

`alignment.compare_tensor` 输出：

| 指标 | 含义 | 用途 |
|---|---|---|
| `max_abs` | 最大绝对误差 | 定位离群点/串号 |
| `mean_abs` | 平均绝对误差 | 整体精度量级 |
| `cosine` | 余弦相似度 | 方向一致性，bf16 下比绝对误差稳健 |
| `mismatch_frac` | 超容限元素占比 | 是否存在局部错误段 |
| `num_nan` | NaN 个数 | 溢出/非法计算告警 |

容限：fp32 用严格 atol；bf16/fp16 记录量级，以余弦与 mismatch 为主。

## 4. 差异定位对照表

| 现象 | 可疑环节 | 排查动作 |
|---|---|---|
| topk_ids 与参考不同 | router 打分/bias/softmax | 对比 logits、bias 位置、renormalize |
| 条目数/偏移对不上 | dispatch 元数据 | 检查去重、invalid 过滤、稳定排序 |
| 某专家段输出错乱 | 分组 GEMM / 权重映射 | 单专家段单独喂 eager 对比；查 global→local 映射 |
| 输出幅度整体偏大偏小 | combine 权重 | 核对权重乘法、renorm、scaling factor |
| 局部 NaN/inf | 溢出/空段 | 查激活缩放、空 batch、dtype |
| 仅多 rank 下出错 | 通信 | 对比 dispatch 收发计数、溯源字段、combine 往返 |

## 5. 对齐测试要点

- 同一组随机权重注入两边（router gate、w1、w2）；
- 小尺寸先跑通（E=4, K=2, H=8），再放大；
- 覆盖空专家、单 token、重复命中、padding 槽位；
- 逐层保存中间张量，便于离线复盘。

## 6. 自检

1. 为什么先对齐 eager 参考，而不是直接对 DeepEP？
2. cosine 相似度在低精度下为什么比绝对误差更稳健？
3. 输出串号（局部错乱）最可能来自哪一段？
4. 多 rank 下才出现的差异优先排查什么？
