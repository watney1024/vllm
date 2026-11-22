# L8 Kernel 正确性验证与数值误差排查

> 对应课程 Module 3（第 8 课）。测试：`tests/test_experts_mlp.py`、`tests/test_grouped_gemm.py`、`tests/test_dispatch_combine.py`。

## 1. 验证层次

1. **eager 参考实现**：逐专家 Python 循环矩阵乘，作为数值基线；
2. **kernel vs eager**：Triton/CUDA 路径与基线逐元素对齐；
3. **端到端**：固定输入，路由 → dispatch → 专家 → combine 输出与参考 MoE 对齐。

## 2. 对齐口径

- 精度：fp32 参考；bf16/fp16 训练/推理时记录误差量级（余弦相似度、最大绝对误差、相对误差）；
- 随机种子固定，保证两组输入完全一致；
- 逐 token、逐维对比，不只看全局聚合指标。

## 3. 常见误差来源

| 来源 | 表现 | 排查 |
|---|---|---|
| 索引错位 | 部分 token 输出错乱、局部 NaN | 检查 gather/scatter 下标与排序稳定性 |
| 权重未乘/重复乘 | 输出幅度整体偏大偏小 | 核对 combine 权重与 renormalize |
| 累加精度不足 | 长序列尾差放大 | 累加器改 fp32 |
| 溢出/下溢 | bf16 大值 inf、小值置零 | 检查激活缩放、 routed_scaling_factor |
| 边界 case | 空专家、单 token、top_k=1 出错 | 分段跳过、mask 与空张量测试 |
| 去重错误 | 同 token 同专家重复计入 | 核对聚合权重与条目数 |

## 4. 边界 case 清单

- 某专家 0 个 token；
- 全部 token 只命中一个专家；
- `M=0`（空 batch）；
- H / I 非 2 的幂；
- top_k=1 与 top_k=E；
- 重复命中同一专家（去重路径）；
- padding 槽位（id=-1）。

## 5. 排查流程

```text
输出异常
 ├─ 路由对不对？ 对比 topk_ids / weights 与 eager topk
 ├─ 分发对不对？ 检查 meta 计数、start_loc、bundle 溯源字段
 ├─ 专家对不对？ 单专家段单独喂 eager 与 kernel 对比
 ├─ 汇合对不对？ 固定专家输出为单位基向量，验证还原位置
 └─ 精度问题？   降精度复现 → 定位是 GEMM 累加还是激活溢出
```

## 6. 自检

1. 数值对齐先在哪一层建立基线？
2. bf16 下记录误差应看哪些指标？
3. 索引错位和权重错误的表现有什么不同？
4. 空 batch / 空专家分别怎么测？
