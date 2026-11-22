# L6 Dispatch / Combine Triton Kernel

> 对应课程 Module 3（第 6 课）。代码：`routeep/kernels/dispatch_combine.py`。

## 1. 本节要解决什么

1. dispatch 侧 gather：按路由表把原 token 拷贝到专家分组顺序；
2. combine 侧加权 scatter-add：把专家输出乘权重累加回原 token；
3. 访存模式与 bank conflict 优化思路；
4. 融合 dispatch 减少中间显存读写的方向。

**一句话结论**：这两个算子本质是按索引的读写——gather 是随机读顺序写，scatter-add 是随机写（需原子加）；kernel 逻辑简单，关键在连续访存、向量化加载与避免重复中转。

## 2. 算子语义

### 2.1 Gather（dispatch）

```text
out[i] = x[ token_ids[i] ]      # (M, H) ← (T, H)
```

- 读侧：`token_ids[i]` 随机索引；写侧：`i` 连续；
- 每行整块 H 一次性加载，按 `BLOCK_H` 向量化，mask 处理 H 非 2 的幂。

### 2.2 加权 Scatter-Add（combine）

```text
out[t] += Σ_{i: token_ids[i]=t}  weight[i] · out_i
```

- 写侧随机，同一 token 的 K 个专家条目并发累加，需要 `atomic_add`；
- 权重在 kernel 内广播乘到整行，避免先构造加权中间张量。

## 3. 访存与 bank conflict

- **向量化行加载**：H 维连续，一个程序处理一行，充分利用内存带宽；
- **bank conflict**：scatter 写不同 token 行时，地址落在同一 bank 会串行化；缓解方式是按 token 分桶、相邻条目错开目标行，或依赖原子操作的硬件调度；
- **少中转**：权重乘法融合在 scatter kernel 内，不落地 `weighted_output` 中间张量；
- **dtype**：累加用更高精度（fp32），写回时再转输出 dtype。

## 4. 回退与对齐

- 无 Triton/CUDA 时回退 PyTorch 索引与 `index_add_`，数值语义一致；
- 测试：gather 等价于直接索引；scatter-add 与 `index_add_` 参考实现逐元素对齐；空条目边界单独覆盖。

## 5. 常见误区

1. combine 用普通赋值而非累加，导致同 token 多专家输出互相覆盖；
2. 权重乘法提前落地，多出一份 `(M, H)` 中间显存；
3. 假设 H 对齐，不做 mask，越界读写出错；
4. scatter 不加原子保护，并发累加丢值。

## 6. 自检

1. gather 的读/写侧哪一侧连续？
2. combine 为什么需要原子操作？
3. 权重乘法融合带来什么收益？
4. H 不是 2 的幂时 kernel 如何处理？
