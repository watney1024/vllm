# L7 Grouped GEMM + SwiGLU 融合 Kernel

> 对应课程 Module 3（第 7 课）。代码：`routeep/kernels/grouped_gemm.py`、`routeep/experts/mlp.py`。

## 1. 本节要解决什么

1. 分组 GEMM 的难点：各专家 token 数不同，batch 维度每步变化；
2. 两段 GEMM 与 SwiGLU 激活的融合方式；
3. tiling、访存流水线与调优参数；
4. 权重布局约定。

**一句话结论**：分组 GEMM 按专家分段，每段独立矩阵乘；把 `silu(gate) * up` 融合在 GEMM1 与 GEMM2 之间的 epilogue/prologue，消除激活中间张量的额外读写。

## 2. 权重布局

- `w1 (E, 2I, H)`：gate/up 融合；前 `I` 行 gate，后 `I` 行 up；
- `w2 (E, H, I)`：down projection；
- GEMM1：`gate_up = x @ w1[e]ᵀ → (n, 2I)`；
- 激活：`h = silu(gate) * up`；
- GEMM2：`y = h @ w2[e]ᵀ → (n, H)`。

## 3. 分组 GEMM 结构

- 网格：`(专家数, M 方向 tile 数)`；每个 program 处理一个专家的一段 token；
- 专家段边界由 `expert_start_loc` 给出，空专家直接返回；
- 段内按 `BLOCK_M × BLOCK_N × BLOCK_K` 分块，K 维循环累加；
- 累加器 fp32，写回转输出 dtype。

## 4. SwiGLU 融合

```text
GEMM1 ─► gate │ up ─► silu(gate) ⊙ up ─► GEMM2
        └─ 切片 ─┘    （融合激活，不落地 gate/up 全量中间张量）
```

- 当前实现两段 GEMM 之间在 PyTorch 侧做激活；后续可把激活并入 GEMM1 epilogue 或 GEMM2 prologue，进一步减少 HBM 往返；
- 访存流水线：GEMM2 的加载与 GEMM1 尾部计算重叠，靠 tile 调度与双缓冲调优。

## 5. 调优参数（GPU 实测阶段）

| 参数 | 作用 | 调优方向 |
|---|---|---|
| `BLOCK_M / BLOCK_N` | 计算块形状 | 按 SM 占用与寄存器压力权衡 |
| `BLOCK_K` | K 维分块 | 影响共享内存占用与循环次数 |
| num_stages | 访存流水级数 | 长 K 维加深流水线 |
| 输入布局 | 行优先/转置 | 匹配 tensor core 寻址 |

## 6. 常见误区

1. 把所有专家 token 拼成一个大 GEMM：不同专家权重不同，不能共用一套权重矩阵；
2. 激活中间张量反复读写 HBM；
3. 空专家段不跳过，启动无效计算；
4. 累加器用低精度，长 K 维累计误差放大。

## 7. 自检

1. 分组 GEMM 与普通 batched GEMM 的区别？
2. `w1` 为什么把 gate/up 拼在一起？
3. SwiGLU 公式是什么？融合点在哪两段 GEMM 之间？
4. 空专家在 kernel 里如何处理？
