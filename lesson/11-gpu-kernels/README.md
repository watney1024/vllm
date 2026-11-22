# 课时11 GPU 算子开发：Triton / CuTe / TileLang

> 版本锚点：`main@8c17e46`。代码块标题是片段起点，省略处用 `...` 标注。

## 0. 本节要解决什么

1. 为什么 attention / MoE 里大量手写 kernel；
2. Triton 的编程模型和 CUDA 的差别；
3. vLLM 的 kernel 代码怎么组织、怎么接进模型；
4. 什么时候该用 Triton、CuTe DSL 或 TileLang。

**一句话结论**：Triton 用“块级程序”替代 CUDA 的“线程级编程”，把 shared memory、mask、向量化交给编译器；vLLM 把算子按“通用 Triton → 平台专用 CuTe/TileLang → CUDA”分层，模型层通过 `CustomOp` 调用。

## 1. 为什么 attention 算子成了瓶颈

decode 阶段每生成一个 token，就要把整个 KV cache 读一遍：

```text
算术强度 ≈ O(1) FLOP / byte
```

GPU 的算力远大于显存带宽，所以 decode attention 是**带宽受限**，不是算力受限。优化方向不是“多算”，而是：

- 减少 KV 读取（GQA / MLA / 稀疏 attention）；
- 提高 cache 命中（PagedAttention 的 block 布局）；
- 把多个小 kernel 融合（RMSNorm + quant、RoPE + KV write）。

这也是为什么 vLLM 需要针对不同硬件写不同的 attention kernel。

## 2. Triton vs CUDA vs CuTe / TileLang

| | Triton | CUDA | CuTe DSL | TileLang |
|---|---|---|---|---|
| 抽象层级 | 块级（block） | 线程级（thread） | tile / MMA | tile / pipeline |
| 学习成本 | 低 | 高 | 高 | 中 |
| 控制粒度 | 中 | 最细 | 细 | 中 |
| 典型用途 | attention、activation、quant | 极致优化、通信 | Hopper/Blackwell GEMM | 快速写高性能 kernel |
| vLLM 位置 | `vllm/kernels/triton`、`v1/attention/ops` | `csrc/` | `model_executor/kernels/**/cute_dsl` | `model_executor/kernels/mhc` |

选择建议：

1. 先写 Triton，能覆盖 80% 的场景；
2. 需要 Tensor Core / TMA / warp specialization 时用 CuTe DSL；
3. 需要快速实验新融合模式时用 TileLang；
4. 只有前两者都达不到目标时才写 CUDA。

## 3. Triton 的编程模型

一个 Triton kernel 由三部分组成：

```python
@triton.jit
def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)          # 当前块编号
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)    # 块级加载
    tl.store(y_ptr + offsets, x)    # 块级写回
```

启动：

```python
grid = (triton.cdiv(n, BLOCK_SIZE),)
kernel[grid](x, y, BLOCK_SIZE=1024)
```

关键概念：

- `tl.program_id`：当前 program（块）的编号，不是线程号；
- `tl.arange`：生成块内偏移；
- `mask`：越界保护，`tl.load(..., mask=..., other=0.0)`；
- `tl.constexpr`：编译期常量，决定 unroll 和寄存器占用；
- `num_warps` / `num_stages`：影响并行度和流水线。

## 4. 第一个 Triton 算子：向量加法

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out


x = torch.randn(1 << 20, device="cuda")
y = torch.randn(1 << 20, device="cuda")
print(torch.allclose(add(x, y), x + y))
```

这个 kernel 没有做向量化、没有调 `num_warps`，但已经能跑。Triton 会自动把 `BLOCK_SIZE=1024` 的块映射到线程和寄存器。

## 5. Softmax / Matmul / Attention

- **Softmax**：两遍扫描（max、sum）或在线 softmax（一遍，边算边修正）；
- **Matmul**：分块加载 A/B 到 shared memory，累加 `tl.dot`；
- **Attention**：Q 分块 + K/V 流式加载，在线 softmax 累加输出。

以 decode attention 为例，伪结构：

```text
for each KV block:
    load K_block, V_block
    qk = dot(Q, K_block)
    qk = qk * scale + mask
    m_new = max(m_old, max(qk))
    p = exp(qk - m_new)
    l = l * exp(m_old - m_new) + sum(p)
    acc = acc * exp(m_old - m_new) + dot(p, V_block)
out = acc / l
```

vLLM 的实现会在此基础上加：

- PagedAttention 的 block table 索引；
- GQA / MLA 的 KV 压缩；
- FP8 / INT8 反量化；
- 滑窗 / 稀疏 mask；
- CUDA Graph 兼容的固定形状。

## 6. vLLM 的 kernel 组织

| 目录 | 内容 |
|---|---|
| `vllm/kernels/triton/` | 通用 Triton kernel：activation、norm、quant |
| `vllm/triton_utils/` | Triton 兼容层、autotune 控制、tensor descriptor |
| `vllm/v1/attention/ops/` | attention 相关 Triton/CUDA op（unified/decode/MLA/DCP） |
| `vllm/model_executor/kernels/` | 按算子分类的 kernel 实现：attention、linear、mhc |
| `vllm/model_executor/kernels/linear/cute_dsl/` | CuTe DSL 的 GEMM / router |
| `vllm/model_executor/kernels/mhc/tilelang.py` | TileLang 实现的 MHC kernel |
| `csrc/` | CUDA/C++ kernel：通信、cache、MoE、attention |

一个真实例子是 `vllm/kernels/triton/activation.py` 里的 `_gelu_and_mul_sparse_kernel`（`activation.py:17`）。它用 `@triton.jit` 写 kernel，再用 `torch.library.triton_op` 包成 torch op，这样 Inductor 和 CUDA Graph 都能识别。

## 7. 接进模型：CustomOp 与 PluggableLayer

vLLM 的模型层不会直接调 kernel，而是调 `CustomOp`：

```python title="vllm/model_executor/custom_op.py:102"
class CustomOp(nn.Module):
    """Base class for custom ops.
    Dispatches the forward method to the appropriate backend.
    """
```

`CustomOp` 支持：

- 按平台选择 `forward_cuda` / `forward_rocm` / `forward_cpu`；
- out-of-tree 注册替换实现；
- 与 torch.compile 的 custom op 机制配合。

`PluggableLayer`（`custom_op.py:32`）则用于“整层替换”，比如把 `FusedMoE` 换成另一个实现。

## 8. 最小实验：强制 Triton attention 后端

```bash
vllm serve facebook/opt-125m \
  --attention-backend TRITON_ATTN \
  --max-model-len 256 \
  --enforce-eager
```

然后用 `vllm bench serve` 对比默认后端：

```bash
vllm bench serve --model facebook/opt-125m \
  --base-url http://localhost:8000 \
  --num-prompts 100 --request-rate 10
```

Triton 后端通常比 FlashAttention/FlashInfer 慢，但代码可读、方便改。想验证自己写的 kernel，可以先在 Triton 后端里替换一个 op，再跑同样的 benchmark。

## 9. 常见误区

1. **以为 Triton 一定比 CUDA 慢**。在中小规模、融合场景下，Triton 往往更接近手写 CUDA，且开发快得多。
2. **忽略 mask 和边界**。decode 的序列长度不是 BLOCK_SIZE 的整数倍，越界读写会静默出错。
3. **把所有逻辑塞进一个 kernel**。寄存器压力和 occupancy 会反过来拖慢性能。
4. **不看 dtype**。FP16/BF16/FP8 的累加精度和指令不同，测试要用目标 dtype。
5. **忘了 CUDA Graph**。kernel 里有 CPU 同步或动态分配，会导致 graph 捕获失败。

## 10. 自检 5 问

1. decode attention 为什么是带宽受限？
2. Triton 的 `program_id` 和 CUDA 的 `threadIdx` 有什么区别？
3. `tl.constexpr` 的作用是什么？
4. vLLM 在什么场景下会用 CuTe DSL 而不是 Triton？
5. `CustomOp` 如何帮助 torch.compile？

参考答案：

1. 每生成一个 token 要读全部 KV，计算量小、访存量大。
2. program_id 是块级编号，一个 program 对应一组线程/向量。
3. 编译期常量，影响循环展开、寄存器和代码生成。
4. 需要 Tensor Core / TMA / warp specialization 的高性能 GEMM。
5. 它把 kernel 注册成 torch op，Inductor 可以把它当作黑盒算子，避免错误内联。

## 11. 延伸阅读

- `docs/design/custom_op.md`；
- `docs/design/vllm_ir.md`（已废弃，不要在新 op 中使用）；
- `vllm/v1/attention/ops/triton_unified_attention.py`；
- 下一课：课时12 进入量化，看 kernel 如何与低精度权重配合。
