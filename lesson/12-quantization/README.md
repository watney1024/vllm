# 课时12 vLLM 量化：原理与实现

> 版本锚点：`main@8c17e46`。代码块标题是片段起点，省略处用 `...` 标注。

## 0. 本节要解决什么

1. 定点/浮点、per-tensor/per-channel/per-block 到底差在哪；
2. AWQ / GPTQ / FP8 / MXFP4 各自解决什么问题；
3. vLLM 的量化配置和方法类如何协作；
4. 在线量化和 KV cache 量化怎么用。

**一句话结论**：量化是把权重/激活/KV 从高精度映射到低精度；vLLM 把“怎么读 checkpoint、怎么建权重”放在 `QuantizationConfig`，把“怎么算”放在 `QuantizeMethodBase` 子类（linear / MoE / KV cache）。

## 1. 数值格式

### 1.1 定点数

`INT8` / `INT4` 用整数表示，需要 `scale` 和可选的 `zero_point`：

```text
x_real ≈ (x_int - zero_point) * scale
```

- **对称量化**：`zero_point=0`，适合以 0 为中心的权重；
- **非对称量化**：保留 `zero_point`，适合激活；
- **per-tensor**：整个张量一个 scale；
- **per-channel**：每个输出通道一个 scale，精度更好；
- **per-group**：每 G 个元素一个 scale（如 128），在精度和元数据开销之间折中。

### 1.2 浮点数

`FP8` 有两种常见格式：

| 格式 | 指数位 | 尾数位 | 特点 |
|---|---|---|---|
| E4M3 | 4 | 3 | 动态范围小、精度高，适合权重/激活 |
| E5M2 | 5 | 2 | 动态范围大、精度低，适合梯度 |

`MXFP4` / `NVFP4` 是块浮点：每 16 或 32 个元素共享一个指数（scale），权重用 4 bit 存储。它们比 INT4 更适合大模型权重，因为块内动态范围更合理。

## 2. 常见算法

| 算法 | 核心思想 | 适用 |
|---|---|---|
| AWQ | 用激活幅度找“重要通道”，对权重做 per-channel scaling 后再量化 | 权重量化（W4A16） |
| GPTQ | 逐层用 Hessian 近似最小化重建误差 | 权重量化（W4A16） |
| SmoothQuant | 把激活的离群值迁移到权重，使 W8A8 可行 | W8A8 |
| FP8 | 直接用 FP8 存储/计算，动态或静态 scale | H100+ |
| MXFP4/NVFP4 | 块浮点 4bit，配合 Blackwell | 新一代硬件 |
| KV cache 量化 | 只量化 K/V，减少显存 | 长上下文 |

## 3. vLLM 的量化分层

### 3.1 配置层

`QuantizationMethods`（`vllm/model_executor/layers/quantization/__init__.py:15`）列出了所有可选方法，包括：

```text
awq, fp8, modelopt, gptq_marlin, compressed-tensors, mxfp4,
quark, torchao, online, fp8_per_tensor, nvfp4_per_token, ...
```

`get_quantization_config`（同文件 `:116`）把字符串解析成配置类；`register_quantization_config`（`:65`）允许 out-of-tree 注册。

### 3.2 方法层

`QuantizationConfig`（`base_config.py:121`）负责：

- 从 HuggingFace config 读出量化参数；
- 声明支持的激活 dtype / GPU capability；
- 为每个 layer 返回一个 `QuantizeMethodBase`。

真正干活的是方法类：

| 类 | 作用 | 例子 |
|---|---|---|
| `LinearMethodBase` | 量化 Linear：建权重、apply | `Fp8LinearMethod`（`fp8.py:234`） |
| `FusedMoEMethodBase` | 量化 MoE expert | `Fp8MoEMethod`（`fp8.py:466`） |
| `BaseKVCacheMethod` | 量化 KV cache | `Fp8KVCacheMethod`（`fp8.py:887`） |

每个方法类要实现：

```text
create_weights(...)  -> 建低精度参数 + scale
apply(...)           -> 反量化 / 量化 GEMM
```

### 3.3 模块层

以 FP8 为例：

```python title="vllm/model_executor/layers/quantization/fp8.py:234"
class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.
    """
    def create_weights(self, layer, ...):
        ...
    def apply(self, layer, x, bias=None):
        ...
```

`create_weights` 会创建 `weight`（`float8_e4m3fn`）和 `weight_scale`；`apply` 根据平台选择 `torch._scaled_mm`、Cutlass、DeepGEMM 或 Marlin。

## 4. 在线量化

在线量化指“checkpoint 是 BF16/FP16，启动时现场量化”：

```bash
vllm serve facebook/opt-125m --quantization fp8_per_tensor
```

常用简写（来自 `QuantizationMethods`）：

- `fp8_per_tensor` / `fp8_per_block` / `fp8_per_channel`；
- `int8_per_channel_weight_only`；
- `nvfp4_per_token` / `nvfp4_weight_only`；
- `mxfp8` / `mxfp4_hadamard32`。

在线量化的配置在 `vllm/config/quantization.py:180`（`QuantizationConfigArgs`），实现位于 `layers/quantization/online/`。优点是省去离线量化步骤；缺点是启动时要多花时间，且精度取决于校准/动态策略。

## 5. KV cache 量化

KV cache 量化只改 K/V 的存储精度，不改权重：

```bash
vllm serve facebook/opt-125m --kv-cache-dtype fp8
```

`BaseKVCacheMethod`（`kv_cache.py:45`）负责 scale 的创建和应用。收益：

- KV cache 显存减半，能放更多请求 / 更长上下文；
- attention 读 KV 的带宽减半，decode 可能更快。

代价是精度损失，尤其是长上下文和低比特（INT4）时。KV cache 量化通常和权重量化独立配置。

## 6. 最小实验：FP8 模型的吞吐/精度

```bash
# 终端 1：FP8 checkpoint
vllm serve RedHatAI/Llama-3.2-1B-FP8 --max-model-len 2048

# 终端 2：压测
vllm bench serve --model RedHatAI/Llama-3.2-1B-FP8 \
  --base-url http://localhost:8000 \
  --num-prompts 200 --request-rate 20
```

对比 BF16 版本：

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct --max-model-len 2048
```

重点看三个指标：

- `Output token throughput`：FP8 通常更高；
- `TTFT` / `TPOT`：decode 阶段受 KV 读取影响更大；
- 任务准确率：用 `tests/evals/` 或自己的评测集，不能只看吞吐。

如果 FP8 启动报 `capability` 错误，说明 GPU 不支持 FP8 计算；可以退回 Marlin 权重-only 路径，或换 BF16。

## 7. 常见误区

1. **以为量化一定更快**。小 batch / 短序列时，反量化开销可能抵消收益。
2. **以为 INT4 一定比 FP8 快**。还要看是否有对应 kernel（Marlin / Cutlass / Triton）。
3. **忽略 scale 的粒度**。per-tensor 最省，per-group 精度最好但元数据更多。
4. **把权重和 KV cache 量化混为一谈**。它们是两条独立路径，配置项也不同。
5. **不做精度评测**。量化后的模型可能在某些任务上明显退化，必须跑 eval。

## 8. 自检 5 问

1. per-channel 和 per-group 的区别是什么？
2. AWQ 和 GPTQ 的核心差异是什么？
3. `QuantizationConfig` 和 `QuantizeMethodBase` 各负责什么？
4. 在线量化的优点和代价是什么？
5. KV cache 量化为什么能提升 decode 吞吐？

参考答案：

1. per-channel 每个输出通道一个 scale；per-group 每 G 个元素一个 scale，粒度更细。
2. AWQ 用激活幅度做 scaling 保护重要通道；GPTQ 用 Hessian 逐层最小化重建误差。
3. Config 负责读配置/建方法；Method 负责建权重和执行量化计算。
4. 省去离线量化流程，启动即可用；代价是启动时间和精度不确定性。
5. KV cache 减半后，decode 的访存量减半，带宽瓶颈缓解。

## 9. 延伸阅读

- `docs/features/quantization/`；
- `vllm/model_executor/layers/quantization/fp8.py`；
- `vllm/model_executor/layers/quantization/compressed_tensors/`；
- 下一课：课时13 进入数据并行。
