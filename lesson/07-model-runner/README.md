# 课时07 vLLM 核心组件：ModelRunner 与模型接入

> 版本锚点：`main@8c17e46`。代码块标题是片段起点，省略处用 `...` 标注。

## 0. 本节要解决什么

1. ModelRunner 里为什么有三套“batch”结构；
2. 模型类是怎么被找到、实例化、加载权重的；
3. 一个 `SchedulerOutput` 如何变成 `input_ids / positions / block_table`；
4. 接入一个新模型最少要改哪些地方。

**一句话结论**：`GPUModelRunner` 用 `RequestState` 保存常驻请求状态、用 `InputBuffers` 预分配显存、用 `InputBatch` 表示“这一 step 的视图”；模型加载则是 `get_model_loader → initialize_model → resolve_model_cls → load_weights` 四步。

## 1. 三套 batch 结构

先区分三个容易混的东西：

| 结构 | 生命周期 | 作用 | 源码 |
|---|---|---|---|
| `RequestState` | 请求级，跨 step | 常驻状态：token id、prompt/输出长度、采样状态 | `vllm/v1/worker/gpu/states.py:9` |
| `InputBuffers` | 进程级，预分配 | 固定大小的 GPU 张量池，避免每步 malloc | `vllm/v1/worker/gpu/input_batch.py:23` |
| `InputBatch` | step 级 | 本步 batch 的视图：`req_ids`、`idx_mapping`、`query_start_loc` | `vllm/v1/worker/gpu/input_batch.py:43` |

`RequestState` 用 `req_id_to_index` / `index_to_req_id` 给每个请求分配一个**常驻行号**；请求结束后行号才回收。`InputBatch.idx_mapping` 把“本步 batch 的第 i 行”映射到“常驻状态的第 j 行”。这就是 persistent batch 的核心：请求加入/退出只改映射，不整体搬张量。

`InputBuffers` 的字段都是预分配的：

```python title="vllm/v1/worker/gpu/input_batch.py:23"
class InputBuffers:
    def __init__(self, max_num_reqs: int, max_num_tokens: int, device: torch.device):
        self.input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        self.is_padding = torch.zeros(max_num_tokens, dtype=torch.bool, device=device)
        self.query_start_loc = torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
```

## 2. 模型加载：四步

### 2.1 选 loader

`get_model_loader`（`vllm/model_executor/model_loader/__init__.py:128`）按 `load_config.load_format` 返回 loader：

| load_format | loader | 用途 |
|---|---|---|
| `auto` / `safetensors` | `DefaultModelLoader` | 常规 HuggingFace 权重 |
| `dummy` | `DummyModelLoader` | profile / 测试，不读真实权重 |
| `runai_streamer` | `RunaiModelStreamerLoader` | 流式加载大模型 |
| `tensorizer` | `TensorizerLoader` | 序列化权重 |
| `sharded_state` | `ShardedStateLoader` | 分片权重 |

`DefaultModelLoader` 在 `default_loader.py:58`。

### 2.2 解析模型类

`initialize_model`（`model_loader/utils.py:83`）先拿模型类：

```python title="vllm/model_executor/model_loader/utils.py:83"
def initialize_model(vllm_config, *, prefix="", model_class=None, model_config=None):
    if model_class is None:
        model_class, _ = get_model_architecture(model_config)
    ...
    if "vllm_config" in all_params and "prefix" in all_params:
        with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
            model = model_class(vllm_config=vllm_config, prefix=prefix)
            record_metadata_for_reloading(model)
            return model
    ...
```

`get_model_architecture` 最终走 `ModelRegistry.resolve_model_cls`（`registry.py:1524`）。`ModelRegistry` 是一个 `_ModelRegistry` 实例（`registry.py:1645`），内部维护“架构名 → (模块, 类名)”的映射。比如 `OPTForCausalLM` 映射到 `vllm.model_executor.models.opt`。

### 2.3 加载权重

`DefaultModelLoader.load_model` 做三件事：

1. 调 `initialize_model` 建空模型；
2. 遍历 checkpoint 的权重迭代器；
3. 调 `model.load_weights(weights)`，再 `process_weights_after_loading`。

`load_weights` 是模型类自己的方法，负责把 HuggingFace 的权重名映射到 vLLM 的模块名。不同模型的命名差异（`model.layers.0.self_attn.q_proj` vs `transformer.h.0.attn.c_attn`）都在这里消化。

### 2.4 后处理

`process_weights_after_loading`（`model_loader/utils.py` 末尾）做量化 scale 重排、转置、融合准备等。比如 FP8 模型要在这里把 weight 转成 `float8_e4m3fn` 并算好 scale。

## 3. ModelRunner 的加载入口

`GPUModelRunner.load_model`（`model_runner.py:407`）把上面的流程串起来：

```python title="vllm/v1/worker/gpu/model_runner.py:407"
def load_model(self, load_dummy_weights: bool = False, *args, model=None, **kwargs):
    if model is None and load_dummy_weights:
        self.load_config.load_format = "dummy"
    with DeviceMemoryProfiler() as m:
        model_loader = get_model_loader(self.vllm_config.load_config)
        self.model = model or model_loader.load_model(
            vllm_config=self.vllm_config,
            model_config=self.vllm_config.model_config,
        )
        ...
    self.model_memory_usage = m.consumed_memory
    self.model_state = init_model_state(
        self.vllm_config, self.model, self.encoder_cache, self.device
    )
```

注意 `model_state`：不同模型结构（decoder-only、encoder-decoder、Mamba hybrid）有不同的状态管理策略，`init_model_state` 会选一个 `ModelState` 子类，负责 `prepare_attn` / `compute_sample_logits` 等模型相关逻辑。

## 4. prepare_inputs：从调度结果到模型输入

`prepare_inputs`（`model_runner.py:1494`）是每步最忙的函数。它做：

1. 根据 `scheduler_output.num_scheduled_tokens` 算 `query_start_loc`；
2. 从 `RequestState`  Gather 出本步的 `input_ids` / `positions`；
3. 处理 padding（CUDA Graph 要求固定形状）；
4. 生成 `InputBatch`，其中 `logits_indices` 指向“哪些位置需要算 logits”；
5. 调 `prepare_attn` 生成 block table 和 slot mapping。

一个简化的形状关系：

```text
num_scheduled_tokens = [3, 1, 1]     # 3 个请求
query_start_loc      = [0, 3, 4, 5]
input_ids.shape      = [5]           # 5 个 token
seq_lens             = [128, 64, 7]  # 每个请求当前总长度
logits_indices       = [2, 3, 4]     # 每个请求最后一个 token 的位置
```

`prepare_attn` 返回的 `block_tables` 是 `[num_reqs, max_num_blocks]` 的 int32 张量，`slot_mappings` 是 `[num_tokens]` 的 int64 张量，告诉 attention kernel 每个 token 的 KV 写到哪个物理 slot。

## 5. 接入一个新模型

最小改动清单：

1. **实现模型类**：继承 `nn.Module`，构造函数接受 `vllm_config` 和 `prefix`；
2. **实现 `load_weights`**：把 checkpoint 名字映射到模块参数；
3. **注册架构**：在 `registry.py` 的映射表里加 `"YourModelForCausalLM": ("your_model", "YourModelForCausalLM")`；
4. **实现 attention / MLP 层**：优先复用 `vllm/model_executor/layers/` 里的现有层；
5. **跑 dummy run**：用 `load_format="dummy"` 走一遍 `initialize_model` + `execute_model`，确认形状和 KV cache 对齐。

不需要改 Scheduler、Executor、KV cache manager——只要模型类遵守接口，这些组件是通用的。

## 6. 最小实验：看 prepare_inputs 的输出

```python
import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

original_prepare = GPUModelRunner.prepare_inputs


def traced_prepare(self, scheduler_output, batch_req_state, batch_desc, num_active_loras):
    out = original_prepare(
        self, scheduler_output, batch_req_state, batch_desc, num_active_loras
    )
    print(
        "reqs=", out.req_ids,
        "num_tokens=", out.num_tokens,
        "input_ids=", tuple(out.input_ids.shape),
        "positions=", tuple(out.positions.shape),
        "logits_indices=", out.logits_indices.tolist()[:8],
    )
    return out


GPUModelRunner.prepare_inputs = traced_prepare

llm = LLM(
    model="facebook/opt-125m",
    max_model_len=128,
    enforce_eager=True,
    gpu_memory_utilization=0.3,
)
llm.generate(["Hello world", "Hi"], SamplingParams(max_tokens=3))
```

第一轮会看到两个请求合并成一个 batch，`num_tokens` 等于两个 prompt 长度之和；后续 decode 轮 `num_tokens=2`，每个请求一个 token。

## 7. 常见误区

1. **以为 `InputBatch` 会一直保留**。它只活一个 step；跨 step 的状态在 `RequestState`。
2. **以为模型类可以随便改构造函数**。新式模型必须接受 `vllm_config` 和 `prefix`。
3. **以为 `load_weights` 是通用逻辑**。它是每个模型自己的名字映射，vLLM 不做自动推断。
4. **以为 `prepare_inputs` 只拼 input_ids**。它还要算 positions、query_start_loc、logits_indices、attention metadata。
5. **以为新模型只要注册就能跑**。还要保证 KV cache 的 layer 命名、attention backend 选择、dummy run 都通过。

## 8. 自检 5 问

1. `RequestState` 和 `InputBatch` 的区别是什么？
2. `InputBuffers` 为什么要预分配？
3. `get_model_loader` 根据什么选 loader？
4. `initialize_model` 为什么要求模型类接受 `vllm_config` 和 `prefix`？
5. 接入新模型时，哪一步最容易漏掉？

参考答案：

1. `RequestState` 是跨 step 的常驻状态；`InputBatch` 是本步的 batch 视图。
2. 避免每步 malloc/free，并让 CUDA Graph 捕获固定地址。
3. `LoadConfig.load_format`。
4. 为了统一注入量化、LoRA、编译、prefix 等配置；旧式构造函数已被废弃。
5. `load_weights` 的名字映射和 attention layer 的 KV cache 接口。

## 9. 延伸阅读

- `docs/contributing/model/`：新增模型的官方指南；
- `vllm/model_executor/models/registry.py`：架构映射表；
- `vllm/model_executor/model_loader/default_loader.py`；
- 下一课：课时08 进入 Attention 后端，看 block table 如何被 kernel 消费。
