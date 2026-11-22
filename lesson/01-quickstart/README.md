# 课时01 vLLM 快速上手与环境搭建

> 版本锚点：`main@8c17e46`（2026-11）。本课的目标不是读源码，而是先得到一套“能跑、能测、能复现”的环境。

## 0. 本节要解决什么

1. 用 uv 建一个干净的 vLLM 环境；
2. 跑通离线推理和在线服务；
3. 用自带 benchmark 得到第一组性能数字；
4. 知道常用参数在哪一层生效。

**一句话结论**：vLLM 的入口只有两个——`LLM`（离线）和 `vllm serve`（在线）；先把这两个跑通，再谈调优和源码。

## 1. 环境准备

### 1.1 硬件 / 驱动

- 一张能跑目标模型的 NVIDIA GPU（或 AMD ROCm / Intel XPU / CPU 后端）；
- 驱动和 CUDA 版本满足当前 PyTorch 要求；
- 足够的磁盘空间放模型权重和编译缓存。

先确认：

```bash
nvidia-smi
python3 --version
```

vLLM 只支持较新的 Python（以 `pyproject.toml` 的 `requires-python` 为准）。不要用系统 Python 直接装包。

### 1.2 用 uv 建环境

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

cd /path/to/vllm
git checkout 8c17e46     # 与全文的行号锚点对齐；在 main 上读行号会漂
uv venv --python 3.12
source .venv/bin/activate
```

> 版本以仓库 `pyproject.toml` 为准；如果它要求更高版本，把 `3.12` 换成对应版本。

### 1.3 安装 vLLM

开发/读源码推荐 editable 安装，并优先用预编译产物避免长时间编译：

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

如果你要改 C++/CUDA（`csrc/`），按 `docs/contributing/incremental_build.md` 做增量编译。

### 1.4 安装测试依赖

```bash
uv pip install -r requirements/test/cuda.in
```

非 x86_64 平台把 `cuda.in` 换成对应文件。装完后跑一个最小测试确认环境：

```bash
.venv/bin/python -m pytest tests/models/language/generation/test_common.py -q -k opt-125m
```

（会下载 125M 权重；如果该参数化用例已被调整，选仓库里任意一个最小的模型测试即可。）

## 2. 离线推理

最小示例：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="facebook/opt-125m",
    max_model_len=256,
    gpu_memory_utilization=0.6,
)

params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=64,
)
outputs = llm.generate(["Hello, my name is"], params)

for out in outputs:
    print(out.outputs[0].text)
```

关键参数：

| 参数 | 作用 |
|---|---|
| `model` | HuggingFace 模型名或本地路径 |
| `max_model_len` | 最大上下文长度，直接影响 KV cache 容量 |
| `gpu_memory_utilization` | vLLM 可用的 GPU 显存比例 |
| `tensor_parallel_size` | TP 卡数 |
| `dtype` | `auto` / `float16` / `bfloat16` |
| `enforce_eager` | 关闭 CUDA Graph / 编译，便于调试 |

第一次运行会下载权重。国内网络可以设置 `HF_ENDPOINT` 镜像，或提前用 `huggingface-cli download` 拉到本地。

## 3. 在线服务

### 3.1 启动

```bash
vllm serve facebook/opt-125m \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 512
```

### 3.2 调用

OpenAI 兼容接口：

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Hello, my name is",
    "max_tokens": 32,
    "temperature": 0
  }'
```

Chat 接口：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 32
  }'
```

vLLM 还提供 Anthropic / Cohere / gRPC / Responses 等入口（`vllm/entrypoints/`），本课程以 OpenAI 兼容接口为主。

### 3.3 健康检查

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics | head
```

`/metrics` 是后面性能课的基础，先确认它能返回 Prometheus 文本。

## 4. 性能测试

vLLM 自带 `vllm bench`：

```bash
# 延迟（单请求，串行）
vllm bench latency --model facebook/opt-125m \
  --input-len 128 --output-len 32

# 吞吐（固定并发）
vllm bench throughput --model facebook/opt-125m \
  --input-len 128 --output-len 32 --num-prompts 200

# 在线服务压测
vllm bench serve --model facebook/opt-125m \
  --base-url http://localhost:8000 \
  --num-prompts 200 --request-rate 20
```

第一组数字不用追求好看，重点是建立基线：

| 指标 | 含义 |
|---|---|
| `TTFT` | 首 token 延迟 |
| `TPOT` / `ITL` | 每个输出 token 的间隔 |
| `Output token throughput` | 输出 token 吞吐 |
| `Request throughput` | 请求吞吐 |

把这些数字记下来，后面每次改参数/改代码都用同一组命令对比。

## 5. 配置系统

vLLM 的配置是一个嵌套 dataclass（`VllmConfig`，`vllm/config/vllm.py`）：

```text
VllmConfig
├── ModelConfig
├── CacheConfig
├── ParallelConfig
├── SchedulerConfig
├── CompilationConfig
├── LoadConfig
├── ObservabilityConfig
└── ...
```

配置分两类：

- 大多数字段有对应的**扁平 CLI 参数**，例如 `--max-num-batched-tokens`、`--max-num-seqs`、`--gpu-memory-utilization`；
- 以 JSON 传入的配置组（`--compilation-config`、`--speculative-config`、`--kv-transfer-config` 等）额外支持用点号逐字段覆盖；`-cc` 是 `--compilation-config` 的简写。

例如：

```bash
vllm serve <model> --max-num-batched-tokens 8192
vllm serve <model> -cc.cudagraph_mode=NONE
```

注意不存在 `--scheduler.max-num-batched-tokens` 这种“配置类名.字段”的写法——Scheduler 的字段就是上面那组扁平参数。

Python API 直接传顶层参数或构造配置对象：

```python
from vllm import LLM
llm = LLM(model="facebook/opt-125m", max_num_batched_tokens=8192)
```

## 6. 最小实验：完整跑一遍

```bash
# 1) 离线
VLLM_ENABLE_V1_MULTIPROCESSING=0 .venv/bin/python - <<'PY'
from vllm import LLM, SamplingParams
llm = LLM(model="facebook/opt-125m", max_model_len=256, gpu_memory_utilization=0.5)
out = llm.generate(["Hello"], SamplingParams(max_tokens=8, temperature=0))
print(out[0].outputs[0].text)
PY

# 2) 在线
vllm serve facebook/opt-125m --port 8000 --max-model-len 256 &
sleep 30

# 3) 压测
vllm bench serve --model facebook/opt-125m \
  --base-url http://localhost:8000 \
  --num-prompts 50 --request-rate 5
```

如果第 2 步启动失败，先看日志里的 OOM / CUDA / 端口 / 模型下载错误；不要急着改并行参数。

## 7. 常见问题

1. **OOM**：降低 `gpu_memory_utilization`、`max_model_len`、`max_num_seqs`，或用量化。
2. **启动很慢**：第一次要下载权重、编译、捕获 CUDA Graph；第二次会快很多。
3. **端口被占用**：换 `--port`，或先 `lsof -i :8000`。
4. **模型下载失败**：设置镜像 / 代理，或手动下载后传本地路径。
5. **CUDA 版本不匹配**：按 PyTorch 官方矩阵重装，不要混用系统 CUDA 包。
6. **bench 结果波动大**：固定 `--request-rate`、warmup、关闭其他占用 GPU 的进程。

## 8. 自检 5 问

1. 离线推理和在线服务的入口分别是什么？
2. `gpu_memory_utilization` 影响什么？
3. `vllm bench latency` 和 `vllm bench serve` 的区别是什么？
4. 第一次启动为什么慢？
5. 配置字段如何通过 CLI 覆盖？

参考答案：

1. `LLM` 和 `vllm serve`。
2. vLLM 可用的 GPU 显存比例，进而影响 KV cache block 数量。
3. latency 测单请求串行延迟；serve 测在线服务的吞吐/延迟分布。
4. 下载权重、torch.compile、CUDA Graph 捕获、JIT warmup。
5. 顶层用扁平参数（如 `--max-num-batched-tokens`）；JSON 配置组用点号覆盖（如 `-cc.cudagraph_mode`，`-cc` 是 `--compilation-config` 的简写）。

## 9. 延伸阅读

- `docs/getting_started/quickstart.md`；
- `docs/configuration/optimization.md`；
- `docs/benchmarking/`；
- 下一课：课时02 建立架构总览。
