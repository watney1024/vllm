# 课时19 vLLM 性能分析与瓶颈定位

> 版本锚点：`main@8c17e46`。代码块标题是片段起点，省略处用 `...` 标注。

## 0. 本节要解决什么

1. 先用哪几个指标判断“慢在哪”；
2. Torch Profiler / layerwise / Nsight 各自看什么；
3. vLLM 的 metrics 和日志怎么用；
4. 常见的四类瓶颈如何定位。

**一句话结论**：先用 TTFT / TPOT / 吞吐把问题分到“排队、prefill、decode、通信”四类，再用 profiler 看时间线；不要一上来就抓 nsys。

## 1. 先建立性能模型

| 指标 | 含义 | 主要影响因素 |
|---|---|---|
| TTFT | 首 token 时间 | 排队、prefill、调度、KV 加载 |
| TPOT / ITL | 每个输出 token 的间隔 | decode forward、采样、CPU launch、通信 |
| 吞吐 | tokens/s 或 requests/s | batch 大小、KV 容量、并行策略 |
| Goodput | 满足 SLO 的吞吐 | 尾延迟、排队 |

一次请求的时间可以拆成：

```text
queue_wait + prefill + (decode_step * output_len) + detokenize
```

先用 `vllm bench` 看整体：

```bash
vllm bench serve --model facebook/opt-125m \
  --base-url http://localhost:8000 \
  --num-prompts 200 --request-rate 20
```

如果 TTFT 高但 TPOT 正常，问题在排队或 prefill；如果 TTFT 正常但 TPOT 高，问题在 decode 的单步开销。

## 2. 工具矩阵

| 工具 | 粒度 | 适合 |
|---|---|---|
| `vllm bench` | 端到端 | 压测、参数扫描 |
| Torch Profiler | op / kernel | CPU launch、GPU kernel、memcpy |
| layerwise profile | 模块 | 哪一层慢 |
| Nsight Systems | 时间线 | kernel 重叠、通信、GPU 空闲 |
| Prometheus metrics | 聚合 | 线上监控 |
| iteration details | step | 调度/执行分解 |

## 3. Torch Profiler

开启方式：

```python
from vllm import LLM, SamplingParams
from vllm.config import ProfilerConfig

llm = LLM(
    model="facebook/opt-125m",
    max_model_len=256,
    enforce_eager=True,
    profiler_config=ProfilerConfig(
        profiler="torch",
        torch_profiler_dir="/tmp/vllm-prof",
    ),
)
params = SamplingParams(max_tokens=16, temperature=0, ignore_eos=True)
llm.generate(["Hello"], params)  # warmup

llm.start_profile(profile_prefix="decode")
for _ in range(3):
    llm.generate(["Hello"], params)
llm.stop_profile()
```

产物是 Chrome trace（`.json`），用 `chrome://tracing` 或 Perfetto 打开。

重点看：

- **CPU 行**：kernel launch 之间有没有长间隙（Python 开销 / 调度）；
- **GPU 行**：有没有空闲（等 CPU、等通信、等 KV）；
- **Memcpy**：D2H 拷贝是否在关键路径；
- **NCCL**：all-reduce / all-to-all 是否与计算重叠；
- **kernel 名**：attention / GEMM / quant 的耗时占比。

`TorchProfilerWrapper`（`vllm/profiler/wrapper.py:228`）负责在 worker 侧启动/停止 profiler，并把 trace 写到配置的目录。

## 4. layerwise profile

如果只想知道“哪一层慢”，用 layerwise：

```python
from vllm.profiler.layerwise_profile import layerwise_profile

with layerwise_profile() as prof:
    llm.generate(["Hello"], params)

prof.results.print_summary_table()
```

`layerwise_profile` 是 `torch.profiler.profile` 的子类，退出上下文时把结果挂到 `.results`（`LayerwiseProfileResults`，`layerwise_profile.py:83`），所以打印要走 `prof.results`。它会按模块树聚合耗时，输出每个子模块的 self time / total time，适合发现：

- 某一层异常慢（kernel 选择错误）；
- 量化层反量化开销过大；
- attention 层在不同 seq_len 下的 scaling。

## 5. Nsight Systems

需要系统权限，命令大致是：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  -o /tmp/vllm.nsys-rep \
  python your_script.py
```

然后用 `nsys-ui` 打开。重点看：

- GPU 利用率是否长期低于 50%；
- kernel 之间是否有 gap；
- NCCL kernel 是否与计算重叠；
- H2D/D2H 拷贝是否阻塞。

nsys 对多进程支持有限，通常只 profile rank 0；TP/PP 场景要分别抓取或看 NCCL 时间线。

## 6. Metrics 与日志

vLLM 暴露 Prometheus 指标（`vllm/v1/metrics/prometheus.py`）：

- `vllm:num_requests_running` / `waiting`；
- `vllm:kv_cache_usage_perc`；
- `vllm:time_to_first_token_seconds`；
- `vllm:inter_token_latency_seconds`；
- `vllm:e2e_request_latency_seconds`；
- `vllm:request_success_total`。

排查顺序：

1. `kv_cache_usage_perc` 接近 1 → 显存不足，请求在排队或抢占；
2. `waiting` 持续上升 → 调度/容量问题；
3. TTFT 高但 running 不高 → prefill 太慢或排队策略；
4. TPOT 高 → decode kernel / launch / 通信。

`--enable-per-request-metrics` 可以在响应里带每个请求的耗时分解（见 `docs/features/per_request_metrics.md`）。

## 7. 四类瓶颈的定位

### 7.1 排队 / 容量

现象：waiting 高、KV usage 高、TTFT 高。

处理：

- 降低 `max_num_seqs` / `max_num_batched_tokens`，减少单步时间；
- 开 prefix caching 复用 KV；
- 用 FP8 KV cache 或 KV offload；
- 增加副本 / DP。

### 7.2 prefill 慢

现象：TTFT 高、prefill kernel 占比高。

处理：

- 开 chunked prefill，避免长 prompt 阻塞 decode；
- 换更快的 attention backend；
- 用 TP 分摊 prefill；
- 检查是否被 CPU 预处理（tokenize / mm）拖慢。

### 7.3 decode 慢

现象：TPOT 高、GPU 利用率低、kernel 之间有 gap。

处理：

- 开 CUDA Graph / torch.compile；
- 增大 batch（提高 GPU 利用率）；
- 减少 Python 开销（output processor、logprobs）；
- 检查是否每步都在重新编译/捕获。

### 7.4 通信慢

现象：NCCL kernel 占比高、TP/EP 扩展效率低。

处理：

- 换 all-reduce / all2all 后端；
- 检查 NVLink / IB 拓扑；
- 减少 TP/EP 规模或改用 DP；
- 开通信-计算 overlap（DBO）。

## 8. 常见误区

1. **不 warmup 就 profile**。第一次包含编译、捕获、JIT，数据不可用。
2. **只看平均延迟**。尾延迟（P99）才是 SLO 的关键。
3. **在 profiler 开启时压测**。profiler 本身有开销，吞吐数据会失真。
4. **只抓 rank 0**。TP/PP 下瓶颈可能在别的 rank。
5. **把 GPU 利用率低当成 GPU 慢**。更多时候是在等 CPU 或通信。

## 9. 自检 5 问

1. TTFT 高、TPOT 正常，优先查什么？
2. TPOT 高、GPU 利用率低，优先查什么？
3. layerwise profile 和 nsys 的区别是什么？
4. 为什么要先 warmup 再 profile？
5. KV usage 接近 1 时会发生什么？

参考答案：

1. 排队、prefill、KV 加载。
2. CPU launch、CUDA Graph、batch 大小、通信。
3. layerwise 按模块聚合；nsys 看时间线和重叠。
4. 避免把编译/捕获/JIT 的开销算进稳态。
5. 新请求排队、可能触发抢占/重算，TTFT 和吞吐下降。

## 10. 延伸阅读

- `docs/usage/metrics.md`；
- `docs/design/metrics.md`；
- `vllm/profiler/`；
- 下一课：课时20 进入异步调度、DBO 与 ModelRunner V2。
