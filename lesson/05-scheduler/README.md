# 课时05 vLLM 核心组件：调度器如何实现

> 版本锚点：`main@8c17e46`。代码块标题是片段起点，省略处用 `...` 标注。

## 0. 本节要解决什么

1. Scheduler 到底在调度什么；
2. waiting / running 队列如何流转；
3. chunked prefill、prefix caching、抢占如何共用一套逻辑；
4. AsyncScheduler 和普通 Scheduler 的差别。

**一句话结论**：Scheduler 不区分 prefill 和 decode，它只做一件事——在每步的 token budget 内，让每个请求的 `num_computed_tokens` 尽量追上 `num_tokens_with_spec`。

## 1. 先看数据，不看流程

### 1.1 Request 的两个计数

调度器只关心两个数字：

- `num_computed_tokens`：这个请求已经算过 KV 的 token 数；
- `num_tokens_with_spec`：这个请求最终需要算的 token 数，等于 prompt + 已生成 + 投机草稿。

每步调度的本质：

```text
num_new_tokens = num_tokens_with_spec + 占位符 - num_computed_tokens
```

prefill 是“差很多”，decode 是“差 1”，chunked prefill 是“差很多但 budget 只够一部分”。三者没有本质区别。

### 1.2 RequestStatus

`RequestStatus`（`vllm/v1/request.py:427`）决定请求在哪个队列：

| 状态 | 含义 |
|---|---|
| `WAITING` | 在 waiting 队列，等 token budget 或 KV block |
| `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` | 等 grammar 编译 |
| `WAITING_FOR_REMOTE_KVS` | PD 场景，等远端 KV |
| `WAITING_FOR_STREAMING_REQ` | 流式输入，等下一段输入 |
| `RUNNING` | 已进入 running 列表 |
| `PREEMPTED` | 被抢占，KV 已释放，回到 waiting |
| `FINISHED_*` | 终止状态 |

注意 `PREEMPTED` 之后全部算 finished，这是 `is_finished()` 用 `status > PREEMPTED` 判断的原因。

### 1.3 SchedulerOutput

`Scheduler.schedule` 的返回值只描述“这一 step”：

| 字段 | 含义 |
|---|---|
| `scheduled_new_reqs` | 第一次被调度的请求（需要把完整数据发给 worker） |
| `scheduled_cached_reqs` | 已经在 worker 侧有状态的请求（只发 diff） |
| `num_scheduled_tokens` | `req_id -> 本步算多少 token` |
| `total_num_scheduled_tokens` | 本步总算多少 token |
| `finished_req_ids` | 本步结束的请求，worker 要释放状态 |
| `kv_connector_metadata` | PD / offload 的 KV 传输指令 |

`scheduled_new_reqs` 和 `scheduled_cached_reqs` 的分工是关键：worker 侧缓存了请求状态，后续 step 只传增量，避免每步重传 prompt。

## 2. 五个队列

`Scheduler.__init__`（`scheduler.py:96`）里不是只有一个 waiting 队列：

| 队列 | 作用 |
|---|---|
| `waiting` | 普通等待请求 |
| `kv_holding_waiting` | 已经持有 KV block 的等待请求，必须优先排空 |
| `async_kv_waiting` | 等待异步 KV load 的请求 |
| `deferred_waiting` | 本步跳过 / 被阻塞的请求 |
| `running` | 正在运行的请求列表 |

为什么要拆这么细：一个持有 KV block 的请求如果排在普通 waiting 后面，而前面的请求分配失败触发抢占，就可能出现“该释放的没释放、该跑的跑不了”的死结。所以 block-holder 单独一队，先排空。

```python title="vllm/v1/core/sched/scheduler.py:240"
self.waiting = create_request_queue(self.policy)
self.kv_holding_waiting = create_request_queue(self.policy)
self.async_kv_waiting = create_request_queue(self.policy)
self.deferred_waiting: set[Request] = set()
self.running: list[Request] = []
```

## 3. schedule：先 running，后 waiting

`Scheduler.schedule`（`scheduler.py:702`）分两段：

1. **running 段**（`scheduler.py:778`）：先保证已经在跑的请求；
2. **waiting 段**（`scheduler.py:1040`）：再用剩余 budget 接纳新请求。

### 3.1 token budget

```python title="vllm/v1/core/sched/scheduler.py:728"
token_budget = self.max_num_scheduled_tokens
spec = self.vllm_config.speculative_config
draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
input_budget = self.scheduler_config.max_num_batched_tokens
```

`token_budget` 和 `input_budget` 的区别：`token_budget` 是本步总算力预算，`input_budget` 是输入 token 预算；投机解码会预留 `draft_slots`，避免草稿 token 把输入预算吃光。

### 3.2 running 段

```python title="vllm/v1/core/sched/scheduler.py:778"
# First, schedule the RUNNING requests.
req_index = 0
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
    ...
    num_new_tokens = (
        request.num_tokens_with_spec
        + request.num_output_placeholders
        - request.num_computed_tokens
    )
    if 0 < long_prefill_token_threshold < num_new_tokens:
        num_new_tokens = long_prefill_token_threshold
    num_new_tokens = min(
        num_new_tokens, token_budget, input_budget - draft_slots
    )
    ...
    new_blocks = self.kv_cache_manager.allocate_slots(
        request,
        num_new_tokens,
        num_lookahead_tokens=self.num_lookahead_tokens,
    )
```

三个关键点：

- `min(..., token_budget, ...)` 就是 chunked prefill：预算不够就只算一段；
- `allocate_slots` 失败时，会抢占 `self.running[-1]`（FCFS）或最低优先级请求，然后重试；
- 如果 `num_new_tokens == 0`，用 `continue` 而不是 `break`，允许后面的请求先跑（不严格 FCFS）。

### 3.3 waiting 段

waiting 段先做 prefix cache 查找，再分配 block：

```python title="vllm/v1/core/sched/scheduler.py:1040"
# Next, schedule the WAITING requests.
if not preempted_reqs and self._pause_state != PauseState.PAUSED_ALL:
    ...
    request = request_queue.peek_request()
    ...
    if request.num_computed_tokens == 0:
        (
            new_computed_blocks,
            num_new_local_computed_tokens,
            request.shared_prefix_boundary,
            hit_diverged,
        ) = self._get_local_prefix_cache_hit(request, max_cache_hit_length)
```

命中 prefix cache 后，`num_computed_tokens` 可以直接跳过这一段，只分配新 token 的 block。这也是为什么调度器不需要单独的“prefix caching 分支”：命中只是让 `num_computed_tokens` 变大。

### 3.4 抢占

`_preempt_request`（`scheduler.py:1831`）做四件事：

1. 释放请求的 KV block 和 encoder cache；
2. `request.num_computed_tokens = 0`；
3. 清空投机草稿；
4. 把请求 `prepend` 回 waiting 队列（下次从头发起）。

vLLM v1 只做**重算式抢占**：被抢占的请求不保留 KV，恢复时重新 prefill。实现简单，代价是重复计算。

## 4. update_from_output：结算

`Scheduler.update_from_output`（`scheduler.py:2334`）是调度的另一半。它按 `model_runner_output.req_ids` 的顺序处理：

1. 扣减 `num_in_flight_tokens`；
2. 追加新 token，更新 `num_computed_tokens`；
3. 处理投机解码的接受/拒绝，回滚被拒绝的 token；
4. 判断停止条件（`max_tokens`、stop string、abort、长度上限）；
5. 释放结束请求的 block；
6. 生成 `EngineCoreOutputs`。

停止判断在 `_update_request_with_output` 里。一个容易忽略的点：**投机解码的拒绝会回滚 `num_computed_tokens`**，所以同一 step 里“生成了 3 个 token”不一定意味着“3 个都被接受”。

## 5. AsyncScheduler：把调度提前一步

`AsyncScheduler`（`async_scheduler.py:10`）几乎不改调度逻辑，只重写：

```python title="vllm/v1/core/sched/async_scheduler.py:18"
def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
    super()._update_after_schedule(scheduler_output)
    spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
    ...
    for req_id in scheduler_output.num_scheduled_tokens:
        request = self.requests[req_id]
        if request.is_prefill_chunk:
            continue
        cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
        ...
        request.num_output_placeholders += (
            self.num_sampled_tokens_per_step + cur_num_spec_tokens
        )
        request.num_output_placeholder_steps += 1
        request.spec_token_ids = spec_token_placeholders
        request.next_decode_eligible_step = self.current_step + self.decode_stagger
```

开启 `async_scheduling` 后，EngineCore 可以在上一步 forward 还没结束时调度下一步，用 `num_output_placeholders` / `num_in_flight_tokens` 表示“已经派发但还没拿到结果”的 token。这也是 `pp_size` 步 decode 间隔限制（`next_decode_eligible_step`）存在的原因。

代价是停止条件可能晚一步生效，所以调度器要避免多算一步（`scheduler.py:786` 的 placeholder 判断）。

## 6. 最小实验：观察 chunked prefill 和抢占

```python
import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler

original_schedule = Scheduler.schedule


def traced_schedule(self, *args, **kwargs):
    out = original_schedule(self, *args, **kwargs)
    print(
        f"step={self.current_step:>3} "
        f"scheduled={dict(out.num_scheduled_tokens)} "
        f"running={len(self.running)} waiting={len(self.waiting)}"
    )
    return out


Scheduler.schedule = traced_schedule

llm = LLM(
    model="facebook/opt-125m",
    max_model_len=128,
    max_num_batched_tokens=8,   # 故意调小，触发 chunked prefill
    enforce_eager=True,
    gpu_memory_utilization=0.3,
)
prompts = ["Hello " * 20, "Hi"]
params = SamplingParams(max_tokens=4)
llm.generate(prompts, params)
```

你会看到第一个请求的 prompt 被拆成多个 step（`scheduled={"0": 8}` 重复多次），而第二个请求排在其后。把 `max_num_batched_tokens` 调大，prefill 会合并成一两步。再把 `max_num_seqs` 调到 1，可以观察第二个请求一直留在 waiting。

## 7. 常见误区

1. **以为 Scheduler 有 prefill / decode 两个阶段**。它只有 token budget 和两个计数。
2. **以为 waiting 只有一个队列**。现在有 `kv_holding_waiting` / `async_kv_waiting` / `deferred_waiting`，顺序会影响正确性。
3. **以为抢占会保留 KV**。v1 是重算式抢占，`num_computed_tokens` 直接归零。
4. **以为 `max_num_seqs` 就是 running 长度**。还受 `max_num_active_reqs` 和 streaming input 的占位影响。
5. **以为 async scheduling 只是“多跑一步”**。它改变了 token 的可见性，停止条件和投机回滚都要额外处理。

## 8. 自检 5 问

1. 调度器如何判断一个请求该算多少 token？
2. chunked prefill 是在哪一行生效的？
3. prefix cache 命中后，`SchedulerOutput` 里哪个字段会变小？
4. 抢占一个请求时，为什么要把它 `prepend` 回 waiting？
5. AsyncScheduler 的 placeholders 解决什么问题？

参考答案：

1. `num_tokens_with_spec + 占位符 - num_computed_tokens`，再和 budget 取 min。
2. `num_new_tokens = min(num_new_tokens, token_budget, input_budget - draft_slots)`（`scheduler.py:834` 附近）。
3. `num_scheduled_tokens` 中该请求的值；已命中的 token 不再需要计算。
4. 让它在下一轮优先恢复，避免被新请求饿死。
5. 表示“已派发但还没拿到结果”的 token，允许异步调度下一步而不重复计算或提前停止。

## 9. 延伸阅读

- `docs/design/prefix_caching.md`；
- `vllm/v1/core/sched/output.py` 的 `SchedulerOutput`；
- `vllm/v1/core/sched/async_scheduler.py` 全文（很短）；
- 下一课：课时06 进入 KV cache 管理，看 `allocate_slots` 背后发生了什么。
