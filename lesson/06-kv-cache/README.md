# 课时06 vLLM 核心组件：KV Cache 与 PagedAttention 显存管理

> 版本锚点：`main@8c17e46`。代码块标题是片段起点，省略处用 `...` 标注。

## 0. 本节要解决什么

1. PagedAttention 到底解决什么问题；
2. 一个 KV block 从分配到释放经历了什么；
3. prefix caching 如何在 block 级别共享；
4. Hybrid 模型（滑窗 / Mamba / 交叉注意力）如何共用一套管理器。

**一句话结论**：vLLM 把 KV cache 切成固定大小的物理 block，用请求私有的 block table 做逻辑到物理的映射；`BlockPool` 管物理块的空闲队列和哈希缓存，`KVCacheManager` 管请求级的分配、追加和释放。

## 1. 传统 KV cache 的问题

如果给每个请求预分配一段连续的 `[max_model_len, num_layers, num_kv_heads, head_dim]`：

- **内部碎片**：请求只生成 100 个 token，也要占满 `max_model_len`；
- **外部碎片**：不同长度的请求反复分配/释放，留下大小不一的空洞；
- **无法共享**：两个请求有相同 system prompt，也只能各存一份 KV。

PagedAttention 的思路借自操作系统虚拟内存：

- 物理内存切成固定大小的 block；
- 每个请求维护一张 block table，把逻辑位置映射到物理 block；
- 相同前缀的 block 可以共享，用引用计数管理生命周期。

## 2. 三个核心对象

| 对象 | 职责 | 源码 |
|---|---|---|
| `KVCacheBlock` | 一个物理块的元数据：`block_id`、`ref_cnt`、`block_hash` | `vllm/v1/core/kv_cache_utils.py:160` |
| `BlockPool` | 管理所有物理块：空闲队列、哈希查找、缓存/驱逐 | `vllm/v1/core/block_pool.py:107` |
| `KVCacheManager` | 面向请求：查前缀、分配 slot、释放 | `vllm/v1/core/kv_cache_manager.py:130` |

`KVCacheBlock` 的字段很少：

```python title="vllm/v1/core/kv_cache_utils.py:160"
class KVCacheBlock:
    """KV-cache block metadata."""
    block_id: int
    ref_cnt: int = 0
    block_hash: BlockHashWithGroupId | None = None
    block_hash_num_tokens: int | None = None
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None
    is_null: bool = False
```

`prev_free_block` / `next_free_block` 说明空闲块是用**侵入式双向链表**串起来的：不需要额外的 deque 节点，删除中间任意一个块是 O(1)。

队列的排序规则要分两种情况（`BlockPool.free_blocks`）：

- **非缓存块**释放时 `prepend` 到队头，分配也从队头取，即 LIFO——优先复用刚释放的块，GPU 局部性更好；
- **缓存块**释放时 `append` 到队尾，按 FIFO 排队，队头即最久未使用的块，需要腾空间时优先被驱逐（LRU 语义）。

`is_null` 是特殊块：当某个位置不需要真实 KV（例如 padding、被滑窗跳过的块）时，用 `block_id=0` 的 null block 占位，避免大量空指针判断。

## 3. BlockPool：物理块的管理者

```python title="vllm/v1/core/block_pool.py:107"
class BlockPool:
    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        ...
    ):
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx, pool=self._weak_self) for idx in range(num_gpu_blocks)
        ]
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        self.cached_block_hash_to_block = BlockHashToBlockMap(self.blocks)
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
        ...
```

三个要点：

1. `num_gpu_blocks` 不是配置里写死的，而是 Worker 在 `determine_available_memory` 里实测出来的（见课时04）；
2. `free_block_queue` 初始按 block id 排列；释放时非缓存块插队头（LIFO 复用），缓存块追加队尾（FIFO/LRU 驱逐）；
3. `cached_block_hash_to_block` 只在 `enable_caching=True` 时真正有内容，它把 hash 映射到一个或多个已缓存的块。

申请和释放：

```python title="vllm/v1/core/block_pool.py:681"
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    if num_blocks > self.get_num_free_blocks():
        raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
    ret = self.free_block_queue.popleft_n(num_blocks)
    ...
    for block in ret:
        block.ref_cnt = 1
    return ret
```

```python title="vllm/v1/core/block_pool.py:798"
def free_blocks(self, ordered_blocks, evict_first: bool = False) -> None:
    for block in ordered_blocks:
        if block.is_null:
            continue
        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            if block.block_hash is None or not self.enable_caching or evict_first:
                # 非缓存块直接回到队头，优先复用
                ...
            else:
                # 缓存块保留在池里，等被再次命中或被驱逐
                ...
```

注意 `ref_cnt`：同一个物理块可能被多个请求共享（prefix cache 命中），只有最后一个引用释放时才真正回到空闲队列。

## 4. 哈希前缀缓存

vLLM 的 prefix caching 是 **block 级哈希**：

```text
block_hash = hash(tokens_in_block, tokens_in_prefix_before_block, extra_keys)
```

- 只对“满块”计算 hash；
- 前缀不同，即使 block 内容相同也不会命中；
- `extra_keys` 包含 LoRA id、多模态 hash、cache salt 等，避免错误共享。

查找在 `KVCacheManager.get_computed_blocks`：

```python title="vllm/v1/core/kv_cache_manager.py:278"
def get_computed_blocks(self, request, max_cache_hit_length=None):
    if not self.prefix_cache_lookup_enabled(request):
        return self.empty_kv_cache_blocks, 0, 0
    ...
    computed_blocks, num_new_computed_tokens, num_uncached = (
        self.coordinator.find_longest_cache_hit(
            request.block_hashes, max_cache_hit_length, request.num_tokens
        )
    )
```

一个容易忽略的细节：即使所有 token 都命中，也要**重算最后一个 token** 来拿 logits，所以 `max_cache_hit_length` 会被限制到 `prompt_length - 1`。这可能导致最后一个 block 被整体重算，因为 `allocate_slots` 要求 block 对齐。

## 5. allocate_slots：一次分配的形状

`KVCacheManager.allocate_slots`（`kv_cache_manager.py:391`）的 docstring 里有一张非常重要的布局图：

```text
----------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
----------------------------------------------------------------------
                                           |   < to be computed >     |
----------------------------------------------------------------------
                       |            < to be allocated >              |
```

- `comp`：请求已经算过的 token；
- `new_comp`：本次命中的 prefix cache token；
- `ext_comp`：外部（PD connector）带来的 token；
- `new`：本次要算的新 token；
- `lookahead`：投机解码预留的草稿槽。

分配时按 block 对齐向上取整，已存在的块只增加 `ref_cnt`，缺的块从 `BlockPool` 取。返回值是 `KVCacheBlocks`（每个 KV cache group 一组 block id），失败返回 `None`，Scheduler 据此触发抢占。

## 6. Hybrid KV Cache：不止一种管理器

不同层的 KV 形状和生命周期可能不同：

| 模型结构 | 管理器 | 行为 |
|---|---|---|
| 普通 full attention | `FullAttentionManager` | 每个 token 都占一个位置 |
| 滑窗注意力 | `SlidingWindowManager` | 只保留窗口内的 block |
| Mamba / SSM | `MambaManager` | 状态块，按 block 边界 checkpoint |
| Encoder-decoder | `CrossAttentionManager` | 交叉注意力 KV 单独管理 |
| Chunked local | `ChunkedLocalAttentionManager` | 按 chunk 保留 |
| 循环缓冲 | `CircularBufferManager` | 固定大小环形复用 |
| HiSparse | `HiSparseSourceManager` 等 | GPU/CPU 分层 |

`KVCacheCoordinator`（`kv_cache_coordinator.py:84`）负责把多个 manager 组合起来：

- `UnitaryKVCacheCoordinator`：所有层同一种 manager；
- `HybridKVCacheCoordinator`：多层多种 manager，需要处理“哪些块可以共享”；
- `KVCacheCoordinatorNoPrefixCache`：关闭前缀缓存时的简化路径。

这也是为什么 `SchedulerOutput` 里的 block id 是一个 **tuple of lists**，而不是单个 list：每个 KV cache group 一组。

## 7. 最小实验：看 prefix cache 命中

```python
import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams

llm = LLM(
    model="facebook/opt-125m",
    max_model_len=256,
    enforce_eager=True,
    gpu_memory_utilization=0.3,
    enable_prefix_caching=True,
)
prompt = "The quick brown fox jumps over the lazy dog. " * 6
params = SamplingParams(max_tokens=1)

first = llm.generate([prompt], params)[0]
second = llm.generate([prompt], params)[0]

print("first cached tokens :", first.num_cached_tokens)
print("second cached tokens:", second.num_cached_tokens)

manager = llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager
print("kv cache usage:", manager.usage)
```

预期：第一次 `num_cached_tokens` 接近 0，第二次接近 prompt 长度（减去最后一个 token 的重算）。如果第二次仍然为 0，检查：

- `enable_prefix_caching` 是否为 True；
- prompt 是否至少有 2 个完整 block（默认 block size 64，见 `CacheConfig.DEFAULT_BLOCK_SIZE`）；
- 是否有随机 salt / LoRA / 多模态 hash 导致 key 不同。

## 8. 常见误区

1. **以为 `block_size` 就是 16**。当前默认是 64（`CacheConfig.DEFAULT_BLOCK_SIZE`），平台/后端还可能覆盖。
2. **以为 prefix cache 是 token 级**。它是 block 级哈希，最后一个不满的块不参与命中。
3. **以为 `ref_cnt` 只用于释放**。共享块的分配、驱逐、prefix cache 命中都依赖它。
4. **以为所有层共用一个 block table**。Hybrid 模型下每个 KV cache group 有自己的 manager 和 block 列表。
5. **以为 `num_gpu_blocks` 可以随便调**。它由实测显存决定；手动改只会改变 OOM 的时机。

## 9. 自检 5 问

1. PagedAttention 相比连续分配，解决了哪三类问题？
2. `BlockPool.get_new_blocks` 从空闲队列的哪一端取块？为什么？
3. prefix cache 的 hash 由哪些部分组成？
4. 为什么全命中时还要重算最后一个 token？
5. Hybrid KV Cache 下，`KVCacheManager` 如何知道每一层用哪种管理器？

参考答案：

1. 内部碎片、外部碎片、跨请求共享。
2. 队头；非缓存块从队头 LIFO 复用，缓存块按 FIFO 排队、队头是最久未使用的块，优先被驱逐。
3. 块内 token + 块前缀 token + extra keys（LoRA、多模态、salt 等）。
4. 最后一个 token 的 logits 只能由 forward 产生，缓存里没有。
5. 通过 `KVCacheConfig.kv_cache_groups` 中的 `manager_config`，由 `get_kv_cache_coordinator` 构建对应的 manager 列表。

## 10. 延伸阅读

- `docs/design/prefix_caching.md`；
- `docs/design/hybrid_kv_cache_manager.md`；
- `docs/design/paged_attention.md`（历史文档，解释原始论文思路）；
- 下一课：课时07 进入 ModelRunner，看 block table 如何变成 attention metadata。
