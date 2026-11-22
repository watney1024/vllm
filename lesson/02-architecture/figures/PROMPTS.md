# 课时02 配图 Prompt

> 用途：为 `lesson/02-architecture/README.md` 生成 5 张正文配图（封面可选）。
> 说明：图像模型对长文本和箭头关系不稳定。建议优先用“无文字底图 + Figma/Excalidraw 叠字”的方式；如果模型支持文字，生成后逐字校对标签。
> 当前状态：6 张（5 张正文配图 + 1 张封面）已用 Seedream 5.5 生成并放入本目录（2026-11-22）。图2-2 的 B/C 栏文字拥挤，图2-3 的步骤说明压字，后续可按下方 prompt 重生成。

## 统一风格前缀

把下面这段放在每个 prompt 最前面，保证 5 张图风格一致：

```text
Clean modern technical infographic, flat vector style, white background,
rounded rectangles with thin 1px borders, thin arrows, generous whitespace,
sans-serif typography, limited palette: blue #2563EB, amber #F59E0B,
emerald #10B981, violet #8B5CF6, slate #64748B, no gradients, no shadows,
no 3D, no photorealistic elements, crisp aligned text, 16:9 aspect ratio.
```

统一负面提示词：

```text
blurry text, gibberish letters, misspelled words, extra labels, watermark,
logo, clutter, dark background, neon, 3D render, photorealistic, people,
decorative icons, excessive detail
```

---

## 图 2-1 分层架构

**文件名**：`figures/02-01-layers.png`
**放在**：第 2 节“分层结构”表格之后
**画面**：纵向 6 层堆叠，左侧下行箭头标 “request”，右侧上行箭头标 “output”。

```text
[统一风格前缀]

A vertical six-layer architecture diagram. Six equal-width rounded
rectangles stacked top to bottom with 22px gaps. Each layer has a bold
title on the left and two or three small gray class chips on the right.

From top to bottom:
1. blue box, title "Entrypoints", chips "LLM" and "OpenAI API";
2. blue box, title "Frontend", chips "LLMEngine / AsyncLLM" and
   "InputProcessor / OutputProcessor";
3. violet box, title "EngineClient", chips "InprocClient",
   "SyncMPClient", "AsyncMPClient";
4. amber box, title "EngineCore", chips "Scheduler" and "KV Cache";
5. amber box, title "Executor", chips "UniProc / Multiproc / Ray";
6. green box, title "Worker + ModelRunner", chips "GPUModelRunner".

On the left margin, one long blue downward arrow spanning all six layers,
labeled "request path". On the right margin, one long green upward arrow,
labeled "output path". A small note next to the Frontend box:
"tokenize / detokenize happen here". Keep all text horizontal and readable.
```

---

## 图 2-2 三种进程拓扑

**文件名**：`figures/02-02-processes.png`
**放在**：第 3 节末尾
**画面**：横向三栏，A/B/C 三种部署形态并排。

```text
[统一风格前缀]

A three-panel comparison diagram, three equal columns labeled "A", "B", "C"
at the top. Each panel is a light gray rounded container with a caption
under it.

Panel A, caption "Offline, TP=1, 2 processes":
two boxes side by side. Left blue box "Frontend process" containing
"LLMEngine" and "SyncMPClient". Right amber box "EngineCore process"
containing "EngineCore", "Scheduler", "UniProcExecutor" and
"GPUModelRunner". A double-headed arrow between them labeled "ZMQ".

Panel B, caption "Offline, TP=2, 4 processes":
left blue box "Frontend" with "LLMEngine" and "SyncMPClient"; middle amber
box "EngineCore" with "Scheduler" and "MultiprocExecutor"; right two green
boxes stacked vertically, both labeled "Worker + GPUModelRunner". Double
arrow from Frontend to EngineCore labeled "ZMQ"; thin arrows from
EngineCore to both workers labeled "RPC".

Panel C, caption "Online, DP=2":
left violet box "API server" with "AsyncLLM" and "AsyncMPClient"; a small
amber box "DPCoordinator" in the middle; right two amber boxes stacked
vertically, each "EngineCore + Scheduler", each connected to a small green
"Worker" box. Arrows: API server to coordinator, coordinator to both
engines.

Use consistent colors: blue for frontend, violet for client/API, amber for
engine, green for worker. Keep labels short and horizontal.
```

---

## 图 2-3 请求生命周期时序图

**文件名**：`figures/02-03-lifecycle.png`
**放在**：第 4 节 Mermaid 时序图之后（或替换）
**建议**：时序图最容易把文字画乱。可以先生成无文字骨架，再用 Excalidraw/Figma 加 lifeline 名称和步骤编号。

```text
[统一风格前缀]

A horizontal sequence diagram with six vertical lifelines, evenly spaced,
each topped with a rounded label box. Lifelines from left to right:
"User", "LLMEngine / AsyncLLM", "EngineClient", "EngineCore",
"Scheduler", "Executor / Worker".

Numbered arrows from left to right, top to bottom:
1. User -> LLMEngine: "generate()";
2. LLMEngine -> EngineClient: "add_request";
3. EngineClient -> EngineCore: "EngineCoreRequest";
4. EngineCore -> Scheduler: "add_request";
5. EngineCore -> Scheduler: "schedule()";
6. Scheduler --> EngineCore: "SchedulerOutput";
7. EngineCore -> Executor / Worker: "execute_model()";
8. EngineCore -> Executor / Worker: "sample_tokens()";
9. Executor / Worker --> EngineCore: "AsyncOutput / ModelRunnerOutput";
10. EngineCore -> Scheduler: "update_from_output()";
11. Scheduler --> LLMEngine: "EngineCoreOutputs";
12. LLMEngine -> User: "RequestOutput".

Draw a large rounded rectangle around steps 4 to 11, labeled
"one engine step (loop until all requests finish)". Use blue arrows for
requests, green for outputs, gray for internal calls. Number every arrow
with a small circle. Keep lifelines thin and vertical; no crossing arrows.
```

---

## 图 2-4 核心数据结构转换链

**文件名**：`figures/02-04-data-structures.png`
**放在**：第 5 节“转换链”代码块之前

```text
[统一风格前缀]

A left-to-right pipeline of six rounded cards, connected by thin arrows.
Each card has a colored header and two or three small gray field chips
under it. Cards from left to right:

1. blue header "EngineCoreRequest", chips "request_id",
   "prompt_token_ids", "sampling_params";
2. violet header "Request", chips "status",
   "num_computed_tokens";
3. amber header "SchedulerOutput", chips "num_scheduled_tokens",
   "finished_req_ids";
4. green header "ModelRunnerOutput", chips "req_ids",
   "sampled_token_ids";
5. amber header "EngineCoreOutputs", chips "outputs",
   "scheduler_stats";
6. blue header "RequestOutput", chips "text", "finish_reason".

Arrow labels above the connections, in order: "InputProcessor",
"schedule()", "forward + sample", "update_from_output()",
"detokenize". Under the pipeline, a thin gray line labeled
"token ids → text" with a small arrow at the end. Keep all card widths
equal and all text horizontal.
```

---

## 图 2-5 forward 与采样/拷贝的时间线

**文件名**：`figures/02-05-timeline.png`
**放在**：第 6.5 节“执行与采样”之后
**画面**：三轨甘特图，突出 CPU 与 GPU 的重叠。

```text
[统一风格前缀]

A horizontal timeline / Gantt diagram with three lanes stacked vertically
and time flowing left to right. Lane labels on the left:
"CPU thread", "GPU compute stream", "GPU copy stream". A subtle vertical
dashed line marks the end of one engine step.

Blocks on the CPU lane, left to right:
"schedule()", "grammar bitmask", "update_from_output()".
Blocks on the GPU compute lane:
"model forward" (long), "sample".
Blocks on the GPU copy stream:
"D2H copy of sampled tokens".

Show overlap: "grammar bitmask" sits directly above the "model forward"
block, and "D2H copy" sits directly below the tail of "sample", both with
dashed vertical alignment guides. Use blue for CPU, amber for GPU compute,
green for GPU copy. Add a small legend at the bottom. Above the timeline,
a bracket spanning "one EngineCore.step()". Keep labels short and
horizontal, no arrows crossing.
```

---

## 可选封面

**文件名**：`figures/02-00-cover.png`
**用途**：课程页头图，不放正文
**画面**：等距视角的 GPU 服务流水线，不出现文字，避免和正文标签冲突。

```text
Clean isometric technical illustration, white background, soft blue and
amber palette, a stylized GPU server rack on the right, a queue of glowing
request cards on the left, and a four-stage pipeline in the middle:
frontend, scheduler, executor, GPU. Thin connecting lines with small
directional arrows. Minimal, no text, no logos, no people, generous
negative space, 2:1 aspect ratio, flat vector style, subtle depth, no
photorealistic rendering.
```

---

## 生成后检查清单

- [ ] 文字没有拼错，尤其是 `EngineCoreOutputs`、`num_computed_tokens`、`GPUModelRunner`；
- [ ] 箭头方向与正文一致：请求向下/向右，输出反向；
- [ ] 三张图的颜色语义一致（蓝=前端、紫=客户端、橙=引擎、绿=Worker）；
- [ ] 图中没有正文没有提到的类名；
- [ ] 5 张图缩到 800px 宽仍能看清标签；
- [ ] 如果文字仍然糊，保留无文字版，用 Excalidraw 重排标签。
