"""Print the vLLM v1 request lifecycle with a tiny monkeypatch tracer.

Run:
    VLLM_ENABLE_V1_MULTIPROCESSING=0 .venv/bin/python trace_lifecycle.py

Requires a GPU that can run vLLM. The model is small (~250 MB) and is
downloaded from HuggingFace on first run.
"""

import functools
import os

# Keep EngineCore in this process; otherwise the hooks below only patch the
# frontend and never see Scheduler / ModelRunner calls.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import time

from vllm import LLM, SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

state = {"depth": 0, "enabled": False}


def hook(cls, name, label, describe=None):
    """Wrap a method so it prints an enter/exit line with call depth."""
    original = getattr(cls, name)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        if not state["enabled"]:
            return original(*args, **kwargs)

        indent = "  " * state["depth"]
        detail = describe(args, kwargs) if describe else ""
        print(f"{indent}-> {label} {detail}")
        state["depth"] += 1
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            state["depth"] -= 1
            elapsed = (time.perf_counter() - start) * 1000
            print(f"{indent}<- {label} {elapsed:.2f} ms")

    setattr(cls, name, wrapper)


hook(LLMEngine, "step", "LLMEngine.step")
hook(EngineCore, "step", "EngineCore.step")
hook(
    Scheduler,
    "schedule",
    "Scheduler.schedule",
    lambda a, k: f"// throttle_prefills={a[1] if len(a) > 1 else k.get('throttle_prefills')}",
)
hook(Scheduler, "update_from_output", "Scheduler.update_from_output")
hook(
    GPUModelRunner,
    "execute_model",
    "GPUModelRunner.execute_model",
    lambda a, k: f"scheduled_tokens={a[1].total_num_scheduled_tokens}",
)
hook(GPUModelRunner, "sample_tokens", "GPUModelRunner.sample_tokens")


def main():
    llm = LLM(
        model="facebook/opt-125m",
        max_model_len=256,
        enforce_eager=True,  # disable CUDA Graph / torch.compile for clean logs
        gpu_memory_utilization=0.3,
    )
    params = SamplingParams(max_tokens=4, temperature=0)

    # Ignore the dummy runs executed during engine startup / profiling.
    state["enabled"] = True
    outputs = llm.generate(["Hello, my name is"], params)
    state["enabled"] = False

    print("\n=== final output ===")
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
