# Episode 1: Benchmarking Local LLMs on Apple Silicon with MLX

> Portfolio entry — fill in [PLACEHOLDERS] with real numbers after the benchmark run.

## What I built

A reproducible benchmark harness that measures **time-to-first-token (TTFT)**, **prefill throughput**,
**decode throughput**, and **peak memory** for local LLMs on Apple Silicon, comparing two runtimes:
`mlx-lm` (Apple's native ML framework) and LM Studio's local API server.

## Why it matters

Running LLMs locally on Apple Silicon is increasingly viable — but "it runs" is not a measurement.
The goal of this project is to build a real eval harness that produces numbers, not vibes.
That harness will grow: Episode 2 adds agent/tool-use correctness eval, which is directly
applicable to production agentic-ops systems I'm building.

## Hardware

- **Machine:** Apple M5 Max, 128GB unified memory, 18 cores
- **OS:** macOS 26.3
- **Key insight:** Unified memory means the model lives in the same RAM pool as the CPU/GPU.
  A 40GB 4-bit 70B model doesn't need a discrete GPU — it just needs RAM.

## Results — MLX (`mlx-lm` 0.31.2) across model sizes

Three prompts per model (short factual, medium reasoning, long generation), averaged.
Machine idle during measurement (see "What I learned" — this matters).

| Model              | Params | Prefill tok/s | Decode tok/s | TTFT  | Peak RAM |
|--------------------|--------|--------------|--------------|-------|---------|
| Llama-3.2-3B-4bit  | 3B     | ~1,625       | ~227         | ~89ms | 2.06GB  |
| Qwen2.5-7B-4bit    | 7B     | ~718         | ~117         | ~115ms| 4.45GB  |

**Scaling:** 2.3× the parameters → ~1.9× slower decode, ~2.2× memory. Near-linear, exactly what
theory predicts for memory-bandwidth-bound 4-bit decode. Both run comfortably faster than human
reading speed on a laptop.

Full results JSON:
[`results/ep01-mlx-llama-3.2-3b-instruct-4bit.json`](../results/ep01-mlx-llama-3.2-3b-instruct-4bit.json) ·
[`results/ep01-mlx-qwen2.5-7b-instruct-4bit.json`](../results/ep01-mlx-qwen2.5-7b-instruct-4bit.json)

## What I learned

- **Prefill vs decode:** Prefill (processing your prompt) is embarrassingly parallelizable;
  decode (generating tokens one at a time) is memory-bandwidth-bound. They scale differently.
- **Warm-up matters:** First run always slower — metal shader compilation. Always discard run 0.
- **Background load contaminates benchmarks:** My first 7B run measured 71 tok/s decode. Re-run on
  an idle machine: 117 tok/s. Same model, same script — the difference was a model download finishing
  in the background, starving the decode loop of disk/network bandwidth. Always benchmark on a quiet
  machine and record system state. This is the line between measurement and vibes.

## Code

[`bench/run_bench.py`](../bench/run_bench.py) — the measurement loop I wrote by hand (learning mode).
[`bench/engines.py`](../bench/engines.py) — MLX and LM Studio adapters.
[`bench/prompts.py`](../bench/prompts.py) — fixed prompt set (fair comparison requires identical inputs).

**GitHub:** [GITHUB_REPO_LINK]

## Next: Episode 2

Does a local 7B model call the right tool for an ops alert?
Adding a task set + scorer to measure agent/tool-use correctness.
