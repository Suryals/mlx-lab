# Episode 1 — X Thread Draft

> ✍️ Real numbers filled in. Tighten the voice to sound like you, then post.
> Delete this header before posting.

---

**Tweet 1 (hook)**
I benchmarked local LLMs on my M5 Max — not through a GUI, through Apple's own ML framework (MLX), so every number is reproducible.

3B model: 227 tokens/sec.
7B model: 117 tokens/sec.
On a laptop. No cloud.

Here's what I measured, and the mistake that almost fooled me 🧵

---

**Tweet 2 (why MLX, not LM Studio/Ollama)**
3 ways to run a local LLM on a Mac:
• LM Studio — GUI, easy, abstracted
• Ollama — CLI, llama.cpp under the hood
• MLX — code, Apple Silicon native

For *benchmarking*, you want the one you can script and measure to the millisecond.
MLX is the dyno. The others are the car.

---

**Tweet 3 (the setup — the 128GB flex)**
Hardware: M5 Max, 128GB unified memory.

Unified memory = CPU and GPU share one RAM pool. A model doesn't need a separate GPU — it just lives in RAM.

That's the whole Apple Silicon AI story: you can run models locally that need a $10k+ GPU on a PC.

---

**Tweet 4 (the numbers)**
Same 3 prompts, MLX, 4-bit quantized:

| Model | Decode | TTFT | RAM |
|-------|--------|------|-----|
| Llama-3.2-3B | 227 tok/s | 89ms | 2.1GB |
| Qwen2.5-7B   | 117 tok/s | 115ms| 4.5GB |

2.3x the params → 1.9x slower, 2.2x memory.
Near-linear. Decode is memory-bandwidth-bound — more weights to stream per token.

---

**Tweet 5 (the mistake — the credibility beat)**
My first 7B run: 71 tok/s.
Re-ran it: 117 tok/s.

Same model. Same script. What changed?

The first run happened while a model download was finishing in the background — starving the decode loop of bandwidth.

Background load contaminates benchmarks. Always measure on a quiet machine. This is the line between measurement and vibes.

---

**Tweet 6 (how I measured it)**
The harness captures, per run:
• TTFT — time to first token (wall clock)
• Prefill tok/s — prompt processing rate
• Decode tok/s — generation rate
• Peak memory — MLX's own high-water mark

Warm-up pass discarded (Metal shader compile). All in ~120 lines of Python.

---

**Tweet 7 (repo + what's next)**
Full code, reproducible:
→ github.com/Suryals/mlx-lab

Next episode: agent/tool-use eval.
Can a local 7B model pick the right tool for an ops alert? That's the number that actually matters for autonomous systems.

Follow along — building in public.

#MLX #AppleSilicon #LocalLLM #AIEval #BuildInPublic
