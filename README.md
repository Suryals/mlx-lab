# mlx-lab

> Local LLM benchmarking & agent eval on Apple Silicon.
> M5 Max / 128GB — measuring what others are guessing.

Build-in-public project. Each episode = a reproducible experiment + an X thread + a portfolio entry.

## Episodes

| # | Focus | Status |
|---|-------|--------|
| [Ep 01](content/ep01-portfolio.md) | Raw speed/memory benchmark (3B + 7B) | ✅ Done |
| [Ep 02](content/ep02-portfolio.md) | How far can one Mac go? Ladder to 123B + memory-ceiling probe | ✅ Done |
| Ep 03 | Agent / tool-use correctness eval | Planned |

## Episode 2 — the scaling ladder (M5 Max, 128GB)

| Params | Decode | TTFT | Peak RAM |
|--------|--------|------|---------|
| 3.2B | 227 tok/s | 89ms | 2.1GB |
| 7.6B | 117 tok/s | 115ms | 4.5GB |
| 14B | 60 tok/s | 151ms | 8.5GB |
| 32B | 27 tok/s | 326ms | 18.6GB |
| 70B | 11 tok/s | 1.8s | 39.9GB |
| 123B | 6.3 tok/s | 4.6s | 69.2GB |

A 123B model ran in 69GB. The memory probe allocated to 128GB without Metal refusing — the advertised
115GB "recommended working set" is a soft limit, not a wall. Full writeup: [content/ep02-portfolio.md](content/ep02-portfolio.md).

```bash
uv run python bench/run_ladder.py      # the full ladder
uv run python bench/probe_ceiling.py   # the memory ceiling probe
```

## Reproduce Episode 1

```bash
# 1. Clone + install
git clone https://github.com/Suryals/mlx-lab
cd mlx-lab
uv venv --python 3.12 && source .venv/bin/activate
uv sync

# 2. Pull the benchmark model (~4GB download)
uv run mlx_lm.generate \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --prompt "hello" --max-tokens 5

# 3. Run the benchmark across model sizes (writes one JSON per model)
uv run python bench/run_bench.py --engine mlx --model mlx-community/Llama-3.2-3B-Instruct-4bit
uv run python bench/run_bench.py --engine mlx --model mlx-community/Qwen2.5-7B-Instruct-4bit

# 4. View results
cat results/ep01-mlx-*.json
```

## Episode 1 results (M5 Max, idle machine)

| Model              | Decode tok/s | TTFT  | Peak RAM |
|--------------------|--------------|-------|---------|
| Llama-3.2-3B-4bit  | ~227         | ~89ms | 2.06GB  |
| Qwen2.5-7B-4bit    | ~117         | ~115ms| 4.45GB  |

See [`content/ep01-portfolio.md`](content/ep01-portfolio.md) for the full writeup.

## Stack

- [mlx-lm](https://github.com/ml-explore/mlx-lm) — Apple Silicon native inference
- [psutil](https://github.com/giampaolo/psutil) — process memory sampling
- [httpx](https://www.python-httpx.org/) — LM Studio API client
- Python 3.12 / [uv](https://docs.astral.sh/uv/)

## Hardware

Apple M5 Max · 128GB unified memory · macOS 26.3
