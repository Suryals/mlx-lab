# mlx-lab

> Local LLM benchmarking & agent eval on Apple Silicon.
> M5 Max / 128GB — measuring what others are guessing.

Build-in-public project. Each episode = a reproducible experiment + an X thread + a portfolio entry.

## Episodes

| # | Focus | Status |
|---|-------|--------|
| [Ep 01](content/ep01-portfolio.md) | Raw speed/memory benchmark: MLX vs LM Studio | 🔨 In progress |
| Ep 02 | Agent / tool-use correctness eval | Planned |

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
