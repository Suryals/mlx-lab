"""
bench/run_bench.py — Episode 1 benchmark runner.

Usage:
    uv run python bench/run_bench.py --engine mlx
    uv run python bench/run_bench.py --engine mlx --model mlx-community/Qwen2.5-7B-Instruct-4bit
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from bench.engines import GenerateResult, mlx_engine
from bench.prompts import PROMPTS


# ─────────────────────────────────────────────
# YOUR CODE — the measurement loop you wrote
# ─────────────────────────────────────────────

def measure_run(result: GenerateResult) -> dict:
    """
    Drain result.token_stream and return timing + quality metrics.

    Returns a dict with keys:
        ttft_ms, prefill_tps, decode_tok_s,
        output_tokens, peak_memory_gb, wall_time_s, output_text
    """
    tokens = []
    t_first_token = None
    prefill_tps = None
    peak_memory_gb = 0.0

    t_start = time.perf_counter()

    for chunk in result.token_stream:
        if t_first_token is None:
            t_first_token = time.perf_counter()
            prefill_tps = chunk.prompt_tps

        tokens.append(chunk.text)
        peak_memory_gb = max(peak_memory_gb, chunk.peak_memory)

    t_end = time.perf_counter()

    ttft_ms       = (t_first_token - t_start) * 1000
    output_tokens = len(tokens)
    decode_tok_s  = output_tokens / (t_end - t_first_token)
    output_text   = "".join(tokens)

    return {
        "ttft_ms":        round(ttft_ms, 2),
        "prefill_tps":    round(prefill_tps, 1),
        "decode_tok_s":   round(decode_tok_s, 1),
        "output_tokens":  output_tokens,
        "peak_memory_gb": round(peak_memory_gb, 3),
        "wall_time_s":    round(t_end - t_start, 3),
        "output_text":    output_text,
    }


# ─────────────────────────────────────────────
# Plumbing — loop, CLI, JSON write
# ─────────────────────────────────────────────

ENGINES = {
    "mlx": mlx_engine,
}

DEFAULT_MODELS = {
    "mlx": "mlx-community/Llama-3.2-3B-Instruct-4bit",
}


def main():
    parser = argparse.ArgumentParser(description="MLX Lab — Episode 1 benchmark")
    parser.add_argument("--engine", choices=list(ENGINES), default="mlx")
    parser.add_argument("--model", default=None, help="Override the default model for the engine")
    parser.add_argument("--no-warmup", action="store_true", default=False)
    args = parser.parse_args()

    engine_fn = ENGINES[args.engine]
    model_id  = args.model or DEFAULT_MODELS[args.engine]

    print(f"\n{'─'*60}")
    print(f"  Engine : {args.engine}")
    print(f"  Model  : {model_id}")
    print(f"  Prompts: {len(PROMPTS)}")
    print(f"{'─'*60}\n")

    # Warm-up: one cheap generation to trigger Metal shader compilation.
    # Skipping it would inflate TTFT on the first real prompt.
    if not args.no_warmup:
        print("Warm-up run (discarded)...")
        warmup = engine_fn(
            model_id=model_id,
            system="You are helpful.",
            user="Say hi.",
            max_tokens=5,
            prompt_id="warmup",
        )
        for _ in warmup.token_stream:
            pass
        print("Warm-up done.\n")

    results = []

    for p in PROMPTS:
        print(f"  [{p['prompt_id']}] running...", end=" ", flush=True)

        gen_result = engine_fn(
            model_id=model_id,
            system=p["system"],
            user=p["user"],
            max_tokens=p["max_tokens"],
            prompt_id=p["prompt_id"],
        )

        metrics = measure_run(gen_result)

        row = {
            "engine":        args.engine,
            "model_id":      model_id,
            "prompt_id":     p["prompt_id"],
            "prompt_tokens": gen_result.prompt_tokens,
            **metrics,
        }
        results.append(row)

        print(
            f"TTFT={metrics['ttft_ms']:.0f}ms  "
            f"prefill={metrics['prefill_tps']:.0f} tok/s  "
            f"decode={metrics['decode_tok_s']:.0f} tok/s  "
            f"mem={metrics['peak_memory_gb']:.2f}GB"
        )

    # Per-model filename so runs don't overwrite each other.
    # "mlx-community/Qwen2.5-7B-Instruct-4bit" -> "qwen2.5-7b-instruct-4bit"
    model_slug = model_id.split("/")[-1].lower()
    out_path = Path(__file__).parent.parent / "results" / f"ep01-{args.engine}-{model_slug}.json"
    out_path.parent.mkdir(exist_ok=True)

    payload = {
        "run_at":   datetime.now(timezone.utc).isoformat(),
        "engine":   args.engine,
        "model_id": model_id,
        "results":  results,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults written → {out_path}\n")


if __name__ == "__main__":
    main()
