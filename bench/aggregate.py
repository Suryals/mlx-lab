"""
bench/aggregate.py — Collapse per-model result files into one dashboard JSON.

Scans results/*-mlx-*.json (one file per model, written by run_bench.py),
averages the per-prompt metrics, attaches a parameter count, and writes
results/dashboard.json — the single file the portfolio dashboard consumes.

Usage:
    uv run python bench/aggregate.py
"""

import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Fallback param counts (billions) when the slug regex can't infer one.
PARAM_OVERRIDES = {
    "llama-3.2-3b-instruct-4bit": 3.2,
    "qwen2.5-7b-instruct-4bit": 7.6,
    "mistral-large-instruct-2407-4bit": 123,  # "2407" is a date, not a size
}


def infer_params_b(model_slug: str) -> float | None:
    """Infer parameter count (billions) from a model slug.

    Matches a number followed by 'b' that is NOT '4bit'/'8bit' (quant suffix).
    e.g. 'llama-3.3-70b-instruct-4bit' -> 70.0
    """
    if model_slug in PARAM_OVERRIDES:
        return PARAM_OVERRIDES[model_slug]
    matches = re.findall(r"(\d+(?:\.\d+)?)b(?!it)", model_slug)
    if matches:
        # The param size is the largest such token in practice.
        return max(float(m) for m in matches)
    return None


def infer_quant_bits(model_slug: str) -> int | None:
    m = re.search(r"(\d+)bit", model_slug)
    return int(m.group(1)) if m else None


def summarize_file(path: Path) -> dict | None:
    data = json.loads(path.read_text())
    rows = data.get("results", [])
    if not rows:
        return None

    def avg(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(statistics.mean(vals), 2) if vals else None

    model_id = data["model_id"]
    slug = model_id.split("/")[-1].lower()

    return {
        "model_id": model_id,
        "model_slug": slug,
        "display_name": model_id.split("/")[-1].replace("-", " "),
        "params_b": infer_params_b(slug),
        "quant_bits": infer_quant_bits(slug),
        "decode_tok_s": avg("decode_tok_s"),
        "prefill_tok_s": avg("prefill_tps"),
        "ttft_ms": avg("ttft_ms"),
        "peak_memory_gb": avg("peak_memory_gb"),
        "run_at": data.get("run_at"),
    }


def main():
    files = sorted(RESULTS_DIR.glob("*-mlx-*.json"))
    files = [f for f in files if f.name != "dashboard.json"]

    models = []
    for f in files:
        summary = summarize_file(f)
        if summary:
            models.append(summary)

    # Sort by param count so the scaling curve reads left-to-right.
    models.sort(key=lambda m: (m["params_b"] is None, m["params_b"] or 0))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "machine": "Apple M5 Max · 128GB unified memory",
        "engine": "mlx-lm",
        "model_count": len(models),
        "models": models,
    }

    out_path = RESULTS_DIR / "dashboard.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"Aggregated {len(models)} models → {out_path}")
    for m in models:
        p = f"{m['params_b']}B" if m["params_b"] else "?B"
        print(f"  {p:>6}  {m['decode_tok_s']:>6} tok/s  {m['peak_memory_gb']:>6} GB  {m['model_slug']}")


if __name__ == "__main__":
    main()
