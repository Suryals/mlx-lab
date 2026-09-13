"""
bench/ep04_sanity.py — Ep 04: 10-alert base-vs-tuned JSON validity gate.

Gate: the tuned model must be `compliant` on >= 9/10 before the full grid runs.
Usage: uv run python bench/ep04_sanity.py [--adapter adapters/ep04-qwen3.5-4b]
"""

import argparse
import json
from pathlib import Path

from bench.ep04_model import Runner
from bench.ep04_score import score_behavior

EVAL = Path("data/ep04/eval.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="adapters/ep04-qwen3.5-4b")
    args = ap.parse_args()
    rows = [json.loads(l) for l in EVAL.read_text().splitlines()][:10]
    tallies = {}
    for label, runner in (("base", Runner()), ("tuned", Runner(adapter_path=args.adapter))):
        ok = 0
        for row in rows:
            g = runner.triage(row["alert"])
            s = score_behavior(g.text)
            ok += s["compliant"]
            print(f"[{label}] compliant={s['compliant']} :: {g.text[:120]!r}")
        tallies[label] = ok
    print(f"\nbase {tallies['base']}/10 compliant · tuned {tallies['tuned']}/10 compliant")
    print("GATE PASS" if tallies["tuned"] >= 9 else "GATE FAIL — fix data or iters before running the grid")


if __name__ == "__main__":
    main()
