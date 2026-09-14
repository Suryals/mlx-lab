"""
bench/ep04_eval.py — Ep 04: run the 2×2 grid and the reorg twist.

Usage:
    # the grid (docs_v1, all four configs, 24 eval alerts)
    uv run python bench/ep04_eval.py --docs v1 --out results/ep04-grid.json

    # the reorg: docs_v2, checkout alerts only, three configs that can be stale or fresh
    uv run python bench/ep04_eval.py --docs v2 --configs base_rag tuned tuned_rag \
        --only-service checkout-svc --out results/ep04-reorg.json

Scoring is deterministic (bench/ep04_score.py). Ground truth comes from the
world that matches --docs: v1 = SERVICES, v2 = reorg(SERVICES).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from bench.ep04_gen_data import triage_answer
from bench.ep04_rag import TfidfIndex, format_context, load_index
from bench.ep04_score import FACT_KEYS, parse_output, score_behavior, score_facts
from bench.ep04_world import ALERT_TEMPLATES, SERVICES, Service, reorg

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "ep04"
ADAPTER = str(ROOT / "adapters" / "ep04-qwen3.5-4b")
CONFIGS = ("base", "base_rag", "tuned", "tuned_rag")


def truth_for(row: dict, services: dict[str, Service]) -> dict:
    """Recompute ground truth from the current world so v2 rows expect borealis."""
    svc = services[row["service"]]
    tmpl = next(t for t in ALERT_TEMPLATES[svc.name] if t.severity == row["truth"]["severity"]
                and t.escalate == row["truth"]["escalate"])
    return triage_answer(svc, tmpl)


def run_config(name: str, runner, rows: list[dict], index: TfidfIndex | None,
               services: dict[str, Service], k: int = 3) -> dict:
    runs = []
    for row in rows:
        context = format_context(index.retrieve(row["alert"], k=k)) if index else None
        g = runner.triage(row["alert"], context=context)
        obj, _ = parse_output(g.text)
        truth = truth_for(row, services)
        runs.append({
            "alert": row["alert"], "service": row["service"], "output": g.text,
            "truth": truth, "behavior": score_behavior(g.text), "facts": score_facts(obj, truth),
            "severity_match": bool(obj is not None and obj.get("severity") == truth["severity"]),
            "ttft_s": round(g.ttft_s, 3), "decode_tps": round(g.decode_tps, 1),
        })
    n = len(runs)
    if n == 0:
        summary = {
            "config": name, "n": 0,
            "compliance_rate": 0.0, "fact_accuracy": 0.0, "severity_accuracy": 0.0,
            "mean_ttft_s": 0.0, "mean_decode_tps": 0.0,
        }
        return {"summary": summary, "runs": runs}
    summary = {
        "config": name, "n": n,
        "compliance_rate": sum(r["behavior"]["compliant"] for r in runs) / n,
        "fact_accuracy": sum(r["facts"]["facts_correct"] for r in runs) / (len(FACT_KEYS) * n),
        "severity_accuracy": sum(r["severity_match"] for r in runs) / n,
        "mean_ttft_s": round(sum(r["ttft_s"] for r in runs) / n, 3),
        "mean_decode_tps": round(sum(r["decode_tps"] for r in runs) / n, 1),
    }
    return {"summary": summary, "runs": runs}


def markdown_grid(results: dict[str, dict]) -> str:
    lines = ["| config | compliance | severity | fact accuracy | mean TTFT | decode tok/s |",
              "|---|---|---|---|---|---|"]
    for c, r in results.items():
        s = r["summary"]
        lines.append(f"| {c} | {s['compliance_rate']:.0%} | {s['severity_accuracy']:.0%} | "
                     f"{s['fact_accuracy']:.0%} | {s['mean_ttft_s']:.2f}s | {s['mean_decode_tps']:.0f} |")
    return "\n".join(lines)


def markdown_reorg(results: dict[str, dict]) -> str:
    lines = ["| config | escalate_to answers (after reorg) | correct |", "|---|---|---|"]
    for c, r in results.items():
        answers = sorted({(parse_output(x["output"])[0] or {}).get("escalate_to", "∅") for x in r["runs"]})
        ok = sum(x["facts"]["escalate_to"] for x in r["runs"])
        lines.append(f"| {c} | {', '.join(str(a) for a in answers)} | {ok}/{len(r['runs'])} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", choices=["v1", "v2"], default="v1")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--only-service")
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from bench.ep04_model import BASE_MODEL, Runner  # imported here so tests need no mlx

    services = SERVICES if args.docs == "v1" else reorg(SERVICES)
    rows = [json.loads(l) for l in (DATA / "eval.jsonl").read_text().splitlines()]
    if args.only_service:
        rows = [r for r in rows if r["service"] == args.only_service]
    if not rows:
        raise SystemExit(f"no eval rows match --only-service {args.only_service!r}")
    index = load_index(DATA / f"docs_{args.docs}")

    runners: dict[str, Runner] = {}
    results = {}
    for cfg in args.configs:
        key = "tuned" if cfg.startswith("tuned") else "base"
        if key not in runners:
            runners[key] = Runner(adapter_path=args.adapter if key == "tuned" else None)
        results[cfg] = run_config(cfg, runners[key], rows, index if cfg.endswith("rag") else None, services)
        print(f"{cfg}: {results[cfg]['summary']}")

    out = {"meta": {"episode": "ep04", "model": BASE_MODEL, "adapter": args.adapter,
                    "docs": args.docs, "n_alerts": len(rows),
                    "timestamp": datetime.now(timezone.utc).isoformat()},
           "results": results}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("\n" + (markdown_reorg(results) if args.docs == "v2" else markdown_grid(results)))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
