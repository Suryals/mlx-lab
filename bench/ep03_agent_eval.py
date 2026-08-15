"""
bench/ep03_agent_eval.py — Ep 03: agent / tool-use correctness eval.

Runs the same 18-task, 7-domain tool-use suite against:
  * a local MLX model (Qwen3.8-27B 4-bit on Apple Silicon), or
  * a hosted model via OpenRouter (Claude Opus 4.6).

Both engines get the identical system prompt, tool schemas, and mock tool
world (bench/ep03_tools.py). Scoring is fully deterministic — no LLM judge.

Usage:
    # Local Qwen via MLX
    uv run python bench/ep03_agent_eval.py --engine mlx \
        --model mlx-community/Qwen3.8-27B-Instruct-4bit --label qwen3.8-27b-4bit

    # Opus 4.6 via OpenRouter (needs OPENROUTER_API_KEY in env or .env)
    uv run python bench/ep03_agent_eval.py --engine openrouter \
        --model anthropic/claude-opus-4.6 --label opus-4.6 --runs 2

Cost guards: MAX_TURNS per task, MAX_TOTAL_CALLS per run — a runaway loop
aborts the run rather than draining credits.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from bench.ep03_tools import TOOL_SCHEMAS, execute_tool

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
TASKS_FILE = ROOT / "bench" / "tasks" / "ep03_tasks.yaml"

MAX_TURNS = 8            # per task: model turns in the agent loop
MAX_TOTAL_CALLS = 250    # per run: hard abort if exceeded (runaway guard)

SYSTEM_PROMPT = (
    "You are an on-call operations assistant for an engineering team. "
    "You have diagnostic tools available. Use a tool when the question requires "
    "live information from a system; answer directly from your own knowledge when "
    "it does not. Be precise and concise. When you have enough information, give "
    "your final answer as plain text."
)


# ---------------------------------------------------------------------------
# Engines — each returns (tool_calls_made, final_text, stats) for one task.
# tool_calls_made: list of {"tool": name, "args": dict} in call order.
# ---------------------------------------------------------------------------

class OpenRouterEngine:
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model: str):
        import httpx
        self.model = model
        key = os.environ.get("OPENROUTER_API_KEY") or _read_dotenv("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY not set (env or .env)")
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {key}"}, timeout=120.0
        )
        self.calls_made = 0

    def run_task(self, prompt: str):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        tool_calls_made, t0 = [], time.time()
        for _ in range(MAX_TURNS):
            self.calls_made += 1
            if self.calls_made > MAX_TOTAL_CALLS:
                raise SystemExit("MAX_TOTAL_CALLS exceeded — aborting run (cost guard)")
            r = self.client.post(self.URL, json={
                "model": self.model,
                "messages": messages,
                "tools": TOOL_SCHEMAS,
                "max_tokens": 1024,
            })
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            messages.append(msg)
            if not msg.get("tool_calls"):
                return tool_calls_made, msg.get("content") or "", {"seconds": time.time() - t0}
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {"_unparseable": tc["function"]["arguments"]}
                tool_calls_made.append({"tool": name, "args": args})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": execute_tool(name, args),
                })
        return tool_calls_made, "", {"seconds": time.time() - t0, "hit_max_turns": True}


class MLXEngine:
    TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

    def __init__(self, model: str):
        from mlx_lm import load
        print(f"loading {model} ...")
        self.model, self.tokenizer = load(model)
        self.calls_made = 0

    def run_task(self, prompt: str):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        tool_calls_made, turns, t0 = [], [], time.time()
        for _ in range(MAX_TURNS):
            self.calls_made += 1
            if self.calls_made > MAX_TOTAL_CALLS:
                raise SystemExit("MAX_TOTAL_CALLS exceeded — aborting run")
            text, turn = self._generate_turn(messages)
            turns.append(turn)
            messages.append({"role": "assistant", "content": text})
            calls = self._parse_tool_calls(text)
            if not calls:
                return tool_calls_made, _strip_think(text), self._stats(turns, t0)
            for name, args in calls:
                tool_calls_made.append({"tool": name, "args": args})
                messages.append({"role": "tool", "content": execute_tool(name, args)})
        return tool_calls_made, "", {**self._stats(turns, t0), "hit_max_turns": True}

    def _generate_turn(self, messages):
        """One model turn via stream_generate, capturing local-inference perf:
        TTFT (wall clock to first chunk), prefill/decode tok/s, peak RAM."""
        from mlx_lm import stream_generate
        prompt = self.tokenizer.apply_chat_template(
            messages, tools=TOOL_SCHEMAS, add_generation_prompt=True, tokenize=False,
        )
        text, ttft, last = "", None, None
        t0 = time.time()
        for chunk in stream_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=1024):
            if ttft is None:
                ttft = time.time() - t0
            text += chunk.text
            last = chunk
        return text, {
            "ttft_s": round(ttft or 0.0, 3),
            "prompt_tokens": getattr(last, "prompt_tokens", None),
            "prefill_tps": round(getattr(last, "prompt_tps", 0.0), 1),
            "generation_tokens": getattr(last, "generation_tokens", None),
            "decode_tps": round(getattr(last, "generation_tps", 0.0), 1),
            "peak_memory_gb": round(getattr(last, "peak_memory", 0.0), 2),
        }

    @staticmethod
    def _stats(turns, t0):
        gen_tok = sum(t["generation_tokens"] or 0 for t in turns)
        gen_time = sum((t["generation_tokens"] or 0) / t["decode_tps"] for t in turns if t["decode_tps"])
        return {
            "seconds": time.time() - t0,
            "turns": len(turns),
            "first_ttft_s": turns[0]["ttft_s"] if turns else None,
            "decode_tps": round(gen_tok / gen_time, 1) if gen_time else None,   # token-weighted across turns
            "prefill_tps": round(sum(t["prefill_tps"] for t in turns) / len(turns), 1) if turns else None,
            "generation_tokens": gen_tok,
            "peak_memory_gb": max((t["peak_memory_gb"] for t in turns), default=None),
            "per_turn": turns,
        }

    # Qwen3.8 XML-style: <tool_call><function=NAME><parameter=KEY>VALUE</parameter>...</function></tool_call>
    XML_FN_RE = re.compile(r"<function=([\w.-]+)>(.*?)</function>", re.DOTALL)
    XML_PARAM_RE = re.compile(r"<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>", re.DOTALL)

    def _parse_tool_calls(self, text: str):
        calls = []
        for raw in self.TOOL_CALL_RE.findall(text):  # Hermes JSON format
            try:
                obj = json.loads(raw)
                calls.append((obj.get("name", ""), obj.get("arguments") or {}))
            except json.JSONDecodeError:
                continue  # malformed call block: counts as no call — the model loses the turn
        for name, body in self.XML_FN_RE.findall(text):  # Qwen3.8 XML format
            calls.append((name, dict(self.XML_PARAM_RE.findall(body))))
        return calls


def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Qwen3.8 sometimes emits reasoning with no opening <think>, just a closing tag
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def _read_dotenv(key: str):
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith(f"{key}="):
                return line.split("=", 1)[1].strip().strip('"')
    return None


# ---------------------------------------------------------------------------
# Scoring — deterministic, no LLM judge.
# ---------------------------------------------------------------------------

def _call_matches(actual: dict, spec: dict) -> bool:
    if actual["tool"] != spec["tool"]:
        return False
    for k, v in (spec.get("args_must_include") or {}).items():
        av = actual["args"].get(k)
        if av is None or str(av).strip().lower() != str(v).strip().lower():
            return False
    return True


def score_task(task: dict, calls: list, answer: str) -> dict:
    mode = task["scoring"]
    expected = task.get("expected", {})
    keywords = expected.get("final_answer_must_mention", [])
    kw_ok = all(str(k).lower() in answer.lower() for k in keywords)

    if mode == "no_tool":
        calls_ok = len(calls) == 0
    elif mode == "exact_sequence":
        specs, i = expected.get("tool_calls", []), 0
        for c in calls:                      # expected must appear in order (extra calls tolerated but recorded)
            if i < len(specs) and _call_matches(c, specs[i]):
                i += 1
        calls_ok = i == len(specs)
    elif mode == "any_order":
        remaining = list(expected.get("tool_calls", []))
        for c in calls:
            for s in remaining:
                if _call_matches(c, s):
                    remaining.remove(s)
                    break
        calls_ok = not remaining
    else:
        raise ValueError(f"unknown scoring mode {mode}")

    n_expected = len(expected.get("tool_calls", []))
    return {
        "pass": calls_ok and kw_ok,
        "calls_ok": calls_ok,
        "answer_ok": kw_ok,
        "extra_calls": max(0, len(calls) - n_expected),
    }


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["mlx", "openrouter"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True, help="short name used in the results filename")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--only", help="comma-separated task IDs, e.g. T01 smoke tests")
    args = ap.parse_args()

    tasks = yaml.safe_load(TASKS_FILE.read_text())["tasks"]
    if args.only:
        wanted = set(args.only.split(","))
        tasks = [t for t in tasks if t["id"] in wanted]

    engine = MLXEngine(args.model) if args.engine == "mlx" else OpenRouterEngine(args.model)

    all_runs = []
    for run_idx in range(args.runs):
        records = []
        for task in tasks:
            calls, answer, stats = engine.run_task(task["prompt"])
            rec = {
                "id": task["id"], "category": task["category"],
                "tool_calls": calls, "final_answer": answer, **stats,
                **score_task(task, calls, answer),
            }
            records.append(rec)
            print(f"run{run_idx + 1} {task['id']:>4} [{task['category']:<15}] "
                  f"{'PASS' if rec['pass'] else 'FAIL':4} "
                  f"calls={len(calls)} extra={rec['extra_calls']} {stats['seconds']:.1f}s")
        all_runs.append(records)

    # summary across runs: task passes in a run; report per-run and per-category
    flat = [r for run in all_runs for r in run]
    by_cat = {}
    for r in flat:
        by_cat.setdefault(r["category"], []).append(r["pass"])
    summary = {
        "model": args.model, "engine": args.engine, "runs": args.runs,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall_pass_rate": round(sum(r["pass"] for r in flat) / len(flat), 3),
        "by_category": {c: round(sum(v) / len(v), 3) for c, v in sorted(by_cat.items())},
        "total_extra_calls": sum(r["extra_calls"] for r in flat),
        "avg_task_seconds": round(sum(r["seconds"] for r in flat) / len(flat), 1),
    }
    # local-inference usability numbers (MLX runs only)
    perf = [r for r in flat if r.get("decode_tps")]
    if perf:
        summary["local_inference"] = {
            "decode_tps_avg": round(sum(r["decode_tps"] for r in perf) / len(perf), 1),
            "prefill_tps_avg": round(sum(r["prefill_tps"] for r in perf) / len(perf), 1),
            "first_ttft_s_avg": round(sum(r["first_ttft_s"] for r in perf) / len(perf), 2),
            "peak_memory_gb": max(r["peak_memory_gb"] for r in perf),
        }
    out = RESULTS / f"ep03-{args.label}.json"
    out.write_text(json.dumps({"summary": summary, "runs": all_runs}, indent=2))
    print(f"\n== {args.label} ==")
    print(json.dumps(summary, indent=2))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
