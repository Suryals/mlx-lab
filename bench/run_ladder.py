"""
bench/run_ladder.py — Autonomous overnight benchmark driver.

Runs the model ladder small -> large, one model fully at a time (download +
benchmark in a single process, so each benchmark runs on an otherwise-idle
machine — no download contamination). After every model it rebuilds the
dashboard, so the portfolio page fills in progressively. Failures (e.g. OOM)
are recorded to results/notes.json and the run continues.

Designed to run detached and unattended:
    cd ~/projects/mlx-lab
    nohup caffeinate -i uv run python bench/run_ladder.py > /tmp/mlx-ladder.log 2>&1 &

`caffeinate -i` prevents idle sleep. Keep the machine plugged in and the lid
open (closing the lid sleeps regardless of caffeinate).
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

# Ordered small -> large. Every entry fits within 128GB; the 8-bit 72B
# (~77GB) is the memory-stress run. (params_b is the headline size.)
LADDER = [
    ("mlx-community/Qwen2.5-14B-Instruct-4bit", 14),
    ("mlx-community/Qwen2.5-32B-Instruct-4bit", 32),
    ("mlx-community/Llama-3.3-70B-Instruct-4bit", 70),
    ("mlx-community/Mistral-Large-Instruct-2407-4bit", 123),
    ("mlx-community/Qwen2.5-72B-Instruct-8bit", 72),
]


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def refresh_results():
    # Results data only (results/dashboard.json). The portfolio page is
    # hand-styled by the author and is NOT regenerated here.
    a = run([sys.executable, "bench/aggregate.py"])
    if a.returncode != 0:
        log(f"  aggregate WARN: {a.stderr[-200:]}")


def record_fail(repo: str, params: int, reason: str):
    notes_path = RESULTS / "notes.json"
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    failed = notes.setdefault("failed", [])
    slug = repo.split("/")[-1].lower()
    short = slug.split("-")[0]
    if not any(f.get("model_slug") == slug for f in failed):
        failed.append({
            "model_slug": slug, "params_b": params,
            "short": short, "reason": reason,
        })
    notes_path.write_text(json.dumps(notes, indent=2))


def classify_failure(proc: subprocess.CompletedProcess) -> str:
    tail = ((proc.stderr or "") + (proc.stdout or "")).lower()[-600:]
    if proc.returncode in (-9, 137) or "metal" in tail or "out of memory" in tail \
            or "insufficient memory" in tail or "failed to allocate" in tail:
        return "OOM"
    return f"exit {proc.returncode}"


def main():
    log(f"=== Ladder start: {len(LADDER)} models ===")
    for repo, params in LADDER:
        log(f"--- {params}B  {repo}  (download + benchmark) ---")
        proc = run([sys.executable, "bench/run_bench.py",
                    "--engine", "mlx", "--model", repo])
        if proc.returncode == 0:
            log(f"  OK {params}B")
        else:
            reason = classify_failure(proc)
            log(f"  FAIL {params}B — {reason}")
            record_fail(repo, params, reason)
        refresh_results()
        log(f"  results refreshed after {params}B")
    log("=== Ladder complete ===")


if __name__ == "__main__":
    main()
