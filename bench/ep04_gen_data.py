"""
bench/ep04_gen_data.py — Ep 04: generate the synthetic NimbusCart artifacts.

  docs_v1/  — runbooks + catalog + escalation map for the v1 world
  docs_v2/  — same, after the reorg (checkout-svc → team-borealis)
  train.jsonl / valid.jsonl — chat-format LoRA data (v1 facts baked in)
  eval.jsonl — held-out alerts with ground truth, never seen in training

Usage:
    uv run python bench/ep04_gen_data.py            # writes everything under data/ep04/
    uv run python bench/ep04_gen_data.py --seed 7   # different alert sampling
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from bench.ep04_world import ALERT_TEMPLATES, REGIONS, SERVICES, AlertTemplate, Service, reorg

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "ep04"

SYSTEM_PROMPT = (
    "You are NimbusCart's on-call triage assistant. Given an alert, reply with "
    "a single JSON object and nothing else, with keys: severity (P1|P2|P3), "
    "affected_service, probable_cause, runbook_ref, escalate_to, escalate (true|false)."
)

# values used for eval alerts are disjoint from training values → no string leakage
TRAIN_VALUES = list(range(1000, 9000, 37))
EVAL_VALUES = [9101, 9202, 9303, 9404]
TRAIN_REGIONS = REGIONS[:3]
EVAL_REGIONS = (REGIONS[3],)


def _catalog(services: dict[str, Service]) -> str:
    lines = ["# NimbusCart service catalog", "",
             "| Service | Owner | Escalation channel | Runbook |",
             "|---|---|---|---|"]
    for s in services.values():
        lines.append(f"| {s.name} | {s.owner} | {s.channel} | {s.runbook} |")
    return "\n".join(lines) + "\n"


def _escalation_map(services: dict[str, Service]) -> str:
    lines = ["# NimbusCart escalation map", "",
             "Page the owning team's on-call channel for any P1 or P2.", ""]
    for s in services.values():
        lines.append(f"## {s.name}")
        lines.append(f"- Owner: {s.owner}")
        lines.append(f"- Escalate to: {s.channel}")
        lines.append(f"- Runbook: {s.runbook}")
        lines.append("")
    return "\n".join(lines)


def _runbook(s: Service) -> str:
    lines = [f"# {s.runbook} — {s.name}", "",
             f"Owner: {s.owner}  ",
             f"Escalation channel: {s.channel}", "",
             "## Known failure mode", s.failure_mode, "",
             "## Symptoms"]
    lines += [f"- {x}" for x in s.symptoms]
    lines += ["", "## Remediation"]
    lines += [f"{i}. {x}" for i, x in enumerate(s.remediation, 1)]
    lines += ["", "## Escalation",
              f"P1/P2: page {s.channel} ({s.owner}). P3: ticket to {s.owner}, no page.", ""]
    return "\n".join(lines)


def build_docs(services: dict[str, Service]) -> dict[str, str]:
    docs = {"service-catalog.md": _catalog(services),
            "escalation-map.md": _escalation_map(services)}
    for s in services.values():
        docs[f"runbook-{s.name}.md"] = _runbook(s)
    return docs


def write_docs(services: dict[str, Service], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in build_docs(services).items():
        (out_dir / name).write_text(text)


def render_alert(tmpl: AlertTemplate, region: str, value: int) -> str:
    return tmpl.text.format(region=region, value=value)


def triage_answer(svc: Service, tmpl: AlertTemplate) -> dict:
    return {
        "severity": tmpl.severity,
        "affected_service": svc.name,
        "probable_cause": svc.failure_mode.lower(),
        "runbook_ref": svc.runbook,
        "escalate_to": svc.channel,
        "escalate": tmpl.escalate,
    }


def _example(svc: Service, tmpl: AlertTemplate, region: str, value: int) -> dict:
    return {"alert": render_alert(tmpl, region, value),
            "truth": triage_answer(svc, tmpl), "service": svc.name}


def gen_examples(services: dict[str, Service], seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    names = list(services)
    out = []
    for i in range(n):
        svc = services[names[i % len(names)]]          # round-robin → balanced
        tmpl = rng.choice(ALERT_TEMPLATES[svc.name])
        out.append(_example(svc, tmpl, rng.choice(TRAIN_REGIONS), rng.choice(TRAIN_VALUES)))
    rng.shuffle(out)
    return out


def gen_eval(services: dict[str, Service]) -> list[dict]:
    out = []
    for svc in services.values():
        for j, tmpl in enumerate(ALERT_TEMPLATES[svc.name]):
            out.append(_example(svc, tmpl, EVAL_REGIONS[0], EVAL_VALUES[j % len(EVAL_VALUES)]))
    return out


def split_train_valid_eval(services: dict[str, Service], seed: int) -> tuple[list, list, list]:
    rows = gen_examples(services, seed=seed, n=400)
    return rows[:360], rows[360:], gen_eval(services)


def to_chat(example: dict) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["alert"]},
        {"role": "assistant", "content": json.dumps(example["truth"], separators=(",", ":"))},
    ]}


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    write_docs(SERVICES, DATA / "docs_v1")
    write_docs(reorg(SERVICES), DATA / "docs_v2")
    train, valid, ev = split_train_valid_eval(SERVICES, seed=args.seed)
    write_jsonl([to_chat(e) for e in train], DATA / "train.jsonl")
    write_jsonl([to_chat(e) for e in valid], DATA / "valid.jsonl")
    write_jsonl(ev, DATA / "eval.jsonl")
    print(f"docs_v1/docs_v2, train={len(train)} valid={len(valid)} eval={len(ev)} → {DATA}")


if __name__ == "__main__":
    main()
