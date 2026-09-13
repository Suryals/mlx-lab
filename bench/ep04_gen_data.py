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

from pathlib import Path

from bench.ep04_world import SERVICES, Service, reorg

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "ep04"


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


def main() -> None:
    write_docs(SERVICES, DATA / "docs_v1")
    write_docs(reorg(SERVICES), DATA / "docs_v2")
    print(f"wrote docs to {DATA}")


if __name__ == "__main__":
    main()
