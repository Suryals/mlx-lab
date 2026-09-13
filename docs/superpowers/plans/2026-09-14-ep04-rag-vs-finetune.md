# Ep 04 RAG vs Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible 2×2 experiment (base / base+RAG / LoRA-tuned / tuned+RAG) on a synthetic on-call triage task, plus a post-training "reorg" run that shows retrieval absorbing a fact change while the fine-tuned adapter goes stale.

**Architecture:** A pure-Python "world" module is the single source of truth for the fictional NimbusCart services; everything else (runbook docs, training/eval JSONL, ground-truth scoring) is derived from it. Retrieval is a dependency-free TF-IDF cosine index over markdown chunks. The model runner wraps mlx-lm `load`/`generate` with an optional LoRA adapter and optional retrieved context, and the eval script runs every config × alert, scores deterministically, and writes results JSON in the repo's existing `results/` style.

**Tech Stack:** Python 3.12, uv, mlx-lm 0.31.3 (`mlx_lm.lora`, `mlx_lm.load`, `mlx_lm.generate`), pytest, PyYAML. Base model `mlx-community/Qwen3.5-4B-4bit`.

**Spec:** `docs/superpowers/specs/2026-09-14-ep04-rag-vs-finetune-design.md`

## Global Constraints

- Fully synthetic domain: company "NimbusCart", six services exactly as in the spec ground-truth table. Nothing may resemble a real employer's stack.
- Runs entirely locally on the M5 Max; no cloud training or hosted models.
- Base model: `mlx-community/Qwen3.5-4B-4bit`. Thinking mode disabled at eval (`enable_thinking=False`) and any `<think>…</think>` block stripped before scoring.
- LoRA: `--fine-tune-type lora --num-layers 16 --batch-size 4 --iters 600 --learning-rate 1e-5 --steps-per-eval 50`, adapter at `adapters/ep04-qwen3.5-4b/`.
- Repo layout: code `bench/ep04_*.py`, data `data/ep04/`, results `results/ep04-*.json`, write-up `content/ep04-portfolio.md`.
- Scored fact fields: `affected_service`, `runbook_ref`, `escalate_to`. Behavior: parses as JSON object, all six keys present, `severity ∈ {P1,P2,P3}`, `escalate` is bool, no prose outside the JSON.
- Every command runs through `uv run …` from the repo root. Commit after each task.
- Retrieval deviation from spec, deliberately chosen here: TF-IDF cosine instead of a neural embedding model. It is still a vector index over chunks, adds zero dependencies, and with six services the alert text retrieves the right runbook trivially. The spec's "retrieval quality is not the story" clause is why this is acceptable; note it in the write-up.

---

## File Structure

| File | Responsibility |
|---|---|
| `bench/ep04_world.py` | The fictional world: `SERVICES` (v1), `reorg()` producing v2, alert templates. Single source of truth. |
| `bench/ep04_gen_data.py` | Renders docs_v1/docs_v2 markdown, generates train/valid/eval JSONL. CLI entrypoint. |
| `bench/ep04_rag.py` | Chunk markdown docs, TF-IDF index, `retrieve(query, k)`. |
| `bench/ep04_score.py` | Parse model output, behavior score, fact score. Pure functions. |
| `bench/ep04_model.py` | mlx-lm wrapper: `Runner(base, adapter_path)`, `triage(alert, context) -> Generation` with timing. |
| `bench/ep04_lora.yaml` | Training config for `mlx_lm.lora --config`. |
| `bench/ep04_sanity.py` | 10-alert base-vs-tuned JSON validity gate. |
| `bench/ep04_eval.py` | Runs the grid and the reorg; writes `results/ep04-grid.json`, `results/ep04-reorg.json`; prints markdown tables. |
| `data/ep04/docs_v1/*.md`, `docs_v2/*.md`, `train.jsonl`, `valid.jsonl`, `eval.jsonl` | Generated artifacts (committed; small). |
| `adapters/ep04-qwen3.5-4b/` | LoRA adapter output (gitignored except `adapter_config.json` and the loss log). |
| `tests/test_ep04_*.py` | Unit tests for world, gen_data, rag, score, eval (with a fake runner). |

---

### Task 1: Test scaffolding + the NimbusCart world module

**Files:**
- Modify: `pyproject.toml` (add pytest dev group, pytest config)
- Create: `tests/__init__.py`, `tests/test_ep04_world.py`, `bench/ep04_world.py`
- Modify: `.gitignore` (adapters)

**Interfaces:**
- Produces: `SERVICES: dict[str, Service]` keyed by service name; `@dataclass(frozen=True) Service(name, owner, channel, runbook, failure_mode, symptoms: tuple[str,...], remediation: tuple[str,...])`; `reorg(services) -> dict[str, Service]` (checkout → team-borealis / `#borealis-oncall`); `ALERT_TEMPLATES: dict[str, list[AlertTemplate]]` where `AlertTemplate(text: str, severity: str, escalate: bool)` and `text` contains `{region}` and `{value}` placeholders; `REGIONS: tuple[str,...]`.

- [ ] **Step 1: Add pytest and config**

In `pyproject.toml` append:

```toml
[dependency-groups]
dev = ["pytest>=8.3"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Run: `uv sync` → Expected: installs pytest, lockfile updated.

Append to `.gitignore` (create if missing):

```
adapters/*/adapters.safetensors
adapters/*/*.safetensors
```

- [ ] **Step 2: Write the failing test**

`tests/__init__.py` — empty file.

`tests/test_ep04_world.py`:

```python
from bench.ep04_world import SERVICES, ALERT_TEMPLATES, REGIONS, reorg


def test_six_services_with_unique_runbooks_and_channels():
    assert len(SERVICES) == 6
    assert set(SERVICES) == {
        "checkout-svc", "payments-gw", "inventory-svc",
        "search-svc", "notify-svc", "auth-svc",
    }
    assert len({s.runbook for s in SERVICES.values()}) == 6
    assert len({s.channel for s in SERVICES.values()}) == 6


def test_checkout_v1_facts_match_spec():
    c = SERVICES["checkout-svc"]
    assert c.owner == "team-atlas"
    assert c.channel == "#atlas-oncall"
    assert c.runbook == "RB-CHK-001"


def test_reorg_changes_only_checkout_owner_and_channel():
    v2 = reorg(SERVICES)
    assert v2["checkout-svc"].owner == "team-borealis"
    assert v2["checkout-svc"].channel == "#borealis-oncall"
    assert v2["checkout-svc"].runbook == "RB-CHK-001"
    for name, svc in SERVICES.items():
        if name != "checkout-svc":
            assert v2[name] == svc


def test_every_service_has_alert_templates_with_placeholders():
    for name in SERVICES:
        templates = ALERT_TEMPLATES[name]
        assert len(templates) >= 3
        for t in templates:
            assert "{region}" in t.text and "{value}" in t.text
            assert t.severity in {"P1", "P2", "P3"}
    assert len(REGIONS) >= 3
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_ep04_world.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.ep04_world'`

- [ ] **Step 4: Write the world module**

`bench/ep04_world.py`:

```python
"""
bench/ep04_world.py — Ep 04: the fictional NimbusCart world.

Single source of truth for services, owners, escalation channels, runbooks,
and alert templates. Everything else in Ep 04 (docs, training data, eval
ground truth) is derived from this file. Entirely synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Service:
    name: str
    owner: str
    channel: str
    runbook: str
    failure_mode: str
    symptoms: tuple[str, ...]
    remediation: tuple[str, ...]


@dataclass(frozen=True)
class AlertTemplate:
    text: str        # contains {region} and {value}
    severity: str    # P1 | P2 | P3
    escalate: bool


REGIONS: tuple[str, ...] = ("us-east-1", "eu-west-1", "ap-south-1", "us-west-2")


SERVICES: dict[str, Service] = {
    "checkout-svc": Service(
        name="checkout-svc", owner="team-atlas", channel="#atlas-oncall",
        runbook="RB-CHK-001",
        failure_mode="DB connection-pool exhaustion during flash sales",
        symptoms=("p99 latency above 800ms", "db_pool_wait_ms climbing",
                  "5xx rate on /cart/confirm"),
        remediation=("Scale checkout-db read replicas by 2",
                     "Raise pool_max from 40 to 80 via config map",
                     "Enable checkout queueing flag if 5xx persists"),
    ),
    "payments-gw": Service(
        name="payments-gw", owner="team-ledger", channel="#ledger-oncall",
        runbook="RB-PAY-001",
        failure_mode="Upstream PSP timeout causing retry storm",
        symptoms=("psp_timeout_rate above 5%", "retry_queue_depth growing",
                  "duplicate authorization warnings"),
        remediation=("Set retry backoff to exponential with 30s cap",
                     "Fail over to secondary PSP region",
                     "Pause retries for cards already authorized"),
    ),
    "inventory-svc": Service(
        name="inventory-svc", owner="team-stockroom", channel="#stockroom-oncall",
        runbook="RB-INV-001",
        failure_mode="Cache/DB drift after bulk import",
        symptoms=("stock_mismatch_count rising", "oversell alerts",
                  "cache_hit_ratio drop after import job"),
        remediation=("Invalidate inventory cache namespace",
                     "Re-run reconciliation job with --dry-run first",
                     "Block bulk imports until drift below 0.1%"),
    ),
    "search-svc": Service(
        name="search-svc", owner="team-lens", channel="#lens-oncall",
        runbook="RB-SRCH-001",
        failure_mode="Index rebuild starving query latency",
        symptoms=("query p95 above 1.5s", "indexer CPU at 100%",
                  "search timeouts on category pages"),
        remediation=("Throttle indexer to 2 shards concurrent",
                     "Route queries to warm replica set",
                     "Defer rebuild to off-peak window"),
    ),
    "notify-svc": Service(
        name="notify-svc", owner="team-beacon", channel="#beacon-oncall",
        runbook="RB-NTF-001",
        failure_mode="Queue backlog when email vendor rate-limits",
        symptoms=("email_queue_depth above 50k", "vendor 429 responses",
                  "notification latency above 10min"),
        remediation=("Switch to secondary email vendor",
                     "Drop marketing tier from queue temporarily",
                     "Raise consumer count from 4 to 12"),
    ),
    "auth-svc": Service(
        name="auth-svc", owner="team-gatekeeper", channel="#gate-oncall",
        runbook="RB-AUTH-001",
        failure_mode="Token-signing key rotation missed",
        symptoms=("token_verify_fail_rate spike", "401 rate above 2%",
                  "JWKS fetch errors"),
        remediation=("Rotate signing key via keyctl rotate --service auth",
                     "Republish JWKS endpoint",
                     "Extend previous key grace period by 1h"),
    ),
}


def reorg(services: dict[str, Service]) -> dict[str, Service]:
    """v2 world: team-atlas dissolved, checkout-svc moves to team-borealis."""
    out = dict(services)
    out["checkout-svc"] = replace(
        services["checkout-svc"], owner="team-borealis", channel="#borealis-oncall"
    )
    return out


ALERT_TEMPLATES: dict[str, list[AlertTemplate]] = {
    "checkout-svc": [
        AlertTemplate("[FIRING] checkout-svc p99 latency {value}ms (threshold 800ms), db_pool_wait_ms=1900, region={region}", "P1", True),
        AlertTemplate("[FIRING] checkout-svc 5xx rate {value}% on /cart/confirm, region={region}", "P1", True),
        AlertTemplate("[WARNING] checkout-svc db_pool_wait_ms={value} trending up over 15m, region={region}", "P2", True),
        AlertTemplate("[INFO] checkout-svc pool utilisation {value}% during scheduled load test, region={region}", "P3", False),
    ],
    "payments-gw": [
        AlertTemplate("[FIRING] payments-gw psp_timeout_rate {value}% (threshold 5%), retry_queue_depth=12000, region={region}", "P1", True),
        AlertTemplate("[FIRING] payments-gw duplicate_authorization_warnings={value} in 5m, region={region}", "P1", True),
        AlertTemplate("[WARNING] payments-gw retry_queue_depth={value} growing, psp latency elevated, region={region}", "P2", True),
        AlertTemplate("[INFO] payments-gw psp_timeout_rate {value}% brief blip, auto-recovered, region={region}", "P3", False),
    ],
    "inventory-svc": [
        AlertTemplate("[FIRING] inventory-svc stock_mismatch_count={value} after bulk import job, oversell alerts firing, region={region}", "P1", True),
        AlertTemplate("[WARNING] inventory-svc cache_hit_ratio dropped to {value}% post-import, region={region}", "P2", True),
        AlertTemplate("[WARNING] inventory-svc oversell_alerts={value} in 10m, region={region}", "P2", True),
        AlertTemplate("[INFO] inventory-svc stock_mismatch_count={value}, within tolerance, region={region}", "P3", False),
    ],
    "search-svc": [
        AlertTemplate("[FIRING] search-svc query p95 {value}ms (threshold 1500ms), indexer CPU 100%, region={region}", "P1", True),
        AlertTemplate("[WARNING] search-svc timeouts on category pages {value}/min during index rebuild, region={region}", "P2", True),
        AlertTemplate("[WARNING] search-svc indexer CPU {value}% for 20m, query latency rising, region={region}", "P2", True),
        AlertTemplate("[INFO] search-svc scheduled index rebuild running, p95 {value}ms, region={region}", "P3", False),
    ],
    "notify-svc": [
        AlertTemplate("[FIRING] notify-svc email_queue_depth={value} (threshold 50000), vendor returning 429, region={region}", "P1", True),
        AlertTemplate("[WARNING] notify-svc notification latency {value}min, vendor rate-limiting, region={region}", "P2", True),
        AlertTemplate("[WARNING] notify-svc vendor 429 responses {value}/min, region={region}", "P2", True),
        AlertTemplate("[INFO] notify-svc email_queue_depth={value}, draining normally, region={region}", "P3", False),
    ],
    "auth-svc": [
        AlertTemplate("[FIRING] auth-svc token_verify_fail_rate {value}% (threshold 2%), JWKS fetch errors, region={region}", "P1", True),
        AlertTemplate("[FIRING] auth-svc 401 rate {value}% across all clients, region={region}", "P1", True),
        AlertTemplate("[WARNING] auth-svc signing key expires in {value}h, rotation not scheduled, region={region}", "P2", True),
        AlertTemplate("[INFO] auth-svc JWKS cache refresh took {value}ms, region={region}", "P3", False),
    ],
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_ep04_world.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .gitignore tests/__init__.py tests/test_ep04_world.py bench/ep04_world.py
git commit -m "Ep 04: pytest scaffold + NimbusCart world module"
```

---

### Task 2: Generate runbook docs (v1 and v2)

**Files:**
- Create: `bench/ep04_gen_data.py` (docs part), `tests/test_ep04_gen_data.py`

**Interfaces:**
- Consumes: `SERVICES`, `reorg`, `Service` from `bench.ep04_world`.
- Produces: `build_docs(services: dict[str, Service]) -> dict[str, str]` mapping relative filename → markdown; `write_docs(services, out_dir: Path) -> None`. Filenames: `service-catalog.md`, `escalation-map.md`, `runbook-<service>.md` ×6 (8 files; the spec's "~12" was an estimate).

- [ ] **Step 1: Write the failing test**

`tests/test_ep04_gen_data.py`:

```python
from pathlib import Path

from bench.ep04_gen_data import build_docs, write_docs
from bench.ep04_world import SERVICES, reorg


def test_build_docs_produces_catalog_escalation_and_one_runbook_per_service():
    docs = build_docs(SERVICES)
    assert "service-catalog.md" in docs
    assert "escalation-map.md" in docs
    for name in SERVICES:
        assert f"runbook-{name}.md" in docs
    assert len(docs) == 8


def test_runbook_contains_facts_the_scorer_checks():
    docs = build_docs(SERVICES)
    rb = docs["runbook-checkout-svc.md"]
    assert "RB-CHK-001" in rb
    assert "#atlas-oncall" in rb
    assert "team-atlas" in rb
    assert "DB connection-pool exhaustion" in rb


def test_v2_docs_differ_from_v1_only_where_reorg_touched():
    v1 = build_docs(SERVICES)
    v2 = build_docs(reorg(SERVICES))
    changed = {k for k in v1 if v1[k] != v2[k]}
    assert changed == {"service-catalog.md", "escalation-map.md", "runbook-checkout-svc.md"}
    assert "#borealis-oncall" in v2["runbook-checkout-svc.md"]
    assert "#atlas-oncall" not in v2["runbook-checkout-svc.md"]


def test_write_docs_writes_files(tmp_path: Path):
    write_docs(SERVICES, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(build_docs(SERVICES))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ep04_gen_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.ep04_gen_data'`

- [ ] **Step 3: Write docs generation**

`bench/ep04_gen_data.py` (first version — data generation is added in Task 3):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ep04_gen_data.py -v`
Expected: 4 passed

- [ ] **Step 5: Generate and commit the docs**

Run: `uv run python bench/ep04_gen_data.py && ls data/ep04/docs_v1 data/ep04/docs_v2 && diff -r data/ep04/docs_v1 data/ep04/docs_v2`
Expected: 8 files in each; diff shows only atlas→borealis lines in three files.

```bash
git add bench/ep04_gen_data.py tests/test_ep04_gen_data.py data/ep04/docs_v1 data/ep04/docs_v2
git commit -m "Ep 04: generate NimbusCart runbook docs v1 + v2 (reorg)"
```

---

### Task 3: Generate training, validation, and eval JSONL

**Files:**
- Modify: `bench/ep04_gen_data.py`
- Modify: `tests/test_ep04_gen_data.py`

**Interfaces:**
- Produces: `SYSTEM_PROMPT: str`; `triage_answer(svc: Service, tmpl: AlertTemplate) -> dict` (the six-key ground-truth JSON); `render_alert(tmpl, region, value) -> str`; `gen_examples(services, seed, n) -> list[dict]` each `{"alert": str, "truth": dict, "service": str}`; `to_chat(example) -> {"messages": [...]}`; `write_jsonl(rows, path)`; `main()` writes `train.jsonl` (360), `valid.jsonl` (40), `eval.jsonl` (24 = 6 services × 4 templates, one per template, with regions/values not used in train).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ep04_gen_data.py`:

```python
import json

from bench.ep04_gen_data import (
    SYSTEM_PROMPT, gen_examples, to_chat, triage_answer, split_train_valid_eval,
)
from bench.ep04_world import ALERT_TEMPLATES

REQUIRED_KEYS = {"severity", "affected_service", "probable_cause",
                 "runbook_ref", "escalate_to", "escalate"}


def test_triage_answer_has_six_keys_and_v1_facts():
    svc = SERVICES["checkout-svc"]
    tmpl = ALERT_TEMPLATES["checkout-svc"][0]
    ans = triage_answer(svc, tmpl)
    assert set(ans) == REQUIRED_KEYS
    assert ans["severity"] == "P1" and ans["escalate"] is True
    assert ans["affected_service"] == "checkout-svc"
    assert ans["runbook_ref"] == "RB-CHK-001"
    assert ans["escalate_to"] == "#atlas-oncall"


def test_gen_examples_is_deterministic_and_covers_all_services():
    a = gen_examples(SERVICES, seed=1, n=120)
    b = gen_examples(SERVICES, seed=1, n=120)
    assert a == b
    assert {e["service"] for e in a} == set(SERVICES)
    assert all(set(e["truth"]) == REQUIRED_KEYS for e in a)


def test_to_chat_produces_assistant_json_only():
    ex = gen_examples(SERVICES, seed=1, n=1)[0]
    chat = to_chat(ex)
    roles = [m["role"] for m in chat["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert chat["messages"][0]["content"] == SYSTEM_PROMPT
    assert chat["messages"][1]["content"] == ex["alert"]
    assert json.loads(chat["messages"][2]["content"]) == ex["truth"]


def test_split_has_no_alert_leakage_and_expected_sizes():
    train, valid, ev = split_train_valid_eval(SERVICES, seed=1)
    assert len(train) == 360 and len(valid) == 40 and len(ev) == 24
    train_alerts = {e["alert"] for e in train} | {e["alert"] for e in valid}
    assert not any(e["alert"] in train_alerts for e in ev)
    assert {e["service"] for e in ev} == set(SERVICES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ep04_gen_data.py -v`
Expected: FAIL with `ImportError: cannot import name 'SYSTEM_PROMPT'`

- [ ] **Step 3: Add data generation**

In `bench/ep04_gen_data.py`, add imports and functions (keep the docs code from Task 2; replace `main`):

```python
import json
import random

from bench.ep04_world import ALERT_TEMPLATES, REGIONS, AlertTemplate

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ep04_gen_data.py -v`
Expected: 8 passed

- [ ] **Step 5: Generate data, eyeball it, commit**

Run: `uv run python bench/ep04_gen_data.py && head -c 600 data/ep04/train.jsonl && wc -l data/ep04/*.jsonl`
Expected: 360 / 40 / 24 lines. Chief reviews ~10 lines of `train.jsonl` for voice and correctness before Task 6.

```bash
git add bench/ep04_gen_data.py tests/test_ep04_gen_data.py data/ep04/*.jsonl
git commit -m "Ep 04: generate LoRA train/valid + held-out eval JSONL"
```

---

### Task 4: TF-IDF retrieval over the docs

**Files:**
- Create: `bench/ep04_rag.py`, `tests/test_ep04_rag.py`

**Interfaces:**
- Produces: `@dataclass Chunk(doc: str, heading: str, text: str)`; `chunk_docs(docs: dict[str, str]) -> list[Chunk]` (split on `#`/`##` headings; catalog table rows become one chunk per row); `class TfidfIndex` with `TfidfIndex.build(chunks) -> TfidfIndex`, `.retrieve(query: str, k: int = 3) -> list[Chunk]`; `load_index(docs_dir: Path) -> TfidfIndex`; `format_context(chunks) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_ep04_rag.py`:

```python
from bench.ep04_gen_data import build_docs
from bench.ep04_rag import TfidfIndex, chunk_docs, format_context
from bench.ep04_world import SERVICES, reorg


def _index(services):
    return TfidfIndex.build(chunk_docs(build_docs(services)))


def test_chunks_carry_doc_and_heading():
    chunks = chunk_docs(build_docs(SERVICES))
    assert any(c.doc == "runbook-checkout-svc.md" and c.heading.startswith("Escalation") for c in chunks)
    assert all(c.text.strip() for c in chunks)


def test_checkout_alert_retrieves_checkout_runbook_first():
    idx = _index(SERVICES)
    hits = idx.retrieve("[FIRING] checkout-svc p99 latency 4200ms db_pool_wait_ms=1900 region=us-west-2", k=3)
    assert hits[0].doc == "runbook-checkout-svc.md"
    assert any("#atlas-oncall" in h.text for h in hits)


def test_v2_index_returns_borealis_not_atlas():
    idx = _index(reorg(SERVICES))
    ctx = format_context(idx.retrieve("checkout-svc 5xx rate 12% on /cart/confirm", k=3))
    assert "#borealis-oncall" in ctx
    assert "#atlas-oncall" not in ctx


def test_each_service_alert_retrieves_its_own_runbook():
    idx = _index(SERVICES)
    for name in SERVICES:
        hits = idx.retrieve(f"[FIRING] {name} something is wrong region=us-west-2", k=3)
        assert hits[0].doc == f"runbook-{name}.md", name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ep04_rag.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.ep04_rag'`

- [ ] **Step 3: Write the retriever**

`bench/ep04_rag.py`:

```python
"""
bench/ep04_rag.py — Ep 04: deliberately boring retrieval.

Chunk markdown by heading, TF-IDF vectors, cosine top-k. Zero dependencies.
Retrieval quality is not the story of Ep 04 — editability is: swap docs_v1
for docs_v2 and the index rebuilds in milliseconds.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9#_./-]+")


def tokenize(text: str) -> list[str]:
    # keep service names ("checkout-svc"), channels ("#atlas-oncall"), metrics ("db_pool_wait_ms")
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Chunk:
    doc: str
    heading: str
    text: str


def chunk_docs(docs: dict[str, str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for name, md in docs.items():
        heading, buf = "", []
        for line in md.splitlines():
            if line.startswith("#"):
                if buf and "".join(buf).strip():
                    chunks.append(Chunk(name, heading, "\n".join(buf).strip()))
                heading, buf = line.lstrip("# ").strip(), []
            else:
                buf.append(line)
        if buf and "".join(buf).strip():
            chunks.append(Chunk(name, heading, "\n".join(buf).strip()))
    # prefix heading so the service name in "# RB-CHK-001 — checkout-svc" is searchable
    return [Chunk(c.doc, c.heading, f"{c.heading}\n{c.text}") for c in chunks]


class TfidfIndex:
    def __init__(self, chunks: list[Chunk], idf: dict[str, float], vecs: list[dict[str, float]]):
        self.chunks, self.idf, self.vecs = chunks, idf, vecs

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "TfidfIndex":
        docs_tokens = [tokenize(c.text) for c in chunks]
        df = Counter(t for toks in docs_tokens for t in set(toks))
        n = len(chunks)
        idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
        vecs = [cls._vec(toks, idf) for toks in docs_tokens]
        return cls(chunks, idf, vecs)

    @staticmethod
    def _vec(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
        tf = Counter(tokens)
        v = {t: c * idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def retrieve(self, query: str, k: int = 3) -> list[Chunk]:
        q = self._vec(tokenize(query), self.idf)
        scored = []
        for chunk, v in zip(self.chunks, self.vecs):
            s = sum(w * v.get(t, 0.0) for t, w in q.items())
            scored.append((s, chunk))
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:k]]


def load_index(docs_dir: Path) -> TfidfIndex:
    docs = {p.name: p.read_text() for p in sorted(docs_dir.glob("*.md"))}
    return TfidfIndex.build(chunk_docs(docs))


def format_context(chunks: list[Chunk]) -> str:
    parts = [f"[{c.doc} › {c.heading}]\n{c.text}" for c in chunks]
    return "Reference material:\n\n" + "\n\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ep04_rag.py -v`
Expected: 4 passed. If `test_each_service_alert_retrieves_its_own_runbook` fails for a service, the catalog table row chunk is outscoring the runbook; fix by making `_catalog` rows headed sections is NOT the answer — instead boost: in `retrieve`, add `+0.5` to the score when the query contains the chunk's doc service name (`c.doc.startswith("runbook-") and c.doc[8:-3] in query`). Keep whichever passes; document it in the module docstring.

- [ ] **Step 5: Commit**

```bash
git add bench/ep04_rag.py tests/test_ep04_rag.py
git commit -m "Ep 04: TF-IDF chunk retriever over runbook docs"
```

---

### Task 5: Deterministic scoring

**Files:**
- Create: `bench/ep04_score.py`, `tests/test_ep04_score.py`

**Interfaces:**
- Produces: `REQUIRED_KEYS`, `FACT_KEYS = ("affected_service", "runbook_ref", "escalate_to")`; `strip_think(text) -> str`; `parse_output(text) -> tuple[dict | None, bool]` returning `(obj, clean)` where `clean` is True only if the whole stripped text is exactly one JSON object (no prose, no code fence); `score_behavior(text) -> dict` with keys `valid_json, all_keys, severity_ok, escalate_bool, no_prose, compliant` (compliant = all True); `score_facts(obj, truth) -> dict` with one bool per fact key plus `facts_correct: int` (0–3).

- [ ] **Step 1: Write the failing tests**

`tests/test_ep04_score.py`:

```python
import json

from bench.ep04_score import parse_output, score_behavior, score_facts, strip_think

GOOD = {"severity": "P1", "affected_service": "checkout-svc",
        "probable_cause": "pool exhausted", "runbook_ref": "RB-CHK-001",
        "escalate_to": "#atlas-oncall", "escalate": True}


def test_strip_think_removes_reasoning_block():
    assert strip_think("<think>\nhmm\n</think>\n{\"a\":1}") == '{"a":1}'
    assert strip_think('{"a":1}') == '{"a":1}'


def test_parse_clean_json():
    obj, clean = parse_output(json.dumps(GOOD))
    assert obj == GOOD and clean is True


def test_parse_json_inside_prose_or_fence_is_not_clean():
    obj, clean = parse_output("Sure! Here is the triage:\n```json\n" + json.dumps(GOOD) + "\n```")
    assert obj == GOOD and clean is False


def test_parse_garbage_returns_none():
    assert parse_output("The service seems down, page someone.") == (None, False)


def test_behavior_score_compliant():
    s = score_behavior(json.dumps(GOOD))
    assert s["compliant"] is True and s["no_prose"] is True


def test_behavior_score_bad_severity_and_prose():
    bad = dict(GOOD, severity="SEV1")
    s = score_behavior("Triage: " + json.dumps(bad))
    assert s["valid_json"] is True
    assert s["severity_ok"] is False and s["no_prose"] is False and s["compliant"] is False


def test_behavior_score_missing_key():
    partial = {k: v for k, v in GOOD.items() if k != "runbook_ref"}
    assert score_behavior(json.dumps(partial))["all_keys"] is False


def test_fact_score_counts_matches():
    truth = dict(GOOD)
    stale = dict(GOOD, escalate_to="#borealis-oncall")
    assert score_facts(GOOD, truth)["facts_correct"] == 3
    f = score_facts(stale, truth)
    assert f["escalate_to"] is False and f["facts_correct"] == 2
    assert score_facts(None, truth)["facts_correct"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ep04_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.ep04_score'`

- [ ] **Step 3: Write the scorer**

`bench/ep04_score.py`:

```python
"""
bench/ep04_score.py — Ep 04: deterministic scoring, no LLM judge.

Behavior axis (what LoRA should fix): is the output exactly one JSON object
with the six keys, a valid severity, a boolean escalate, and no prose?
Fact axis (what RAG should fix): do affected_service / runbook_ref /
escalate_to match the ground truth for that alert?
"""

from __future__ import annotations

import json
import re

REQUIRED_KEYS = ("severity", "affected_service", "probable_cause",
                 "runbook_ref", "escalate_to", "escalate")
FACT_KEYS = ("affected_service", "runbook_ref", "escalate_to")
SEVERITIES = {"P1", "P2", "P3"}

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def strip_think(text: str) -> str:
    return _THINK.sub("", text).strip()


def _first_json_object(text: str) -> dict | None:
    # scan for the first balanced {...} that parses as a dict
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def parse_output(text: str) -> tuple[dict | None, bool]:
    """Return (parsed object or None, clean) — clean means the whole output is one JSON object."""
    t = strip_think(text)
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj, True
    except json.JSONDecodeError:
        pass
    m = _FENCE.search(t)
    if m:
        obj = _first_json_object(m.group(1))
        if obj is not None:
            return obj, False
    return _first_json_object(t), False


def score_behavior(text: str) -> dict:
    obj, clean = parse_output(text)
    valid = obj is not None
    all_keys = valid and all(k in obj for k in REQUIRED_KEYS)
    severity_ok = valid and obj.get("severity") in SEVERITIES
    escalate_bool = valid and isinstance(obj.get("escalate"), bool)
    no_prose = clean
    return {
        "valid_json": valid, "all_keys": all_keys, "severity_ok": severity_ok,
        "escalate_bool": escalate_bool, "no_prose": no_prose,
        "compliant": bool(valid and all_keys and severity_ok and escalate_bool and no_prose),
    }


def score_facts(obj: dict | None, truth: dict) -> dict:
    out = {k: bool(obj is not None and obj.get(k) == truth[k]) for k in FACT_KEYS}
    out["facts_correct"] = sum(out.values())
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ep04_score.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add bench/ep04_score.py tests/test_ep04_score.py
git commit -m "Ep 04: deterministic behavior + fact scorer"
```

---

### Task 6: Model runner (mlx-lm wrapper) + base-model smoke test

**Files:**
- Create: `bench/ep04_model.py`

**Interfaces:**
- Produces: `BASE_MODEL = "mlx-community/Qwen3.5-4B-4bit"`; `@dataclass Generation(text: str, ttft_s: float, decode_tps: float, output_tokens: int)`; `class Runner` with `Runner(model_id: str = BASE_MODEL, adapter_path: str | None = None)`, `.triage(alert: str, context: str | None = None, max_tokens: int = 200) -> Generation`, `.label: str` (`"base"` or `"tuned"`). Prompt layout: system = `SYSTEM_PROMPT` (+ `"\n\n" + context` when given), user = alert. Chat template applied with `add_generation_prompt=True, enable_thinking=False`.

No unit test (requires the 4B model); the deliverable is a smoke run whose output Chief inspects.

- [ ] **Step 1: Download the base model**

Run: `uv run python -c "from mlx_lm import load; load('mlx-community/Qwen3.5-4B-4bit'); print('ok')"`
Expected: downloads ~2.5GB to the HF cache, prints `ok`. If `enable_thinking` is unsupported by the chat template you will see no error here — checked in Step 3.

- [ ] **Step 2: Write the runner**

`bench/ep04_model.py`:

```python
"""
bench/ep04_model.py — Ep 04: thin mlx-lm wrapper.

One code path for all four grid configs:
  base        Runner()
  base+RAG    Runner().triage(alert, context=retrieved)
  tuned       Runner(adapter_path="adapters/ep04-qwen3.5-4b")
  tuned+RAG   Runner(adapter_path=...).triage(alert, context=retrieved)

Thinking mode is disabled via the chat template so reasoning traces never
enter the JSON-compliance score. Timing is measured here (TTFT, decode tok/s)
because it is a secondary result in the write-up.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from mlx_lm import load, stream_generate

from bench.ep04_gen_data import SYSTEM_PROMPT

BASE_MODEL = "mlx-community/Qwen3.5-4B-4bit"


@dataclass
class Generation:
    text: str
    ttft_s: float
    decode_tps: float
    output_tokens: int


class Runner:
    def __init__(self, model_id: str = BASE_MODEL, adapter_path: str | None = None):
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.label = "tuned" if adapter_path else "base"
        self.model, self.tokenizer = load(model_id, adapter_path=adapter_path)

    def _prompt(self, alert: str, context: str | None) -> str:
        system = SYSTEM_PROMPT if context is None else f"{SYSTEM_PROMPT}\n\n{context}"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": alert}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:  # template without a thinking switch
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)

    def triage(self, alert: str, context: str | None = None, max_tokens: int = 200) -> Generation:
        prompt = self._prompt(alert, context)
        t0 = time.perf_counter()
        first, last, n, text = None, t0, 0, ""
        for r in stream_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens):
            now = time.perf_counter()
            if first is None:
                first = now
            last, n, text = now, n + 1, text + r.text
        ttft = (first or last) - t0
        decode = (n - 1) / (last - first) if first and n > 1 and last > first else 0.0
        return Generation(text=text.strip(), ttft_s=ttft, decode_tps=decode, output_tokens=n)
```

- [ ] **Step 3: Smoke-run the base model on three eval alerts**

Run:

```bash
uv run python - <<'EOF'
import json
from bench.ep04_model import Runner
from bench.ep04_score import score_behavior
rows = [json.loads(l) for l in open("data/ep04/eval.jsonl")][:3]
r = Runner()
for row in rows:
    g = r.triage(row["alert"])
    print("ALERT:", row["alert"]); print("OUT:", g.text[:300]); print(score_behavior(g.text), f"ttft={g.ttft_s:.2f}s tps={g.decode_tps:.0f}\n")
EOF
```

Expected: three generations, no `<think>` text in output (if there is, the template ignored `enable_thinking`; `strip_think` in the scorer still protects the score, but note it). Base compliance is expected to be mixed — that is the point of config ①.

- [ ] **Step 4: Commit**

```bash
git add bench/ep04_model.py
git commit -m "Ep 04: mlx-lm runner with optional LoRA adapter + retrieved context"
```

---

### Task 7: LoRA training + sanity gate

**Files:**
- Create: `bench/ep04_lora.yaml`, `bench/ep04_sanity.py`
- Produces: `adapters/ep04-qwen3.5-4b/adapters.safetensors` (gitignored), `adapters/ep04-qwen3.5-4b/adapter_config.json`, `results/ep04-lora-train.log` (committed).

**Interfaces:**
- Consumes: `Runner`, `score_behavior`, `data/ep04/{train,valid}.jsonl`.

- [ ] **Step 1: Write the training config**

`bench/ep04_lora.yaml`:

```yaml
# Ep 04 LoRA config — run with: uv run mlx_lm.lora --config bench/ep04_lora.yaml
model: mlx-community/Qwen3.5-4B-4bit
train: true
data: data/ep04            # expects train.jsonl + valid.jsonl (chat format)
adapter_path: adapters/ep04-qwen3.5-4b
fine_tune_type: lora
num_layers: 16
batch_size: 4
iters: 600
learning_rate: 1.0e-5
steps_per_report: 10
steps_per_eval: 50
val_batches: 10
save_every: 100
max_seq_length: 512
mask_prompt: true          # loss only on the assistant JSON
seed: 42
lora_parameters:
  rank: 8
  scale: 20.0
  dropout: 0.0
```

- [ ] **Step 2: Train (Chief runs this by hand and watches the loss)**

Run: `mkdir -p results && uv run mlx_lm.lora --config bench/ep04_lora.yaml 2>&1 | tee results/ep04-lora-train.log`
Expected: ~10 minutes on the M5 Max. Train loss should drop from ~2–3 toward <0.3; val loss printed every 50 iters should fall and flatten. If val loss rises after ~300 iters, stop and rerun with `iters: 300`.
Record in the log header which iteration count was used if changed.

- [ ] **Step 3: Write the sanity gate**

`bench/ep04_sanity.py`:

```python
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
```

- [ ] **Step 4: Run the gate**

Run: `uv run python bench/ep04_sanity.py`
Expected: `GATE PASS`. If FAIL: inspect the tuned outputs printed; the usual causes are (a) `<think>` blocks — confirm `enable_thinking=False` took effect in `Runner._prompt`; (b) under-training — raise `iters` to 900; (c) output truncated — raise `max_tokens` to 300.

- [ ] **Step 5: Commit**

```bash
git add bench/ep04_lora.yaml bench/ep04_sanity.py results/ep04-lora-train.log adapters/ep04-qwen3.5-4b/adapter_config.json
git commit -m "Ep 04: LoRA config, training log, sanity gate"
```

---

### Task 8: The grid + reorg evaluation script

**Files:**
- Create: `bench/ep04_eval.py`, `tests/test_ep04_eval.py`

**Interfaces:**
- Consumes: `Runner` (injectable — the script accepts any object with `.triage(alert, context)` returning something with `.text/.ttft_s/.decode_tps`), `load_index`, `format_context`, `score_behavior`, `score_facts`, `parse_output`.
- Produces: `CONFIGS = ("base", "base_rag", "tuned", "tuned_rag")`; `run_config(name, runner, rows, index) -> dict` with `summary` (`n`, `compliance_rate`, `fact_accuracy`, `mean_ttft_s`, `mean_decode_tps`) and `runs` (per-alert records: `alert`, `service`, `output`, `behavior`, `facts`, `ttft_s`, `decode_tps`); `truth_for(row, services)`; `markdown_grid(results) -> str`; `markdown_reorg(results) -> str`; CLI `--docs v1|v2 --configs ... --only-service checkout-svc --out results/ep04-grid.json`.

- [ ] **Step 1: Write the failing tests (fake runner, no model)**

`tests/test_ep04_eval.py`:

```python
import json

from bench.ep04_eval import CONFIGS, markdown_grid, run_config, truth_for
from bench.ep04_gen_data import build_docs, gen_eval
from bench.ep04_rag import TfidfIndex, chunk_docs
from bench.ep04_world import SERVICES, reorg


class FakeGen:
    def __init__(self, text): self.text, self.ttft_s, self.decode_tps, self.output_tokens = text, 0.1, 50.0, 30


class FakeRunner:
    """Echoes the escalation channel it sees in context, else a stale default; tuned → clean JSON."""
    def __init__(self, tuned: bool): self.tuned = tuned
    def triage(self, alert, context=None, max_tokens=200):
        channel = "#atlas-oncall"
        if context:
            for tok in context.split():
                if tok.startswith("#") and tok.endswith("-oncall"):
                    channel = tok
                    break
        obj = {"severity": "P1", "affected_service": alert.split()[1],
               "probable_cause": "x", "runbook_ref": "RB-CHK-001",
               "escalate_to": channel, "escalate": True}
        text = json.dumps(obj)
        return FakeGen(text if self.tuned else "Here you go: " + text)


def _rows():
    return [r for r in gen_eval(SERVICES) if r["service"] == "checkout-svc"]


def test_truth_for_uses_current_world():
    row = _rows()[0]
    assert truth_for(row, SERVICES)["escalate_to"] == "#atlas-oncall"
    assert truth_for(row, reorg(SERVICES))["escalate_to"] == "#borealis-oncall"


def test_run_config_scores_behavior_and_facts():
    idx = TfidfIndex.build(chunk_docs(build_docs(SERVICES)))
    res = run_config("tuned_rag", FakeRunner(tuned=True), _rows(), idx, SERVICES)
    assert res["summary"]["n"] == 4
    assert res["summary"]["compliance_rate"] == 1.0
    assert res["summary"]["fact_accuracy"] == 1.0
    res_base = run_config("base", FakeRunner(tuned=False), _rows(), None, SERVICES)
    assert res_base["summary"]["compliance_rate"] == 0.0


def test_reorg_stale_vs_fresh():
    v2 = reorg(SERVICES)
    idx2 = TfidfIndex.build(chunk_docs(build_docs(v2)))
    stale = run_config("tuned", FakeRunner(tuned=True), _rows(), None, v2)
    fresh = run_config("tuned_rag", FakeRunner(tuned=True), _rows(), idx2, v2)
    assert stale["summary"]["fact_accuracy"] < fresh["summary"]["fact_accuracy"]
    assert all(r["facts"]["escalate_to"] is False for r in stale["runs"])
    assert all(r["facts"]["escalate_to"] is True for r in fresh["runs"])


def test_markdown_grid_lists_all_configs():
    idx = TfidfIndex.build(chunk_docs(build_docs(SERVICES)))
    results = {c: run_config(c, FakeRunner("tuned" in c), _rows(), idx if "rag" in c else None, SERVICES)
               for c in CONFIGS}
    md = markdown_grid(results)
    for c in CONFIGS:
        assert c in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ep04_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.ep04_eval'`

- [ ] **Step 3: Write the eval script**

`bench/ep04_eval.py`:

```python
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
from bench.ep04_score import parse_output, score_behavior, score_facts
from bench.ep04_world import ALERT_TEMPLATES, SERVICES, Service, reorg

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "ep04"
ADAPTER = "adapters/ep04-qwen3.5-4b"
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
            "ttft_s": round(g.ttft_s, 3), "decode_tps": round(g.decode_tps, 1),
        })
    n = len(runs)
    summary = {
        "config": name, "n": n,
        "compliance_rate": sum(r["behavior"]["compliant"] for r in runs) / n,
        "fact_accuracy": sum(r["facts"]["facts_correct"] for r in runs) / (3 * n),
        "mean_ttft_s": round(sum(r["ttft_s"] for r in runs) / n, 3),
        "mean_decode_tps": round(sum(r["decode_tps"] for r in runs) / n, 1),
    }
    return {"summary": summary, "runs": runs}


def markdown_grid(results: dict[str, dict]) -> str:
    lines = ["| config | compliance | fact accuracy | mean TTFT | decode tok/s |", "|---|---|---|---|---|"]
    for c, r in results.items():
        s = r["summary"]
        lines.append(f"| {c} | {s['compliance_rate']:.0%} | {s['fact_accuracy']:.0%} | "
                     f"{s['mean_ttft_s']:.2f}s | {s['mean_decode_tps']:.0f} |")
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -v`
Expected: all Ep 04 tests pass (world 4, gen_data 8, rag 4, score 8, eval 4 = 28).

- [ ] **Step 5: Commit**

```bash
git add bench/ep04_eval.py tests/test_ep04_eval.py
git commit -m "Ep 04: grid + reorg evaluation with deterministic scoring"
```

---

### Task 9: Run the experiment and record results

**Files:**
- Create: `results/ep04-grid.json`, `results/ep04-reorg.json`, `results/ep04-summary.md`

- [ ] **Step 1: Run the grid**

Run: `uv run python bench/ep04_eval.py --docs v1 --out results/ep04-grid.json`
Expected: ~24 alerts × 4 configs ≈ 96 generations, a few minutes. Printed markdown grid. Sanity expectations: `base_rag` fact accuracy ≫ `base`; `tuned` compliance ≫ `base`; `tuned_rag` best on both.

- [ ] **Step 2: Run the reorg**

Run: `uv run python bench/ep04_eval.py --docs v2 --configs base_rag tuned tuned_rag --only-service checkout-svc --out results/ep04-reorg.json`
Expected: `tuned` answers `#atlas-oncall` (0/4 correct); `base_rag` and `tuned_rag` answer `#borealis-oncall`. If `tuned_rag` is also stale, that is the RAFT finding from the spec — keep it, it goes in the post.

- [ ] **Step 3: Write the summary file**

Create `results/ep04-summary.md` by pasting both printed tables under headings `## Grid (docs_v1)` and `## Reorg (docs_v2, checkout-svc)`, plus the final train/val loss line from `results/ep04-lora-train.log` and iteration count used.

- [ ] **Step 4: Commit**

```bash
git add results/ep04-grid.json results/ep04-reorg.json results/ep04-summary.md
git commit -m "Ep 04: grid + reorg results (Qwen3.5-4B, LoRA vs RAG)"
```

---

### Task 10: Portfolio write-up skeleton + README row

**Files:**
- Create: `content/ep04-portfolio.md`
- Modify: `README.md` (episodes table + reproduce section)

- [ ] **Step 1: Write the portfolio skeleton with real numbers**

`content/ep04-portfolio.md` — sections, each filled from `results/ep04-summary.md` (no placeholders: copy the actual tables):

```markdown
# Ep 04 — Qwen3.5-4B: LoRA vs RAG on the same alert

**Framework under test:** Paolo Perrone, *RAG vs Fine-Tuning — When to Use Each in Production*
(https://theaiengineer.substack.com/p/rag-vs-fine-tuning). His rule: RAG changes what the model
sees, fine-tuning changes how it behaves, production wants both. This episode checks it on a
fully synthetic on-call desk ("NimbusCart"), locally, on an M5 Max.

## Setup
- Base: mlx-community/Qwen3.5-4B-4bit · LoRA rank 8, 16 layers, <iters> iters, lr 1e-5 (mlx-lm 0.31.3)
- Knowledge: 8 markdown runbook/catalog docs → TF-IDF top-3 retrieval
- Behavior: 360 alert→triage-JSON pairs (v1 facts baked in), 40 validation
- Eval: 24 held-out alerts, deterministic scoring (JSON compliance · fact accuracy)

## The grid
<paste markdown_grid table>

## The reorg
<paste markdown_reorg table>

## What moved which axis
<3–5 sentences from the numbers>

## Caveats
- Retrieval is TF-IDF, not neural embeddings — chosen because the point is editability, not recall.
- One model, one LoRA config, one seed. Gemma 4 E4B rerun is the stretch goal.
- Fully synthetic world; numbers describe this task, not "AIOps in general".
```

Replace every `<…>` with the real content before committing.

- [ ] **Step 2: Update README**

In the Episodes table add:

```markdown
| [Ep 04](content/ep04-portfolio.md) | Qwen3.5-4B: LoRA vs RAG on the same alert — 2×2 grid + post-training reorg | ✅ Done |
```

Under the reproduce sections add:

```bash
uv run python bench/ep04_gen_data.py                       # docs + jsonl
uv run mlx_lm.lora --config bench/ep04_lora.yaml            # ~10 min on M5 Max
uv run python bench/ep04_sanity.py                          # gate
uv run python bench/ep04_eval.py --docs v1 --out results/ep04-grid.json
uv run python bench/ep04_eval.py --docs v2 --configs base_rag tuned tuned_rag --only-service checkout-svc --out results/ep04-reorg.json
```

- [ ] **Step 3: Commit**

```bash
git add content/ep04-portfolio.md README.md
git commit -m "Ep 04: portfolio write-up + README"
```

Article, X thread and LinkedIn visuals follow from `content/ep04-portfolio.md` and are content work, not part of this plan.

---

## Self-review notes

- **Spec coverage:** world/ground truth (T1), docs v1/v2 (T2), training data + eval set + thinking-off (T3, T6), RAG (T4), scoring three axes incl. latency (T5, T6, T8), LoRA config/run/sanity gate (T7), grid + reorg + results JSON (T8, T9), write-up + README (T10). Gemma 4 stretch intentionally not planned.
- **Deviation from spec, stated up front:** TF-IDF instead of neural embeddings; 8 docs instead of "~12".
- **Type consistency:** `Runner.triage(alert, context, max_tokens)` in T6 matches the fake in T8; `score_facts(obj, truth)` signature consistent T5→T8; `truth_for` uses `triage_answer` from T3.
