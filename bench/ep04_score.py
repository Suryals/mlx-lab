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
