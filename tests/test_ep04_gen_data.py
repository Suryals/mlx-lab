import json
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
