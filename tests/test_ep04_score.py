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
