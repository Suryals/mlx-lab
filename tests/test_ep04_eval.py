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


def test_run_config_handles_empty_rows():
    res = run_config("base", FakeRunner(tuned=False), [], None, SERVICES)
    assert res["summary"]["n"] == 0
    assert res["summary"]["compliance_rate"] == 0.0
    assert res["summary"]["fact_accuracy"] == 0.0
    assert res["summary"]["severity_accuracy"] == 0.0


def test_severity_accuracy_tracks_severity_match_separately_from_facts():
    res = run_config("tuned", FakeRunner(tuned=True), _rows(), None, SERVICES)
    assert res["summary"]["severity_accuracy"] == 0.5
    mismatches = [r for r in res["runs"] if r["truth"]["severity"] in ("P2", "P3")]
    assert len(mismatches) == 2
    assert all(r["severity_match"] is False for r in mismatches)


def test_markdown_grid_lists_all_configs():
    idx = TfidfIndex.build(chunk_docs(build_docs(SERVICES)))
    results = {c: run_config(c, FakeRunner("tuned" in c), _rows(), idx if "rag" in c else None, SERVICES)
               for c in CONFIGS}
    md = markdown_grid(results)
    for c in CONFIGS:
        assert c in md
