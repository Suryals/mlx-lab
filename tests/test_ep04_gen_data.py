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
