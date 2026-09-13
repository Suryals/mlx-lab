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
