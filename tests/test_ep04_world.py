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
