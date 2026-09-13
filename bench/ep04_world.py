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
