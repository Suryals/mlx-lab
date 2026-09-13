# RB-SRCH-001 — search-svc

Owner: team-lens  
Escalation channel: #lens-oncall

## Known failure mode
Index rebuild starving query latency

## Symptoms
- query p95 above 1.5s
- indexer CPU at 100%
- search timeouts on category pages

## Remediation
1. Throttle indexer to 2 shards concurrent
2. Route queries to warm replica set
3. Defer rebuild to off-peak window

## Escalation
P1/P2: page #lens-oncall (team-lens). P3: ticket to team-lens, no page.
