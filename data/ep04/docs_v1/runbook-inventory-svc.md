# RB-INV-001 — inventory-svc

Owner: team-stockroom  
Escalation channel: #stockroom-oncall

## Known failure mode
Cache/DB drift after bulk import

## Symptoms
- stock_mismatch_count rising
- oversell alerts
- cache_hit_ratio drop after import job

## Remediation
1. Invalidate inventory cache namespace
2. Re-run reconciliation job with --dry-run first
3. Block bulk imports until drift below 0.1%

## Escalation
P1/P2: page #stockroom-oncall (team-stockroom). P3: ticket to team-stockroom, no page.
