# RB-CHK-001 — checkout-svc

Owner: team-borealis  
Escalation channel: #borealis-oncall

## Known failure mode
DB connection-pool exhaustion during flash sales

## Symptoms
- p99 latency above 800ms
- db_pool_wait_ms climbing
- 5xx rate on /cart/confirm

## Remediation
1. Scale checkout-db read replicas by 2
2. Raise pool_max from 40 to 80 via config map
3. Enable checkout queueing flag if 5xx persists

## Escalation
P1/P2: page #borealis-oncall (team-borealis). P3: ticket to team-borealis, no page.
