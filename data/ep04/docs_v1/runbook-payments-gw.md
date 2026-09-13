# RB-PAY-001 — payments-gw

Owner: team-ledger  
Escalation channel: #ledger-oncall

## Known failure mode
Upstream PSP timeout causing retry storm

## Symptoms
- psp_timeout_rate above 5%
- retry_queue_depth growing
- duplicate authorization warnings

## Remediation
1. Set retry backoff to exponential with 30s cap
2. Fail over to secondary PSP region
3. Pause retries for cards already authorized

## Escalation
P1/P2: page #ledger-oncall (team-ledger). P3: ticket to team-ledger, no page.
