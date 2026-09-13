# RB-NTF-001 — notify-svc

Owner: team-beacon  
Escalation channel: #beacon-oncall

## Known failure mode
Queue backlog when email vendor rate-limits

## Symptoms
- email_queue_depth above 50k
- vendor 429 responses
- notification latency above 10min

## Remediation
1. Switch to secondary email vendor
2. Drop marketing tier from queue temporarily
3. Raise consumer count from 4 to 12

## Escalation
P1/P2: page #beacon-oncall (team-beacon). P3: ticket to team-beacon, no page.
