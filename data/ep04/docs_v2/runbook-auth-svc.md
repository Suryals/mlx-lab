# RB-AUTH-001 — auth-svc

Owner: team-gatekeeper  
Escalation channel: #gate-oncall

## Known failure mode
Token-signing key rotation missed

## Symptoms
- token_verify_fail_rate spike
- 401 rate above 2%
- JWKS fetch errors

## Remediation
1. Rotate signing key via keyctl rotate --service auth
2. Republish JWKS endpoint
3. Extend previous key grace period by 1h

## Escalation
P1/P2: page #gate-oncall (team-gatekeeper). P3: ticket to team-gatekeeper, no page.
