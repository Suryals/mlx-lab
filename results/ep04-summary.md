# Ep 04 results — Qwen3.5-4B: LoRA vs RAG on the same alert

Base: `mlx-community/Qwen3.5-4B-4bit` · LoRA rank 8, last 16/32 layers, 600 iters (loss hit 0.000 at iter 70), lr 1e-5, mlx-lm 0.31.3 · M5 Max 128GB.
Eval: 24 held-out alerts (6 services × 4 templates, unseen region/values). Scoring deterministic (`bench/ep04_score.py`).
Sanity gate: base 10/10, tuned 10/10 JSON-compliant.

## Grid (docs_v1)

| config | compliance | severity | fact accuracy | mean TTFT | decode tok/s |
|---|---|---|---|---|---|
| base | 100% | 71% | 33% | 0.13s | 168 |
| base_rag | 100% | 75% | 92% | 0.15s | 167 |
| tuned | 100% | 100% | 100% | 0.16s | 91 |
| tuned_rag | 100% | 88% | 97% | 0.18s | 92 |

Fact accuracy broken down (affected_service / runbook_ref / escalate_to):

| config | service | runbook | channel |
|---|---|---|---|
| base | 100% | 0% | 0% |
| base_rag | 100% | 88% | 88% |
| tuned | 100% | 100% | 100% |
| tuned_rag | 100% | 92% | 100% |

The base model always names the right service (it is in the alert text) and always emits valid JSON, but invents runbook ids (`RUNBOOK-DB-POOL-LATENCY`) and channels (`platform-lead`) — 0/24 on both. Retrieval fixes the facts (0% → 88%) and leaves severity where it was (71% → 75%). The adapter fixes severity (71% → 100%) — the house severity policy is learned behaviour, not a lookup.

**Caveat on the severity gap:** eval alerts carry magnitude-nonsensical values (value ranges are template-agnostic, not clamped per template) — e.g. "5xx rate 9202%", "[INFO] … pool utilisation 9404%", "signing key expires in 9303h". Some of the base model's severity misses are it reading the (nonsense) number instead of the `[FIRING]/[WARNING]/[INFO]` tag, while the adapter learned to trust the tag. The severity-policy claim still stands; the size of the 71% → 100% gap is inflated by this artefact.

## Reorg (docs_v2, checkout-svc, no retraining)

team-atlas dissolved → checkout-svc owned by team-borealis, escalation `#borealis-oncall`. One markdown edit.

| config | escalate_to answers (after reorg) | correct |
|---|---|---|
| base_rag | #borealis-oncall | 4/4 |
| tuned | #atlas-oncall | 0/4 |
| tuned_rag | #borealis-oncall | 4/4 |

The tuned-only model pages a team that no longer exists, 4/4 times, with perfect confidence and perfect JSON. Both retrieval configs follow the doc edit immediately. The RAFT-style risk (adapter overpowering retrieved context) did not materialise on this task.

## Caveats worth keeping

- **Decode speed:** tuned configs run at ~91 tok/s vs ~168 for base — the LoRA adapter is applied unfused at inference. `mlx_lm.fuse` would remove that gap; not measured here.
- **tuned_rag < tuned on v1:** 97% vs 100% facts, 88% vs 100% severity. Not memorised behaviour being perturbed by context in general — every runbook miss across the RAG configs (3 in base_rag, 2 in tuned_rag) is the model echoing back the doc filename `runbook-<svc>.md`, copied straight from the `[doc › heading]` label `format_context` prefixes to each retrieved chunk. (base_rag's two channel misses are a different mix-up: `team-atlas`, an owner not a channel, and `platform-lead`.) It's a context-format artefact worth fixing in a follow-up, not evidence the adapter's learned policy is fragile. On v2 tuned_rag is the only config that is both current and on-policy.
- **Base compliance was already 100%** with thinking disabled — the spec's "fine-tune rescues JSON format" axis did not show on this model; the visible behaviour gain is severity policy and compact voice.
- Loss reached 0.000 by iter 70; 600 iters was ~9× more than needed. Mac slept twice during the run — visible as the it/s collapse at iters 340–410 and 480–520, and the stall inside the iter-250 validation progress bar in `ep04-lora-train.log` (the log has no wall-clock timestamps).
- Retrieval is TF-IDF over 8 docs; one model, one seed, 24 alerts. Numbers describe this task, not "AIOps in general".
