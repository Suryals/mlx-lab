# Ep 04 results — Qwen3.5-4B: LoRA vs RAG on the same alert

Base: `mlx-community/Qwen3.5-4B-4bit` · LoRA rank 8, last 16/36 layers, 600 iters (loss hit 0.000 at iter 70), lr 1e-5, mlx-lm 0.31.3 · M5 Max 128GB.
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
- **tuned_rag < tuned on v1:** 97% vs 100% facts, 88% vs 100% severity. Retrieved context slightly perturbs the memorised behaviour (2 runbook misses, 3 severity misses out of 24). On v2 it is the only config that is both current and on-policy.
- **Base compliance was already 100%** with thinking disabled — the spec's "fine-tune rescues JSON format" axis did not show on this model; the visible behaviour gain is severity policy and compact voice.
- Loss reached 0.000 by iter 70; 600 iters was ~9× more than needed. Mac slept twice during the run (see `ep04-lora-train.log` checkpoint timestamps).
- Retrieval is TF-IDF over 8 docs; one model, one seed, 24 alerts. Numbers describe this task, not "AIOps in general".
