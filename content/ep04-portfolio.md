# Ep 04 — Qwen3.5-4B: LoRA vs RAG on the same alert

**Framework under test:** Paolo Perrone, *RAG vs Fine-Tuning — When to Use Each in Production*
(https://theaiengineer.substack.com/p/rag-vs-fine-tuning). His rule: RAG changes what the model
sees, fine-tuning changes how it behaves, production wants both. This episode checks it on a
fully synthetic on-call desk ("NimbusCart"), locally, on an M5 Max.

## Setup
- Base: mlx-community/Qwen3.5-4B-4bit · LoRA rank 8, 16 layers, 600 iters, lr 1e-5 (mlx-lm 0.31.3)
- Knowledge: 8 markdown runbook/catalog docs (spec estimated ~12) → TF-IDF top-3 retrieval
- Behavior: 360 alert→triage-JSON pairs (v1 facts baked in), 40 validation
- Eval: 24 held-out alerts (6 services × 4 templates, unseen region/values), deterministic scoring
  (JSON compliance · severity · fact accuracy). The 24 eval alerts reuse the same 24 templates as
  training with an unseen region and unseen values (in-distribution) — the tuned model's 100% facts
  on v1 is template recall, not generalisation.
- Sanity gate before scoring: base 10/10, tuned 10/10 JSON-compliant

## The grid

Grid (docs_v1):

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

## The reorg

team-atlas dissolved → checkout-svc owned by team-borealis, escalation `#borealis-oncall`. One
markdown edit, no retraining.

| config | escalate_to answers (after reorg) | correct |
|---|---|---|
| base_rag | #borealis-oncall | 4/4 |
| tuned | #atlas-oncall | 0/4 |
| tuned_rag | #borealis-oncall | 4/4 |

## What moved which axis

The base model always emits valid JSON (100% compliance) but invents facts it was never given —
0/24 on both runbook id and escalation channel. Retrieval is what fixes that: fact accuracy jumps
33% → 92% while severity barely moves (71% → 75%), because the retrieved docs carry facts, not
house judgment calls. The adapter is what fixes severity: 71% → 100% — the on-call severity policy
is learned behaviour, not something lookup can supply. That gap is partly inflated by a confound,
though: eval alert values are template-agnostic, so some read as magnitude-nonsense ("5xx rate
9202%", "[INFO] … pool utilisation 9404%", "signing key expires in 9303h"). Part of what the base
model is missing is it reading the number instead of the `[FIRING]/[WARNING]/[INFO]` tag, where the
adapter learned to trust the tag — the "severity is learned behaviour" claim holds, the 71% → 100%
gap is just larger than the clean version of this experiment would show. After the reorg the split holds exactly
where it should: the tuned-only model keeps paging the dissolved team 4/4 times with perfect JSON
and full confidence, while both retrieval configs (with or without the adapter) follow the one-line
doc edit 4/4 times. Fine-tuning bakes in judgment; retrieval keeps facts current — neither
substitutes for the other on this task.

## Caveats
- **Decode speed:** tuned configs run at ~91 tok/s vs ~168 for base — the LoRA adapter is applied
  unfused at inference. `mlx_lm.fuse` would remove that gap; not measured here.
- **tuned_rag < tuned on v1:** 97% vs 100% facts, 88% vs 100% severity. This isn't retrieved context
  generally perturbing memorised behaviour — every runbook miss in the RAG configs (3 in base_rag, 2
  in tuned_rag) is the model emitting the doc filename `runbook-<svc>.md`, copied verbatim from the
  `[doc › heading]` label that `format_context` prefixes to each chunk. (base_rag's two channel
  misses are separate: `team-atlas`, an owner not a channel, and `platform-lead`.) A context-format
  artefact worth fixing in a follow-up, not fragility in the learned policy. On v2 it is the
  only config that is both current and on-policy.
- **Base compliance was already 100%** with thinking disabled — the spec's "fine-tune rescues JSON
  format" axis did not show on this model; the visible behaviour gain is severity policy and
  compact voice.
- Loss reached 0.000 by iter 70; 600 iters was ~9x more than needed. The Mac slept twice during the
  run — visible as the it/s collapse at iters 340–410 and 480–520, and the stall inside the
  iter-250 validation progress bar, in `ep04-lora-train.log` (no wall-clock timestamps in the log).
- Retrieval is TF-IDF over 8 docs; one model, one seed, 24 alerts. Numbers describe this task, not
  "AIOps in general".
