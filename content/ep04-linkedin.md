# Ep 04 — LinkedIn post (light image: content/ep04-linkedin-card.html → PNG)

We fine-tuned an ops agent. It scored 100%. Then a team moved.

Every AIOps design review I sit in is about retrieval. Fine-tuning barely comes up. So I spent a weekend testing the one rule everyone quotes but nobody checks: RAG for knowledge, fine-tuning for behaviour.

The setup, all local on a Mac:
- A synthetic incident-triage agent for a fictional shop. Alert in, one JSON out: severity, service, runbook, which team to route to.
- Built four ways: base model, base + RAG, LoRA fine-tuned, fine-tuned + RAG. Qwen3.5-4B, 24 held-out alerts, deterministic scoring.

Day one, the fine-tuned agent swept the board. 100% on everything. That's the cell that gets promoted to prod.

Day two, we dissolved one team and changed one line in the service catalog. Nobody retrained anything, because nobody thinks of a triage agent as something that needs retraining when a team moves.

The fine-tuned agent routed every incident to the dead channel. 4 out of 4. Perfect JSON, full confidence, and no human in the loop to notice.

Both configs with retrieval followed the edit. 4 out of 4.

What the adapter did keep through the reorg: the severity policy. 100% on both days. The base model, even with retrieval, guessed severity at ~73%. Retrieval never touched that number.

So the boundary isn't about effort. If it lives in a system of record (owners, channels, thresholds, runbooks), retrieve it. If it's how the agent is supposed to behave (the escalation rule, the output contract), train it. An agent that acts without a human needs both.

Full write-up, code, results and the caveats (there are real ones: the "held-out" set is in-distribution, and the severity gap is partly a confound) in the comments.

---
Comment 1: Framework under test: Paolo Perrone's "RAG vs Fine-Tuning — When to Use Each in Production" on The AI Engineer. Clean rule, worth reading before you start a training run: https://theaiengineer.substack.com/p/rag-vs-fine-tuning

Comment 2: Repo, results JSON, training log and the five commands to reproduce: https://github.com/Suryals/mlx-lab (Ep 04)

Comment 3: The write-up with the day-one/day-two tables and everything we'd say against ourselves: [suryal.dev article link]
