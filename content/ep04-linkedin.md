# Ep 04 — LinkedIn post (light image: content/ep04-linkedin-card.html → PNG)

Do we still need fine-tuning, when we have powerful models and a well-defined RAG architecture?

I half believed the answer was no. Every AIOps design review this year has been about retrieval. Plain RAG, hybrid, graph, agentic. Fine-tuning comes up and someone says "the frontier model already does that." So I spent a weekend finding out, on an ops task, locally on a Mac.

The setup: a synthetic incident-triage agent for a fictional shop. Alert in, one JSON out: severity, service, runbook, which team to route to. No human reads it first. Built four ways with Qwen3.5-4B: base, base + RAG, LoRA fine-tuned, both. 24 held-out alerts, deterministic scoring.

Day one, the fine-tuned agent swept the board. 100% on everything. Retrieval looked like a tax. That's the cell that gets promoted to prod.

Day two, I dissolved one team and changed one line in the service catalog. Nobody retrained anything, because nobody thinks of a triage agent as something that needs retraining when a team moves.

The fine-tuned agent routed every incident to the dead channel. 4 out of 4. Perfect JSON, full confidence, nobody in the loop to notice. Both configs with retrieval followed the edit. 4 out of 4.

Here's the part I didn't expect. What the adapter kept through the reorg was the severity policy: 100% on both days. The base model, even with retrieval, guessed severity at ~73%. Retrieval never touched that number. Powerful model, good RAG, and it still didn't know how we escalate.

So, yes, we still need fine-tuning. Just not for what most people reach for it for.

If it lives in a system of record (owners, channels, thresholds, runbooks), retrieve it. If it's how the agent is supposed to behave (the escalation rule, the output contract), train it. An agent that acts without a human needs both.

Full write-up, code, results and the caveats (there are real ones: the "held-out" set is in-distribution, and the severity gap is partly a confound) in the comments.

---
Comment 1: Framework under test: Paolo Perrone's "RAG vs Fine-Tuning — When to Use Each in Production" on The AI Engineer. Clean rule, worth reading before you start a training run: https://theaiengineer.substack.com/p/rag-vs-fine-tuning

Comment 2: Repo, results JSON, training log and the five commands to reproduce: https://github.com/Suryals/mlx-lab (Ep 04)

Comment 3: The write-up with the day-one/day-two tables and everything I'd say against myself: [suryal.dev article link]
