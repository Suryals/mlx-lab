I fine-tuned an ops agent. It scored 100% on severity and the measured fact fields. Then I changed which team owned checkout.

That is where the RAG vs fine-tuning debate became concrete for me.

The experiment: Qwen3.5-4B on an M5 Max, a fictional organization, 24 synthetic alerts. One triage task, four configurations: base, base + RAG, fine-tuned, and fine-tuned + RAG.

On the original facts, fine-tuning alone topped every measured column. Base + RAG reached 75% severity and 92% fact accuracy.

You could look at that table and choose fine-tuning.

Then checkout moved to a different team. I updated the catalog and reran four alerts. No retraining.

New destination correct:
• Fine-tuned: 0/4
• Base + RAG: 4/4
• Fine-tuned + RAG: 4/4

The adapter kept returning the old channel. The JSON still passed the format check.

This was an offline test. In an autonomous workflow, the same mistake could send an incident to the former team without anyone reviewing the answer first.

Retrieval followed the change. But the combined setup still missed one severity label, just like base + RAG. “Use both” did not automatically solve the task.

The sequence I would follow now:
Base + RAG → golden set → diagnose the gaps → fine-tune if justified.

Fix sources and retrieval when evidence is missing. Test prompts and rules when behavior is wrong. Add an adapter only when a held-out evaluation shows a worthwhile gain, with acceptable regressions and serving costs.

Production may need both. Let the evidence decide when.

A golden set means representative, human-reviewed cases with trusted outcomes, kept separate from training data. RAG needs evaluation too.

No golden set? I’d stay with base + RAG while building one.

The limits matter: reused training templates, some unrealistic alert values, and just four catalog-change cases. This illustrates a failure mode; it does not establish a production winner.

Diagram attached. Experiment, code, and article details in the comments.

---
Comment 1:
The framing was inspired by Paolo Perrone’s article: https://theaiengineer.substack.com/p/rag-vs-fine-tuning

Comment 2:
Code and raw results: https://github.com/Suryals/mlx-lab — Episode 04 includes the four-configuration grid, catalog-change experiment, and deterministic scoring.

Comment 3 — draft, add the published article URL before posting:
Full article: “RAG or fine-tuning? One agent, four configurations, one catalog change.” Includes the results, limitations, and the golden-set decision gate.

Image: theme-diagram.png
Alt text: Start with base plus RAG. Build a golden evaluation set. Without one, keep improving RAG and collecting reviewed cases. With one, diagnose errors and test simpler fixes. Add fine-tuning while retaining RAG only if held-out results justify it. After a catalog edit without retraining in a four-alert synthetic test, tuned-only returned the new destination zero times; both retrieval configurations did so four times. The combined setup still missed one severity label. Qwen3.5-4B on M5 Max; the original grid used 24 alerts.
