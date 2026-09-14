# Ep 04 LinkedIn — final (unicode bold applied)

𝗜 𝗳𝗶𝗻𝗲-𝘁𝘂𝗻𝗲𝗱 𝗮𝗻 𝗼𝗽𝘀 𝗮𝗴𝗲𝗻𝘁. 𝗜𝘁 𝘀𝗰𝗼𝗿𝗲𝗱 𝟭𝟬𝟬%. 𝗧𝗵𝗲𝗻 𝗜 𝗰𝗵𝗮𝗻𝗴𝗲𝗱 𝘄𝗵𝗶𝗰𝗵 𝘁𝗲𝗮𝗺 𝗼𝘄𝗻𝗲𝗱 𝗰𝗵𝗲𝗰𝗸𝗼𝘂𝘁.

That is where the RAG vs fine-tuning debate became concrete for me. The debate usually ends with a team picking a side. One approach becomes the architecture, and every failure becomes a reason to invest more in it.

The experiment: Qwen3.5-4B on an M5 Max, a fictional organization, 24 synthetic alerts. One triage task, four configurations: base, base + RAG, fine-tuned, and fine-tuned + RAG. Deterministic scoring, no LLM judge.

On the original facts, fine-tuning alone topped every measured column. Base + RAG reached 75% severity and 92% fact accuracy. You could look at that table and choose fine-tuning.

Then checkout moved to a different team. I updated the catalog and reran four alerts. No retraining.

New destination correct:
• Fine-tuned: 𝟬/𝟰
• Base + RAG: 𝟰/𝟰
• Fine-tuned + RAG: 𝟰/𝟰

The adapter kept returning the old channel. The JSON still passed the format check. In an autonomous workflow, that record would have routed an incident to the former team without anyone reviewing it first.

Retrieval followed the change. But "use both" wasn't automatic either: the combined setup still missed one severity label, same as base + RAG.

𝗧𝗵𝗲 𝘀𝗲𝗾𝘂𝗲𝗻𝗰𝗲 𝗜 𝘄𝗼𝘂𝗹𝗱 𝗳𝗼𝗹𝗹𝗼𝘄 𝗻𝗼𝘄: 𝗯𝗮𝘀𝗲 + 𝗥𝗔𝗚 → 𝗴𝗼𝗹𝗱𝗲𝗻 𝘀𝗲𝘁 → 𝗱𝗶𝗮𝗴𝗻𝗼𝘀𝗲 𝘁𝗵𝗲 𝗴𝗮𝗽𝘀 → 𝗳𝗶𝗻𝗲-𝘁𝘂𝗻𝗲 𝗼𝗻𝗹𝘆 𝗶𝗳 𝗷𝘂𝘀𝘁𝗶𝗳𝗶𝗲𝗱.

Fix sources and retrieval when evidence is missing. Test prompts and rules when behaviour is wrong. Add an adapter only when a held-out evaluation shows a worthwhile gain, with acceptable regressions and serving cost.

Production may need both. Let the evidence decide when.

A golden set means representative, human-reviewed cases with trusted outcomes, kept separate from training data. RAG needs evaluation too.

𝗡𝗼 𝗴𝗼𝗹𝗱𝗲𝗻 𝘀𝗲𝘁? 𝗦𝘁𝗮𝘆 𝘄𝗶𝘁𝗵 𝗯𝗮𝘀𝗲 + 𝗥𝗔𝗚 𝘄𝗵𝗶𝗹𝗲 𝘆𝗼𝘂 𝗯𝘂𝗶𝗹𝗱 𝗼𝗻𝗲.

The limits matter: reused training templates, some unrealistic alert values, four catalog-change cases. This illustrates a failure mode; it does not crown a production winner.

Diagram attached. Article, code and raw results in the comments.

---
Comment 1 — Full article, with the results, limits and the golden-set gate:
https://suryal.dev/articles/rag-or-fine-tuning.html

Comment 2 — The framing was inspired by Paolo Perrone's "RAG vs Fine-Tuning — When to Use Each in Production":
https://theaiengineer.substack.com/p/rag-vs-fine-tuning

Comment 3 — Code, training log and raw results (Episode 04, five commands to reproduce):
https://github.com/Suryals/mlx-lab

Image: theme-diagram.png
