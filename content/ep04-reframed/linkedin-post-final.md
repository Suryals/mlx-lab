# Ep 04 LinkedIn — final (debate-first opener, unicode bold applied)

𝗥𝗔𝗚 𝗼𝗿 𝗳𝗶𝗻𝗲-𝘁𝘂𝗻𝗶𝗻𝗴? 𝗧𝗵𝗲 𝗱𝗲𝗯𝗮𝘁𝗲 𝘂𝘀𝘂𝗮𝗹𝗹𝘆 𝗲𝗻𝗱𝘀 𝘄𝗶𝘁𝗵 𝗮 𝘁𝗲𝗮𝗺 𝗽𝗶𝗰𝗸𝗶𝗻𝗴 𝗼𝗻𝗲.

One approach becomes the architecture, and every failure becomes a reason to invest more in it. With today's models and mature retrieval patterns, fine-tuning is often the side that gets dropped. I wanted to know what that costs, so I built one agent both ways.

The experiment: Qwen3.5-4B on an M5 Max, a fictional organization, 24 synthetic alerts. One incident-triage task, four configurations: base, base + RAG, fine-tuned, and fine-tuned + RAG. Deterministic scoring, no LLM judge.

On the original facts, fine-tuning alone topped every measured column. Base + RAG reached 75% severity and 92% fact accuracy. You could look at that table and choose fine-tuning.

Then checkout moved to a different team. I updated the catalog and reran four alerts. No retraining.

New destination correct:
• Fine-tuned: 𝟬/𝟰
• Base + RAG: 𝟰/𝟰
• Fine-tuned + RAG: 𝟰/𝟰

The adapter kept returning the old channel. The JSON still passed the format check. In an autonomous workflow, that record would have routed an incident to the former team without anyone reviewing it first.

Retrieval followed the change. But "use both" wasn't automatic either: the combined setup still missed one severity label, same as base + RAG.

𝗣𝗿𝗼𝗱𝘂𝗰𝘁𝗶𝗼𝗻 𝗺𝗮𝘆 𝗻𝗲𝗲𝗱 𝗯𝗼𝘁𝗵. 𝗧𝗵𝗲 𝗾𝘂𝗲𝘀𝘁𝗶𝗼𝗻 𝗶𝘀 𝘁𝗵𝗲 𝗼𝗿𝗱𝗲𝗿, 𝗮𝗻𝗱 𝘄𝗵𝗮𝘁 𝗲𝘃𝗶𝗱𝗲𝗻𝗰𝗲 𝗲𝗮𝗿𝗻𝘀 𝘁𝗵𝗲 𝘀𝗲𝗰𝗼𝗻𝗱 𝘀𝘁𝗲𝗽.

The sequence I would follow now: base + RAG → golden set → diagnose the gaps → fine-tune only if justified. Fix sources and retrieval when evidence is missing. Test prompts and rules when behaviour is wrong. Add an adapter only when a held-out evaluation shows a worthwhile gain, with acceptable regressions and serving cost.

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
