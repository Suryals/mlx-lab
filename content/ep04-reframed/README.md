# Episode 04 — revised editorial direction

Open `article.html` for the designed article. `article.md` is the editable copy. `linkedin.md` contains the matching post, three comment drafts, and image alt text. `theme-diagram.svg` is the editable vector master; `theme-diagram.png` is the LinkedIn export. Both article and post use this single theme diagram. The article renders the SVG inline with site typography and colors; the standalone social exports retain their original palette.

Story: the either/or debate → a tempting snapshot winner → changing facts expose the limitation → start with base + RAG → build a golden set → add fine-tuning only for a measured benefit. Production may need both; the experiment does not establish that both always win.

Review corrections:
- The old draft claimed the combined configuration was right on facts and policy after the reorg. Raw results show 4/4 correct destinations but 3/4 severity labels.
- Removed unmeasured confidence claims and language implying real incidents were routed. This was offline generation on synthetic alerts.
- Retained both original-snapshot and reorg results, including the combined configuration’s weaker original-snapshot scores.
- Separated severity accuracy from full policy correctness: correctness of the escalation boolean and probable cause is not captured by the headline accuracy measures.
- Described the golden-set gate as a proposed engineering approach. The experiment did not test it or an optimized prompt baseline.
- Distinguished the synthetic held-out set from a representative, reviewed production evaluation set.

All new work is contained in this folder. Existing content, experiment data, and code were left intact. No training or inference was rerun.

Regenerate the HTML and diagram exports with `python3 build.py` from this folder (Pillow required). The diagram uses code-native vector shapes and text, with a matching raster export; no generated imagery is needed. The article uses the same Google Fonts stylesheet as the live site, with system fallbacks.

Narrative revision: lead with the debate and experiment, delay the recommendation until after the catalog change, and place the single diagram alongside the resulting decision path. LinkedIn opens with the observed failure and uses no hashtags. Comment 3 needs the eventual published article URL. The diagram now identifies the model, hardware, 24-alert grid, and the no-retraining condition.

## suryal.dev design alignment

Matched against https://suryal.dev/articles/qwen3-8-27b-vs-claude-opus-4-6.html and the live `/style.css` and `/articles.css` on 2026-09-14. `site-style.css` and `site-articles.css` are local snapshots of those shared styles. `article.css` contains the Episode 04 table, quote, inline diagram, and accessibility additions. `article-template.html` provides the matching navigation, article header, metadata, tags, footer, mobile menu, and reading progress bar. `build.py` combines that template with the Markdown copy.

The title is presented as a question plus an italic subtitle. Fraunces headings, IBM Plex Sans body text, IBM Plex Mono metadata, warm paper (#FBFAF7), ink (#17160F), and blue (#1F4BA8) match the live site. The article remains a local draft; no site deployment or original-site files were changed. Links to navigation use the live domain. Experiment artifact links retain local repository paths and should be remapped when publishing.

Validation: headless Chromium compared computed desktop styles against the live article; checked widths 390, 768, and 1440 for page overflow; verified mobile menu and Escape; visually reviewed desktop and diagram rendering. `preview-desktop.png`, `preview-mobile.png`, and `preview-diagram.png` show the rendered draft.

Published: https://suryal.dev/articles/rag-or-fine-tuning.html (2026-09-15). The site copy inlines article.css and links the site's own style.css/articles.css; the diagram lives at suryal.dev/images/ep04-theme-diagram.{png,svg}.
