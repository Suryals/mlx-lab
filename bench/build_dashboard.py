"""
bench/build_dashboard.py — Render the benchmark dashboard into the portfolio.

Reads results/dashboard.json (from aggregate.py) and an optional
results/notes.json (breaking-point annotations), then writes a fully static
article page — charts are server-rendered inline SVG, no JS chart library —
into the portfolio repo, matching its editorial design system.

Build-in-public loop:
    run_bench.py  ->  aggregate.py  ->  build_dashboard.py

Usage:
    uv run python bench/build_dashboard.py
"""

import html
import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
PORTFOLIO = Path.home() / "ai-engineer-portfolio"
OUT_PATH = PORTFOLIO / "articles" / "mlx-breaking-point.html"

ACCENT = "#FF4A1C"
INK_SOFT = "#4A463B"
INK_FAINT = "#8E8775"
LINE = "rgba(23, 21, 15, 0.16)"


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def hbar_chart(rows: list[dict], value_key: str, unit: str, color: str) -> str:
    """Render a horizontal bar chart as inline SVG.

    rows: [{label, value, sub}] — value None renders as a hatched 'failed' bar.
    """
    if not rows:
        return '<p class="mlxd-empty">No data yet.</p>'

    row_h = 44
    gap = 14
    label_w = 132
    chart_w = 720
    bar_max_w = chart_w - label_w - 90
    height = len(rows) * (row_h + gap)

    vals = [r["value"] for r in rows if r["value"] is not None]
    vmax = max(vals) if vals else 1

    parts = [
        f'<svg viewBox="0 0 {chart_w} {height}" role="img" '
        f'class="mlxd-svg" preserveAspectRatio="xMinYMin meet">'
    ]
    for i, r in enumerate(rows):
        y = i * (row_h + gap)
        cy = y + row_h / 2
        label = esc(r["label"])
        sub = esc(r.get("sub", ""))
        parts.append(
            f'<text x="0" y="{cy - 2}" class="mlxd-rowlabel">{label}</text>'
        )
        if sub:
            parts.append(
                f'<text x="0" y="{cy + 14}" class="mlxd-rowsub">{sub}</text>'
            )
        if r["value"] is None:
            # failed / OOM bar — hatched stub
            parts.append(
                f'<rect x="{label_w}" y="{y + 6}" width="60" height="{row_h - 12}" '
                f'rx="4" fill="none" stroke="{INK_FAINT}" stroke-dasharray="4 3"/>'
            )
            parts.append(
                f'<text x="{label_w + 72}" y="{cy + 4}" class="mlxd-failval">OOM</text>'
            )
        else:
            w = max(6, (r["value"] / vmax) * bar_max_w)
            parts.append(
                f'<rect x="{label_w}" y="{y + 6}" width="{w:.1f}" height="{row_h - 12}" '
                f'rx="4" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{label_w + w + 10:.1f}" y="{cy + 4}" class="mlxd-val">'
                f'{r["value"]:g}<tspan class="mlxd-unit"> {esc(unit)}</tspan></text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def build_table(models: list[dict]) -> str:
    head = (
        "<tr><th>Model</th><th>Params</th><th>Quant</th>"
        "<th>Decode</th><th>Prefill</th><th>TTFT</th><th>Peak RAM</th></tr>"
    )
    body = []
    for m in models:
        params = f"{m['params_b']:g}B" if m.get("params_b") else "—"
        quant = f"{m['quant_bits']}-bit" if m.get("quant_bits") else "—"
        decode = f"{m['decode_tok_s']:g} tok/s" if m.get("decode_tok_s") else "—"
        prefill = f"{m['prefill_tok_s']:g} tok/s" if m.get("prefill_tok_s") else "—"
        ttft = f"{m['ttft_ms']:g} ms" if m.get("ttft_ms") else "—"
        mem = f"{m['peak_memory_gb']:g} GB" if m.get("peak_memory_gb") else "—"
        body.append(
            f"<tr><td class='mlxd-model'>{esc(m['display_name'])}</td>"
            f"<td>{params}</td><td>{quant}</td><td>{decode}</td>"
            f"<td>{prefill}</td><td>{ttft}</td><td>{mem}</td></tr>"
        )
    return f"<table class='mlxd-table'><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"


def render(data: dict, notes: dict) -> str:
    models = data["models"]
    measured = [m for m in models if m.get("decode_tok_s")]

    # Hero stat: largest successfully-run model.
    biggest = max(measured, key=lambda m: m.get("params_b") or 0) if measured else None

    # Build chart rows (measured + any failed/ceiling entries from notes).
    decode_rows = [
        {"label": f"{m['params_b']:g}B", "sub": m["display_name"].split()[0],
         "value": m["decode_tok_s"]}
        for m in models if m.get("params_b")
    ]
    mem_rows = [
        {"label": f"{m['params_b']:g}B", "sub": m["display_name"].split()[0],
         "value": m["peak_memory_gb"]}
        for m in models if m.get("params_b")
    ]
    # Append breaking-point failures from notes.json
    for f in notes.get("failed", []):
        decode_rows.append({"label": f"{f['params_b']:g}B", "sub": f.get("short", ""), "value": None})
        mem_rows.append({"label": f"{f['params_b']:g}B", "sub": f.get("short", ""), "value": None})

    decode_rows.sort(key=lambda r: float(r["label"].rstrip("B")))
    mem_rows.sort(key=lambda r: float(r["label"].rstrip("B")))

    decode_svg = hbar_chart(decode_rows, "decode_tok_s", "tok/s", ACCENT)
    mem_svg = hbar_chart(mem_rows, "peak_memory_gb", "GB", INK_SOFT)
    table = build_table(models)

    gen_date = datetime.now(timezone.utc).strftime("%B %Y")
    machine = esc(data.get("machine", "Apple M5 Max · 128GB"))

    hero = ""
    if biggest:
        hero = f"""
                <div class="mlxd-hero">
                    <div class="mlxd-hero-stat">
                        <span class="mlxd-hero-num">{biggest['params_b']:g}B</span>
                        <span class="mlxd-hero-cap">largest model run locally</span>
                    </div>
                    <div class="mlxd-hero-stat">
                        <span class="mlxd-hero-num">{biggest['decode_tok_s']:g}<i> tok/s</i></span>
                        <span class="mlxd-hero-cap">decode throughput at {biggest['params_b']:g}B</span>
                    </div>
                    <div class="mlxd-hero-stat">
                        <span class="mlxd-hero-num">{biggest['peak_memory_gb']:g}<i> GB</i></span>
                        <span class="mlxd-hero-cap">peak unified memory used</span>
                    </div>
                </div>"""

    ceiling_note = notes.get("ceiling_html", "")

    return TEMPLATE.format(
        machine=machine,
        gen_date=gen_date,
        model_count=len(measured),
        hero=hero,
        decode_svg=decode_svg,
        mem_svg=mem_svg,
        table=table,
        ceiling_note=ceiling_note,
        accent=ACCENT,
        ink_soft=INK_SOFT,
        ink_faint=INK_FAINT,
        line=LINE,
        data_json=json.dumps(data, indent=0),
    )


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>How Far Can One Mac Go? Local LLM Limits on an M5 Max — Suryaprakash Lakshmanan</title>
    <meta name="description" content="A reproducible benchmark pushing local LLM inference to the limit on an Apple M5 Max with 128GB unified memory — decode throughput, memory, and the breaking point, measured with MLX.">
    <link rel="canonical" href="https://suryal.dev/articles/mlx-breaking-point.html">
    <meta property="og:type" content="article">
    <meta property="og:title" content="How Far Can One Mac Go? Local LLM Limits on an M5 Max">
    <meta property="og:description" content="Pushing local LLM inference to the breaking point on a 128GB M5 Max — measured, charted, reproducible.">
    <meta property="og:url" content="https://suryal.dev/articles/mlx-breaking-point.html">
    <meta property="og:image" content="https://suryal.dev/images/og-image.png">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:image" content="https://suryal.dev/images/og-image.png">
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><rect width=%22100%22 height=%22100%22 fill=%22%23F1EEE6%22/><text x=%2250%22 y=%2272%22 font-size=%2270%22 font-family=%22Georgia,serif%22 text-anchor=%22middle%22 fill=%22%2317150F%22>S</text><circle cx=%2278%22 cy=%2228%22 r=%2210%22 fill=%22%23FF4A1C%22/></svg>">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Hanken+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="../style.css">
    <link rel="stylesheet" href="../articles.css">
    <style>
    /* ---- MLX dashboard (scoped) ---- */
    .mlxd-hero{{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:{line};
        border:1px solid {line};border-radius:14px;overflow:hidden;margin:34px 0 8px}}
    .mlxd-hero-stat{{background:var(--paper);padding:26px 24px;display:flex;flex-direction:column;gap:8px}}
    .mlxd-hero-num{{font-family:var(--serif);font-size:clamp(38px,6vw,56px);line-height:.9;color:var(--ink)}}
    .mlxd-hero-num i{{font-size:.42em;font-style:normal;color:{ink_faint};font-family:var(--mono)}}
    .mlxd-hero-cap{{font-family:var(--mono);font-size:11.5px;letter-spacing:.04em;
        text-transform:uppercase;color:{ink_faint}}}
    .mlxd-chartblock{{margin:30px 0 10px}}
    .mlxd-chart-h{{font-family:var(--mono);font-size:12px;letter-spacing:.06em;text-transform:uppercase;
        color:{accent};margin:0 0 6px}}
    .mlxd-chart-sub{{font-family:var(--mono);font-size:12px;color:{ink_faint};margin:0 0 16px}}
    .mlxd-svg{{width:100%;height:auto;display:block}}
    .mlxd-rowlabel{{font-family:var(--serif);font-size:21px;fill:var(--ink)}}
    .mlxd-rowsub{{font-family:var(--mono);font-size:10.5px;fill:{ink_faint};text-transform:uppercase;letter-spacing:.03em}}
    .mlxd-val{{font-family:var(--mono);font-size:14px;fill:var(--ink);font-weight:500}}
    .mlxd-unit{{fill:{ink_faint};font-size:11px}}
    .mlxd-failval{{font-family:var(--mono);font-size:12px;fill:{ink_faint};letter-spacing:.05em}}
    .mlxd-table{{width:100%;border-collapse:collapse;margin:8px 0 6px;font-family:var(--mono);font-size:13px}}
    .mlxd-table th{{text-align:right;padding:10px 12px;border-bottom:1.5px solid {line};
        color:{ink_faint};font-weight:500;text-transform:uppercase;font-size:10.5px;letter-spacing:.05em}}
    .mlxd-table th:first-child,.mlxd-table td:first-child{{text-align:left}}
    .mlxd-table td{{text-align:right;padding:11px 12px;border-bottom:1px solid {line};color:var(--ink-soft)}}
    .mlxd-model{{color:var(--ink)!important;font-weight:500}}
    .mlxd-table tr:hover td{{background:var(--paper-2)}}
    .mlxd-caption{{font-family:var(--mono);font-size:11.5px;color:{ink_faint};margin-top:10px}}
    @media(max-width:680px){{.mlxd-hero{{grid-template-columns:1fr}}
        .mlxd-table{{font-size:11px}}.mlxd-table th,.mlxd-table td{{padding:8px 7px}}}}
    </style>
    <script type="application/ld+json">
    {{
      "@context": "https://schema.org",
      "@type": "TechArticle",
      "headline": "How Far Can One Mac Go? Local LLM Limits on an M5 Max",
      "description": "A reproducible benchmark pushing local LLM inference to the limit on a 128GB Apple M5 Max, measured with MLX.",
      "datePublished": "2026-06-12",
      "author": {{ "@type": "Person", "name": "Suryaprakash Lakshmanan", "url": "https://suryal.dev/" }},
      "publisher": {{ "@type": "Person", "name": "Suryaprakash Lakshmanan" }},
      "mainEntityOfPage": "https://suryal.dev/articles/mlx-breaking-point.html",
      "keywords": "MLX, Apple Silicon, Local LLM, Benchmark, M5 Max, Unified Memory, AI Eval",
      "image": "https://suryal.dev/images/og-image.png"
    }}
    </script>
</head>
<body>
    <div class="progress" id="progress" aria-hidden="true"></div>
    <div class="cursor" id="cursor" aria-hidden="true"></div>

    <header class="topbar" id="topbar">
        <div class="bar">
            <a href="../index.html#top" class="mark">
                <span class="mark-name">Surya</span>
                <span class="mark-dot"></span>
            </a>
            <nav class="nav" aria-label="Primary">
                <a href="../index.html#work"><i>01</i> Work</a>
                <a href="../index.html#capabilities"><i>02</i> Capabilities</a>
                <a href="../index.html#experience"><i>03</i> Experience</a>
                <a href="../index.html#writing"><i>04</i> Writing</a>
                <a href="../index.html#contact"><i>05</i> Contact</a>
            </nav>
            <a href="index.html" class="topbar-cta">All articles ↗</a>
            <button class="menu-btn" id="menuBtn" aria-label="Toggle menu" aria-expanded="false">
                <span></span><span></span>
            </button>
        </div>
    </header>
    <div class="mobile-menu" id="mobileMenu" aria-hidden="true">
        <a href="../index.html#work"><i>01</i> Work</a>
        <a href="../index.html#capabilities"><i>02</i> Capabilities</a>
        <a href="../index.html#experience"><i>03</i> Experience</a>
        <a href="../index.html#writing"><i>04</i> Writing</a>
        <a href="../index.html#contact"><i>05</i> Contact</a>
    </div>

    <main class="article">
        <div class="wrap">
            <header class="article-header">
                <p class="breadcrumb"><a href="index.html">Articles</a> <span>/</span> SRE / Platform</p>
                <span class="article-cat" style="margin-top:14px">Local AI · Benchmarking</span>
                <h1 class="article-title">How Far Can One Mac Go?</h1>
                <p class="article-sub">Pushing local LLM inference to the breaking point on an M5 Max with 128GB of unified memory — measured, charted, and reproducible with MLX.</p>
                <div class="article-meta">
                    <span class="who">Suryaprakash Lakshmanan</span>
                    <span class="sep">·</span><span>{gen_date}</span>
                    <span class="sep">·</span><span>Live dashboard · {model_count} models</span>
                </div>
                <div class="article-tags">
                    <span>MLX</span><span>Apple Silicon</span><span>Local LLM</span><span>Benchmark</span><span>M5 Max</span><span>AI Eval</span>
                </div>
                {hero}
                <p class="mlxd-caption">{machine} · engine: mlx-lm · generated {gen_date}. Every number reproducible — <a href="https://github.com/Suryals/mlx-lab" target="_blank" rel="noopener">github.com/Suryals/mlx-lab</a>.</p>
            </header>

            <article class="article-body">
                <p>The pitch for Apple Silicon and AI is <em>unified memory</em>: the CPU and GPU share one pool, so a model just needs RAM — not a discrete GPU with its own VRAM ceiling. A 128GB M5 Max should, in theory, run models that need a small rack of datacenter GPUs. This is the test of that theory. I ran a ladder of models, all 4-bit quantized, through the same benchmark harness on an idle machine, and measured where it stops being comfortable — and where it stops entirely.</p>

                <h2>Decode throughput vs model size</h2>
                <p>Decode — generating tokens one at a time — is memory-bandwidth-bound. More weights to stream per token means proportionally slower generation. This is the number that decides whether a model is usable interactively.</p>
                <div class="mlxd-chartblock">
                    <p class="mlxd-chart-h">Decode throughput</p>
                    <p class="mlxd-chart-sub">tokens / second · higher is better · 4-bit quantized</p>
                    {decode_svg}
                </div>

                <h2>Memory is the real ceiling</h2>
                <p>Throughput degrades gracefully. Memory does not — when a model's weights plus its KV cache exceed what the system can hand to Metal, it doesn't slow down, it fails. That wall is the whole point of the 128GB machine.</p>
                <div class="mlxd-chartblock">
                    <p class="mlxd-chart-h">Peak unified memory</p>
                    <p class="mlxd-chart-sub">GB resident during generation · the 128GB ceiling is the limit</p>
                    {mem_svg}
                </div>

                {ceiling_note}

                <h2>The full numbers</h2>
                <p>Three prompts per model (short factual, medium reasoning, long generation), averaged, on an idle machine. A warm-up pass is discarded to absorb Metal shader compilation.</p>
                {table}
                <p class="mlxd-caption">Decode = generation rate · Prefill = prompt-processing rate · TTFT = time to first token · Peak RAM from MLX's own high-water mark.</p>

                <h2>How it was measured</h2>
                <p>The harness loads each model through <code>mlx-lm</code>, streams the generation, and brackets the timing with a stopwatch placed at the first token (for TTFT) and across the decode loop (for throughput). Peak memory comes straight from MLX's allocator, not an external sampler, so it reflects exactly what the runtime held. Every model sees an identical prompt set — a fair comparison requires identical inputs. The full code, the per-model result JSON, and this dashboard's data are all in the <a href="https://github.com/Suryals/mlx-lab" target="_blank" rel="noopener">mlx-lab repo</a>.</p>
                <p>This is Episode 2 of a build-in-public series on running and evaluating local models on Apple Silicon. Next: can a local model pick the right tool for an operations alert — the number that actually matters for autonomous ops.</p>
            </article>

            <section class="discussion">
                <span class="eyebrow">Discussion</span>
                <h3>Join the conversation on LinkedIn</h3>
                <p>The code lives on GitHub; the discussion — what you'd run on 128GB, where you'd push it next — happens on LinkedIn.</p>
                <a class="li-button" href="https://www.linkedin.com/in/suryaprakash-lakshmanan-068a7684/recent-activity/all/" target="_blank" rel="noopener">
                    <span class="li-badge">in</span> Join the discussion on LinkedIn
                </a>
            </section>

            <section class="related">
                <span class="eyebrow">Keep reading</span>
                <div class="related-grid">
                    <a class="rel-card" href="https://github.com/Suryals/mlx-lab" target="_blank" rel="noopener">
                        <span class="rc-cat">The code</span>
                        <h4>mlx-lab on GitHub</h4>
                        <p>The benchmark harness, per-model results, and this dashboard's data — all reproducible.</p>
                        <span class="rc-go">View the repo →</span>
                    </a>
                    <a class="rel-card" href="index.html">
                        <span class="rc-cat">The library</span>
                        <h4>More long-form</h4>
                        <p>Enterprise AI architecture, MCP & agentic operations, platform engineering.</p>
                        <span class="rc-go">Browse all articles →</span>
                    </a>
                </div>
            </section>
        </div>
    </main>

    <footer class="footer">
        <div class="wrap footer-inner">
            <div class="footer-col">
                <span class="footer-mark">Suryaprakash Lakshmanan<b>.</b></span>
                <p>Staff AI Platform &amp; Cloud Engineer — Chennai, India</p>
            </div>
            <div class="footer-col footer-colophon">
                <span>© <span id="year">2026</span> — All rights reserved</span>
                <span><a href="index.html" style="color:inherit">Articles</a> · <a href="../index.html" style="color:inherit">Home</a></span>
            </div>
        </div>
    </footer>

    <!-- Benchmark data (machine-readable, mirrors github.com/Suryals/mlx-lab/results/dashboard.json) -->
    <script type="application/json" id="bench-data">
{data_json}
    </script>
    <script src="../script.js"></script>
    <script src="../articles.js"></script>
</body>
</html>
"""


def main():
    # The portfolio page is hand-styled by the author; this generator only
    # writes it when explicitly asked, so automated runs (the ladder driver)
    # never clobber the author's restyle. Set MLX_WRITE_PORTFOLIO=1 to write.
    import os
    if os.environ.get("MLX_WRITE_PORTFOLIO") != "1":
        print("build_dashboard: skipping portfolio write "
              "(set MLX_WRITE_PORTFOLIO=1 to generate the page).")
        return

    data = json.loads((RESULTS_DIR / "dashboard.json").read_text())
    notes_path = RESULTS_DIR / "notes.json"
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}

    if not PORTFOLIO.exists():
        raise SystemExit(f"Portfolio not found at {PORTFOLIO}")

    html_out = render(data, notes)
    OUT_PATH.write_text(html_out)
    print(f"Dashboard written → {OUT_PATH}")
    print(f"  models charted: {data['model_count']}  (measured: {len([m for m in data['models'] if m.get('decode_tok_s')])})")


if __name__ == "__main__":
    main()
