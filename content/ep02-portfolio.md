# Episode 2: How Far Can One Mac Go?

> Portfolio writeup. The styled dashboard page is built separately from `results/dashboard.json`.

## The question

Apple Silicon's pitch for AI is unified memory: CPU and GPU share one pool, so a model just needs
RAM — not a discrete GPU with its own VRAM ceiling. A 128GB M5 Max should, in theory, run models
that need a small rack of datacenter GPUs. This episode tests that theory to destruction: a ladder
of models from 3B to 123B parameters through one benchmark harness, plus a memory probe that
allocates until the hardware refuses.

## The scaling curve

All 4-bit quantized, three prompts each, averaged, on an idle machine.

| Params | Decode | Prefill | TTFT | Peak RAM |
|--------|--------|---------|------|---------|
| 3.2B   | 227 tok/s | 1,625 tok/s | 89ms  | 2.1GB  |
| 7.6B   | 117 tok/s | 718 tok/s   | 115ms | 4.5GB  |
| 14B    | 60 tok/s  | 450 tok/s   | 151ms | 8.5GB  |
| 32B    | 27 tok/s  | 191 tok/s   | 326ms | 18.6GB |
| 70B    | 11 tok/s  | 38 tok/s    | 1.8s  | 39.9GB |
| 123B   | 6.3 tok/s | 11 tok/s    | 4.6s  | 69.2GB |

## Three findings

**1. Decode falls almost inversely with size — and that's the boring part.**
38× the parameters, ~36× slower decode. Memory-bandwidth-bound, exactly as theory predicts.

**2. TTFT is the real cost of going big, not throughput.**
Time-to-first-token explodes from 89ms (3B) to **4.6 seconds** (123B) — a 52× jump. Prefill on a huge
model is the brutal part. Decode at 123B is a usable 6 tok/s; the 4.6s you wait *before* it starts is
what makes it batch-only. The practical cliff for interactive use is ~32B.

**3. Memory is dead linear** — ~0.56GB per billion params at 4-bit. This makes the ceiling computable,
which leads to the headline.

## The breaking point that wasn't

The biggest model — Mistral-Large 123B — used **69GB on a 128GB machine**. Half the tank. Nothing
OOM'd. So I went looking for the actual wall with a probe that allocates GPU memory in 4GB chunks
until Metal refuses.

- **Total unified memory:** 137.4 GB
- **Metal's "recommended working set":** 115.4 GB
- **Max single buffer:** 86.6 GB
- **Probe result:** allocated to **128GB with no failure** — stopped at a self-imposed 125GB safety
  cap, not a hardware limit.

**Apple's advertised 115GB working set is a soft recommendation, not a wall.** MLX handed out memory
well past it. The true ceiling is essentially physical RAM minus OS overhead — far above the number
the API reports. A PC with a 24GB GPU can't load a 70B model at all; this laptop ran 123B with 60GB
to spare.

## The ops war story (build-in-public means showing the mess)

The one model that didn't make it: the **8-bit 72B**. Its download failed three different ways —
`hf-xet` hung on resume after a sleep, then HTTP read-timeouts, then a dropped connection mid-shard.
Lessons banked:
- `hf-xet` is fast on a clean run but doesn't recover from interrupted partials; `HF_HUB_DISABLE_XET=1`
  falls back to reliable parallel HTTPS.
- `HF_HUB_DOWNLOAD_TIMEOUT=120` beats the default short timeout on flaky large shards.
- **Every restart spawns new orphaned partial files** — restarting is not free; let a stable download finish.

It was cut loose. The 4-bit ladder already tells the whole story.

## Reproduce it

Harness, per-model result JSON, the memory probe, and this dashboard's data:
**[github.com/Suryals/mlx-lab](https://github.com/Suryals/mlx-lab)**

```bash
uv run python bench/run_ladder.py      # the full ladder
uv run python bench/probe_ceiling.py   # the memory ceiling probe
uv run python bench/aggregate.py       # build results/dashboard.json
```

## Next: Episode 3

Speed is table stakes. The number that decides whether a local model can drive an autonomous ops
loop is **correctness** — does it pick the right tool, with the right arguments, for a real alert?
That's the next eval.
