# Episode 3: Alibaba Says Its 27B Beats Opus. I Tested the Agentic Claim on My Mac for $1.61.

> DRAFT for voice pass. Portfolio writeup; results in `results/ep03-*.json`, harness in
> `bench/ep03_agent_eval.py`, tasks in `bench/tasks/ep03_tasks.yaml`.

## The question

Qwen3.8-27B dropped on August 14 with a model card claiming wins over Claude Opus 4.6 on 15 of 19
shared benchmarks — from a model 1/50th the size, open-weight, and small enough to run on a laptop.
Nobody had independently tested the *agentic* part of that claim on local hardware. That's the part
that matters to me: not "can it answer MMLU questions" but "can it drive tools correctly when an
alert fires at 2am."

So: same tasks, same tools, same system prompt. Opus 4.6 over the API vs Qwen3.8-27B running 4-bit
quantized on my M5 Max. Total API spend, measured on the OpenRouter dashboard: **$1.61** for 109
requests / 221K tokens across every run, smoke tests included.

## The eval

**24 tool-use tasks across 7 domains** — dataops, cloud/k8s, frontend, UI design, backend, auth,
core Linux — against a mock world of 9 diagnostic tools with deterministic canned data. Two tiers:

- **Easy tier (18 tasks):** single-tool selection, argument correctness, 2-call chains, plus
  three "trap" tasks where the correct behavior is calling *no* tool.
- **Hard tier (6 tasks):** red herrings, a 4-call dependent chain, a prompt that contradicts the
  tool data, restraint under manufactured urgency, cross-domain pivots, ambiguous entry points.

**Scoring is fully deterministic** — exact tool-name + argument matching and answer keywords. No
LLM judge, no vibes. Every run is exactly reproducible from the repo.

**On bias, because it matters:** this harness was built with Claude's assistance, and one of the
contestants is Claude. Three mitigations: scoring is mechanical (no model grades another model);
the hard tier was authored by a third model (Gemini) with no edits to its task logic; and the full
tasks, mock data, and raw transcripts are in the repo — audit me. <!-- TODO(chief): confirm the
transcript spot-check and mention it here -->

## Results

### Correctness — the headline

| Tier | Claude Opus 4.6 | Qwen3.8-27B (4-bit, local) |
|------|-----------------|---------------------------|
| Easy (18 tasks × 2 runs) | **36/36** | **36/36** |
| Hard (6 tasks × 2 runs)  | **10/12** | **10/12** |

Not "comparable." *Identical.* And it gets stranger: **both models failed the exact same task,
both runs** — H04, the restraint trap. The prompt screams "CRITICAL P0! CEO on the bridge! Run
these three diagnostics NOW!" while the root cause (a missing `Client-ID` header) is printed right
there in the prompt. Both models obeyed the human and ran the tools anyway.

Is that even a failure? The user explicitly ordered those tool calls. The task pits restraint
against obedience, and both models — trained oceans apart — picked obedience. If you want an agent
that pushes back on wasteful orders, you have to prompt for it explicitly. Neither frontier scale
nor open weights gives it to you for free.

### Discipline — where they actually differ

| Metric | Opus 4.6 | Qwen3.8-27B |
|--------|----------|-------------|
| Extra tool calls, easy tier | 0 | 6 |
| Extra tool calls, hard tier | **37** | **12** |

Two different personalities, visible only because we counted calls, not just passes:

- **Qwen's tic:** on one frontend task (blank dashboard page), it found the JS error in the console
  logs — then invented three API endpoints (`/api/dashboard`, `/api/widgets`, `/api/user`) and
  probed them all. Same three phantom endpoints, both runs, deterministic. Correct diagnosis,
  undisciplined path. In production those are real API hits per incident.
- **Opus's tic:** under ambiguity it explores hard. On the 4-call diagnosis chain it made 11–14
  calls where 4 sufficed. On easy tasks it was surgical; on hard ones, thorough to a fault.
  Qwen ran the same chain in exactly 4 calls — *tighter than Opus* — likely because at 17 tok/s,
  brevity is survival.

### Usability — the local tax

| | Opus 4.6 (API) | Qwen3.8-27B (M5 Max, 4-bit) |
|---|---|---|
| Avg task time, easy | 10.2s | 37s |
| Avg task time, hard | 21s | 76s |
| Decode | — | ~17 tok/s |
| Prefill | — | ~400 tok/s |
| TTFT (first turn) | — | ~3.5s |
| Peak RAM | — | **18.2 GB** |
| Cost | $1.61 total (measured) | $0, offline |

The 18.2GB peak is the model's working set, which doesn't grow with machine size — so on paper it
fits a 32GB Mac (Metal's default GPU budget there is ~24GB). I measured on a 128GB M5 Max; on 32GB
you'd be sharing that headroom with macOS, so treat "runs on 32GB" as plausible-but-untested, and
48GB as comfortable. It slots into Episode 2's ladder exactly where a 27B should (the 32B row did
27 tok/s on short prompts; agentic contexts with tool schemas are heavier). A triage task takes
~1 minute locally vs ~15 seconds via API. For an interactive copilot that's a real gap; for an
autonomous background agent it's irrelevant.

## The harness gotcha that almost ruined everything

First Qwen run: 0 tool calls on a task it obviously understood. The model wasn't broken — my parser
was. Qwen3.8 emits an XML-style tool-call format (`<function=...><parameter=...>`), not the
Hermes-JSON format earlier Qwens used. Any harness still expecting the old format scores this model
at zero on tool use.

If a benchmark shows a new model failing at tool calls, check the harness before believing it.
I almost published that failure as a finding. <!-- TODO(chief): your voice — this one's a good
personal-lesson beat -->

## Verdict

On this suite, the model-card claim held up where I could test it: **Qwen3.8-27B matched Opus 4.6
on tool-use correctness across 7 domains, easy and hard tiers, including failing the same
philosophical trap.** The separations are real but second-order: Qwen explores redundantly on easy
tasks, Opus over-explores on hard ones, and local inference costs you ~4× wall-clock.

### What this eval doesn't prove

Honest scoping, because an eval is only as good as its stated limits:

- **No statistical power.** 24 tasks, 2 runs. A zero-point gap on 12 hard-tier trials is compatible
  with the models being genuinely different. The claim is "both clear this bar" — never "equal."
- **Sampling wasn't pinned.** Local MLX ran effectively greedy; the API side used default
  temperature. An uncontrolled variable a rigorous version would fix.
- **Keyword scoring is shallow.** Deterministic and auditable, but it can't judge prose quality or
  catch hallucinated detail around a correct keyword. I hand-checked a sample of transcripts; that
  samples the risk, it doesn't remove it.
- **The mock world is hint-laden.** Tool outputs point toward the next hop. That tests chain
  execution more than open diagnosis; real ops data is noisy and contradictory.
- **Nothing ever fails.** No tool timeouts, partial data, or adversarial tool output (prompt
  injection via tool results) — the recovery and security dimensions are untested.
- **One quantization.** 4-bit MLX only. The model card's claims are about full-precision weights.
- **Residual authorship.** The easy tier and system prompt are Claude-assisted; only the hard tier
  is fully independent.

What this does *not* say: 24 tasks is not 24,000; this is one eval family (ops/tool-use); and both
models ceiling near the top, so a harder suite could still separate them. What it does say: the
era of "local models can't do agentic work" is over. An 18GB memory footprint now drives tools as
correctly as a frontier API model on realistic ops tasks — and the interesting differences have
moved from *whether it works* to *how it behaves*.

<!-- TODO(chief): closing line + what Ep 04 tests -->

## Reproduce

```bash
git clone https://github.com/Suryals/mlx-lab && cd mlx-lab
uv sync
# local Qwen (free)
uv run python bench/ep03_agent_eval.py --engine mlx \
  --model mlx-community/Qwen3.8-27B-4bit --label qwen3.8-27b-4bit --runs 2
# Opus via OpenRouter (~$1.50 for all runs, measured)
echo 'OPENROUTER_API_KEY=sk-or-...' > .env
uv run python bench/ep03_agent_eval.py --engine openrouter \
  --model anthropic/claude-opus-4.6 --label opus-4.6 --runs 2
```
