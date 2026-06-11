"""
bench/engines.py — Engine adapters for the benchmark.

Each engine exposes a single function:

    generate(model_id: str, system: str, user: str, max_tokens: int) -> GenerateResult

`GenerateResult` carries everything the measurement loop needs; it does NOT
do the timing — that's intentionally left to run_bench.py so Chief can focus
on the measurement logic there.

Engines implemented:
  - mlx_engine  : mlx-lm direct (Apple Silicon native, measures prefill separately)
  - lmstudio_engine : LM Studio local server (OpenAI-compatible REST at localhost:1234)

NOTE: lmstudio_engine requires LM Studio running with the API server enabled
(Server > Start Server, default port 1234) and the target model loaded.
"""

from __future__ import annotations

import dataclasses
from typing import Iterator


# ──────────────────────────────────────────────
# Shared result type
# ──────────────────────────────────────────────

@dataclasses.dataclass
class GenerateResult:
    """Raw output from a generate call — no timing here."""
    engine: str           # "mlx" | "lmstudio"
    model_id: str
    prompt_id: str
    prompt_tokens: int    # tokens in the prompt (prefill count)
    output_text: str
    output_tokens: int    # tokens generated (decode count)
    # token iterator used by run_bench.py to time TTFT and decode
    # None for non-streaming engines (lmstudio_engine yields None here,
    # returning full output in output_text instead)
    token_stream: Iterator[str] | None = None


# ──────────────────────────────────────────────
# MLX engine
# ──────────────────────────────────────────────

def mlx_engine(
    model_id: str,
    system: str,
    user: str,
    max_tokens: int,
    prompt_id: str = "unknown",
) -> GenerateResult:
    """
    Run inference via mlx-lm directly.

    mlx-lm separates prefill (processing the prompt tokens) from decode
    (generating new tokens) under the hood. We get a streaming generator
    via mlx_lm.stream_generate that yields one token-string at a time —
    this is what run_bench.py will iterate to measure TTFT and decode tok/s.

    model_id: HuggingFace repo id, e.g. "mlx-community/Qwen2.5-7B-Instruct-4bit"
    """
    from mlx_lm import load, stream_generate  # type: ignore

    model, tokenizer = load(model_id)

    # Build the chat-style prompt
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    # Apply the model's chat template if available, else fall back to plain concat
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = f"System: {system}\nUser: {user}\nAssistant:"

    # Count prompt tokens
    prompt_tokens = len(tokenizer.encode(prompt_text))

    # Return a streaming generator — run_bench.py will iterate it
    gen = stream_generate(model, tokenizer, prompt=prompt_text, max_tokens=max_tokens)

    return GenerateResult(
        engine="mlx",
        model_id=model_id,
        prompt_id=prompt_id,
        prompt_tokens=prompt_tokens,
        output_text="",      # run_bench.py fills this as it drains the stream
        output_tokens=0,     # run_bench.py fills this too
        token_stream=gen,
    )


# ──────────────────────────────────────────────
# LM Studio engine (OpenAI-compatible REST API)
# ──────────────────────────────────────────────

def lmstudio_engine(
    model_id: str,
    system: str,
    user: str,
    max_tokens: int,
    prompt_id: str = "unknown",
    base_url: str = "http://localhost:1234/v1",
) -> GenerateResult:
    """
    Run inference via LM Studio's local OpenAI-compatible server.

    LM Studio does NOT expose prefill timing via its API, so we treat the
    whole call as a single block: wall time from first byte to last byte.
    TTFT is measured from request-send to first streamed token.

    model_id: the model name exactly as shown in LM Studio's loaded model panel,
              e.g. "qwen2.5-7b-instruct" (LM Studio uses its own naming).
    """
    import httpx

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }

    # We collect streamed tokens into a generator that run_bench.py iterates.
    # This keeps the TTFT measurement in the same place as the MLX engine.
    def _stream() -> Iterator[str]:
        with httpx.stream(
            "POST",
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    import json
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta

    # We don't know prompt_tokens without calling the tokenizer separately.
    # LM Studio returns usage in the final chunk (non-streaming only).
    # For now we set -1 and note this in results.
    return GenerateResult(
        engine="lmstudio",
        model_id=model_id,
        prompt_id=prompt_id,
        prompt_tokens=-1,   # not available from streaming API
        output_text="",
        output_tokens=0,
        token_stream=_stream(),
    )
