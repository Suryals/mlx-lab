"""
bench/ep04_model.py — Ep 04: thin mlx-lm wrapper.

One code path for all four grid configs:
  base        Runner()
  base+RAG    Runner().triage(alert, context=retrieved)
  tuned       Runner(adapter_path="adapters/ep04-qwen3.5-4b")
  tuned+RAG   Runner(adapter_path=...).triage(alert, context=retrieved)

Thinking mode is disabled via the chat template so reasoning traces never
enter the JSON-compliance score. Timing is measured here (TTFT, decode tok/s)
because it is a secondary result in the write-up.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from mlx_lm import load, stream_generate

from bench.ep04_gen_data import SYSTEM_PROMPT

BASE_MODEL = "mlx-community/Qwen3.5-4B-4bit"


@dataclass
class Generation:
    text: str
    ttft_s: float
    decode_tps: float
    output_tokens: int


class Runner:
    def __init__(self, model_id: str = BASE_MODEL, adapter_path: str | None = None):
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.label = "tuned" if adapter_path else "base"
        self.model, self.tokenizer = load(model_id, adapter_path=adapter_path)

    def _prompt(self, alert: str, context: str | None) -> str:
        system = SYSTEM_PROMPT if context is None else f"{SYSTEM_PROMPT}\n\n{context}"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": alert}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError as e:  # template without a thinking switch
            if "enable_thinking" not in str(e):
                raise
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)

    def triage(self, alert: str, context: str | None = None, max_tokens: int = 200) -> Generation:
        prompt = self._prompt(alert, context)
        t0 = time.perf_counter()
        first, last, n, text = None, t0, 0, ""
        for r in stream_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens):
            now = time.perf_counter()
            if first is None:
                first = now
            last, n, text = now, n + 1, text + r.text
        ttft = (first or last) - t0
        decode = (n - 1) / (last - first) if first and n > 1 and last > first else 0.0
        return Generation(text=text.strip(), ttft_s=ttft, decode_tps=decode, output_tokens=n)
