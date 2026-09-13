"""
bench/ep04_rag.py — Ep 04: deliberately boring retrieval.

Chunk markdown by heading, TF-IDF vectors, cosine top-k. Zero dependencies.
Retrieval quality is not the story of Ep 04 — editability is: swap docs_v1
for docs_v2 and the index rebuilds in milliseconds.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9#_./-]+")


def tokenize(text: str) -> list[str]:
    # keep service names ("checkout-svc"), channels ("#atlas-oncall"), metrics ("db_pool_wait_ms")
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Chunk:
    doc: str
    heading: str
    text: str


def chunk_docs(docs: dict[str, str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for name, md in docs.items():
        heading, buf = "", []
        for line in md.splitlines():
            if line.startswith("#"):
                if buf and "".join(buf).strip():
                    chunks.append(Chunk(name, heading, "\n".join(buf).strip()))
                heading, buf = line.lstrip("# ").strip(), []
            else:
                buf.append(line)
        if buf and "".join(buf).strip():
            chunks.append(Chunk(name, heading, "\n".join(buf).strip()))
    # prefix heading so the service name in "# RB-CHK-001 — checkout-svc" is searchable
    return [Chunk(c.doc, c.heading, f"{c.heading}\n{c.text}") for c in chunks]


class TfidfIndex:
    def __init__(self, chunks: list[Chunk], idf: dict[str, float], vecs: list[dict[str, float]]):
        self.chunks, self.idf, self.vecs = chunks, idf, vecs

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "TfidfIndex":
        docs_tokens = [tokenize(c.text) for c in chunks]
        df = Counter(t for toks in docs_tokens for t in set(toks))
        n = len(chunks)
        idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
        vecs = [cls._vec(toks, idf) for toks in docs_tokens]
        return cls(chunks, idf, vecs)

    @staticmethod
    def _vec(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
        tf = Counter(tokens)
        v = {t: c * idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def retrieve(self, query: str, k: int = 3) -> list[Chunk]:
        q = self._vec(tokenize(query), self.idf)
        scored = []
        for chunk, v in zip(self.chunks, self.vecs):
            s = sum(w * v.get(t, 0.0) for t, w in q.items())
            scored.append((s, chunk))
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:k]]


def load_index(docs_dir: Path) -> TfidfIndex:
    docs = {p.name: p.read_text() for p in sorted(docs_dir.glob("*.md"))}
    return TfidfIndex.build(chunk_docs(docs))


def format_context(chunks: list[Chunk]) -> str:
    parts = [f"[{c.doc} › {c.heading}]\n{c.text}" for c in chunks]
    return "Reference material:\n\n" + "\n\n".join(parts)
