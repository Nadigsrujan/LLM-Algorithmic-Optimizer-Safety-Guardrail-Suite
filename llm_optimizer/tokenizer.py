"""
Real LLM tokenization (Improvement 1).

Provides a single source of truth for token counting used by the knapsack
packer, statistics, and every UI metric. Uses OpenAI's `tiktoken` BPE encoder
when available (the same family of byte-pair tokenisation real LLMs use), and
gracefully falls back to the classic ~4-chars/token heuristic when tiktoken is
not installed or fails to initialise — so the project still runs fully offline
with zero hard dependencies.

The knapsack algorithm is weight-agnostic: swapping the heuristic for real BPE
counts changes only the numbers, never the algorithm.
"""
from __future__ import annotations

import math
from functools import lru_cache

# Default encoding. cl100k_base backs gpt-3.5/gpt-4 and is a good general proxy
# for modern BPE tokenisers (including most local models, approximately).
_DEFAULT_ENCODING = "cl100k_base"


@lru_cache(maxsize=4)
def _get_encoder(encoding_name: str = _DEFAULT_ENCODING):
    """Lazily build (and cache) a tiktoken encoder. Returns None on any failure
    so callers transparently fall back to the heuristic."""
    try:
        import tiktoken
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return None


def tokenizer_backend(encoding_name: str = _DEFAULT_ENCODING) -> str:
    """Report which backend is active — handy for the UI to show honesty about
    whether counts are real BPE or the heuristic."""
    return f"tiktoken:{encoding_name}" if _get_encoder(encoding_name) else "heuristic(~4 chars/token)"


def _heuristic_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4)) if text else 0


def count_tokens(text: str, encoding_name: str = _DEFAULT_ENCODING) -> int:
    """Count tokens in `text`.

    Uses tiktoken's real BPE tokeniser when available; otherwise the
    ~4-chars/token heuristic. Always returns >= 0 (0 for empty text)."""
    if not text:
        return 0
    enc = _get_encoder(encoding_name)
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return _heuristic_tokens(text)


if __name__ == "__main__":
    samples = [
        "",
        "Hello world, this is a test.",
        "The Q3 budget review is scheduled for next Tuesday afternoon.",
    ]
    print("backend:", tokenizer_backend())
    for s in samples:
        print(f"  {count_tokens(s):>4}  tokens  <- {s!r}")
