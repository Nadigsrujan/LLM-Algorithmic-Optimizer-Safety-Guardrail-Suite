"""
Ollama integration (Improvement 6).

Sends the optimizer's packed, budget-safe context to a genuine local LLM and
returns its reply — closing the loop so the demo ends with a real model answer.

Strategy (robust + offline-friendly):
  1. Prefer the official `ollama` Python package (`ollama.chat`).
  2. Fall back to Ollama's local REST API at http://localhost:11434 using only
     the standard library (urllib) — no extra dependency.
Both talk to the same locally running Ollama daemon; nothing leaves the machine.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:1b"


@dataclass
class LLMResponse:
    content: str
    model: str
    ok: bool = True
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    backend: str = ""           # "ollama-pkg" or "ollama-http"


# --- backend detection -------------------------------------------------------
def _have_pkg() -> bool:
    try:
        import ollama  # noqa: F401
        return True
    except Exception:
        return False


def is_available() -> bool:
    """True if a local Ollama daemon answers on its REST port."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def list_models() -> List[str]:
    """Return the names of locally pulled Ollama models (empty if unreachable)."""
    # Try the package first.
    if _have_pkg():
        try:
            import ollama
            data = ollama.list()
            models = data.get("models", []) if isinstance(data, dict) else getattr(data, "models", [])
            names = []
            for m in models:
                name = m.get("model") or m.get("name") if isinstance(m, dict) else getattr(m, "model", None)
                if name:
                    names.append(name)
            if names:
                return names
        except Exception:
            pass
    # HTTP fallback.
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [m["name"] for m in data.get("models", []) if "name" in m]
    except Exception:
        return []


# --- the main entry point ----------------------------------------------------
def _query_http(prompt: str, model_name: str, timeout: float) -> str:
    payload = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["message"]["content"]


def query_llm(prompt: str, model_name: str = DEFAULT_MODEL,
              timeout: float = 120.0) -> LLMResponse:
    """Send `prompt` to a local Ollama `model_name` and return its reply.

    Never raises — a failure (daemon down, model missing) is reported via the
    returned LLMResponse so the UI can degrade gracefully."""
    t0 = time.perf_counter()

    if _have_pkg():
        try:
            import ollama
            resp = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            content = resp["message"]["content"]
            return LLMResponse(content=content, model=model_name, ok=True,
                               elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                               backend="ollama-pkg")
        except Exception as e:
            pkg_err = f"{type(e).__name__}: {e}"
    else:
        pkg_err = "ollama package not installed"

    # HTTP fallback.
    try:
        content = _query_http(prompt, model_name, timeout)
        return LLMResponse(content=content, model=model_name, ok=True,
                           elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                           backend="ollama-http")
    except urllib.error.URLError as e:
        return LLMResponse(content="", model=model_name, ok=False,
                           error=f"Cannot reach Ollama at {OLLAMA_HOST} ({e}). "
                                 f"Is `ollama serve` running? (pkg path: {pkg_err})",
                           elapsed_ms=(time.perf_counter() - t0) * 1000.0)
    except Exception as e:
        return LLMResponse(content="", model=model_name, ok=False,
                           error=f"{type(e).__name__}: {e} (pkg path: {pkg_err})",
                           elapsed_ms=(time.perf_counter() - t0) * 1000.0)


if __name__ == "__main__":
    print("Ollama available:", is_available())
    models = list_models()
    print("Local models:", models)
    if models:
        model = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]
        print(f"Querying {model} ...")
        r = query_llm("Reply with exactly: OK", model_name=model)
        print(f"  ok={r.ok}  backend={r.backend}  {r.elapsed_ms:.0f}ms")
        print("  ->", (r.content or r.error)[:200])
