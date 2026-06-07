"""Ollama integration: connectivity + response retrieval.

These tests are SKIPPED automatically when no local Ollama daemon is running,
so the suite stays green offline. Run `ollama serve` (and pull a small model
like `llama3.2:1b`) to exercise them live.
"""
import pytest

import llm as ollama_llm

_ONLINE = ollama_llm.is_available()
online_only = pytest.mark.skipif(not _ONLINE, reason="Ollama daemon not running")


def test_is_available_returns_bool():
    assert isinstance(ollama_llm.is_available(), bool)


def test_list_models_returns_list():
    assert isinstance(ollama_llm.list_models(), list)


def test_query_llm_never_raises_when_offline():
    # Even pointed at a bogus model, query_llm reports via the dataclass.
    resp = ollama_llm.query_llm("hi", model_name="definitely-not-a-real-model-xyz",
                                timeout=5)
    assert hasattr(resp, "ok")
    assert isinstance(resp.ok, bool)


@online_only
def test_connectivity():
    assert ollama_llm.is_available() is True
    assert len(ollama_llm.list_models()) >= 1


@online_only
def test_response_retrieval():
    models = ollama_llm.list_models()
    model = ollama_llm.DEFAULT_MODEL if ollama_llm.DEFAULT_MODEL in models else models[0]
    resp = ollama_llm.query_llm("Reply with exactly: OK", model_name=model)
    assert resp.ok is True
    assert isinstance(resp.content, str)
    assert resp.content.strip() != ""
    assert resp.elapsed_ms > 0
