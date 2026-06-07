"""Pipeline: block splitting, token weights, TF-IDF relevance scoring,
QA-mode classification, mandatory reservation, and context-dependency bonus."""
import pytest

from pipeline import (
    build_snippets,
    build_pipeline_input,
    classify_blocks,
    is_question_block,
    has_context_reference,
    split_blocks,
    estimate_tokens,
    scoring_backend,
    CONTEXT_DEPENDENCY_BONUS,
)


# ── split_blocks ─────────────────────────────────────────────────────────────

def test_split_on_blank_lines():
    text = "para one\n\npara two\n\npara three"
    assert split_blocks(text) == ["para one", "para two", "para three"]


def test_split_fallback_to_lines():
    text = "line one\nline two\nline three"
    assert split_blocks(text) == ["line one", "line two", "line three"]


def test_split_strips_whitespace():
    text = "  hello  \n\n  world  "
    assert split_blocks(text) == ["hello", "world"]


# ── estimate_tokens / scoring_backend ────────────────────────────────────────

def test_estimate_tokens_positive():
    assert estimate_tokens("a reasonably long sentence about budgets") > 0


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_backend_string():
    b = scoring_backend()
    assert "tfidf" in b or "keyword" in b


# ── is_question_block ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "What are the key priorities?",                         # ends with ?
    "How does this algorithm work?",                        # starts with How
    "Q: Identify the main issues.",                         # Q: label
    "Question 1: Describe the approach.",                   # Question label
    "Identify the key technical priorities for AI.",        # starts with Identify
    "Explain why this matters for deployment.",              # starts with Explain
    "Based on the passage above, identify the key points.", # context-ref starter
    "Using the information provided, analyze the results.", # context-ref starter
    "From the article above, describe the method.",         # context-ref starter
    "According to the passage, what changed?",              # context-ref + ?
])
def test_is_question_block_positive(text):
    assert is_question_block(text) is True, f"Expected question, got context for: {text!r}"


@pytest.mark.parametrize("text", [
    "Artificial intelligence is transforming enterprise software.",
    "Key priorities include data governance and scalability.",
    "The budget review is scheduled for next Tuesday.",
    "AI deployment teams report that security review cycles are critical blockers.",
])
def test_is_question_block_negative(text):
    assert is_question_block(text) is False, f"Expected context, got question for: {text!r}"


# ── has_context_reference ─────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "Based on the passage above, identify the main themes.",
    "Using the information provided, explain the trade-offs.",
    "From the article above, what conclusions can be drawn?",
    "According to the text, what is the primary finding?",
    "Referring to the passage, describe the method.",
    "As mentioned above, the results show clear improvement.",
])
def test_has_context_reference_positive(text):
    assert has_context_reference(text) is True


@pytest.mark.parametrize("text", [
    "What are the main priorities for AI deployment?",
    "Describe the key challenges in machine learning.",
    "How does tokenization affect LLM performance?",
])
def test_has_context_reference_negative(text):
    assert has_context_reference(text) is False


# ── classify_blocks ───────────────────────────────────────────────────────────

def test_classify_splits_qa_correctly():
    blocks = [
        "AI is transforming enterprise software.",           # context
        "Key priorities include data governance.",           # context
        "Based on the passage above, identify priorities.",  # question
    ]
    ctx, qs, dep = classify_blocks(blocks)
    assert len(ctx) == 2
    assert len(qs) == 1
    assert dep is True  # question has context reference


def test_classify_no_questions():
    blocks = ["Block one.", "Block two.", "Block three."]
    ctx, qs, dep = classify_blocks(blocks)
    assert len(ctx) == 3
    assert qs == []
    assert dep is False


def test_classify_multiple_questions():
    blocks = [
        "This is context text.",
        "What is the first point?",
        "How does this relate to the second?",
    ]
    ctx, qs, dep = classify_blocks(blocks)
    assert len(ctx) == 1
    assert len(qs) == 2


def test_classify_context_dep_without_ref_phrase():
    # "Explain why this matters" is a question but has no "passage above" phrase
    blocks = ["Some context.", "Explain why this matters for deployment."]
    ctx, qs, dep = classify_blocks(blocks)
    assert len(qs) == 1
    assert dep is False    # no explicit passage reference


# ── build_pipeline_input — QA mode ───────────────────────────────────────────

QA_TEXT = (
    "Artificial intelligence is transforming enterprise software. "
    "Key priorities include data governance and interpretability.\n\n"
    "AI deployment teams report that security review cycles are blockers.\n\n"
    "Based on the passage above, identify the key technical priorities."
)


def test_qa_mode_activated():
    pi = build_pipeline_input(QA_TEXT, "")
    assert pi.qa_mode is True
    assert pi.context_dependency is True


def test_question_never_in_context_snippets():
    pi = build_pipeline_input(QA_TEXT, "")
    ctx_texts = {s.text for s in pi.context_snippets}
    # The "Based on the passage above" block must NOT be in context snippets.
    assert not any("Based on the passage" in t for t in ctx_texts)


def test_mandatory_snippets_contain_question():
    pi = build_pipeline_input(QA_TEXT, "")
    assert len(pi.mandatory_snippets) >= 1
    assert any("Based on the passage" in s.text for s in pi.mandatory_snippets)


def test_mandatory_tokens_reserved():
    pi = build_pipeline_input(QA_TEXT, "")
    assert pi.mandatory_tokens > 0


def test_context_dependency_bonus_applied():
    pi = build_pipeline_input(QA_TEXT, "")
    # Each context snippet should have score > 1 (base) + bonus (50).
    for s in pi.context_snippets:
        assert s.score > CONTEXT_DEPENDENCY_BONUS, (
            f"Expected score > {CONTEXT_DEPENDENCY_BONUS}, got {s.score} for {s.text[:40]!r}"
        )


def test_no_qa_mode_for_plain_text():
    text = (
        "The Q3 budget review is on Tuesday.\n\n"
        "Final numbers due before the Friday deadline."
    )
    pi = build_pipeline_input(text, "budget deadline")
    assert pi.qa_mode is False
    assert pi.mandatory_snippets == []
    assert len(pi.context_snippets) == 2


# ── Critical regression: context must not be dropped when budget is tight ────

def test_question_cannot_starve_context():
    """The root bug: budget=30, question=21 tok, context=22+19 tok.
    Old code: question wins, context starved.
    Fix: mandatory tokens (21) reserved first; remaining=9, knapsack picks
    whichever context block fits — but the question NEVER competes."""
    pi = build_pipeline_input(QA_TEXT, "")
    from algorithms.heapsort import prioritize
    from algorithms.knapsack import knapsack_pack

    # Tight budget that previously caused full context drop.
    BUDGET = 30
    mandatory_toks = pi.mandatory_tokens
    remaining = max(0, BUDGET - mandatory_toks)

    ordered = prioritize(pi.context_snippets)
    pack = knapsack_pack(
        [s.tokens for s in ordered],
        [s.score  for s in ordered],
        remaining,
    )
    # The question must be in mandatory, not in knapsack output.
    q_texts = {s.text for s in pi.mandatory_snippets}
    selected_texts = {ordered[i].text for i in pack.selected_indices}
    assert not (q_texts & selected_texts), (
        "Question text appeared in knapsack output — it must be mandatory only."
    )
    # Knapsack respects the reduced budget.
    assert pack.total_tokens <= remaining


# ── build_snippets (legacy) ───────────────────────────────────────────────────

def test_snippets_have_tokens_and_scores():
    text = "The budget deadline is Friday.\n\nUnrelated coffee chatter here."
    snips = build_snippets(text, "budget deadline")
    assert len(snips) == 2
    for s in snips:
        assert s.tokens > 0
        assert s.score >= 1.0


def test_relevant_block_scores_higher():
    text = (
        "The quarterly budget deadline is this Friday afternoon.\n\n"
        "The office coffee machine makes a strange noise sometimes."
    )
    snips = build_snippets(text, "budget deadline Friday")
    by_idx = {s.idx: s.score for s in snips}
    assert by_idx[0] > by_idx[1]


def test_empty_input():
    assert build_pipeline_input("", "query").context_snippets == []
    assert build_snippets("", "query") == []
