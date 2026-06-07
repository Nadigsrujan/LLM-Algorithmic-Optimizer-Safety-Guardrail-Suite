"""
Glue layer shared by the UI: splits raw input into blocks, estimates token
cost (knapsack weight) via the real tokenizer, and computes a relevance score
(knapsack value) using TF-IDF cosine similarity to a focus query plus a recency
bonus.

Improvement 1: token counts come from `tokenizer.count_tokens` (tiktoken).
Improvement 2: relevance uses sklearn's TF-IDF + cosine similarity, with a
               dependency-free keyword-overlap fallback so the tool still runs
               offline if sklearn is unavailable. Heapsort/knapsack are
               unchanged — only the scoring function changed.

Fix (QA-mode): the pipeline now classifies blocks as *context* vs *mandatory
               question*. Questions are never put into the knapsack and their
               token cost is reserved upfront, so the optimizer can never drop
               the context a question depends on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from algorithms.heapsort import Snippet
from tokenizer import count_tokens

# --- optional TF-IDF backend -------------------------------------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False


def scoring_backend() -> str:
    return "tfidf+cosine (sklearn)" if _HAVE_SKLEARN else "keyword-overlap (fallback)"


# ── Block splitting ───────────────────────────────────────────────────────────

def split_blocks(text: str) -> List[str]:
    """Split on blank lines first; fall back to single lines. Blocks are the
    knapsack 'items' (a paragraph or a chat message)."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) <= 1:
        blocks = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return blocks


def estimate_tokens(text: str) -> int:
    """Token weight for the knapsack — delegates to the real tokenizer module
    (tiktoken when available, ~4-chars/token heuristic otherwise)."""
    return count_tokens(text)


# ── Scoring helpers ───────────────────────────────────────────────────────────

_WORD = re.compile(r"[a-z0-9']+")


def _words(s: str):
    return _WORD.findall(s.lower())


def _recency_bonus(idx: int, n_blocks: int) -> float:
    """Later blocks (higher idx) are 'more recent' and get a small boost 0..3."""
    if n_blocks <= 0:
        return 0.0
    return round(3.0 * (idx + 1) / n_blocks, 4)


def _tfidf_similarities(blocks: List[str], focus_query: str) -> List[float]:
    """Cosine similarity between each block and the focus query in TF-IDF space.
    Returns a list of similarities in [0, 1], one per block."""
    if not _HAVE_SKLEARN or not focus_query.strip():
        return [0.0] * len(blocks)
    corpus = blocks + [focus_query]
    try:
        vec = TfidfVectorizer(lowercase=True, stop_words="english")
        matrix = vec.fit_transform(corpus)
        query_vec = matrix[-1]
        block_matrix = matrix[:-1]
        sims = cosine_similarity(block_matrix, query_vec).ravel()
        return [float(s) for s in sims]
    except Exception:
        return [0.0] * len(blocks)


def _keyword_similarity(block: str, focus_terms: set) -> float:
    """Fallback similarity in [0, 1]: fraction of focus terms present in block."""
    if not focus_terms:
        return 0.0
    bw = set(_words(block))
    if not bw:
        return 0.0
    overlap = sum(1 for w in focus_terms if w in bw)
    return overlap / len(focus_terms)


# ── QA-mode classification (Fix) ─────────────────────────────────────────────

# A block is a "question" if it starts with any of these indicators or ends
# with "?". We deliberately use a conservative set to avoid mislabelling a
# context paragraph that happens to contain a question word mid-sentence.
_QUESTION_STARTERS = re.compile(
    r"^\s*(?:"
    r"Q\s*[\d.]*\s*[:\-]|"            # Q:  Q1:  Q.
    r"Question\s*[\d.]*\s*[:\-]|"     # Question:  Question 1:
    r"What\b|Which\b|How\b|Why\b|"
    r"When\b|Where\b|Who\b|"
    r"Identify\b|Explain\b|Describe\b|"
    r"Compare\b|Analy[sz]e\b|Discuss\b|"
    r"Evaluate\b|Calculate\b|Find\b|"
    r"List\b|State\b|Define\b|Outline\b"
    r")",
    re.IGNORECASE,
)

# Blocks that START with a context-reference phrase (e.g. "Based on the passage
# above, identify...") are themselves question/directive blocks — they reference
# the passage AND contain an action directive, so they must be mandatory.
_CONTEXT_REF_STARTER = re.compile(
    r"^\s*(?:"
    r"based on (?:the )?(?:passage|text|article|excerpt|case study|information|context|above|following)|"
    r"using (?:the )?(?:passage|text|information|context|excerpt)|"
    r"refer(?:ring)? to (?:the )?(?:passage|text|following)|"
    r"according to (?:the )?(?:passage|text|article|excerpt)|"
    r"from (?:the )?(?:passage|text|article|excerpt|case study)|"
    r"given (?:the )?(?:passage|text|context|information)|"
    r"in (?:the )?(?:passage|text) above|"
    r"as (?:stated|mentioned|described|discussed) (?:in|above)"
    r")",
    re.IGNORECASE,
)

# Phrases that mean "this question explicitly needs the passage to answer".
_CONTEXT_REF_PHRASES: List[str] = [
    "based on the passage",
    "based on the text",
    "based on the above",
    "based on the following",
    "using the information",
    "using the passage",
    "refer to the following",
    "referring to the passage",
    "from the passage",
    "from the article",
    "from the text",
    "the passage above",
    "the text above",
    "the article above",
    "given the context",
    "given the passage",
    "according to the passage",
    "according to the text",
    "from the case study",
    "in the passage",
    "mentioned in the passage",
    "described above",
    "discussed above",
    "from the excerpt",
    "as stated in the passage",
    "as mentioned above",
]

# Score bonus added to ALL context blocks when the question explicitly depends
# on the passage — this makes them outcompete even a keyword-rich question in
# the knapsack (which the question never enters anyway in QA mode).
CONTEXT_DEPENDENCY_BONUS: float = 50.0


def is_question_block(text: str) -> bool:
    """Return True if `text` looks like a question / prompt directive.

    Three detection paths:
    1. Block ends with "?" (explicit question mark).
    2. Block starts with a recognised question word / label (What, How, Q:, …).
    3. Block starts with a context-reference phrase followed by a directive
       ("Based on the passage above, identify …") — these blocks are themselves
       the question even though they open with a reference phrase.
    """
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    if _QUESTION_STARTERS.match(stripped):
        return True
    # A block that STARTS with a context-reference phrase is the question/directive.
    if _CONTEXT_REF_STARTER.match(stripped):
        return True
    return False


def has_context_reference(text: str) -> bool:
    """Return True if `text` contains phrases that explicitly depend on a
    context passage being present."""
    lower = text.lower()
    return any(phrase in lower for phrase in _CONTEXT_REF_PHRASES)


def classify_blocks(
    blocks: List[str],
) -> Tuple[List[str], List[str], bool]:
    """Split `blocks` into (context_blocks, question_blocks, context_dep).

    context_blocks  — passage / information blocks (go into heapsort + knapsack).
    question_blocks — question / directive blocks (always mandatory, never dropped).
    context_dep     — True when a question phrase explicitly references the passage,
                      which triggers a large score bonus on all context blocks so
                      the knapsack will strongly prefer keeping them.
    """
    ctx: List[str] = []
    qs: List[str] = []
    for b in blocks:
        (qs if is_question_block(b) else ctx).append(b)
    context_dep = has_context_reference("\n".join(qs))
    return ctx, qs, context_dep


# ── Pipeline input dataclass ─────────────────────────────────────────────────

@dataclass
class PipelineInput:
    """Fully classified, scored, and weighted input ready for the optimizer."""
    # Context blocks — go through Heapsort → Knapsack.
    context_snippets: List[Snippet] = field(default_factory=list)
    # Question / directive blocks — always kept, tokens reserved upfront.
    mandatory_snippets: List[Snippet] = field(default_factory=list)
    # True when the question explicitly says "based on the passage / text above".
    context_dependency: bool = False
    # True when at least one question block was detected (activates QA mode).
    qa_mode: bool = False

    @property
    def mandatory_tokens(self) -> int:
        return sum(s.tokens for s in self.mandatory_snippets)

    @property
    def all_context_tokens(self) -> int:
        return sum(s.tokens for s in self.context_snippets)


def build_pipeline_input(text: str, focus_query: str) -> PipelineInput:
    """Build a QA-aware PipelineInput.

    *Normal mode* (no question detected): all blocks compete in the knapsack
    — identical to the original behaviour.

    *QA mode* (at least one question block detected):
      • Question blocks → ``mandatory_snippets`` (tokens reserved from the budget,
        never fed to the knapsack, always included in the final prompt).
      • Context blocks → ``context_snippets`` (heapsorted, then knapsacked against
        the *remaining* budget after mandatory tokens are subtracted).
      • If the question references the passage ("based on the passage above …"),
        every context block gets ``+CONTEXT_DEPENDENCY_BONUS`` added to its score
        so the knapsack strongly prefers keeping them.
    """
    raw_blocks = split_blocks(text)
    if not raw_blocks:
        return PipelineInput()

    ctx_blocks, q_blocks, context_dep = classify_blocks(raw_blocks)
    qa_mode = bool(q_blocks)

    n_all = len(raw_blocks)
    # Map block text → original position for consistent recency scoring.
    global_idx: dict[str, int] = {b: i for i, b in enumerate(raw_blocks)}

    # ── Score context blocks ──────────────────────────────────────────────
    # Use the focus query; fall back to the question text itself when the user
    # hasn't supplied a separate focus query (empty string).
    effective_query = focus_query.strip() or " ".join(q_blocks)

    if _HAVE_SKLEARN and ctx_blocks:
        sims = _tfidf_similarities(ctx_blocks, effective_query)
    else:
        focus_terms = set(_words(effective_query))
        sims = [_keyword_similarity(b, focus_terms) for b in ctx_blocks]

    bonus = CONTEXT_DEPENDENCY_BONUS if context_dep else 0.0

    ctx_snippets: List[Snippet] = []
    for local_i, (b, sim) in enumerate(zip(ctx_blocks, sims)):
        gi = global_idx.get(b, local_i)
        score = round(1.0 + sim * 100.0 + _recency_bonus(gi, n_all) + bonus, 2)
        ctx_snippets.append(Snippet(idx=gi, text=b, tokens=count_tokens(b), score=score))

    # ── Mandatory snippets (questions) ────────────────────────────────────
    # Scored only for display (shown in Panel B); they never enter the knapsack.
    q_snippets: List[Snippet] = []
    for b in q_blocks:
        gi = global_idx.get(b, len(raw_blocks))
        q_snippets.append(Snippet(idx=gi, text=b, tokens=count_tokens(b), score=0.0))

    return PipelineInput(
        context_snippets=ctx_snippets,
        mandatory_snippets=q_snippets,
        context_dependency=context_dep,
        qa_mode=qa_mode,
    )


# ── Legacy helper (kept for backward-compat; app.py uses build_pipeline_input) ─

def build_snippets(text: str, focus_query: str) -> List[Snippet]:
    """Split, score, and weigh all blocks without QA classification.
    Preserved for standalone tests and the __main__ demo below."""
    blocks = split_blocks(text)
    n = len(blocks)
    if n == 0:
        return []
    if _HAVE_SKLEARN:
        sims = _tfidf_similarities(blocks, focus_query)
    else:
        focus_terms = set(_words(focus_query))
        sims = [_keyword_similarity(b, focus_terms) for b in blocks]
    out: List[Snippet] = []
    for i, (b, sim) in enumerate(zip(blocks, sims)):
        score = round(1.0 + sim * 100.0 + _recency_bonus(i, n), 2)
        out.append(Snippet(idx=i, text=b, tokens=count_tokens(b), score=score))
    return out


if __name__ == "__main__":
    QA = (
        "Artificial intelligence is transforming enterprise software. "
        "Key priorities include data governance, model interpretability, "
        "and regulatory compliance.\n\n"
        "AI deployment teams report that security review cycles and MLOps "
        "tooling maturity are critical blockers.\n\n"
        "Based on the passage above, identify the key technical priorities "
        "for enterprise AI deployment."
    )
    print("scoring backend:", scoring_backend())
    pi = build_pipeline_input(QA, "")
    print(f"QA mode: {pi.qa_mode}  context_dep: {pi.context_dependency}")
    print(f"mandatory tokens reserved: {pi.mandatory_tokens}")
    print("context snippets:")
    for s in pi.context_snippets:
        print(f"  blk{s.idx} tok={s.tokens:>3} score={s.score:>7} {s.text[:50]!r}")
    print("mandatory snippets (question):")
    for s in pi.mandatory_snippets:
        print(f"  blk{s.idx} tok={s.tokens:>3} score=MANDATORY  {s.text[:50]!r}")
