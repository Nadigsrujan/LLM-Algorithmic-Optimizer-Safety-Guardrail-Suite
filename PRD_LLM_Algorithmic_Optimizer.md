# Product Requirements Document (PRD)
## Local LLM Algorithmic Optimizer & Safety Guardrail Suite

| Field | Value |
|-------|-------|
| **Document type** | Product Requirements Document |
| **Project type** | Academic — Design & Analysis of Algorithms (DAA) course project |
| **Version** | 1.0 |
| **Status** | Approved for implementation |
| **Platform** | Local desktop application (offline, no cloud) |
| **Primary language** | Python 3.10+ |
| **UI framework** | Streamlit (web-local) or Tkinter (desktop) |

---

## 1. Overview

The Local LLM Algorithmic Optimizer is an offline pre-processing layer that sits
**in front of** a local Large Language Model (LLM) chat interface. Before any
text reaches the model, the tool guards it against malicious input, ranks its
content by relevance, packs the most valuable content into the model's fixed
token budget, and compresses whatever overflows for cheap local storage.

The project's purpose is twofold:
1. **Practical:** solve the universal local-LLM problem of the fixed context
   window without losing the most important information.
2. **Academic:** demonstrate four classic algorithm-design paradigms (per
   Levitin's *Introduction to the Design and Analysis of Algorithms*) applied to
   a single coherent, visually demonstrable system.

The entire application runs locally. It requires no cloud APIs, no network
calls, and no servers beyond a one-time package install.

---

## 2. Problem statement

Every LLM has a hard **context window limit** — a maximum number of tokens it
can process at once. When a user pastes a long conversation history or several
reference documents, three failure modes occur:

1. **Overflow** — the input exceeds the limit and the request fails or silently
   truncates, losing information.
2. **Injection** — malicious users embed adversarial phrases (e.g. *"ignore all
   previous instructions"*) to hijack the model.
3. **Storage bloat** — chat logs that drop out of the active window still
   consume disk space.

A naïve solution (truncate the oldest text, scan inputs with nested loops,
store logs raw) is slow, insecure, and lossy. This project replaces each naïve
step with an asymptotically efficient, well-understood algorithm.

---

## 3. Goals & non-goals

### 3.1 Goals
- Guarantee no input exceeding the token budget is ever sent to the model.
- Maximise the *relevance* of the content that is sent, not just its quantity.
- Detect and block known prompt-injection phrases before execution.
- Compress dropped/cold conversation history losslessly to save disk space.
- Make every algorithm **visible** in the UI for evaluation/demonstration.
- Run fully offline with minimal dependencies.

### 3.2 Non-goals
- Building or training an LLM. The tool is a pre-processor; the model itself is
  out of scope (though a local model may be optionally attached).
- Production-grade tokenisation. A character-based token estimate is acceptable;
  exact BPE tokenisation is an optional enhancement.
- Semantic relevance via embeddings. A keyword + recency heuristic is sufficient
  for the core deliverable.
- Networked/multi-user operation.

---

## 4. Users & use cases

| User | Use case |
|------|----------|
| **Student / presenter** | Demonstrates each DAA paradigm live on a projector during evaluation. |
| **Evaluator / professor** | Inspects algorithm behaviour (shift tables, memo states, Huffman tree) to verify correctness and understanding. |
| **End user (conceptual)** | Pastes a long prompt into a local chat and receives an optimized, budget-safe, guarded prompt. |

**Primary use case flow:** user pastes text → clicks *Optimize & Execute* → tool
guards, ranks, packs, and compresses → UI shows what was kept, dropped, blocked,
and saved.

---

## 5. Syllabus mapping (academic requirement)

This is the defining requirement of the project: each functional module **must**
implement one specific DAA paradigm.

| # | Paradigm (Levitin) | Algorithm | Functional role |
|---|--------------------|-----------|-----------------|
| 1 | Dynamic Programming | 0/1 Knapsack via memory functions (top-down memoization) | Pack maximum-relevance blocks into the token budget |
| 2 | Space–Time Trade-offs (Input Enhancement) | Horspool / Boyer–Moore string matching | Detect and block prompt-injection phrases |
| 3 | Transform & Conquer | Heapsort (bottom-up heap construction) | Rank text snippets by relevance |
| 4 | Greedy Technique | Huffman coding | Compress dropped chat history |

---

## 6. Functional requirements

### FR-1 — Prompt Injection Guardrail (Space–Time Trade-offs)
- **FR-1.1** The system shall scan all raw user input against a configurable list
  of banned exploit phrases.
- **FR-1.2** Matching shall use Horspool's algorithm: a bad-character **shift
  table** is pre-computed from each pattern (input enhancement), enabling the
  scanner to skip characters instead of sliding one position at a time.
- **FR-1.3** If any banned phrase is found, the pipeline shall **halt** and send
  nothing downstream.
- **FR-1.4** The system shall display the shift table and scan telemetry
  (comparisons, shifts, characters skipped) for the matched/scanned pattern.
- **FR-1.5** Matching shall be case-insensitive by default.

### FR-2 — Snippet Prioritisation (Transform & Conquer)
- **FR-2.1** The system shall split raw input into discrete blocks (paragraphs or
  chat messages).
- **FR-2.2** Each block shall receive a relevance **score** based on keyword
  overlap with a user-supplied focus query plus a recency bonus.
- **FR-2.3** Blocks shall be ordered by score, descending, using **Heapsort**
  (explicit bottom-up max-heap construction, then repeated max-extraction).
- **FR-2.4** The ordered list shall be displayed in the UI.

### FR-3 — Context Token Packing (Dynamic Programming)
- **FR-3.1** Each block shall be treated as a knapsack item with weight = token
  count and value = relevance score.
- **FR-3.2** The system shall select the subset of blocks that maximises total
  relevance without exceeding the configurable token capacity, using the
  **0/1 Knapsack** algorithm solved top-down with **memoization** (memory
  functions).
- **FR-3.3** The system shall reconstruct and display the **kept** vs **dropped**
  blocks and report total value, tokens used, and number of memoized states.
- **FR-3.4** The token capacity shall be user-configurable (default 1024).

### FR-4 — Cold-Storage Compression (Greedy)
- **FR-4.1** Blocks dropped by the knapsack shall be routed to compression.
- **FR-4.2** The system shall tally character frequencies, build an explicit
  **Huffman tree** via a greedy least-frequency merge, and assign prefix-free
  codes.
- **FR-4.3** The system shall encode the dropped text, pack the bit-string into
  real bytes, and report original size, compressed size, and percentage saved.
- **FR-4.4** The system shall display the Huffman tree and code table, and allow
  the compressed archive to be downloaded as a binary file.

### FR-5 — User Interface
- **FR-5.1** A text area shall accept the raw prompt/history.
- **FR-5.2** Controls shall allow editing the token limit, focus query, and
  banned phrase list.
- **FR-5.3** A single **Optimize & Execute** action shall run the full pipeline.
- **FR-5.4** Three logical panels shall be shown: (A) Input & Guardrail,
  (B) DAA Engine view (heapsort order + knapsack result), (C) Compression output.

---

## 7. Non-functional requirements

| ID | Requirement |
|----|-------------|
| NFR-1 | **Offline:** no network calls at runtime after install. |
| NFR-2 | **Performance:** guardrail scan and full pipeline complete in well under one second for inputs up to ~10 KB. |
| NFR-3 | **Transparency:** every algorithm exposes inspectable internal state (tables, trees, memo counts) for evaluation. |
| NFR-4 | **Modularity:** each algorithm lives in its own module, independently runnable and testable. |
| NFR-5 | **Portability:** pure standard-library algorithm core; UI layer is the only external dependency. |
| NFR-6 | **Determinism:** identical input + settings produce identical output (tie-breaking in Huffman is deterministic). |

---

## 8. System architecture & data flow

```
              ┌──────────────────────────────────────────────┐
  RAW INPUT → │ 1. GUARDRAIL  (Horspool / Space–Time)         │
              │    banned-phrase scan with shift tables       │
              └───────────────┬──────────────────────────────┘
                              │ safe?  ──no──► HALT (block + show table)
                              │ yes
              ┌───────────────▼──────────────────────────────┐
              │ 2. SCORE      (pipeline)                      │
              │    split into blocks, token estimate,         │
              │    relevance = keyword overlap + recency      │
              └───────────────┬──────────────────────────────┘
              ┌───────────────▼──────────────────────────────┐
              │ 3. PRIORITISE (Heapsort / Transform&Conquer)  │
              │    rank blocks by relevance, descending       │
              └───────────────┬──────────────────────────────┘
              ┌───────────────▼──────────────────────────────┐
              │ 4. PACK       (Knapsack / Dynamic Programming)│
              │    pick max-relevance subset ≤ token budget   │
              └──────────┬───────────────────────┬───────────┘
                  KEPT blocks               DROPPED blocks
                  │                              │
                  ▼                              ▼
            sent to LLM            ┌─────────────────────────────┐
                                   │ 5. COMPRESS (Huffman/Greedy)│
                                   │    encode → .huff archive   │
                                   └─────────────────────────────┘
```

---

## 9. Algorithm specifications & complexity

| Algorithm | Time complexity | Space complexity | Key detail |
|-----------|-----------------|------------------|------------|
| **0/1 Knapsack** (memoized) | O(n · W) | O(n · W) | n = #blocks, W = token capacity. Pseudo-polynomial. Top-down recursion caches each (item, capacity) state once; selection reconstructed by re-walking the memo. |
| **Horspool** | Best/avg sub-linear; worst O(n · m) | O(alphabet size) | Pattern pre-processed into a bad-character shift table; mismatches jump ahead by the table entry for the aligned text character. |
| **Heapsort** | O(n log n) (all cases) | O(1) auxiliary | Heap built bottom-up in O(n); n extractions at O(log n) each; in-place. |
| **Huffman** | O(k log k) | O(k) | k = #distinct characters. Greedy: repeatedly merge two lowest-frequency nodes via a min-heap; yields a provably optimal prefix code. |

---

## 10. Technical stack & project structure

**Stack:** Python 3.10+, Streamlit (UI). Algorithm core uses only the standard
library (`heapq`, `collections`, `dataclasses`, `re`, `math`).

```
llm_optimizer/
├── app.py                  # Streamlit UI — three-panel demo, runs the pipeline
├── pipeline.py             # block splitting, token estimate, relevance scoring
├── requirements.txt        # streamlit
├── README.md
└── algorithms/
    ├── __init__.py
    ├── knapsack.py         # Dynamic Programming  — context token packing
    ├── horspool.py         # Space–Time Trade-offs — injection guardrail
    ├── heapsort.py         # Transform & Conquer   — snippet prioritisation
    └── huffman.py          # Greedy                — history compression
```

Each `algorithms/*.py` module includes a `__main__` self-test so any single
algorithm can be demonstrated in isolation.

**Setup & run:**
```bash
cd llm_optimizer
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 11. UI requirements (three-panel layout)

```
+-------------------------------------------------------------------------+
|              LOCAL LLM ALGORITHMIC OPTIMIZER SUITE                      |
+------------------------------------+------------------------------------+
| PANEL A: USER INPUT & GUARDRAILS   | PANEL B: THE DAA ENGINE VIEW       |
| - long prompt text box             | - Horspool status: CLEAN/BLOCKED   |
| - Optimize & Execute button        |   + characters skipped             |
| - context limit field (1024)       | - Heapsort relevance order         |
|                                    | - Knapsack: kept [..] / dropped[..]|
+------------------------------------+------------------------------------+
| PANEL C: COMPRESSION OUTPUT                                             |
| - Huffman telemetry: original | compressed | % saved                   |
| - Huffman tree + code table; download .huff archive                    |
+-------------------------------------------------------------------------+
```

---

## 12. Acceptance criteria (demo / evaluation)

| ID | Scenario | Expected result |
|----|----------|-----------------|
| AC-1 | Submit benign text | Panel A shows **CLEAN** with characters-skipped count; pipeline proceeds. |
| AC-2 | Submit text containing a banned phrase | Panel A turns **red**, pipeline halts, shift table displayed. |
| AC-3 | Submit text exceeding the token limit | Knapsack drops lowest-value/heaviest blocks; kept blocks ≤ budget. |
| AC-4 | Lower the token limit live | More blocks are dropped; total tokens stay ≤ the new limit. |
| AC-5 | Inspect compression panel | Reports a positive % saved and shows a valid Huffman tree; `.huff` downloads. |
| AC-6 | Run any `algorithms/*.py` directly | Prints a correct standalone self-test. |

---

## 13. Risks, assumptions & limitations

| Item | Description | Mitigation |
|------|-------------|------------|
| Token estimation | ~4 chars/token heuristic ≠ true BPE tokenisation. | Algorithm is weight-agnostic; swap in `tiktoken` for exact counts. |
| Multi-pattern matching | Horspool runs once per banned phrase (not single-pass). | Acceptable for modest blocklists; Aho–Corasick noted as an optimisation. |
| Knapsack scalability | NP-hard in general; DP is pseudo-polynomial in W. | Fine because W (e.g. 1024) is small and fixed. |
| Relevance heuristic | Keyword+recency is not semantic. | Replaceable with embeddings; not core to the DAA demonstration. |
| Huffman archive | Stores padding header; code table not serialised. | Reported ratio counts payload bits (standard demo metric). |

---

## 14. Future enhancements (out of scope for v1)
- Huffman **decoder** for lossless `.huff` → original round-trip.
- Attach a real local model (Ollama / `llama-cpp-python`) at the pipeline's end.
- Exact tokenisation via `tiktoken`.
- Embedding-based relevance scoring.
- Aho–Corasick guardrail mode with a benchmark against Horspool to quantify the
  space–time trade-off.

---

## 15. Deliverables
- Source code: `app.py`, `pipeline.py`, and four `algorithms/` modules.
- `requirements.txt` and `README.md`.
- This PRD.
- A live demo following the acceptance-criteria scenarios.

---

*End of document.*
