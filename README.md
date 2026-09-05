# Local LLM Algorithmic Optimizer & Safety Guardrail Suite

A Design & Analysis of Algorithms  project that applies four classic
algorithm paradigms as a real pre-processing layer in front of a local LLM
chat box, now upgraded with real tokenisation, TF-IDF relevance, a
self-describing Huffman archive, an Aho-Corasick benchmark, and a live local
LLM (Ollama) at the end of the pipeline. It runs fully offline — every upgrade
has a graceful fallback.

The tool takes raw text (a long conversation history or pasted documents),
**guards** it against prompt-injection, **prioritises** the most relevant
blocks, **packs** them to fit a fixed token budget, **compresses** whatever
spills over, and finally **answers** with a real local model.

> **The application code lives in [`llm_optimizer/`](llm_optimizer/).**
> See [`llm_optimizer/README.md`](llm_optimizer/README.md) for the full
> syllabus mapping, complexity analysis, project structure, setup, demo script,
> and honest technical notes. The full spec is in
> [`PRD_LLM_Algorithmic_Optimizer.md`](PRD_LLM_Algorithmic_Optimizer.md).

---

## Syllabus mapping (the core of the project)

| # | Paradigm (Levitin) | Algorithm | Role in the tool | Module |
|---|--------------------|-----------|------------------|--------|
| 1 | Dynamic Programming | 0/1 Knapsack (top-down memory functions) | Pack max-relevance blocks into the token budget | `algorithms/knapsack.py` |
| 2 | Space–Time Trade-offs (Input Enhancement) | Horspool string matching (**primary guardrail**) | Block prompt-injection phrases | `algorithms/horspool.py` |
| 3 | Transform & Conquer | Heapsort (bottom-up heap construction) | Rank snippets by relevance | `algorithms/heapsort.py` |
| 4 | Greedy Technique | Huffman coding | Compress dropped chat history | `algorithms/huffman.py` |

Plus an optional **Aho-Corasick** companion (`algorithms/aho_corasick.py`) used
*only to benchmark* the multi-pattern space–time trade-off against Horspool.

## Production upgrades

1. **Real BPE tokenization** — `tokenizer.py` (tiktoken, heuristic fallback).
2. **TF-IDF relevance scoring** — `pipeline.py` (scikit-learn, keyword fallback).
3. **Self-describing `.huff` archive** — `algorithms/huffman.py` (`compress`).
4. **Lossless Huffman decompression** — `decompress` + `verify_roundtrip`.
5. **Aho-Corasick benchmark mode** — Horspool stays primary; AC compares.
6. **Ollama integration** — `llm.py`, plus a Raw-vs-Optimized comparison panel.

## Quick start

```bash
cd llm_optimizer
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py            # or: python -m streamlit run app.py
pytest                          # run the test suite
```

For a real model answer at the end of the pipeline: 

```bash
ollama serve && ollama pull llama3.2:1b
```
