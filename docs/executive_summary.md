# Executive Summary — LLM Systematic Review Method Comparison

**MSK | Goel Lab**
**Project**: Comparing Boolean Search, Standard RAG, and Agentic RAG for systematic review of LLM optimization strategies in radiology/pathology feature extraction

---

## Overview

This project conducts a head-to-head comparison of three retrieval methodologies used to identify literature on LLM optimization strategies for extracting clinically relevant features from radiology and pathology source documents.

The central question is not which *LLM strategy* works best for feature extraction, but which *systematic review method* most effectively and reproducibly surfaces that literature.

---

## Methods Summary

### Method 1 — Boolean Search (Gold Standard)
- Databases: PubMed/MEDLINE, Embase, Web of Science, Scopus
- Search strings co-designed with librarian using MeSH/thesaurus terms + free-text variants
- Protocol registered on PROSPERO; PRISMA-compliant reporting
- Two independent reviewers for title/abstract screening
- Deterministic and fully reproducible
- See: `search/boolean/`, `docs/prisma/`

### Method 2 — Agentic RAG
- LLM agent (GPT-4o via LangGraph) decomposes review question into sub-queries
- Iteratively retrieves, re-ranks, and evaluates chunks; adjusts strategy dynamically
- Sources: embedded corpus of rad/path LLM papers + live PubMed API calls
- Non-deterministic; stochasticity controlled via `LLM_TEMPERATURE=0.0`
- See: `src/methods/agentic_rag/`, `notebooks/04_agentic_rag_pipeline.ipynb`

### Method 3 — Standard RAG
- Static vector store (FAISS/Chroma) of embedded corpus
- Single query → cosine/ANN similarity → top-k chunks → LLM synthesis
- Fast and reproducible given fixed embeddings
- See: `src/methods/standard_rag/`, `notebooks/03_standard_rag_pipeline.ipynb`

---

## Code Map

| Component | Path |
|-----------|------|
| Corpus ingestion | `notebooks/01_corpus_preparation.ipynb` |
| Boolean search execution | `notebooks/02_boolean_search.ipynb` |
| Standard RAG pipeline | `src/methods/standard_rag/pipeline.py` |
| Agentic RAG pipeline | `src/methods/agentic_rag/agent.py` |
| Feature extraction | `src/extraction/extractor.py` |
| Evaluation metrics | `src/evaluation/metrics.py` |
| Method comparison | `notebooks/06_method_comparison.ipynb` |
| Figures / tables | `notebooks/07_reporting_and_figures.ipynb` |

---

## Key Evaluation Metrics

- **Precision** — of retrieved articles, what fraction are relevant?
- **Recall** — of all relevant articles (gold set), what fraction were retrieved?
- **F1** — harmonic mean of precision and recall
- **Coverage of key themes** — thematic overlap with manually curated reference set
- **Reproducibility** — variance across repeated runs (RAG methods)
- **Time-to-results** — wall-clock time per method
- **Screening burden** — number of records requiring human review

---

## Data Flow

```
Raw rad/path PDFs (PHI)
        ↓  [NB01: corpus_preparation]
Deidentified text + metadata (data_private/)
        ↓  [NB01]
Embedded corpus + non-PHI CSVs (data/processed/, data/features/)
        ↓
  ┌─────────────────────────────────┐
  │  Method 1: Boolean Search       │ → search/results/
  │  Method 2: Agentic RAG          │ → data_private/method_outputs/agentic_rag/
  │  Method 3: Standard RAG         │ → data_private/method_outputs/standard_rag/
  └─────────────────────────────────┘
        ↓  [NB05, NB06]
Comparison metrics (data/processed/method_comparison.csv)
        ↓  [NB07]
Figures + tables (reports/)
```

---

*(Last updated: 2026-04)*
