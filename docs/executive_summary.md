# Executive Summary — LLM Systematic Review

**MSK | Goel Lab**
**Project**: Change in LLM accuracy following optimization in summarization of oncologic histories: a systematic review

---

## Overview

This project conducts a PRISMA-compliant systematic review evaluating the change in large language model (LLM) accuracy following optimization strategies (prompt engineering, fine-tuning, and retrieval-augmented generation) for summarizing oncologic histories from radiology and pathology free-text reports.

The central question is how LLM optimization techniques affect accuracy in generating structured clinical summaries from unstructured radiology and pathology reports.

---

## Methods Summary

### Boolean Database Search
- Databases: PubMed, Embase, Web of Science, Scopus, IEEE Xplore, ACM Digital Library
- Search strings adapted to each database's controlled vocabulary (MeSH, Emtree) and syntax
- Protocol registered on PROSPERO; PRISMA-compliant reporting
- Two independent reviewers (RJ and SB) for title/abstract screening and full-text review
- Interrater agreement assessed using Cohen's kappa coefficient
- Deterministic and fully reproducible
- See: `search/boolean/`, `docs/prisma/`

---

## Code Map

| Component | Path |
|-----------|------|
| Boolean search execution | `notebooks/02_boolean_search.ipynb` |
| Feature extraction | `src/extraction/extractor.py` |
| Evaluation metrics | `src/evaluation/metrics.py` |
| Figures / tables | `notebooks/07_reporting_and_figures.ipynb` |

---

## Key Evaluation Metrics

- **Baseline accuracy** — pre-optimization LLM performance
- **Post-optimization accuracy** — LLM performance after prompt engineering, fine-tuning, or RAG
- **Change in accuracy** — absolute and relative difference pre- vs. post-optimization
- **Screening burden** — number of records requiring human review per stage

---

## Data Flow

```
Boolean search across 6 databases (search/results/)
        ↓  [Deduplication]
Unique records for screening
        ↓  [Title/abstract screening — 2 independent reviewers]
Records for full-text review
        ↓  [Full-text assessment — 2 independent reviewers]
Included studies
        ↓  [Data extraction]
Summary tables + figures (reports/)
```

---

*(Last updated: 2026-05)*
