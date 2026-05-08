# LLM Systematic Review — Method Comparison

**MSK | Goel Lab**

## Description

This repository supports a systematic review that compares three retrieval methods—Boolean search, standard RAG, and agentic RAG—for identifying and evaluating literature on LLM optimization strategies used for radiology and pathology feature extraction.

---

## Research Question

> Which retrieval method — Boolean database search, standard RAG, or agentic RAG — most effectively and reproducibly identifies literature on LLM optimization strategies for feature extraction from radiology and pathology reports?

---

## Methods Under Comparison

| # | Method | Description |
|---|--------|-------------|
| 1 | **Boolean Search (Gold Standard)** | Structured Boolean queries across PubMed/MEDLINE, Embase, Web of Science, Scopus. PRISMA-compliant. PROSPERO-registered. |
| 2 | **Agentic RAG** | LLM agent autonomously decomposes topic into sub-questions, iteratively retrieves and re-ranks chunks, adjusts strategy dynamically. |
| 3 | **Standard RAG** | Static vector store (cosine/ANN similarity). Single query → retrieval pass → LLM synthesis. |

---

## Project Structure

```
llm_systematic_review/          ← PROJECT_ROOT
├── notebooks/                  7 Jupyter notebooks (01–07) with MSK | Goel Lab headers
├── data/
│   ├── processed/              Non-PHI CSVs: search results, extracted features      ← committed
│   ├── features/               Embeddings, feature vectors from rad/path docs        ← committed
│   ├── splits/                 Eval splits for RAG methods                           ← committed
│   └── corpus/                 Source document metadata (non-PHI)                    ← committed
├── docs/
│   ├── executive_summary.md    Technical overview + code map
│   ├── dataset_metadata.md     YAML front-matter document corpus specification
│   ├── prisma/                 PRISMA flow diagrams and checklists
│   ├── manuscript/             Methods, results, discussion drafts
│   └── manuscript_components/  Abstract, appendix, supplementary methods, cover letter
├── search/
│   ├── boolean/                Boolean search strings per database + search log
│   └── results/                Raw search results per database (pubmed, embase, etc.)
├── reports/                    Exported figures and tables for manuscript             ← committed
├── prompts/
│   ├── prompt_library.csv      Prompt versions with metadata
│   ├── library/                Frozen prompt templates
│   └── generated/              Agent-derived prompts
├── references/                 Academic papers, background literature (see REFERENCES_INDEX.md)
├── src/
│   ├── methods/
│   │   ├── boolean_search/     Boolean search utilities + deduplication
│   │   ├── standard_rag/       Standard RAG pipeline
│   │   └── agentic_rag/        Agentic RAG pipeline
│   ├── evaluation/             Cross-method evaluation metrics
│   ├── extraction/             Feature extraction from rad/path documents
│   └── utils/                  Shared utilities (parsing, logging, I/O)
├── eval/                       Evaluation schemas and metric definitions
├── models/                     Model configurations
├── experiments/                Run tracking (run_id, method, prompt_id, results)
├── tools/
│   └── watch_repo.sh           Local macOS notifications for remote repo changes
├── .env.example                API key + path template (copy to .env — never commit .env)
├── pyproject.toml              Project metadata + dependencies
├── requirements.txt            Pip-compatible dependency list
└── LICENCE                     MIT
```

**Data private directory** (local only — never committed):
```
data_private/
├── raw/                        Source PDFs with PHI (rad/path reports)
├── deidentified/               Redacted PDFs + document_mapping.csv
├── extracted_text/             Per-document deidentified .txt files
└── method_outputs/             Per-method extraction outputs for comparison
```

---

## Setup

```bash
# 1. Clone repo
git clone <repo-url>
cd llm_systematic_review

# 2. Create environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys and local data paths
```

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_corpus_preparation.ipynb` | Ingest and preprocess rad/path source documents |
| `02_boolean_search.ipynb` | Method 1: Boolean search execution, deduplication, PRISMA screening |
| `03_standard_rag_pipeline.ipynb` | Method 3: Standard RAG — embedding, retrieval, synthesis, NIAH sweep |
| `04_agentic_rag_pipeline.ipynb` | Method 2: Agentic RAG — agent loop, sub-query decomposition, 3× reproducibility |
| `05_feature_extraction_eval.ipynb` | Extract features from rad/path docs via each method |
| `06_method_comparison.ipynb` | Cross-method comparison: precision, recall, F1, coverage |
| `07_reporting_and_figures.ipynb` | Final figures, tables, and manuscript outputs |
| `08_agentic_niah_stress_test.ipynb` | **Exploratory**: NIAH stress test for agentic pipeline — iteration convergence, sub-query quality, cross-method comparison |

---

## Exploratory Analysis — Needle-in-a-Haystack (NIAH)

Adapted from [Kamradt (2023)](https://github.com/gkamradt/LLMTest_NeedleInAHaystack).

A factual "needle" is inserted at varying depths (10–90%) through haystacks of increasing length (2k–32k tokens). Both RAG pipelines are asked to retrieve it.

The **agentic pipeline** extends the standard (depth × context_length) grid with two extra dimensions:

| Dimension | Standard RAG | Agentic RAG |
|-----------|:-----------:|:-----------:|
| Needle retrieved | ✓ | ✓ |
| Needle in answer | ✓ | ✓ |
| DeepEval faithfulness | ✓ | ✓ |
| Sub-query targets needle | — | ✓ |
| Iteration at first retrieval | — | ✓ |

Key files:
- `src/methods/agentic_rag/niah.py` — `AgenticNIAHTester` class
- `notebooks/08_agentic_niah_stress_test.ipynb` — full sweep + 5 figures + cross-method comparison
- `eval/metrics/metric_definitions.yaml` — formal definitions for all NIAH metrics

---

## PROSPERO Registration

- Protocol registered prior to search execution
- Registration ID: *(to be completed)*
- PRISMA checklist: `docs/prisma/`

---

## Citation

*(To be completed upon publication)*

---

## Licence

MIT — see `LICENCE`
