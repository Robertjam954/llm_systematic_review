---
title: Dataset Metadata — LLM Systematic Review Corpus
project: llm_systematic_review
version: "0.1"
created: 2026-04
maintainer: MSK | Goel Lab
phi_present: false
committed: true
---

# Dataset Metadata

## Document Corpus

| Field | Value |
|-------|-------|
| **Domain** | Radiology and Pathology reports / LLM optimization literature |
| **Document types** | PubMed abstracts, full-text articles (PDF), preprints |
| **PHI status** | Source PDFs in `data_private/raw/` only; all committed data is deidentified |
| **Languages** | English |
| **Date range** | 2018–2026 (LLM/NLP era for clinical NLP) |

---

## Search Corpus (`data/corpus/`)

Metadata-only records (title, abstract, authors, DOI, source database) for all candidate documents retrieved across the three methods.

| Column | Type | Description |
|--------|------|-------------|
| `doc_id` | str | Unique document identifier (hash of DOI or title) |
| `title` | str | Article title |
| `abstract` | str | Full abstract text |
| `authors` | str | Semicolon-separated author list |
| `year` | int | Publication year |
| `doi` | str | Digital Object Identifier |
| `pmid` | str | PubMed ID (if applicable) |
| `source_db` | str | Source database (pubmed, embase, scopus, wos, corpus) |
| `retrieval_method` | str | `boolean` / `standard_rag` / `agentic_rag` |
| `retrieval_run_id` | str | Links to `experiments/` run record |
| `included_screening` | bool | Passed title/abstract screening |
| `included_fulltext` | bool | Passed full-text inclusion/exclusion |

---

## Processed Data (`data/processed/`)

| File | Description |
|------|-------------|
| `screening_results.csv` | Title/abstract screening decisions (both reviewers) |
| `inclusion_decisions.csv` | Full-text inclusion/exclusion with reasons |
| `extracted_features.csv` | Structured feature extraction from included articles |
| `method_comparison.csv` | Precision, recall, F1, coverage per method per run |
| `prompt_library.csv` | Prompt versions used across runs |

---

## Feature Data (`data/features/`)

| File | Description |
|------|-------------|
| `corpus_embeddings.npy` | NumPy array of document embeddings (dim = `EMBEDDING_DIMENSION`) |
| `corpus_embedding_index.csv` | Maps row index → `doc_id` |
| `chunk_embeddings.npy` | Chunk-level embeddings for RAG retrieval |
| `chunk_index.csv` | Maps chunk index → `doc_id`, chunk position, text |

---

## Splits (`data/splits/`)

| File | Description |
|------|-------------|
| `gold_standard_set.csv` | Manually curated reference set for recall evaluation |
| `eval_query_set.csv` | Standardised query set used across all three methods |
