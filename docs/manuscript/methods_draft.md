# Methods — Draft

## 2. Methods

### 2.1 Study Design

We conducted a methodological comparison study evaluating three retrieval strategies for performing a systematic review on LLM optimization strategies for extracting clinically relevant features from radiology and pathology source documents. The three methods were: (1) structured Boolean database search (gold standard), (2) standard retrieval-augmented generation (RAG), and (3) agentic RAG.

### 2.2 Research Question (PICO/SPIDER Framework)

*(to be completed)*

**Topic**: What LLM optimization strategies (prompt engineering, fine-tuning, chain-of-thought, retrieval augmentation, etc.) have been proposed or evaluated for extracting structured clinical features from unstructured radiology reports and/or pathology reports?

### 2.3 Method 1 — Boolean Search (Gold Standard)

#### 2.3.1 Database Selection
Searches were conducted across: PubMed/MEDLINE, Embase, Web of Science, and Scopus.

#### 2.3.2 Search String Development
Boolean search strings were co-designed with an information specialist using a combination of controlled vocabulary (MeSH terms for PubMed; Emtree for Embase) and free-text synonyms, connected with Boolean operators (AND, OR, NOT). See `search/boolean/` for full strings per database.

#### 2.3.3 Search Execution
Searches were executed on *(date)*. Date limits: 2018–2026 (English language only). Results were exported in RIS format and deduplicated using *(tool)*.

#### 2.3.4 Screening
Title/abstract screening and full-text review were performed by two independent reviewers using *(tool)*. Conflicts were resolved by discussion or a third reviewer. Inclusion/exclusion criteria are specified in `docs/manuscript/inclusion_criteria.md`.

### 2.4 Method 2 — Agentic RAG

An LLM-powered agent (GPT-4o, temperature = 0.0) was implemented using LangGraph. The agent:
1. Decomposed the review question into sub-queries via chain-of-thought reasoning
2. Iteratively retrieved candidate documents from the embedded corpus and via PubMed API
3. Evaluated retrieved chunks for relevance and adjusted the search strategy
4. Synthesised findings across retrieved content

Agent configuration is documented in `src/methods/agentic_rag/` and run parameters in `experiments/`.

### 2.5 Method 3 — Standard RAG

A static vector store (FAISS) was constructed from the embedded corpus (`data/features/`). For each standardised query (`data/splits/eval_query_set.csv`):
1. The query was embedded using `text-embedding-3-large`
2. Top-k chunks (k = *(value)*) were retrieved via cosine similarity
3. Retrieved chunks were passed to GPT-4o for synthesis

### 2.6 Evaluation

Evaluation was performed against a manually curated gold standard reference set (`data/splits/gold_standard_set.csv`). Metrics: precision, recall, F1, thematic coverage, time-to-results, and screening burden. See `src/evaluation/metrics.py` and `notebooks/06_method_comparison.ipynb`.

---

*(Draft — to be expanded)*
