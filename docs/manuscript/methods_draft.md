# Methods — Draft

## 2. Methods

### 2.1 Study Design

This is a PRISMA-compliant systematic review evaluating changes in large language model (LLM) accuracy following optimization strategies in the summarization of oncologic histories from radiology and pathology free-text reports.

### 2.2 Eligibility Criteria

#### 2.2.1 Inclusion Criteria

- Studies evaluating large language model (LLM) performance in generating structured clinical summaries from radiology and/or pathology free-text reports
- Studies that assess optimization techniques, including: prompt engineering, fine-tuning, or retrieval-augmented generation (RAG)
- Studies that evaluate performance and include:
  - baseline accuracy (pre-optimization)
  - post-optimization accuracy
  - change in accuracy (explicitly or derivable)
- Studies published between May 2016 and May 2026

#### 2.2.2 Exclusion Criteria

- Studies without both abstract and full text available
- Studies that do not report both pre- and post-optimization accuracy
- Studies that do not use radiologic or pathologic reports as input data
- Studies published prior to May 2016
- Studies limited to pure benchmarking without clinical text summarization/extraction tasks

### 2.3 Information Sources

Searches were conducted across the following databases:
- PubMed
- Embase
- Web of Science
- IEEE Xplore
- ACM Digital Library
- Scopus

### 2.4 Search Strategy

Search strings include combinations of the following terms:

**LLM terms**: "large language model" OR "LLM" OR "language model" OR "foundation model" OR "transformer" OR "GPT" OR "BERT"

**Task terms**: "summarization" OR "information extraction" OR "clinical extraction"

**Domain terms**: "radiology" OR "pathology" OR "clinical"

**Performance terms**: "accuracy" OR "performance" OR "hallucination" OR "fabrication"

**Optimization terms**: "prompt engineering" OR "fine tuning" OR "RAG" OR "retrieval augmented generation"

Search strategies were adapted to each database's controlled vocabulary (e.g., MeSH, Emtree) and syntax. Full search strings per database are available in `search/boolean/`.

#### 2.4.1 Search Execution Results (May 3, 2025)

| Database | Records Retrieved | Date | Notes |
|----------|------------------|------|-------|
| PubMed | 28,546 | May 3, 2025 | Filters: full text available, 2016–2026 |
| Embase | 430 | May 3, 2025 | Time range: 2016–2026 |
| Scopus | 336 | May 3, 2025 | PUBYEAR 2016–2026 |
| IEEE Xplore | 2,580 | May 2025 | Filters: 2016–2026 |
| Web of Science | *(to be completed)* | | |
| ACM Digital Library | *(to be completed)* | | |

Results were exported and will be deduplicated using DOI-first matching, then title+year fuzzy match.

### 2.5 Selection Process

Study selection was conducted in two stages by two independent reviewers (RJ and SB).

**Stage 1 — Title/Abstract Screening**: Titles and abstracts of all retrieved records were screened independently by both reviewers to assess eligibility. To minimize the risk of premature exclusion, any disagreements at this stage were resolved conservatively by retaining the study for full-text review. Interrater agreement during this stage was assessed using Cohen's kappa coefficient.

**Stage 2 — Full-Text Review**: Full-text articles were independently assessed for eligibility by the same two reviewers. Discrepancies were resolved through discussion and consensus. Reasons for exclusion at the full-text stage were recorded. No automation tools were used in the screening process.

### 2.6 Data Collection and Extraction

*(to be completed)*

### 2.7 Risk of Bias Assessment

*(to be completed)*

### 2.8 Synthesis

*(to be completed)*

---

*(Draft — to be expanded)*
