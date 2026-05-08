# LLM Systematic Review

**MSK | Goel Lab**

A PRISMA-compliant systematic review evaluating the change in large language model (LLM) accuracy following optimization strategies in the summarization of oncologic histories from radiology and pathology free-text reports.

---

## Research Question

> What is the change in LLM accuracy following optimization (prompt engineering, fine-tuning, or retrieval-augmented generation) in the summarization of oncologic histories from radiology and/or pathology free-text reports?

---

## Project Structure

```
llm_systematic_review/          ← PROJECT_ROOT
├── notebooks/                  Jupyter notebooks for analysis
├── data/
│   ├── processed/              Non-PHI CSVs: search results, extracted features      ← committed
│   ├── features/               Embeddings, feature vectors from rad/path docs        ← committed
│   ├── splits/                 Screening splits                                      ← committed
│   └── corpus/                 Source document metadata (non-PHI)                    ← committed
├── docs/
│   ├── executive_summary.md    Technical overview + code map
│   ├── dataset_metadata.md     YAML front-matter document corpus specification
│   ├── prisma/                 PRISMA flow diagrams and checklists
│   ├── manuscript/             Methods, results, discussion drafts
│   └── manuscript_components/  Abstract, appendix, supplementary methods, cover letter
├── search/
│   ├── boolean/                Boolean search strings per database + search log
│   └── results/                Raw search results per database (pubmed, embase, scopus, ieee_xplore, acm, web_of_science)
├── reports/                    Exported figures and tables for manuscript             ← committed
├── references/                 Academic papers, background literature (see REFERENCES_INDEX.md)
├── src/
│   ├── methods/
│   │   └── boolean_search/     Boolean search utilities + deduplication
│   ├── evaluation/             Evaluation metrics
│   ├── extraction/             Feature extraction from rad/path documents
│   └── utils/                  Shared utilities (parsing, logging, I/O)
├── eval/                       Evaluation schemas and metric definitions
├── models/                     Model configurations
├── experiments/                Run tracking
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
└── screening/                  Screening outputs per stage
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
| `02_boolean_search.ipynb` | Boolean search execution, deduplication, PRISMA screening |
| `05_feature_extraction_eval.ipynb` | Extract and evaluate features from included studies |
| `07_reporting_and_figures.ipynb` | Final figures, tables, and manuscript outputs |

---

## Search Databases

| Database | Search Date | Records Retrieved |
|----------|------------|------------------|
| PubMed | May 3, 2025 | 28,546 |
| Embase | May 3, 2025 | 430 |
| Scopus | May 3, 2025 | 336 |
| IEEE Xplore | May 2025 | 2,580 |
| Web of Science | *(to be completed)* | |
| ACM Digital Library | *(to be completed)* | |

Full search strings per database: `search/boolean/`

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
