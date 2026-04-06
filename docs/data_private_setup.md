# DATA_PRIVATE_DIR — Local Setup Guide

This directory lives **outside the repository** and is **never committed**.
It holds PHI-containing source documents and method output texts.

---

## Recommended Location

| Platform | Path |
|----------|------|
| macOS | `~/data_private/llm_systematic_review/` |
| Windows | `C:\Users\<you>\data_private\llm_systematic_review\` |

Set the path in your `.env` file:
```
DATA_PRIVATE_DIR=/path/to/data_private/llm_systematic_review
```

---

## Directory Structure

```
data_private/
├── raw/                        Source PDFs (radiology + pathology reports with PHI)
│   └── *.pdf
├── deidentified/               Redacted PDFs + mapping file
│   ├── *.pdf
│   └── document_mapping.csv    columns: original_filename, deidentified_filename, doc_id
├── extracted_text/             Per-document deidentified .txt files
│   └── <doc_id>.txt
└── method_outputs/             Per-method synthesis + retrieval outputs
    ├── boolean/
    │   └── screening_full_texts.csv
    ├── standard_rag/
    │   └── synthesis_runs/
    │       └── <run_id>_synthesis.txt
    └── agentic_rag/
        └── synthesis_runs/
            └── <run_id>_synthesis.txt
```

---

## Deidentification Protocol

1. Source PDFs in `raw/` are processed to remove all 18 HIPAA identifiers
2. Output redacted PDFs saved to `deidentified/`
3. `document_mapping.csv` records the mapping (kept local — contains linkage information)
4. Text extracted from redacted PDFs → `extracted_text/<doc_id>.txt`

---

## Notes

- `data_private/` should be excluded from Time Machine / cloud backup if it contains PHI
- Access restricted to approved personnel under IRB protocol
- Add `data_private/` to `.gitignore` (already done at project root)
