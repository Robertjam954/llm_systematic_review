"""
Boolean search fetcher — queries PubMed via NCBI E-utilities (Biopython Entrez).
Extend with Embase/Scopus API clients as needed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from Bio import Entrez
from loguru import logger


def configure_entrez(email: str, api_key: Optional[str] = None) -> None:
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key


def search_pubmed(query: str, max_results: int = 10_000) -> list[str]:
    """Return a list of PubMed IDs matching *query*."""
    logger.info(f"Searching PubMed: max_results={max_results}")
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, usehistory="y")
    record = Entrez.read(handle)
    handle.close()
    pmids: list[str] = record["IdList"]
    logger.info(f"PubMed returned {len(pmids)} results")
    return pmids


def fetch_pubmed_records(pmids: list[str], batch_size: int = 200) -> list[dict]:
    """Fetch abstracts + metadata for a list of PMIDs."""
    records: list[dict] = []
    for start in range(0, len(pmids), batch_size):
        batch = pmids[start : start + batch_size]
        handle = Entrez.efetch(db="pubmed", id=",".join(batch), rettype="xml", retmode="xml")
        fetched = Entrez.read(handle)
        handle.close()
        for article in fetched.get("PubmedArticle", []):
            records.append(_parse_pubmed_article(article))
        time.sleep(0.34)  # Respect NCBI rate limit (~3 req/s without key, 10/s with key)
        logger.debug(f"Fetched batch {start}–{start + len(batch)}")
    return records


def _parse_pubmed_article(article: dict) -> dict:
    try:
        medline = article["MedlineCitation"]
        art = medline["Article"]
        title = str(art.get("ArticleTitle", ""))
        abstract = " ".join(
            str(t) for t in art.get("Abstract", {}).get("AbstractText", [])
        )
        pmid = str(medline["PMID"])
        year = str(
            art.get("Journal", {})
            .get("JournalIssue", {})
            .get("PubDate", {})
            .get("Year", "")
        )
        authors_list = art.get("AuthorList", [])
        authors = "; ".join(
            f"{a.get('LastName', '')} {a.get('Initials', '')}".strip()
            for a in authors_list
        )
        doi = ""
        for loc_id in art.get("ELocationID", []):
            if loc_id.attributes.get("EIdType") == "doi":
                doi = str(loc_id)
        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "doi": doi,
            "source_db": "pubmed",
        }
    except Exception as exc:
        logger.warning(f"Failed to parse article: {exc}")
        return {}


def save_results(records: list[dict], output_path: Path) -> None:
    df = pd.DataFrame([r for r in records if r])
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} records to {output_path}")


def load_ris(ris_path: Path) -> pd.DataFrame:
    """Parse an exported RIS file (e.g. from Embase/Scopus) using rispy."""
    import rispy

    with open(ris_path, encoding="utf-8") as fh:
        entries = rispy.load(fh)
    rows = []
    for e in entries:
        rows.append(
            {
                "title": e.get("title", e.get("primary_title", "")),
                "abstract": e.get("abstract", ""),
                "authors": "; ".join(e.get("authors", [])),
                "year": e.get("year", ""),
                "doi": e.get("doi", ""),
                "pmid": e.get("pmid", ""),
                "source_db": e.get("database_provider", "unknown"),
            }
        )
    return pd.DataFrame(rows)
