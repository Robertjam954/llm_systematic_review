"""
Deduplication utilities for merging search results across databases.
Priority: DOI match > PMID match > fuzzy title+year match.
"""

from __future__ import annotations

import hashlib

import pandas as pd
from loguru import logger


def dedup_by_doi(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.copy()
    df["doi_norm"] = df["doi"].str.strip().str.lower().str.replace(r"https?://doi\.org/", "", regex=True)
    df = df.drop_duplicates(subset=["doi_norm"], keep="first")
    logger.info(f"DOI dedup: {before} → {len(df)} records")
    return df


def dedup_by_pmid(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.copy()
    mask = df["pmid"].notna() & (df["pmid"] != "")
    df_with_pmid = df[mask].drop_duplicates(subset=["pmid"], keep="first")
    df_no_pmid = df[~mask]
    df = pd.concat([df_with_pmid, df_no_pmid], ignore_index=True)
    logger.info(f"PMID dedup: {before} → {len(df)} records")
    return df


def dedup_by_title_year(df: pd.DataFrame, threshold: float = 0.92) -> pd.DataFrame:
    """Fuzzy title+year deduplication using sequence matching."""
    from difflib import SequenceMatcher

    df = df.copy()
    df["title_norm"] = df["title"].str.strip().str.lower()
    keep = [True] * len(df)
    rows = df[["title_norm", "year"]].to_dict("records")
    for i in range(len(rows)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(rows)):
            if not keep[j]:
                continue
            if rows[i]["year"] != rows[j]["year"]:
                continue
            sim = SequenceMatcher(None, rows[i]["title_norm"], rows[j]["title_norm"]).ratio()
            if sim >= threshold:
                keep[j] = False
    before = len(df)
    df = df[keep].reset_index(drop=True)
    logger.info(f"Title+year fuzzy dedup: {before} → {len(df)} records")
    return df


def assign_doc_id(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a stable doc_id (SHA-1 of DOI or title)."""
    df = df.copy()

    def _hash(row: pd.Series) -> str:
        key = row["doi"] if row.get("doi") else row.get("title", "")
        return hashlib.sha1(str(key).encode()).hexdigest()[:12]

    df["doc_id"] = df.apply(_hash, axis=1)
    return df


def full_dedup_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = dedup_by_doi(df)
    df = dedup_by_pmid(df)
    df = dedup_by_title_year(df)
    df = assign_doc_id(df)
    return df
