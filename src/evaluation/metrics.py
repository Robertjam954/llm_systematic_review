"""
Evaluation metrics for comparing retrieval methods.
All metrics are computed against a manually curated gold standard set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score, precision_score, recall_score


@dataclass
class RetrievalMetrics:
    method: str
    run_id: Optional[str]
    precision: float
    recall: float
    f1: float
    n_retrieved: int
    n_gold: int
    n_true_positive: int
    thematic_coverage: Optional[float] = None
    time_seconds: Optional[float] = None
    screening_burden: Optional[int] = None


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    gold_ids: list[str],
    method: str,
    run_id: Optional[str] = None,
    time_seconds: Optional[float] = None,
    screening_burden: Optional[int] = None,
) -> RetrievalMetrics:
    """
    Compute precision, recall, F1 at the document level.

    Parameters
    ----------
    retrieved_ids : list of doc_ids returned by the method
    gold_ids : list of doc_ids in the gold standard set
    """
    retrieved_set = set(retrieved_ids)
    gold_set = set(gold_ids)
    all_ids = list(retrieved_set | gold_set)

    y_true = [1 if d in gold_set else 0 for d in all_ids]
    y_pred = [1 if d in retrieved_set else 0 for d in all_ids]

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    tp = len(retrieved_set & gold_set)

    logger.info(
        f"[{method}] P={p:.3f} R={r:.3f} F1={f:.3f} "
        f"TP={tp} retrieved={len(retrieved_set)} gold={len(gold_set)}"
    )
    return RetrievalMetrics(
        method=method,
        run_id=run_id,
        precision=round(p, 4),
        recall=round(r, 4),
        f1=round(f, 4),
        n_retrieved=len(retrieved_set),
        n_gold=len(gold_set),
        n_true_positive=tp,
        time_seconds=time_seconds,
        screening_burden=screening_burden,
    )


def compute_thematic_coverage(
    retrieved_texts: list[str],
    theme_keywords: dict[str, list[str]],
) -> dict[str, bool]:
    """
    Check whether each predefined theme is represented in retrieved texts.

    Parameters
    ----------
    retrieved_texts : list of abstract/chunk texts
    theme_keywords : mapping of theme_name → list of keywords (any match = covered)
    """
    combined = " ".join(retrieved_texts).lower()
    coverage = {}
    for theme, keywords in theme_keywords.items():
        coverage[theme] = any(kw.lower() in combined for kw in keywords)
    covered = sum(coverage.values())
    logger.info(f"Thematic coverage: {covered}/{len(theme_keywords)} themes covered")
    return coverage


def metrics_to_dataframe(metrics_list: list[RetrievalMetrics]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": m.method,
                "run_id": m.run_id,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "n_retrieved": m.n_retrieved,
                "n_gold": m.n_gold,
                "n_true_positive": m.n_true_positive,
                "thematic_coverage": m.thematic_coverage,
                "time_seconds": m.time_seconds,
                "screening_burden": m.screening_burden,
            }
            for m in metrics_list
        ]
    )
