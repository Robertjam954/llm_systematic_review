"""
Feature extractor for radiology and pathology source documents.
Uses LLM with structured prompts to extract predefined clinical features.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


EXTRACTION_SCHEMA = {
    "document_type": "radiology_report | pathology_report | other",
    "modality": "CT | MRI | PET | X-ray | ultrasound | histology | cytology | other",
    "anatomic_site": "free text",
    "llm_method": "prompt_engineering | fine_tuning | rag | chain_of_thought | instruction_tuning | other",
    "llm_model": "free text (e.g. GPT-4, Llama-3, BERT)",
    "task": "NER | classification | summarisation | structured_extraction | question_answering | other",
    "performance_metric": "free text (e.g. F1=0.91)",
    "dataset": "free text",
    "sample_size": "integer or null",
    "key_finding": "one-sentence summary",
}


class FeatureExtractor:
    """
    LLM-powered structured feature extractor for included articles.

    Parameters
    ----------
    prompt_template : str
        Prompt template with {schema}, {title}, {abstract} placeholders.
    model : str
        OpenAI model to use.
    temperature : float
        LLM temperature (0.0 recommended for deterministic extraction).
    """

    def __init__(
        self,
        prompt_template: str,
        model: str = "gpt-4o",
        temperature: float = 0.0,
    ) -> None:
        self.prompt_template = prompt_template
        self.model = model
        self.temperature = temperature
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def extract_from_text(self, title: str, abstract: str, doc_id: Optional[str] = None) -> dict:
        prompt = self.prompt_template.format(
            schema=json.dumps(EXTRACTION_SCHEMA, indent=2),
            title=title,
            abstract=abstract,
        )
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            logger.warning(f"JSON decode failed for doc_id={doc_id}; returning raw text")
            result = {"raw_response": response.choices[0].message.content}
        result["doc_id"] = doc_id
        return result

    def extract_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run extraction over a DataFrame with columns: doc_id, title, abstract.
        Returns a DataFrame of extracted features.
        """
        rows = []
        for _, row in df.iterrows():
            logger.debug(f"Extracting features for doc_id={row.get('doc_id', '?')}")
            result = self.extract_from_text(
                title=str(row.get("title", "")),
                abstract=str(row.get("abstract", "")),
                doc_id=str(row.get("doc_id", "")),
            )
            rows.append(result)
        return pd.DataFrame(rows)

    @staticmethod
    def load_prompt_template(prompt_path: Path) -> str:
        return prompt_path.read_text(encoding="utf-8")
