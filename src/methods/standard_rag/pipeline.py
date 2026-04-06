"""
Standard (non-agentic) RAG pipeline.
Single query → embed → cosine/ANN retrieval → LLM synthesis.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from openai import OpenAI


class StandardRAGPipeline:
    """
    Thin wrapper around a FAISS vector store for single-pass retrieval + synthesis.

    Usage
    -----
    pipeline = StandardRAGPipeline.from_index(index_path, chunk_index_path)
    result = pipeline.run(query="...", prompt_template=PROMPT, top_k=20)
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        chunk_index: pd.DataFrame,
        embedding_model: str = "text-embedding-3-large",
        llm_model: str = "gpt-4o",
        llm_temperature: float = 0.0,
    ) -> None:
        self.embeddings = embeddings
        self.chunk_index = chunk_index
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @classmethod
    def from_index(cls, embeddings_path: Path, chunk_index_path: Path, **kwargs) -> "StandardRAGPipeline":
        embeddings = np.load(embeddings_path)
        chunk_index = pd.read_csv(chunk_index_path)
        logger.info(f"Loaded {len(embeddings)} chunk embeddings from {embeddings_path}")
        return cls(embeddings=embeddings, chunk_index=chunk_index, **kwargs)

    def embed_query(self, query: str) -> np.ndarray:
        response = self._client.embeddings.create(input=[query], model=self.embedding_model)
        return np.array(response.data[0].embedding, dtype=np.float32)

    def retrieve(self, query: str, top_k: int = 20) -> pd.DataFrame:
        query_vec = self.embed_query(query)
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)
        norms = np.where(norms == 0, 1e-9, norms)
        scores = (self.embeddings @ query_vec) / norms
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = self.chunk_index.iloc[top_idx].copy()
        results["similarity_score"] = scores[top_idx]
        return results.reset_index(drop=True)

    def synthesise(self, query: str, retrieved_chunks: pd.DataFrame, prompt_template: str) -> str:
        context = "\n\n---\n\n".join(
            f"[{row.get('doc_id', i)}] {row.get('text', '')}"
            for i, row in retrieved_chunks.iterrows()
        )
        prompt = prompt_template.format(query=query, context=context)
        response = self._client.chat.completions.create(
            model=self.llm_model,
            temperature=self.llm_temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def run(
        self,
        query: str,
        prompt_template: str,
        top_k: int = 20,
        run_id: Optional[str] = None,
    ) -> dict:
        logger.info(f"StandardRAG run | query='{query[:80]}...' top_k={top_k}")
        retrieved = self.retrieve(query, top_k=top_k)
        synthesis = self.synthesise(query, retrieved, prompt_template)
        return {
            "run_id": run_id,
            "query": query,
            "top_k": top_k,
            "n_retrieved": len(retrieved),
            "retrieved_doc_ids": retrieved["doc_id"].tolist() if "doc_id" in retrieved.columns else [],
            "synthesis": synthesis,
        }
