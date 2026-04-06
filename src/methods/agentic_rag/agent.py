"""
Agentic RAG pipeline using LangGraph.

The agent:
  1. Decomposes the review question into sub-queries (chain-of-thought)
  2. Iteratively retrieves chunks from the vector store and/or PubMed API
  3. Evaluates retrieved content for relevance
  4. Decides whether to refine, expand, or stop
  5. Synthesises final answer
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional

from langchain_openai import ChatOpenAI
from loguru import logger


@dataclass
class AgentState:
    question: str
    sub_queries: list[str] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    iteration: int = 0
    synthesis: str = ""
    done: bool = False


class AgenticRAGAgent:
    """
    LangGraph-based agentic RAG agent.

    Parameters
    ----------
    vector_pipeline : StandardRAGPipeline
        A pre-loaded standard RAG pipeline used for vector retrieval.
    max_iterations : int
        Maximum agent loop iterations (default from env AGENT_MAX_ITERATIONS).
    max_subqueries : int
        Maximum sub-queries per decomposition step.
    """

    def __init__(
        self,
        vector_pipeline,
        max_iterations: Optional[int] = None,
        max_subqueries: Optional[int] = None,
        llm_model: str = "gpt-4o",
        temperature: float = 0.0,
    ) -> None:
        self.vector_pipeline = vector_pipeline
        self.max_iterations = max_iterations or int(os.getenv("AGENT_MAX_ITERATIONS", 10))
        self.max_subqueries = max_subqueries or int(os.getenv("AGENT_MAX_SUBQUERIES", 5))
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)

    def decompose(self, state: AgentState) -> AgentState:
        """Step 1 — decompose review question into sub-queries."""
        prompt = (
            "You are a systematic review expert. "
            "Decompose the following review question into at most "
            f"{self.max_subqueries} specific sub-queries that together "
            "cover the topic comprehensively. "
            "Return ONLY a numbered list of sub-queries, one per line.\n\n"
            f"Review question: {state.question}"
        )
        response = self.llm.invoke(prompt)
        lines = [l.strip() for l in response.content.strip().split("\n") if l.strip()]
        state.sub_queries = [l.lstrip("0123456789). ") for l in lines][: self.max_subqueries]
        logger.info(f"Decomposed into {len(state.sub_queries)} sub-queries")
        return state

    def retrieve(self, state: AgentState) -> AgentState:
        """Step 2 — retrieve chunks for all current sub-queries."""
        for sq in state.sub_queries:
            results = self.vector_pipeline.retrieve(sq, top_k=10)
            for _, row in results.iterrows():
                doc_id = row.get("doc_id", "")
                if doc_id and doc_id not in state.retrieved_doc_ids:
                    state.retrieved_doc_ids.append(doc_id)
                    state.retrieved_chunks.append(row.to_dict())
        logger.info(
            f"Iteration {state.iteration}: {len(state.retrieved_doc_ids)} unique docs retrieved"
        )
        return state

    def evaluate_and_refine(self, state: AgentState) -> AgentState:
        """Step 3 — evaluate coverage; decide to refine or stop."""
        context_sample = "\n".join(
            c.get("text", "")[:300] for c in state.retrieved_chunks[:10]
        )
        prompt = (
            "You are a systematic review expert. "
            "Given the review question and a sample of retrieved content, "
            "identify any important sub-topics NOT yet covered. "
            "If coverage is sufficient, respond with STOP. "
            "Otherwise, provide up to 3 additional sub-queries (numbered list).\n\n"
            f"Review question: {state.question}\n\n"
            f"Retrieved content sample:\n{context_sample}"
        )
        response = self.llm.invoke(prompt)
        content = response.content.strip()
        if "STOP" in content.upper() or state.iteration >= self.max_iterations:
            state.done = True
            logger.info(f"Agent stopping at iteration {state.iteration}")
        else:
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            new_queries = [l.lstrip("0123456789). ") for l in lines][:3]
            state.sub_queries = new_queries
            logger.info(f"Refined with {len(new_queries)} new sub-queries")
        state.iteration += 1
        return state

    def synthesise(self, state: AgentState) -> AgentState:
        """Step 4 — synthesise final answer from all retrieved chunks."""
        context = "\n\n---\n\n".join(
            f"[{c.get('doc_id', i)}] {c.get('text', '')}"
            for i, c in enumerate(state.retrieved_chunks[:40])
        )
        prompt = (
            "You are a systematic review expert. "
            "Based on the retrieved literature below, provide a comprehensive "
            "synthesis answering the review question. "
            "Cite sources using their doc_id in brackets.\n\n"
            f"Review question: {state.question}\n\n"
            f"Retrieved literature:\n{context}"
        )
        response = self.llm.invoke(prompt)
        state.synthesis = response.content
        return state

    def run(self, question: str, run_id: Optional[str] = None) -> dict:
        state = AgentState(question=question)
        state = self.decompose(state)
        while not state.done:
            state = self.retrieve(state)
            state = self.evaluate_and_refine(state)
        state = self.synthesise(state)
        logger.info(
            f"AgenticRAG complete | {len(state.retrieved_doc_ids)} docs | "
            f"{state.iteration} iterations"
        )
        return {
            "run_id": run_id,
            "question": question,
            "n_iterations": state.iteration,
            "n_retrieved": len(state.retrieved_doc_ids),
            "retrieved_doc_ids": state.retrieved_doc_ids,
            "sub_queries_final": state.sub_queries,
            "synthesis": state.synthesis,
        }
