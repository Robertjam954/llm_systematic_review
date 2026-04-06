"""
Needle-in-a-Haystack (NIAH) stress test for the Agentic RAG pipeline.

Adapted from: https://github.com/gkamradt/LLMTest_NeedleInAHaystack

Beyond the standard (depth × context_length) grid used for the static RAG pipeline,
the agentic NIAH adds two extra dimensions:

  1. Sub-query targeting  — does the agent's decomposition produce a sub-query that
                            semantically targets the needle topic?
  2. Iteration convergence — which iteration (1, 2, ...) first retrieves the needle chunk?

This lets you answer:
  • Does multi-pass retrieval rescue the needle in contexts where single-pass fails?
  • Does the agent converge faster (fewer iterations) when the needle is shallow?
  • Does the agent's query-refinement step help when the needle is deeply buried?
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger
from openai import OpenAI

# ---------------------------------------------------------------------------
# Domain-specific needle (same corpus as standard RAG NIAH for comparability)
# ---------------------------------------------------------------------------

DEFAULT_NEEDLE = (
    "A landmark RCT (Smith et al., 2019) found that the intervention "
    "reduced all-cause mortality by 42% (95% CI 31–51%, p<0.001), "
    "making it the largest effect size reported in this indication to date."
)
DEFAULT_QUESTION = (
    "What was the largest effect size reported for all-cause mortality "
    "reduction in the systematic review corpus?"
)
DEFAULT_EXPECTED = (
    "Smith et al. (2019) reported a 42% reduction in all-cause mortality "
    "(95% CI 31–51%, p<0.001)."
)

# Biomedical prose haystack filler
_HAYSTACK_FILLER = (
    "Multiple systematic reviews have examined the efficacy of pharmacological "
    "and non-pharmacological interventions across diverse patient populations. "
    "Heterogeneity in study design, outcome measurement, and follow-up duration "
    "limits direct comparability of findings. Meta-analytic pooling was performed "
    "using random-effects models to account for between-study variance. "
    "Publication bias was assessed via funnel plots and Egger's test. "
    "Risk of bias was evaluated independently by two reviewers using the "
    "Cochrane Risk of Bias 2.0 tool. Subgroup analyses stratified by age, "
    "sex, and comorbidity burden revealed no significant interaction effects. "
    "Future high-quality trials with standardised outcome reporting are needed. "
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class NIAHCellResult:
    context_length: int
    depth: float
    subquery_targets_needle: bool
    needle_retrieved: bool
    needle_in_synthesis: bool
    iteration_retrieved: Optional[int]
    n_iterations: int
    faithfulness_score: Optional[float]
    latency_s: float
    answer_snippet: str
    sub_queries: list[str] = field(default_factory=list)

    def passed(self) -> bool:
        return self.needle_retrieved and self.needle_in_synthesis

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_length": self.context_length,
            "depth": self.depth,
            "subquery_targets_needle": self.subquery_targets_needle,
            "needle_retrieved": self.needle_retrieved,
            "needle_in_synthesis": self.needle_in_synthesis,
            "iteration_retrieved": self.iteration_retrieved,
            "n_iterations": self.n_iterations,
            "faithfulness_score": self.faithfulness_score,
            "latency_s": self.latency_s,
            "answer_snippet": self.answer_snippet,
            "sub_queries": self.sub_queries,
            "passed": self.passed(),
        }


# ---------------------------------------------------------------------------
# Core tester
# ---------------------------------------------------------------------------

class AgenticNIAHTester:
    """
    NIAH stress tester for the AgenticRAGAgent.

    Parameters
    ----------
    agent : AgenticRAGAgent
        An initialised agentic RAG agent wrapping a vector store.
    context_lengths : list[int]
        Target haystack sizes in tokens (approx. chars / 4).
    depths : list[float]
        Needle insertion depths as fractions 0.0 (start) → 1.0 (end).
    needle : str
        The factual sentence to hide in the haystack.
    question : str
        The retrieval question whose answer requires the needle.
    expected : str
        Gold-standard answer for faithfulness scoring.
    use_deepeval : bool
        Whether to run DeepEval faithfulness scoring per cell.
    eval_model : str
        Judge model for DeepEval (default: gpt-4o).
    """

    def __init__(
        self,
        agent,
        context_lengths: list[int] | None = None,
        depths: list[float] | None = None,
        needle: str = DEFAULT_NEEDLE,
        question: str = DEFAULT_QUESTION,
        expected: str = DEFAULT_EXPECTED,
        use_deepeval: bool = True,
        eval_model: str = "gpt-4o",
    ) -> None:
        self.agent = agent
        self.context_lengths = context_lengths or [2_000, 8_000, 16_000, 32_000]
        self.depths = depths or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.needle = needle
        self.question = question
        self.expected = expected
        self.use_deepeval = use_deepeval
        self.eval_model = eval_model
        self._openai = OpenAI()

    # ------------------------------------------------------------------
    # Haystack construction
    # ------------------------------------------------------------------

    def _build_haystack(self, target_chars: int, depth: float) -> str:
        """Build a haystack of ~target_chars with needle at depth fraction."""
        reps = max(1, target_chars // len(_HAYSTACK_FILLER) + 2)
        base = (_HAYSTACK_FILLER * reps)[:target_chars]

        insert_pos = int(len(base) * depth)
        snap = base.rfind(". ", 0, insert_pos)
        insert_pos = snap + 2 if snap != -1 else insert_pos

        return base[:insert_pos] + " " + self.needle + " " + base[insert_pos:]

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _needle_in_text(text: str, needle_key: str = "42% (95% CI 31") -> bool:
        return needle_key in text

    def _subquery_targets_needle(self, sub_queries: list[str]) -> bool:
        """
        Check whether any sub-query semantically targets the needle topic.
        Uses a lightweight keyword check + optional embedding similarity.
        """
        needle_keywords = {"mortality", "42%", "effect size", "RCT", "smith", "survival"}
        for sq in sub_queries:
            sq_lower = sq.lower()
            if any(kw in sq_lower for kw in needle_keywords):
                return True
        return False

    def _faithfulness_score(
        self,
        question: str,
        answer: str,
        retrieval_context: list[str],
    ) -> Optional[float]:
        if not self.use_deepeval:
            return None
        try:
            from deepeval import evaluate
            from deepeval.test_case import LLMTestCase
            from deepeval.metrics import FaithfulnessMetric

            metric = FaithfulnessMetric(
                model=self.eval_model, threshold=0.7, include_reason=True
            )
            tc = LLMTestCase(
                input=question,
                actual_output=answer,
                retrieval_context=retrieval_context,
                expected_output=self.expected,
            )
            metric.measure(tc)
            return metric.score
        except Exception as exc:
            logger.warning(f"DeepEval scoring failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Instrumented agent run — captures per-iteration retrieval events
    # ------------------------------------------------------------------

    def _run_agent_instrumented(self, haystack_text: str) -> dict[str, Any]:
        """
        Temporarily inject the haystack document into the agent's vector store,
        run the agent with instrumentation, then remove it.

        Returns a dict with:
          sub_queries, retrieved_doc_ids_per_iter, synthesis,
          iteration_retrieved, n_iterations, retrieval_contexts
        """
        from langchain_core.documents import Document

        # Build a single haystack LangChain Document and inject via the
        # agent's underlying vector_pipeline. If it exposes a LangChain
        # Chroma store we use add_documents; otherwise we wrap a retrieval shim.
        vp = self.agent.vector_pipeline
        doc_ids_injected: list[str] = []

        # Try to access the underlying Chroma store directly
        chroma_store = getattr(vp, "_chroma", None) or getattr(vp, "vs", None)

        if chroma_store is not None:
            hay_doc = Document(
                page_content=haystack_text,
                metadata={
                    "source": "niah_haystack",
                    "title":  "NIAH haystack",
                    "year":   "2024",
                    "doc_id": "niah_tmp",
                },
            )
            # Chunk the haystack through the standard splitter
            try:
                from src.methods.standard_rag.langchain_pipeline import chunk_documents
                hay_chunks = chunk_documents([hay_doc])
            except ImportError:
                hay_chunks = [hay_doc]

            doc_ids_injected = chroma_store.add_documents(hay_chunks)
            logger.debug(f"Injected {len(doc_ids_injected)} haystack chunks into store")

        # Patch the agent's retrieve step to track per-iteration retrieval
        iter_log: list[dict] = []   # [{iter: int, doc_ids: list, texts: list}]
        original_retrieve = self.agent.retrieve

        def instrumented_retrieve(state):
            state = original_retrieve(state)
            iter_log.append({
                "iter": state.iteration,
                "doc_ids": list(state.retrieved_doc_ids),
                "texts": [c.get("text", "") for c in state.retrieved_chunks],
            })
            return state

        self.agent.retrieve = instrumented_retrieve

        try:
            result = self.agent.run(question=self.question)
        finally:
            self.agent.retrieve = original_retrieve
            if doc_ids_injected and chroma_store is not None:
                chroma_store.delete(ids=doc_ids_injected)
                logger.debug("Removed injected haystack chunks")

        # Determine which iteration first retrieved the needle
        iteration_retrieved: Optional[int] = None
        for log_entry in iter_log:
            if any(self._needle_in_text(t) for t in log_entry["texts"]):
                iteration_retrieved = log_entry["iter"]
                break

        all_texts = [t for log in iter_log for t in log["texts"]]

        return {
            "sub_queries":         result.get("sub_queries_final", []),
            "synthesis":           result.get("synthesis", ""),
            "n_iterations":        result.get("n_iterations", 0),
            "iteration_retrieved": iteration_retrieved,
            "retrieval_contexts":  all_texts,
        }

    # ------------------------------------------------------------------
    # Single grid cell
    # ------------------------------------------------------------------

    def run_cell(self, context_length: int, depth: float) -> NIAHCellResult:
        cell_key = f"ctx{context_length}_d{int(depth * 100)}"
        logger.info(f"[AgenticNIAH] {cell_key} — ctx={context_length} depth={depth:.0%}")

        haystack = self._build_haystack(context_length * 4, depth)

        t0 = time.perf_counter()
        run = self._run_agent_instrumented(haystack)
        latency = round(time.perf_counter() - t0, 2)

        subquery_hit = self._subquery_targets_needle(run["sub_queries"])
        needle_retrieved = self._needle_in_text(
            " ".join(run["retrieval_contexts"])
        )
        needle_in_synthesis = self._needle_in_text(run["synthesis"])

        faith = self._faithfulness_score(
            self.question, run["synthesis"], run["retrieval_contexts"][:8]
        )

        result = NIAHCellResult(
            context_length=context_length,
            depth=depth,
            subquery_targets_needle=subquery_hit,
            needle_retrieved=needle_retrieved,
            needle_in_synthesis=needle_in_synthesis,
            iteration_retrieved=run["iteration_retrieved"],
            n_iterations=run["n_iterations"],
            faithfulness_score=faith,
            latency_s=latency,
            answer_snippet=run["synthesis"][:300],
            sub_queries=run["sub_queries"],
        )

        status = "✓ PASS" if result.passed() else "✗ FAIL"
        logger.info(
            f"  {status} | sub_q_hit={subquery_hit} retrieved={needle_retrieved} "
            f"in_synthesis={needle_in_synthesis} iter={run['iteration_retrieved']} "
            f"n_iter={run['n_iterations']} latency={latency}s faith={faith}"
        )
        return result

    # ------------------------------------------------------------------
    # Full sweep
    # ------------------------------------------------------------------

    def run_sweep(self) -> dict[str, NIAHCellResult]:
        """
        Run the full (context_length × depth) grid.
        Returns a dict keyed by 'ctx{N}_d{D}' → NIAHCellResult.
        """
        total = len(self.context_lengths) * len(self.depths)
        results: dict[str, NIAHCellResult] = {}
        run = 0

        for ctx in self.context_lengths:
            for depth in self.depths:
                run += 1
                logger.info(f"[AgenticNIAH] Cell {run}/{total}")
                key = f"ctx{ctx}_d{int(depth * 100)}"
                results[key] = self.run_cell(ctx, depth)

        self._print_heatmap(results)
        return results

    # ------------------------------------------------------------------
    # CLI heatmap
    # ------------------------------------------------------------------

    def _print_heatmap(self, results: dict[str, NIAHCellResult]) -> None:
        depth_labels = [f"{int(d * 100):>4}%" for d in self.depths]
        print("\n── Agentic NIAH Heatmap (✓=pass ✗=fail | iter=retrieval iteration) ──")
        print("ctx\\depth  " + "  ".join(depth_labels))
        for ctx in self.context_lengths:
            row = []
            for d in self.depths:
                key = f"ctx{ctx}_d{int(d * 100)}"
                r = results.get(key)
                if r is None:
                    row.append("  ?  ")
                elif r.passed():
                    it = r.iteration_retrieved or "?"
                    row.append(f" ✓i{it} ")
                else:
                    row.append("  ✗  ")
            print(f"{ctx:>9}  " + "".join(row))
        print()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save(results: dict[str, NIAHCellResult], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v.to_dict() for k, v in results.items()}
        path.write_text(json.dumps(payload, indent=2, default=str))
        logger.info(f"Agentic NIAH results saved → {path}")

    @staticmethod
    def load(path: Path) -> dict[str, dict]:
        return json.loads(path.read_text())

    # ------------------------------------------------------------------
    # Cross-method comparison helper
    # ------------------------------------------------------------------

    @staticmethod
    def compare_with_standard(
        agentic_results: dict[str, NIAHCellResult],
        standard_results_path: Path,
    ) -> pd.DataFrame:
        """
        Produce a cell-by-cell comparison DataFrame.
        standard_results_path: JSON from langchain_pipeline.save_results()
        """
        std = json.loads(standard_results_path.read_text())
        rows = []
        for key, ag in agentic_results.items():
            st = std.get(key, {})
            rows.append({
                "cell": key,
                "context_length": ag.context_length,
                "depth": ag.depth,
                # Standard RAG
                "std_retrieved": st.get("needle_retrieved", None),
                "std_in_answer": st.get("needle_in_answer", None),
                "std_passed":    st.get("needle_retrieved") and st.get("needle_in_answer"),
                "std_latency":   st.get("latency_s"),
                "std_faith":     st.get("faithfulness"),
                # Agentic RAG
                "agt_retrieved":    ag.needle_retrieved,
                "agt_in_synthesis": ag.needle_in_synthesis,
                "agt_passed":       ag.passed(),
                "agt_latency":      ag.latency_s,
                "agt_faith":        ag.faithfulness_score,
                "agt_iter_hit":     ag.iteration_retrieved,
                "agt_n_iter":       ag.n_iterations,
                "agt_subquery_hit": ag.subquery_targets_needle,
                # Delta
                "agentic_advantage": ag.passed() and not (
                    st.get("needle_retrieved") and st.get("needle_in_answer")
                ),
                "standard_advantage": (
                    st.get("needle_retrieved") and st.get("needle_in_answer")
                ) and not ag.passed(),
            })
        return pd.DataFrame(rows)
