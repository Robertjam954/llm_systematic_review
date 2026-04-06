"""
Systematic Review — Standard RAG Pipeline
==========================================
Stack (per LangChain docs, April 2026):
  • langchain + langchain-text-splitters  (orchestration)
  • langchain-openai                       (embeddings + chat model)
  • langchain-chroma                       (vector store)
  • deepeval                               (Confident AI evaluation framework)

Pipeline:
  1. Ingest   → PubMedLoader / PyPDFDirectoryLoader / JSON
  2. Split    → RecursiveCharacterTextSplitter (section-aware separators)
  3. Store    → Chroma vector store  (text-embedding-3-small, persistent)
  4. Retrieve → similarity_search, k=8
  5. Generate → gpt-4o, temperature=0, structured review prompt
  6. Evaluate → DeepEval: Faithfulness, AnswerRelevancy, ContextualPrecision,
                          ContextualRecall, ContextualRelevancy
  7. NIAH     → Needle-in-a-Haystack: sweeps depth × context-length grid
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import os
import json
import random
import time
from pathlib import Path
from typing import Any

# ── env ─────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# ── LangChain (current import paths as of langchain 0.3.x) ─────────────────
#   Ref: https://docs.langchain.com/oss/python/langchain/rag
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    PubMedLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── DeepEval / Confident AI ─────────────────────────────────────────────────
#   Ref: https://deepeval.com/docs/getting-started-rag
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CHROMA_DIR      = "./chroma_db"
COLLECTION      = "systematic_review"
EMBED_MODEL     = "text-embedding-3-small"   # 1536-dim, $0.02/1M tokens
LLM_MODEL       = "gpt-4o"
TOP_K           = 8
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150

# DeepEval judge model (can differ from pipeline LLM)
EVAL_MODEL      = "gpt-4o"

# NIAH sweep grid
NIAH_CONTEXT_LENGTHS = [2_000, 8_000, 16_000, 32_000]   # tokens (approx chars/4)
NIAH_DEPTHS          = [0.1, 0.25, 0.5, 0.75, 0.9]      # fraction through haystack


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — INGEST
# ─────────────────────────────────────────────────────────────────────────────

def load_from_pubmed(query: str, max_docs: int = 100) -> list[Document]:
    """
    Pull abstracts from PubMed via Entrez.
    Use the same Boolean string as your gold-standard DB search for alignment.

    LangChain docs → Document Loaders → PubMedLoader
    https://docs.langchain.com/oss/python/integrations/document_loaders/pubmed
    """
    loader = PubMedLoader(query=query, load_max_docs=max_docs)
    docs = loader.load()
    print(f"[ingest:pubmed] {len(docs)} abstracts loaded")
    return docs


def load_from_pdfs(pdf_dir: str) -> list[Document]:
    """
    Bulk-load a directory of PDFs (downloaded from Embase / WoS / Cochrane).
    LangChain docs → Document Loaders → PyPDFDirectoryLoader
    """
    loader = PyPDFDirectoryLoader(pdf_dir, glob="**/*.pdf")
    docs = loader.load()
    print(f"[ingest:pdf] {len(docs)} pages loaded from {pdf_dir}")
    return docs


def load_from_json(json_path: str) -> list[Document]:
    """
    Load from a Covidence / Rayyan JSON export.
    Expected: [{"title": str, "abstract": str, "authors": str, "year": int}, ...]
    """
    records = json.loads(Path(json_path).read_text())
    docs = [
        Document(
            page_content=f"Title: {r.get('title','')}\n\nAbstract: {r.get('abstract','')}",
            metadata={
                "title":   r.get("title", ""),
                "authors": r.get("authors", ""),
                "year":    str(r.get("year", "")),
                "source":  "json_export",
            },
        )
        for r in records
    ]
    print(f"[ingest:json] {len(docs)} records loaded from {json_path}")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — CHUNK
# Section-aware splitting: tries biomedical section headers before
# falling back to paragraph → sentence → word.
# LangChain docs → Text Splitters → RecursiveCharacterTextSplitter
# https://docs.langchain.com/oss/python/langchain/retrieval#text_splitters
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_SEPS = [
    "\nAbstract\n", "\nINTRODUCTION\n", "\nIntroduction\n",
    "\nMETHODS\n",  "\nMethods\n",       "\nMaterials and Methods\n",
    "\nRESULTS\n",  "\nResults\n",
    "\nDISCUSSION\n", "\nDiscussion\n",
    "\nCONCLUSION\n", "\nConclusion\n",
    "\nREFERENCES\n", "\n\n", "\n", ". ", " ",
]

def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=_SECTION_SEPS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    # tag each chunk with its index within its source
    counters: dict[str, int] = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        counters[src] = counters.get(src, 0) + 1
        c.metadata["chunk_index"] = counters[src]
    print(f"[chunk] {len(docs)} docs → {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 + 4 — EMBED + STORE
# LangChain docs → Vector Stores → Chroma
# https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
# ─────────────────────────────────────────────────────────────────────────────

def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)


def build_vector_store(chunks: list[Document]) -> Chroma:
    """Embed and persist all chunks. Run once; use load_vector_store() after."""
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings(),
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    print(f"[store] {vs._collection.count()} vectors persisted → {CHROMA_DIR}")
    return vs


def load_vector_store() -> Chroma:
    """Load existing ChromaDB collection from disk (no re-embedding)."""
    vs = Chroma(
        collection_name=COLLECTION,
        embedding_function=_embeddings(),
        persist_directory=CHROMA_DIR,
    )
    print(f"[store] Loaded {vs._collection.count()} vectors from {CHROMA_DIR}")
    return vs


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — RAG CHAIN (LCEL)
# Built with LangChain Expression Language (LCEL) as shown in the official
# LangChain RAG tutorial (two-step RAG chain variant).
# Ref: https://docs.langchain.com/oss/python/langchain/rag#rag-chains
# ─────────────────────────────────────────────────────────────────────────────

_REVIEW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a systematic review researcher.\n"
        "Answer using ONLY the retrieved study excerpts in <context>.\n"
        "For each claim, cite the source title and year in parentheses.\n"
        "If the evidence is insufficient, say so — do not speculate.\n\n"
        "Structure your answer as:\n"
        "1. Summary of evidence (2–3 sentences)\n"
        "2. Key findings (bullet list, cited)\n"
        "3. Gaps and limitations\n"
        "4. PICO elements for formal meta-analysis\n\n"
        "<context>\n{context}\n</context>"
    )),
    ("human", "{question}"),
])


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{doc.metadata.get('title', doc.metadata.get('source', 'unknown'))} "
        f"({doc.metadata.get('year', 'n.d.')})] {doc.page_content}"
        for doc in docs
    )


def build_rag_chain(vs: Chroma):
    """
    LCEL two-step RAG chain:
      retriever → format_docs ─┐
                               ├→ prompt → llm → str
      passthrough ─────────────┘

    Returns a callable: chain.invoke({"question": "..."})
    Also returns the retriever for direct use in evaluation.
    """
    retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | _REVIEW_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def run_query(chain, retriever, question: str) -> dict[str, Any]:
    """
    Run a question through the RAG chain.
    Returns answer, retrieval_context (list[str]), and source metadata.
    """
    answer = chain.invoke(question)
    retrieved_docs = retriever.invoke(question)
    retrieval_context = [d.page_content for d in retrieved_docs]
    sources = [
        {
            "title":   d.metadata.get("title", d.metadata.get("source", "?")),
            "year":    d.metadata.get("year", ""),
            "excerpt": d.page_content[:200] + "…",
        }
        for d in retrieved_docs
    ]
    return {
        "question":          question,
        "answer":            answer,
        "retrieval_context": retrieval_context,   # kept as list[str] for DeepEval
        "sources":           sources,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — EVALUATION WITH DeepEval / CONFIDENT AI
#
# 5 core RAG metrics (LLM-as-judge, gpt-4o):
#   • Faithfulness       — claims in answer supported by retrieval context
#   • AnswerRelevancy    — answer addresses the question
#   • ContextualPrecision  — relevant chunks ranked higher than irrelevant ones
#   • ContextualRecall   — retrieval context covers the expected answer
#   • ContextualRelevancy — retrieval context is relevant to the question
#
# Ref: https://deepeval.com/docs/getting-started-rag
#      https://deepeval.com/docs/metrics-faithfulness
#      https://deepeval.com/docs/metrics-contextual-recall
# ─────────────────────────────────────────────────────────────────────────────

def make_metrics(threshold: float = 0.7) -> list:
    kwargs = dict(model=EVAL_MODEL, threshold=threshold, include_reason=True)
    return [
        FaithfulnessMetric(**kwargs),
        AnswerRelevancyMetric(**kwargs),
        ContextualPrecisionMetric(**kwargs),
        ContextualRecallMetric(**kwargs),
        ContextualRelevancyMetric(**kwargs),
    ]


def build_test_case(
    result: dict[str, Any],
    expected_output: str | None = None,
) -> LLMTestCase:
    """
    Construct a DeepEval LLMTestCase from a RAG run result.

    expected_output is required for ContextualPrecision and ContextualRecall;
    provide the ideal answer (from a gold-standard screened dataset if available).
    """
    return LLMTestCase(
        input=result["question"],
        actual_output=result["answer"],
        retrieval_context=result["retrieval_context"],
        expected_output=expected_output or result["answer"],  # fallback to actual
    )


def evaluate_pipeline(
    chain,
    retriever,
    qa_pairs: list[dict],           # [{"question": str, "expected": str}, ...]
    threshold: float = 0.7,
) -> dict[str, Any]:
    """
    Run end-to-end evaluation across a dataset of QA pairs.

    qa_pairs: list of dicts with keys 'question' and optionally 'expected'
    Returns aggregate scores per metric and per-case detail.
    """
    metrics = make_metrics(threshold)
    test_cases: list[LLMTestCase] = []

    print(f"\n[eval] Running {len(qa_pairs)} test cases …")
    for pair in qa_pairs:
        result = run_query(chain, retriever, pair["question"])
        tc = build_test_case(result, expected_output=pair.get("expected"))
        test_cases.append(tc)

    # evaluate() runs all metrics against all test cases, parallel by default
    eval_result = evaluate(test_cases=test_cases, metrics=metrics)

    # aggregate scores
    agg: dict[str, list[float]] = {}
    for tc in test_cases:
        for m in metrics:
            name = type(m).__name__
            agg.setdefault(name, []).append(m.score or 0.0)

    summary = {name: round(sum(scores) / len(scores), 4)
               for name, scores in agg.items()}

    print("\n[eval] Mean scores:")
    for name, score in summary.items():
        status = "✓" if score >= threshold else "✗"
        print(f"  {status} {name}: {score:.3f}")

    return {"summary": summary, "eval_result": eval_result}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7 — NEEDLE IN A HAYSTACK (NIAH)
#
# Concept (Kamradt, 2023):
#   Insert a specific factual "needle" sentence at depth D% into a background
#   "haystack" of unrelated text. Ask the pipeline to retrieve the needle.
#   Sweep across context lengths × depths to build a 2-D performance heatmap.
#
# In a RAG context, this tests:
#   a) Whether the RETRIEVER surfaces the chunk containing the needle.
#   b) Whether the GENERATOR correctly quotes / uses the needle in its answer.
#
# Ref: https://github.com/gkamradt/LLMTest_NeedleInAHaystack
#      https://blog.langchain.com/multi-needle-in-a-haystack/
# ─────────────────────────────────────────────────────────────────────────────

# Domain-appropriate needle for a systematic review corpus
NIAH_NEEDLE = (
    "A landmark RCT (Smith et al., 2019) found that the intervention "
    "reduced all-cause mortality by 42% (95% CI 31–51%, p<0.001), "
    "making it the largest effect size reported in this indication to date."
)
NIAH_QUESTION = (
    "What was the largest effect size reported for all-cause mortality "
    "reduction in the systematic review corpus?"
)
NIAH_EXPECTED = (
    "Smith et al. (2019) reported a 42% reduction in all-cause mortality "
    "(95% CI 31–51%, p<0.001)."
)

# Background haystack text: repeated filler mimicking biomedical prose
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


def _build_haystack(target_chars: int, needle: str, depth: float) -> str:
    """
    Build a haystack of approximately `target_chars` characters,
    with `needle` inserted at `depth` fraction (0.0 = start, 1.0 = end).
    """
    filler_reps = max(1, target_chars // len(_HAYSTACK_FILLER) + 2)
    base = _HAYSTACK_FILLER * filler_reps
    base = base[:target_chars]

    insert_pos = int(len(base) * depth)
    # snap to nearest sentence boundary
    snap = base.rfind(". ", 0, insert_pos)
    insert_pos = snap + 2 if snap != -1 else insert_pos

    haystack = base[:insert_pos] + " " + needle + " " + base[insert_pos:]
    return haystack


def _needle_retrieved(retrieval_context: list[str], needle: str) -> bool:
    """Return True if any retrieved chunk contains the needle (substring match)."""
    needle_key = "42% (95% CI 31"     # unique numeric fragment
    return any(needle_key in chunk for chunk in retrieval_context)


def _needle_in_answer(answer: str) -> bool:
    """Return True if the answer correctly references the needle fact."""
    return "42%" in answer and "Smith" in answer


def run_niah_test(
    vs: Chroma,
    chain,
    retriever,
    context_lengths: list[int] = NIAH_CONTEXT_LENGTHS,
    depths: list[float] = NIAH_DEPTHS,
    needle: str = NIAH_NEEDLE,
    question: str = NIAH_QUESTION,
    expected: str = NIAH_EXPECTED,
    use_deepeval: bool = True,
) -> dict[str, Any]:
    """
    Full NIAH sweep across (context_length × depth) grid.

    For each cell:
      1. Build a haystack document of `context_length` chars with needle at `depth`.
      2. Add it as a temporary document to ChromaDB.
      3. Query the pipeline with the retrieval question.
      4. Record: needle_retrieved (bool), needle_in_answer (bool),
                 latency (s), and optionally DeepEval Faithfulness score.
      5. Remove the temporary document.

    Returns a results dict keyed by (context_length, depth).
    """
    results: dict[str, Any] = {}
    faithfulness_metric = FaithfulnessMetric(
        model=EVAL_MODEL, threshold=0.7, include_reason=True
    ) if use_deepeval else None

    total = len(context_lengths) * len(depths)
    run = 0

    for ctx_len in context_lengths:
        for depth in depths:
            run += 1
            cell_key = f"ctx{ctx_len}_d{int(depth*100)}"
            print(f"\n[niah] {run}/{total} — ctx_len={ctx_len} chars, depth={depth:.0%}")

            # ── 1. Build haystack document ───────────────────────────────────
            haystack_text = _build_haystack(ctx_len * 4, needle, depth)
            haystack_doc = Document(
                page_content=haystack_text,
                metadata={
                    "source":      "niah_haystack",
                    "title":       f"NIAH haystack ctx={ctx_len} d={depth}",
                    "year":        "2024",
                    "niah_cell":   cell_key,
                },
            )

            # ── 2. Temporarily add to vector store ───────────────────────────
            # Split the haystack so it actually goes through the chunker
            hay_chunks = chunk_documents([haystack_doc])
            doc_ids = vs.add_documents(hay_chunks)

            # ── 3. Query ─────────────────────────────────────────────────────
            t0 = time.perf_counter()
            result = run_query(chain, retriever, question)
            latency = round(time.perf_counter() - t0, 3)

            # ── 4. Score ─────────────────────────────────────────────────────
            retrieved = _needle_retrieved(result["retrieval_context"], needle)
            in_answer = _needle_in_answer(result["answer"])

            faith_score: float | None = None
            if use_deepeval and faithfulness_metric:
                tc = LLMTestCase(
                    input=question,
                    actual_output=result["answer"],
                    retrieval_context=result["retrieval_context"],
                    expected_output=expected,
                )
                faithfulness_metric.measure(tc)
                faith_score = faithfulness_metric.score

            results[cell_key] = {
                "context_length": ctx_len,
                "depth":          depth,
                "needle_retrieved": retrieved,
                "needle_in_answer": in_answer,
                "faithfulness":     faith_score,
                "latency_s":        latency,
                "answer_snippet":   result["answer"][:300],
            }

            status = "✓ PASS" if (retrieved and in_answer) else "✗ FAIL"
            print(
                f"  {status} | retrieved={retrieved} in_answer={in_answer} "
                f"latency={latency}s faith={faith_score}"
            )

            # ── 5. Remove temporary docs ──────────────────────────────────────
            vs.delete(ids=doc_ids)

    _print_niah_heatmap(results, context_lengths, depths)
    return results


def _print_niah_heatmap(
    results: dict[str, Any],
    context_lengths: list[int],
    depths: list[float],
) -> None:
    """Print a simple ASCII heatmap of NIAH pass/fail."""
    print("\n── NIAH Heatmap (✓=pass, ✗=fail) ──")
    depth_labels = [f"{int(d*100):>4}%" for d in depths]
    print("ctx\\depth  " + "  ".join(depth_labels))
    for ctx in context_lengths:
        row = []
        for d in depths:
            key = f"ctx{ctx}_d{int(d*100)}"
            r = results.get(key, {})
            ok = r.get("needle_retrieved") and r.get("needle_in_answer")
            row.append("  ✓ " if ok else "  ✗ ")
        print(f"{ctx:>9}  " + "".join(row))
    print()


def save_results(results: dict[str, Any], path: str = "niah_results.json") -> None:
    Path(path).write_text(json.dumps(results, indent=2, default=str))
    print(f"[niah] Results saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. Ingest ─────────────────────────────────────────────────────────
    pubmed_query = (
        "(machine learning[MH] OR deep learning[TW] OR artificial intelligence[MH]) "
        "AND (clinical diagnosis[MH] OR diagnostic accuracy[MH]) "
        "AND (2018:2024[dp])"
    )
    docs = load_from_pubmed(pubmed_query, max_docs=100)
    # docs = load_from_pdfs("./papers")
    # docs = load_from_json("./covidence_export.json")

    # ── 2. Chunk ──────────────────────────────────────────────────────────
    chunks = chunk_documents(docs)

    # ── 3+4. Embed + Store (comment out after first run) ──────────────────
    vs = build_vector_store(chunks)
    # vs = load_vector_store()

    # ── 5. Build chain ────────────────────────────────────────────────────
    chain, retriever = build_rag_chain(vs)

    # ── Quick retrieval test ──────────────────────────────────────────────
    test_q = "What is the diagnostic accuracy of ML models for early disease detection?"
    result = run_query(chain, retriever, test_q)
    print("\n" + "="*60)
    print("ANSWER:", result["answer"][:500])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  • {s['title']} ({s['year']})")

    # ── 6. DeepEval evaluation ────────────────────────────────────────────
    # Provide expected outputs from your gold-standard screened dataset.
    # Fallback: expected = actual_output (tests faithfulness & relevancy only).
    qa_pairs = [
        {
            "question": "What is the diagnostic accuracy of ML for early detection?",
            "expected": (
                "Machine learning models for early disease detection report "
                "AUC values between 0.80 and 0.97 across included studies."
            ),
        },
        {
            "question": "What study designs are most commonly reported?",
            "expected": (
                "Retrospective cohort studies and case-control designs dominate "
                "the included literature."
            ),
        },
        {
            "question": "What limitations do included studies most often report?",
            "expected": (
                "Small sample sizes, single-centre designs, and lack of external "
                "validation are the most commonly cited limitations."
            ),
        },
    ]

    eval_summary = evaluate_pipeline(chain, retriever, qa_pairs, threshold=0.7)
    print("\n[eval] Aggregate scores:", eval_summary["summary"])

    # ── 7. Needle-in-a-Haystack ───────────────────────────────────────────
    niah_results = run_niah_test(
        vs=vs,
        chain=chain,
        retriever=retriever,
        context_lengths=NIAH_CONTEXT_LENGTHS,
        depths=NIAH_DEPTHS,
        use_deepeval=True,
    )
    save_results(niah_results)
