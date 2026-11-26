# ⏳ Temporal & Contradiction-Aware Claim-Diff RAG (`corrective-rag-agent`)

[![Python 3.11](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Vector%20Store-Qdrant-red.svg?logo=qdrant)](https://qdrant.tech/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Siamese%20NLI-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-5%20Passed-brightgreen)](tests/)

Production **LangGraph Corrective RAG Engine** solving enterprise **truth-decay** across multi-version PDFs, regulatory rulings, board resolutions, and contract amendments.

## Problem Solved
Naive RAG chunks conflicting versions of corporate policies together and averages outdated rules with binding superseding directives. `corrective-rag-agent` retrieves across real PDFs via **Qdrant Hybrid Search + BM25 RRF**, rejects entity-trap distractors via domain-agnostic pairwise grading, runs a trained **PyTorch Siamese NLI model** over competing claims, resolves conflicts via **Authority Tier (`REGULATORY_BINDING > BOARD_RESOLUTION > EXECUTED_CONTRACT > INTERNAL_MEMO`) + Effective Date**, and synthesizes citation-locked LLM answers with an automated **LangGraph self-correction retry gate**.

## System Architecture
1. **PDF Ingestion & Qdrant Hybrid Index ([`app/ingestion/doc_loader.py`](app/ingestion/doc_loader.py), [`app/retrieval/hybrid_index.py`](app/retrieval/hybrid_index.py)):** Multi-page `pypdf` ingestion, Qdrant cosine vector search (`384-d` `all-MiniLM-L6-v2` / SVD fallback) + payload authority/date filtering fused with `BM25Okapi` via Reciprocal Rank Fusion ($k=60$).
2. **LangGraph Cyclic Workflow ([`app/graph/crag_workflow.py`](app/graph/crag_workflow.py)):**
   - **Pairwise ML Distractor Grader ([`training/train_distractor_grader.py`](training/train_distractor_grader.py)):** Unseen-document `GroupKFold` classifier with domain-agnostic predicate-vs-entity overlap features (zero filename or entity hardcoding).
   - **Generic Multi-Hop Decomposition:** LLM JSON entity extraction with statistical NP/identifier fallback.
   - **LLM Grounding & Self-Correction Loop ([`app/llm/client.py`](app/llm/client.py)):** Groq/OpenAI/Ollama synthesis + SSE streaming (`/api/v1/query/stream`) + post-generation `[doc_id p.X]` citation & numeric verification with bounded retry.
3. **PyTorch Siamese NLI Contradiction Engine ([`training/train_contradiction_nli.py`](training/train_contradiction_nli.py), [`app/claims/contradiction_engine.py`](app/claims/contradiction_engine.py)):** Trained on **real open benchmarks** (`kiddothe2b/contract-nli` + `nyu-mll/multi_nli` `promptID`-disjoint split) with zero synthetic templates or train/test leakage.

## Verified Benchmark Results (`benchmarks/ablation_results.json`)
- **PyTorch Siamese NLI Holdout (`N = 3,945` unseen pairs across ContractNLI + MultiNLI):**
  - ContractNLI Holdout Accuracy: **0.7906** | Combined 3-Class Test Accuracy: **0.6383** (Macro-F1: **0.6321**)
- **Unseen-Document Pairwise Distractor Grader (`GroupKFold`):** Accuracy = **0.9167**, F1 = **0.8372**
- **End-to-End Claim-Diff Pipeline (`groq/qwen/qwen3.8-27b`, 8 multi-page PDFs):**
  - Authoritative Doc Recall: **100.0%** | Contradiction Detection Rate: **100.0%**
  - Distractor Rejection Rate: **100.0%** | Grounding Faithfulness: **100.0%** | Avg Latency: **1891.55 ms**

## Quickstart
```bash
pip install -r requirements.txt
python training/train_contradiction_nli.py
python training/train_distractor_grader.py
python benchmarks/run_ablation.py
pytest -v tests/
```

Run FastAPI + Streamlit (or Docker Compose with Qdrant):
```bash
uvicorn app.main:app --port 8001
streamlit run streamlit_app.py
# or: docker compose up --build
```
