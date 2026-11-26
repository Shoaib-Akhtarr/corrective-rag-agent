# Corrective RAG Ablation Report

- **Corpus:** 50 documents | **Queries:** 8

| Method | Multi-Hop Recall | Distractor Rejection | Faithfulness |
|---|---:|---:|---:|
| Naive TF-IDF Top-5 (No Grader, No Multi-Hop) | 87.1% | 92.9% | N/A |
| BM25 + Multi-Hop Entity Decomposer + Trained ML Distractor Grader | 100.0% | 100.0% | 100.0% |
