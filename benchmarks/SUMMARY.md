# Temporal & Contradiction-Aware Claim-Diff RAG Benchmark

- **Ingested PDFs:** 8 multi-page PDFs (12 chunks in Qdrant + BM25 RRF)
- **PyTorch NLI Contradiction Model (`ContractNLI` + `MNLI` holdout, N=3945):** Test Accuracy = **0.6383**, Macro-F1 = **0.6321** (ContractNLI-only Acc = **0.7906**)
- **Authoritative Doc Recall:** **100.0%**
- **Contradiction Detection Rate:** **100.0%**
- **Distractor Rejection Rate:** **100.0%**
- **Grounding Faithfulness:** **100.0%**
