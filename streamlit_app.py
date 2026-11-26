import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.graph.crag_workflow import crag_agent
from app.retrieval.hybrid_index import hybrid_index

st.set_page_config(page_title="Temporal Claim-Diff RAG", page_icon="⏳", layout="wide")
st.title("⏳ Temporal & Contradiction-Aware Claim-Diff RAG")
st.caption("Real PDF/MD/TXT Ingestion · PyTorch Neural NLI Contradiction Detection · Authority + Date Conflict Resolution")

with st.sidebar:
    st.header("Ingest Real Documents")
    uploaded = st.file_uploader("Upload PDF, Markdown, or TXT", type=["pdf", "md", "txt"], accept_multiple_files=True)
    if uploaded:
        for uf in uploaded:
            added = hybrid_index.ingest_bytes(uf.name, uf.read())
            st.success(f"Indexed `{uf.name}` ({added} chunks)")

    docs_count = len({c["doc_id"] for c in hybrid_index.chunks})
    st.metric("Documents Indexed", docs_count)
    st.metric("Total Page Chunks", len(hybrid_index.chunks))

presets = [
    "What is the current default vector storage layer in eu-central-2, and which earlier policy did it supersede?",
    "Who authorizes HelixDB encryption key rotations in eu-central-2, and is the chair compliant with EMA DORA?",
    "What GPU unit threshold triggers SLA penalty credits for Meridian AI Labs, and what credit was issued?",
    "Does Apex Cloud Infrastructure have legal liability for Luminos Labs' patents?",
]
chosen = st.selectbox("Sample Contradiction / Multi-Hop Query:", presets)
query = st.text_area("Query across ingested documents:", value=chosen, height=80)

if st.button("Run Claim-Diff Pipeline", type="primary"):
    state = crag_agent.run(query)
    diff = state["claim_diff"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verified Chunks", len(state["graded_docs"]))
    c2.metric("Distractors Rejected", len(state["rejected_distractors"]))
    c3.metric("Contradictions Resolved", diff["contradictions_detected"])
    c4.metric("Grounded", "YES" if state["is_grounded"] else "NO")

    st.markdown(state["draft_answer"])

    if diff["claim_lineage_diff"]:
        st.subheader("Claim Lineage Diff (Truth-Decay Audit)")
        st.dataframe(pd.DataFrame(diff["claim_lineage_diff"]), use_container_width=True)

    with st.expander("Pipeline Audit Trail"):
        for step in state["audit_trail"]:
            st.code(step, language="text")
