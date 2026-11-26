import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.graph.crag_workflow import crag_agent
from app.retrieval.store import store
from app.evaluate import run_benchmark

st.set_page_config(page_title="Corrective RAG (CRAG) Agent", page_icon="🛡️", layout="wide")
st.title("🛡️ Self-Reflective Corrective RAG (CRAG) Agent")
st.caption("Multi-Hop Entity Decomposition · Adversarial Distractor Rejection · Hallucination Verification")

with st.sidebar:
    st.header("Corpus & Benchmark")
    st.metric("Indexed Documents", len(store.documents))
    st.metric("Benchmark Queries", len(store.benchmark_queries))
    if st.button("Run Full CRAG Benchmark", use_container_width=True):
        with st.spinner("Running MultiHop-RAG benchmark..."):
            report = run_benchmark()
        st.success("Benchmark Complete")
        st.metric("Multi-Hop Doc Recall", f"{report['multi_hop_doc_recall']}%")
        st.metric("Distractor Rejection", f"{report['distractor_rejection_rate']}%")
        st.metric("Faithfulness", f"{report['grounding_faithfulness']}%")
        st.metric("Avg Latency", f"{report['avg_latency_ms']} ms")

preset_queries = [q["query"] for q in store.benchmark_queries]
selected_preset = st.selectbox("Choose a Multi-Hop Benchmark Query:", ["Custom..."] + preset_queries)
query = st.text_area(
    "Enter your multi-hop query:",
    value=preset_queries[0] if selected_preset == "Custom..." else selected_preset,
    height=80,
)

if st.button("Run Corrective RAG Pipeline", type="primary"):
    state = crag_agent.run(query)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hops Executed", state["hop_count"])
    c2.metric("Verified Docs", len(state["graded_docs"]))
    c3.metric("Distractors Rejected", len(state["rejected_distractors"]))
    c4.metric("Faithfulness", f"{state['faithfulness_score'] * 100:.0f}%")

    if state["is_grounded"]:
        st.success("Hallucination Gate: GROUNDED (all citations verified)")
    else:
        st.error("Hallucination Gate: UNGROUNDED")

    st.subheader("Synthesized Answer")
    st.markdown(state["draft_answer"])

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Verified Supporting Documents")
        for doc in state["graded_docs"]:
            with st.expander(f"[{doc['doc_id']}] {doc['title']} ({doc['category']})"):
                st.write(doc["content"])

    with col_right:
        st.subheader("Rejected Adversarial Distractors")
        if state["rejected_distractors"]:
            for did in state["rejected_distractors"]:
                st.warning(f"Rejected `{did}` (domain mismatch / distractor)")
        else:
            st.info("No distractors triggered in top-k candidates.")

        st.subheader("Audit Trail")
        for step in state["audit_trail"]:
            st.code(step, language="text")
