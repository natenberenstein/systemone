"""Streamlit support-operations lab. Run: streamlit run app.py"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

import streamlit as st

from demo.data import SAMPLES, Ticket
from demo.laya_gateway import make_gateway
from demo.workflow import build_workflow


st.set_page_config(page_title="Laya + LangGraph | Support Ops Lab", page_icon="🧭", layout="wide")


@st.cache_resource(show_spinner="Loading Laya checkpoint…")
def gateway():
    return make_gateway()


@st.cache_resource(show_spinner=False)
def openai_service():
    from demo.openai_agent import OpenAIService

    return OpenAIService()


st.title("Support Ops Lab")
st.caption("Laya handles bounded decisions; LangGraph invokes a reasoning agent only where the route needs one.")

with st.sidebar:
    st.subheader("Demo settings")
    source = "Remote laya-serve" if os.getenv("LAYA_BASE_URL") else "Local Laya checkpoint"
    st.write(f"**Laya:** {source}")
    st.write(f"**OpenAI:** {'configured' if os.getenv('OPENAI_API_KEY') else 'not configured'}")
    threshold = st.slider("Second-judgment threshold", 0.0, 1.0, 0.85, 0.01,
                          help="If the chosen label probability is below this value, ask OpenAI for a second judgment. This is a demo setting, not a calibrated production cutoff.")
    st.info("Local mode groups the sample reports into one Laya batch. The remote laya-serve API sends one HTTP request per report.")

st.write("Route a mixed queue of synthetic support tickets. The labels are reference answers for this demo, not a production benchmark.")
if st.button("Run Laya triage", type="primary"):
    try:
        with st.spinner("Classifying the queue…"):
            st.session_state.batch = gateway().predict_batch([ticket.report for ticket in SAMPLES])
    except Exception as exc:
        st.error(f"Laya triage failed: {type(exc).__name__}: {exc}")

batch = st.session_state.get("batch")
if batch:
    if len(batch.decisions) != len(SAMPLES):
        st.error("The number of Laya decisions does not match the ticket count.")
        st.stop()
    rows = []
    for ticket, decision in zip(SAMPLES, batch.decisions):
        rows.append({
            "Ticket": ticket.name,
            "Expected": ticket.expected,
            "Laya": decision.choice,
            "Chosen probability": round(decision.answer_confidence, 3),
            "Next step": "OpenAI second judgment" if decision.answer_confidence < threshold else ("OpenAI investigation" if decision.choice == "technical" else f"{decision.choice} queue"),
            "Correct?": decision.choice == ticket.expected,
        })
    accepted = sum(d.answer_confidence >= threshold for d in batch.decisions)
    correct = sum(row["Correct?"] for row in rows)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickets", len(SAMPLES))
    c2.metric("Laya matches", f"{correct}/{len(SAMPLES)}")
    c3.metric("Above threshold", f"{accepted}/{len(SAMPLES)}")
    c4.metric("Batch wall time", f"{batch.wall_seconds:.2f}s")
    st.caption(f"{batch.mode}; {batch.inference_calls} inference request(s). Initial checkpoint loading is included in this first-run timing.")
    st.dataframe(rows, width="stretch", hide_index=True)
    errors = [row["Ticket"] for row in rows if not row["Correct?"]]
    if errors:
        st.warning("Mismatches are visible: " + ", ".join(errors) + ". A confidence cutoff does not guarantee all errors are caught.")

st.divider()
st.subheader("Follow one ticket through LangGraph")
names = [ticket.name for ticket in SAMPLES] + ["Custom report"]
selected = st.selectbox("Ticket", names, index=4)
custom = st.text_area("Custom report", placeholder="Describe a support issue…") if selected == "Custom report" else None
if selected != "Custom report":
    ticket = next(ticket for ticket in SAMPLES if ticket.name == selected)
    st.code(ticket.report, language="text")
else:
    ticket = Ticket("Custom report", (custom or "").strip(), "unknown")

if st.button("Run workflow"):
    if not ticket.report:
        st.warning("Enter a report first.")
    else:
        try:
            with st.spinner("Running Laya and any selected agent step…"):
                prior = batch.decisions[SAMPLES.index(ticket)] if batch and ticket in SAMPLES else None
                ai = openai_service() if os.getenv("OPENAI_API_KEY") else None
                workflow = build_workflow(gateway(), ai.fallback if ai else None, ai.investigate if ai else None)
                result = workflow.invoke({"report": ticket.report, "threshold": threshold, **({"decision": prior} if prior else {})})
                st.session_state.last_result = result
                st.session_state.last_report = ticket.report
        except Exception as exc:
            st.error(f"Workflow failed: {type(exc).__name__}: {exc}")

if st.session_state.get("last_result"):
    result = st.session_state.last_result
    st.write("**Decision trace**")
    st.write(" → ".join(result["steps"]))
    st.write("**Outcome**")
    st.write(result["answer"])
    st.caption(f"Knowledge-base topics: {', '.join(result.get('sources', [])) or 'none'} · Tool calls: {result.get('tool_calls', 0)} · OpenAI stages: {result.get('llm_calls', 0)}")
