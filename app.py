"""Streamlit Laya + LangGraph lab. Run: streamlit run app.py"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

import streamlit as st

from demo.coding_data import CASES, FIXTURE_ROOT
from demo.coding_workflow import build_coding_workflow, compare_batch, compare_complete, stream_result
from demo.data import SAMPLES, Ticket
from demo.laya_gateway import make_gateway
from demo.workflow import build_workflow

st.set_page_config(page_title="System One Lab | Laya + LangGraph", page_icon="🧭", layout="wide")


@st.cache_resource(show_spinner="Loading Laya checkpoint…")
def gateway():
    return make_gateway()


@st.cache_resource(show_spinner=False)
def support_ai():
    from demo.openai_agent import OpenAIService
    return OpenAIService()


@st.cache_resource(show_spinner=False)
def coding_ai():
    from demo.coding_agent import CodingAI
    return CodingAI()


def graph_status(name: str, update: dict, status) -> None:
    """Show real LangGraph node updates, not a trace composed afterward."""
    step = update.get("steps", [name])[-1]
    status.write(f"**{name}** · {step}")


def render_coding() -> None:
    st.subheader("Coding-agent failure triage")
    st.write("Laya chooses a bounded CI investigation route. LangGraph sends reproducible code failures to an OpenAI agent with read-only repository and test tools.")
    threshold = st.slider("Coding fallback probability", 0.0, 1.0, 0.55, 0.01, key="coding_threshold",
                          help="Exploratory cutoff for this fixture. It was not calibrated on held-out failures.")
    st.caption("Three cart tests genuinely fail; the environment, flaky-test, and truncated-log examples are synthetic summaries. Predictions and errors remain visible.")
    if st.button("Compare triage on all 8 failures", type="primary", key="coding_batch"):
        try:
            with st.spinner("Running a Laya batch, keyword rules, and one OpenAI classification per case…"):
                st.session_state.coding_batch_result = (threshold, compare_batch(
                    gateway(), coding_ai() if os.getenv("OPENAI_API_KEY") else None, threshold))
        except Exception as exc:
            st.error(f"Comparison failed: {type(exc).__name__}: {exc}")
    saved_comparison = st.session_state.get("coding_batch_result")
    comparison = saved_comparison[1] if saved_comparison and saved_comparison[0] == threshold else None
    if saved_comparison and comparison is None:
        st.info("The cutoff changed. Run the triage comparison again to update the results.")
    if comparison:
        st.dataframe(comparison["summary"], hide_index=True, width="stretch")
        st.dataframe(comparison["rows"], hide_index=True, width="stretch")
        with st.expander("Cutoff sweep on these 8 examples"):
            st.dataframe(comparison["threshold_sweep"], hide_index=True, width="stretch")
            st.caption("This is an in-sample diagnostic, not a calibrated cutoff. Validate a policy on separate failures before deployment.")
        st.caption(
            f"Laya batch: {comparison['laya_batch_seconds']:.2f}s, {comparison['laya_requests']} request(s). "
            "The fallback lane reuses the OpenAI-only classifications to avoid duplicate billing. "
            "Its time is a sum of observed stage durations, not an independent end-to-end wall time."
        )

    st.divider()
    selected = st.selectbox("Failure to investigate", [case.name for case in CASES], key="coding_case")
    case = next(case for case in CASES if case.name == selected)
    st.code(case.log, language="text")
    with st.expander("Inspect the fixture repository"):
        st.write(f"Linked test: `{case.test_id or 'none — synthetic CI example'}`")
        st.code((FIXTURE_ROOT / "cart.py").read_text(), language="python")
        st.code((FIXTURE_ROOT / "tests/test_cart.py").read_text(), language="python")

    left, right = st.columns(2)
    with left:
        run_one = st.button("Run Laya-guided workflow", key="coding_run_one")
    with right:
        run_three = st.button("Compare complete workflows", key="coding_run_three",
                              disabled=not bool(os.getenv("OPENAI_API_KEY")))
    if run_one:
        try:
            ai = coding_ai() if os.getenv("OPENAI_API_KEY") else None
            graph = build_coding_workflow(gateway(), ai)
            with st.status("Running LangGraph…", expanded=True) as status:
                result = stream_result(
                    graph, {"case": case, "strategy": "laya", "threshold": threshold},
                    lambda name, update: graph_status(name, update, status),
                )
                status.update(label="Workflow complete", state="complete")
            st.session_state.coding_result = (selected, threshold, result)
        except Exception as exc:
            st.error(f"Workflow failed: {type(exc).__name__}: {exc}")
    cached = st.session_state.get("coding_result")
    if cached and cached[:2] == (selected, threshold):
        result = cached[2]
        st.write("**Outcome:**", result["answer"])
        st.caption(f"Route: {result['route']} · Sources: {', '.join(result.get('sources', [])) or 'none'} · "
                   f"OpenAI calls: {result['metrics'].get('model_calls', 0)} · "
                   f"Tokens: {result['metrics'].get('input_tokens', 0)} in / {result['metrics'].get('output_tokens', 0)} out · "
                   f"Tool calls: {result['metrics'].get('tool_calls', 0)}")

    if run_three:
        try:
            with st.status("Running three complete workflows…", expanded=True) as status:
                results = compare_complete(
                    case, gateway(), coding_ai(), threshold,
                    lambda strategy, name, update: graph_status(f"{strategy} / {name}", update, status),
                )
                status.update(label="Three workflows complete", state="complete")
            st.session_state.coding_complete = (selected, threshold, results)
        except Exception as exc:
            st.error(f"Full comparison failed: {type(exc).__name__}: {exc}")
    complete = st.session_state.get("coding_complete")
    if complete and complete[:2] == (selected, threshold):
        results = complete[2]
        st.dataframe([{
            "Strategy": row["strategy"], "Route": row["route"], "Route matches?": row["correct"],
            "Wall time (s)": round(row["wall_seconds"], 2),
            "Laya requests": row["metrics"].get("laya_calls", 0),
            "OpenAI calls": row["metrics"].get("model_calls", 0),
            "Input tokens": row["metrics"].get("input_tokens", 0),
            "Output tokens": row["metrics"].get("output_tokens", 0),
            "Tool calls": row["metrics"].get("tool_calls", 0),
        } for row in results], hide_index=True, width="stretch")
        for row in results:
            with st.expander(f"{row['strategy']}: {row['route']} — outcome and path"):
                st.write(" → ".join(row["steps"]))
                st.write(row["answer"])


def render_support() -> None:
    st.subheader("Support-operations example")
    st.write("A companion example: classify 14 synthetic support tickets, then follow one through LangGraph.")
    threshold = st.slider("Support fallback probability", 0.0, 1.0, 0.85, 0.01, key="support_threshold",
                          help="Exploratory demo setting, not a calibrated production cutoff.")
    if st.button("Run support triage", key="support_batch"):
        try:
            with st.spinner("Classifying the queue…"):
                st.session_state.support_batch_result = gateway().predict_batch([ticket.report for ticket in SAMPLES])
        except Exception as exc:
            st.error(f"Laya triage failed: {type(exc).__name__}: {exc}")
    batch = st.session_state.get("support_batch_result")
    if batch:
        rows = [{"Ticket": ticket.name, "Expected": ticket.expected, "Laya": decision.choice,
                 "P(chosen)": round(decision.answer_confidence, 3),
                 "Next step": "second judgment" if decision.answer_confidence < threshold else decision.choice,
                 "Correct?": decision.choice == ticket.expected}
                for ticket, decision in zip(SAMPLES, batch.decisions)]
        c1, c2, c3 = st.columns(3)
        c1.metric("Tickets", len(rows))
        c2.metric("Laya matches", f"{sum(row['Correct?'] for row in rows)}/{len(rows)}")
        c3.metric("Above cutoff", f"{sum(d.answer_confidence >= threshold for d in batch.decisions)}/{len(rows)}")
        st.caption(f"{batch.mode}; {batch.inference_calls} inference request(s); {batch.wall_seconds:.2f}s wall time.")
        st.dataframe(rows, hide_index=True, width="stretch")

    selected = st.selectbox("Support ticket", [ticket.name for ticket in SAMPLES] + ["Custom report"], index=4)
    custom = st.text_area("Custom report") if selected == "Custom report" else None
    ticket = (next(ticket for ticket in SAMPLES if ticket.name == selected) if selected != "Custom report"
              else Ticket("Custom report", (custom or "").strip(), "unknown"))
    if ticket.report:
        st.code(ticket.report, language="text")
    if st.button("Run support workflow", key="support_run"):
        if not ticket.report:
            st.warning("Enter a report first.")
        else:
            try:
                prior = batch.decisions[SAMPLES.index(ticket)] if batch and ticket in SAMPLES else None
                ai = support_ai() if os.getenv("OPENAI_API_KEY") else None
                graph = build_workflow(gateway(), ai.fallback if ai else None, ai.investigate if ai else None)
                with st.status("Running LangGraph…", expanded=True) as status:
                    result = stream_result(
                        graph, {"report": ticket.report, "threshold": threshold,
                                **({"decision": prior} if prior else {})},
                        lambda name, update: graph_status(name, update, status),
                    )
                    status.update(label="Workflow complete", state="complete")
                st.session_state.support_result = (ticket.report, threshold, result)
            except Exception as exc:
                st.error(f"Workflow failed: {type(exc).__name__}: {exc}")
    cached = st.session_state.get("support_result")
    if cached and cached[:2] == (ticket.report, threshold):
        result = cached[2]
        st.write("**Outcome:**", result["answer"])
        st.caption(f"Sources: {', '.join(result.get('sources', [])) or 'none'} · "
                   f"Tool calls: {result.get('tool_calls', 0)} · OpenAI stages: {result.get('llm_calls', 0)}")


st.title("System One Lab")
st.caption("An observable Laya → LangGraph → coding-agent handoff, with baselines and real tool calls.")
with st.sidebar:
    st.subheader("Runtime")
    backend = ("Remote batch adapter" if os.getenv("LAYA_BATCH_BASE_URL") else
               "Remote laya-serve" if os.getenv("LAYA_BASE_URL") else "Local Laya checkpoint")
    st.write(f"**Laya:** {backend}")
    st.write(f"**OpenAI:** {'configured' if os.getenv('OPENAI_API_KEY') else 'not configured'}")
    st.caption("Changing an endpoint or key requires restarting Streamlit. The batch adapter is optional; laya-serve remains supported.")

coding_tab, support_tab = st.tabs(["Coding agent", "Support triage"])
with coding_tab:
    render_coding()
with support_tab:
    render_support()
