"""LangGraph: a bounded Laya decision followed by selective agent work."""

from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from demo.data import KNOWLEDGE_BASE
from demo.laya_gateway import Decision, LayaGateway


class Investigation(TypedDict):
    answer: str
    sources: list[str]
    tool_calls: int


class WorkflowState(TypedDict, total=False):
    report: str
    threshold: float
    decision: Decision
    route: str
    answer: str
    sources: list[str]
    tool_calls: int
    llm_calls: int
    steps: list[str]


def build_workflow(
    gateway: LayaGateway,
    fallback: Callable[[str], str] | None = None,
    investigate: Callable[[str], Investigation] | None = None,
):
    def triage(state: WorkflowState) -> dict:
        decision = state.get("decision")
        if decision is None:
            decision = gateway.predict_batch([state["report"]]).decisions[0]
        return {"decision": decision, "steps": [f"Laya: {decision.choice} ({decision.answer_confidence:.1%})"]}

    def after_triage(state: WorkflowState) -> str:
        return "fallback" if state["decision"].answer_confidence < state["threshold"] else "dispatch"

    def fallback_node(state: WorkflowState) -> dict:
        if fallback is None:
            return {"route": "manual", "answer": "Low-confidence decision: add OPENAI_API_KEY for an LLM second judgment, or review manually.", "llm_calls": 0, "steps": state["steps"] + ["Fallback unavailable → manual review"]}
        route = fallback(state["report"])
        if route not in {"billing", "technical", "other"}:
            raise ValueError(f"Unexpected OpenAI fallback category: {route!r}")
        return {"route": route, "llm_calls": 1, "steps": state["steps"] + [f"OpenAI second judgment: {route}"]}

    def dispatch(state: WorkflowState) -> dict:
        route = state.get("route", state["decision"].choice)
        return {"route": route, "steps": state["steps"] + [f"Dispatch: {route}"]}

    def after_dispatch(state: WorkflowState) -> str:
        return state["route"]

    def billing(state: WorkflowState) -> dict:
        return {"answer": "Assign to billing for verification. " + KNOWLEDGE_BASE["billing"], "sources": ["billing"], "tool_calls": 0, "steps": state["steps"] + ["Billing queue: fixed policy"]}

    def technical(state: WorkflowState) -> dict:
        if investigate is None:
            return {"answer": "Technical ticket queued. Add OPENAI_API_KEY to run the knowledge-base investigation agent.", "sources": [], "tool_calls": 0, "steps": state["steps"] + ["Investigation agent unavailable"]}
        result = investigate(state["report"])
        return {"answer": result["answer"], "sources": result["sources"], "tool_calls": result["tool_calls"], "llm_calls": state.get("llm_calls", 0) + 1, "steps": state["steps"] + [f"OpenAI agent: {result['tool_calls']} knowledge-base tool call(s)"]}

    def other(state: WorkflowState) -> dict:
        return {"answer": "Send to general support for manual review; no automated customer response.", "sources": [], "tool_calls": 0, "steps": state["steps"] + ["General queue"]}

    graph = StateGraph(WorkflowState)
    graph.add_node("triage", triage)
    graph.add_node("fallback", fallback_node)
    graph.add_node("dispatch", dispatch)
    graph.add_node("billing", billing)
    graph.add_node("technical", technical)
    graph.add_node("other", other)
    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", after_triage, {"fallback": "fallback", "dispatch": "dispatch"})
    graph.add_edge("fallback", "dispatch")
    graph.add_conditional_edges("dispatch", after_dispatch, {"billing": "billing", "technical": "technical", "other": "other", "manual": END})
    for route in ("billing", "technical", "other"):
        graph.add_edge(route, END)
    return graph.compile()
