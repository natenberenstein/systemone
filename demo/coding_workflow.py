"""Three comparable CI-failure routes and a streamed LangGraph investigation."""

from time import perf_counter
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from demo.coding_data import CASES, CODING_QUESTION, CodingCase
from demo.laya_gateway import Decision, LayaGateway


class CodingAIProtocol:
    def classify(self, log: str) -> dict: ...
    def investigate(self, case: CodingCase) -> dict: ...


class CodingState(TypedDict, total=False):
    case: CodingCase
    strategy: str
    threshold: float
    decision: Decision
    route: str
    answer: str
    sources: list[str]
    steps: list[str]
    metrics: dict


def rule_route(log: str) -> str:
    """Intentionally small, transparent keyword baseline."""
    evidence = log.lower()
    if "modulenotfounderror" in evidence:
        return "environment"
    if "intermittent" in evidence or ("passed on" in evidence and "rerun" in evidence):
        return "test_flake"
    if "assertionerror" in evidence or "did not raise" in evidence:
        return "code_bug"
    return "unknown"


def _add_metrics(old: dict, new: dict) -> dict:
    result = old.copy()
    for name in ("model_calls", "input_tokens", "output_tokens", "tool_calls", "laya_calls"):
        result[name] = result.get(name, 0) + new.get(name, 0)
    result["stage_seconds"] = result.get("stage_seconds", 0.0) + new.get("wall_seconds", 0.0)
    return result


def build_coding_workflow(gateway: LayaGateway, ai: CodingAIProtocol | None):
    def triage(state: CodingState) -> dict:
        case = state["case"]
        strategy = state["strategy"]
        if strategy == "laya":
            batch = gateway.predict_batch([case.log], CODING_QUESTION, "route")
            decision = batch.decisions[0]
            return {
                "decision": decision,
                "route": decision.choice,
                "steps": [f"Laya: {decision.choice} ({decision.answer_confidence:.1%})"],
                "metrics": _add_metrics({}, {"laya_calls": batch.inference_calls, "wall_seconds": batch.wall_seconds}),
            }
        if strategy == "openai":
            if ai is None:
                raise ValueError("OPENAI_API_KEY is needed for the OpenAI-only strategy")
            classified = ai.classify(case.log)
            return {"route": classified["route"], "steps": [f"OpenAI classifier: {classified['route']}"],
                    "metrics": _add_metrics({}, classified)}
        if strategy == "rules":
            route = rule_route(case.log)
            return {"route": route, "steps": [f"Keyword rules: {route}"], "metrics": _add_metrics({}, {})}
        raise ValueError(f"Unknown strategy: {strategy}")

    def after_triage(state: CodingState) -> str:
        if state["strategy"] == "laya" and state["decision"].answer_confidence < state["threshold"]:
            return "fallback"
        return "dispatch"

    def fallback(state: CodingState) -> dict:
        if ai is None:
            return {"route": "unknown", "steps": state["steps"] + ["OpenAI unavailable → request more evidence"]}
        classified = ai.classify(state["case"].log)
        return {"route": classified["route"], "metrics": _add_metrics(state["metrics"], classified),
                "steps": state["steps"] + [f"OpenAI second judgment: {classified['route']}"]}

    def dispatch(state: CodingState) -> dict:
        return {"steps": state["steps"] + [f"Dispatch: {state['route']}"]}

    def code_bug(state: CodingState) -> dict:
        case = state["case"]
        if not case.test_id:
            return {"answer": "No executable fixture test is linked; request a reproduction before suggesting a fix.",
                    "sources": [], "steps": state["steps"] + ["Manual code review"]}
        if ai is None:
            return {"answer": "A code failure is queued; add OPENAI_API_KEY for repository investigation.",
                    "sources": [], "steps": state["steps"] + ["Coding agent unavailable"]}
        investigated = ai.investigate(case)
        return {"answer": investigated["answer"], "sources": investigated["sources"],
                "metrics": _add_metrics(state["metrics"], investigated),
                "steps": state["steps"] + [f"Coding agent: {investigated['tool_calls']} tool call(s)"]}

    def environment(state: CodingState) -> dict:
        return {"answer": "Check runner Python version, installed test dependencies, and CI configuration before changing product code.",
                "sources": [], "steps": state["steps"] + ["Environment checklist"]}

    def test_flake(state: CodingState) -> dict:
        return {"answer": "Record rerun outcomes and timing; inspect test isolation and race conditions before changing product code.",
                "sources": [], "steps": state["steps"] + ["Flaky-test checklist"]}

    def unknown(state: CodingState) -> dict:
        return {"answer": "Request the complete failing command, stack trace, and reproduction details.",
                "sources": [], "steps": state["steps"] + ["More evidence needed"]}

    graph = StateGraph(CodingState)
    for name, fn in (("triage", triage), ("fallback", fallback), ("dispatch", dispatch),
                     ("code_bug", code_bug), ("environment", environment),
                     ("test_flake", test_flake), ("unknown", unknown)):
        graph.add_node(name, fn)
    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", after_triage, {"fallback": "fallback", "dispatch": "dispatch"})
    graph.add_edge("fallback", "dispatch")
    graph.add_conditional_edges("dispatch", lambda s: s["route"],
                                {route: route for route in CODING_QUESTION["criteria"]})
    for route in CODING_QUESTION["criteria"]:
        graph.add_edge(route, END)
    return graph.compile()


def stream_result(graph, inputs: dict, on_update: Callable[[str, dict], None] | None = None) -> dict:
    """Consume real LangGraph node updates and reconstruct the final state."""
    state = inputs.copy()
    for part in graph.stream(inputs, stream_mode="updates", version="v2"):
        if part["type"] != "updates":
            continue
        for name, update in part["data"].items():
            state.update(update)
            if on_update:
                on_update(name, update)
    return state


def compare_batch(gateway: LayaGateway, ai: CodingAIProtocol | None, threshold: float) -> dict:
    """One Laya batch and one OpenAI-only pass; reuse those OpenAI outputs for fallback."""
    rules_start = perf_counter()
    rules = [rule_route(case.log) for case in CASES]
    rules_seconds = perf_counter() - rules_start
    batch = gateway.predict_batch([case.log for case in CASES], CODING_QUESTION, "route")
    openai_results = [ai.classify(case.log) for case in CASES] if ai else None
    rows = []
    for index, (case, decision) in enumerate(zip(CASES, batch.decisions)):
        fallback = decision.answer_confidence < threshold
        llm = openai_results[index] if openai_results else None
        rows.append({
            "Case": case.name, "Expected": case.expected,
            "Laya": decision.choice, "P(chosen)": round(decision.answer_confidence, 3),
            "Laya + fallback": llm["route"] if fallback and llm else ("manual" if fallback else decision.choice),
            "Rules": rules[index],
            "OpenAI only": llm["route"] if llm else "unavailable",
            "Fallback?": fallback,
        })
    fallback_results = [openai_results[i] for i, row in enumerate(rows) if row["Fallback?"]] if openai_results else []
    summaries = []
    for label in ("Laya", "Laya + fallback", "Rules", "OpenAI only"):
        if label == "OpenAI only" and not ai:
            continue
        correct = sum(row[label] == row["Expected"] for row in rows)
        selected = fallback_results if label == "Laya + fallback" else (openai_results if label == "OpenAI only" else [])
        stage_seconds = (batch.wall_seconds if label.startswith("Laya") else rules_seconds if label == "Rules" else 0.0)
        stage_seconds += sum(item["wall_seconds"] for item in selected)
        summaries.append({"Strategy": label, "Matches": f"{correct}/{len(rows)}",
                          "OpenAI calls": sum(item["model_calls"] for item in selected),
                          "Input tokens": sum(item["input_tokens"] for item in selected),
                          "Output tokens": sum(item["output_tokens"] for item in selected),
                          "Stage time sum (s)": round(stage_seconds, 2)})
    threshold_sweep = []
    for cutoff in (0.0, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9):
        accepted = [(case, decision) for case, decision in zip(CASES, batch.decisions)
                    if decision.answer_confidence >= cutoff]
        accepted_correct = sum(case.expected == decision.choice for case, decision in accepted)
        threshold_sweep.append({"Cutoff": cutoff, "Accepted": len(accepted),
                                "Correct accepted": accepted_correct,
                                "Errors accepted": len(accepted) - accepted_correct,
                                "Fallbacks": len(CASES) - len(accepted)})
    return {"rows": rows, "summary": summaries, "laya_batch_seconds": batch.wall_seconds,
            "laya_requests": batch.inference_calls,
            "openai_model_calls": sum(item["model_calls"] for item in openai_results) if openai_results else 0,
            "openai_input_tokens": sum(item["input_tokens"] for item in openai_results) if openai_results else 0,
            "openai_output_tokens": sum(item["output_tokens"] for item in openai_results) if openai_results else 0,
            "openai_seconds": sum(item["wall_seconds"] for item in openai_results) if openai_results else 0.0,
            "fallback_count": sum(row["Fallback?"] for row in rows),
            "threshold_sweep": threshold_sweep}


def compare_complete(case: CodingCase, gateway: LayaGateway, ai: CodingAIProtocol, threshold: float,
                     on_update: Callable[[str, str, dict], None] | None = None) -> list[dict]:
    results = []
    for strategy in ("laya", "openai", "rules"):
        graph = build_coding_workflow(gateway, ai)
        start = perf_counter()
        result = stream_result(graph, {"case": case, "strategy": strategy, "threshold": threshold},
                               (lambda name, update: on_update(strategy, name, update)) if on_update else None)
        results.append({"strategy": strategy, "route": result["route"], "correct": result["route"] == case.expected,
                        "wall_seconds": perf_counter() - start, "metrics": result["metrics"],
                        "answer": result["answer"], "steps": result["steps"], "sources": result.get("sources", [])})
    return results
