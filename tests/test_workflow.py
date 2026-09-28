from demo.laya_gateway import BatchResult, Decision
from demo.workflow import build_workflow


class FakeGateway:
    def __init__(self, decision):
        self.decision = decision
        self.calls = 0

    def predict_batch(self, reports):
        self.calls += 1
        return BatchResult([self.decision] * len(reports), 0, "fake", 1)


def test_confident_billing_skips_openai():
    gateway = FakeGateway(Decision("billing", .98, {"billing": .98}))
    result = build_workflow(gateway, fallback=lambda _: (_ for _ in ()).throw(AssertionError("fallback called")),
                            investigate=lambda _: (_ for _ in ()).throw(AssertionError("agent called"))).invoke(
                                {"report": "Duplicate charge", "threshold": .85})
    assert result["route"] == "billing"
    assert result.get("llm_calls", 0) == 0
    assert result["sources"] == ["billing"]
    assert gateway.calls == 1


def test_uncertain_ticket_gets_second_judgment_then_agent():
    gateway = FakeGateway(Decision("billing", .62, {"billing": .62, "technical": .38}))
    calls = []

    def fallback(report):
        calls.append("fallback")
        return "technical"

    def investigate(report):
        calls.append("investigate")
        return {"answer": "Check the API playbook", "sources": ["api"], "tool_calls": 1}

    result = build_workflow(gateway, fallback, investigate).invoke(
        {"report": "Orders returns 500", "threshold": .85})
    assert result["route"] == "technical"
    assert result["llm_calls"] == 2
    assert result["tool_calls"] == 1
    assert calls == ["fallback", "investigate"]


def test_missing_key_routes_low_confidence_to_manual():
    gateway = FakeGateway(Decision("other", .51, {"other": .51}))
    result = build_workflow(gateway).invoke({"report": "Ambiguous", "threshold": .85})
    assert result["route"] == "manual"
    assert result["llm_calls"] == 0
    assert "manual" in result["answer"].lower()


def test_batch_decision_is_reused():
    gateway = FakeGateway(Decision("technical", .97, {"technical": .97}))
    result = build_workflow(gateway).invoke({"report": "Cannot log in", "threshold": .85,
                                              "decision": gateway.decision})
    assert gateway.calls == 0
    assert result["route"] == "technical"
