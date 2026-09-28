from demo.coding_data import CASES, CODING_QUESTION
from demo.coding_workflow import build_coding_workflow, compare_batch, rule_route, stream_result
from demo.laya_gateway import BatchResult, Decision


class FakeGateway:
    def __init__(self, choice="code_bug", probability=.9):
        self.decision = Decision(choice, probability, {choice: probability})
        self.calls = 0

    def predict_batch(self, reports, question=CODING_QUESTION, question_key="route"):
        self.calls += 1
        assert question_key == "route"
        return BatchResult([self.decision] * len(reports), .01, "fake batch", 1)


class FakeAI:
    def __init__(self, route="code_bug"):
        self.route = route
        self.classifications = 0
        self.investigations = 0

    def classify(self, log):
        self.classifications += 1
        return {"route": self.route, "model_calls": 1, "input_tokens": 15,
                "output_tokens": 3, "wall_seconds": .1}

    def investigate(self, case):
        self.investigations += 1
        return {"answer": "Fix the percentage calculation", "sources": ["file:cart.py"],
                "tool_calls": 3, "model_calls": 2, "input_tokens": 40,
                "output_tokens": 8, "wall_seconds": .2}


def test_streamed_coding_graph_uses_laya_then_real_fallback_and_agent_nodes():
    gateway = FakeGateway(probability=.4)
    ai = FakeAI()
    updates = []
    result = stream_result(
        build_coding_workflow(gateway, ai),
        {"case": CASES[0], "strategy": "laya", "threshold": .55},
        lambda name, _: updates.append(name),
    )
    assert updates == ["triage", "fallback", "dispatch", "code_bug"]
    assert result["route"] == "code_bug"
    assert result["metrics"]["model_calls"] == 3
    assert result["metrics"]["tool_calls"] == 3
    assert result["metrics"]["laya_calls"] == 1
    assert ai.classifications == ai.investigations == 1


def test_rules_route_environment_without_agent():
    result = build_coding_workflow(FakeGateway(), FakeAI()).invoke(
        {"case": CASES[3], "strategy": "rules", "threshold": .55})
    assert result["route"] == "environment"
    assert result["metrics"]["model_calls"] == 0
    assert "dependencies" in result["answer"]


def test_batch_comparison_reuses_openai_outputs_for_fallback():
    gateway = FakeGateway(probability=.4)
    ai = FakeAI()
    result = compare_batch(gateway, ai, .55)
    assert gateway.calls == 1
    assert ai.classifications == len(CASES)
    assert result["fallback_count"] == len(CASES)
    assert all(row["Laya + fallback"] == "code_bug" for row in result["rows"])
    assert result["threshold_sweep"][0]["Accepted"] == len(CASES)
    assert result["threshold_sweep"][-1]["Fallbacks"] == len(CASES)


def test_rule_baseline_is_explicit_and_not_perfect():
    predictions = [rule_route(case.log) for case in CASES]
    assert len(predictions) == len(CASES)
    assert predictions[0] == "code_bug"
    assert predictions[-1] == "unknown"
    assert any(prediction != case.expected for prediction, case in zip(predictions, CASES))
