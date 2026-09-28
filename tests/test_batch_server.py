import httpx
from fastapi.testclient import TestClient

from batch_server import create_app
from demo.coding_data import CODING_QUESTION
from demo.laya_gateway import BatchHttpLayaGateway, BatchResult, Decision


class FakeGateway:
    def __init__(self):
        self.calls = []

    def predict_batch(self, reports, question, question_key):
        self.calls.append((reports, question_key))
        return BatchResult([Decision("code_bug", .8, {"code_bug": .8}) for _ in reports], .02, "fake", 1)


def test_batch_server_auth_and_one_request(monkeypatch):
    monkeypatch.setenv("LAYA_API_KEY", "test-secret")
    gateway = FakeGateway()
    app = create_app(gateway)
    body = {"reports": ["failure one", "failure two"], "question": CODING_QUESTION, "question_key": "route"}
    with TestClient(app) as client:
        assert client.post("/v1/batch", json=body).status_code == 401
        response = client.post("/v1/batch", json=body, headers={"Authorization": "Bearer test-secret"})
    assert response.status_code == 200
    assert len(response.json()["decisions"]) == 2
    assert len(gateway.calls) == 1


def test_batch_http_gateway_uses_one_transport_call():
    requests = []

    def handle_two(request):
        requests.append(request)
        return httpx.Response(200, json={"decisions": [
            {"choice": "code_bug", "answer_confidence": .8, "probabilities": {"code_bug": .8}},
            {"choice": "code_bug", "answer_confidence": .8, "probabilities": {"code_bug": .8}},
        ]})

    gateway = BatchHttpLayaGateway("https://batch.example")
    gateway.client = httpx.Client(transport=httpx.MockTransport(handle_two))
    result = gateway.predict_batch(["failure one", "failure two"], CODING_QUESTION, "route")
    assert len(result.decisions) == 2
    assert result.inference_calls == 1
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/batch"
