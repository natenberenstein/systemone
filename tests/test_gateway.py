import httpx
import pytest

from demo.laya_gateway import Decision, RemoteLayaGateway


def test_confidence_is_chosen_probability_not_entropy_score():
    decision = Decision.from_response({"answers": {"queue": {
        "choice": "billing", "answer_confidence": .71, "confidence": .28,
        "probabilities": {"billing": .71, "technical": .2, "other": .09},
    }}})
    assert decision.answer_confidence == .71


def test_remote_gateway_calls_one_request_per_report():
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json={"answers": {"queue": {
            "choice": "other", "answer_confidence": .9,
            "probabilities": {"other": .9},
        }}})

    gateway = RemoteLayaGateway("https://laya.example", "secret")
    gateway.client = httpx.Client(transport=httpx.MockTransport(handle))
    result = gateway.predict_batch(["a", "b"])
    assert result.mode == "remote sequential HTTP"
    assert result.inference_calls == 2
    assert len(seen) == 2
    assert seen[0].url.path == "/v1/systemone"
    assert seen[0].read().decode().find('"report":"a"') >= 0


def test_invalid_remote_url_is_rejected():
    with pytest.raises(ValueError):
        RemoteLayaGateway("laya.example")
