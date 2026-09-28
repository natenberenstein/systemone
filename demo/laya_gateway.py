"""One decision interface for resident local Laya and remote laya-serve."""

from dataclasses import dataclass
import os
from threading import Lock
from time import perf_counter
from typing import Protocol

import httpx

from demo.data import QUESTION


@dataclass(frozen=True)
class Decision:
    choice: str
    answer_confidence: float
    probabilities: dict[str, float]

    @classmethod
    def from_response(cls, response: dict, question: dict = QUESTION, question_key: str = "queue") -> "Decision":
        answer = response["answers"][question_key]
        choice = answer["choice"]
        if choice not in question["criteria"]:
            raise ValueError(f"Unexpected Laya category: {choice!r}")
        probabilities = answer.get("probabilities", {})
        # answer_confidence is the chosen label's probability. Laya's generic
        # `confidence` is an entropy measure and must not be used here.
        confidence = answer.get("answer_confidence", probabilities.get(choice))
        if confidence is None:
            raise ValueError("Laya returned no answer probability")
        return cls(choice, float(confidence), probabilities)


@dataclass(frozen=True)
class BatchResult:
    decisions: list[Decision]
    wall_seconds: float
    mode: str
    inference_calls: int


class LayaGateway(Protocol):
    def predict_batch(self, reports: list[str], question: dict = QUESTION, question_key: str = "queue") -> BatchResult: ...


class LocalLayaGateway:
    def __init__(self, device: str = "cpu"):
        if device == "cpu":
            import torch

            torch.set_num_threads(int(os.getenv("LAYA_THREADS", "4")))
        from laya.router import Router

        self.router = Router(device=device, default="english", preload=False)
        self._predict_lock = Lock()

    def predict_batch(self, reports: list[str], question: dict = QUESTION, question_key: str = "queue") -> BatchResult:
        if not reports:
            return BatchResult([], 0, "local batched inference", 0)
        requests = [
            {"state": {"report": report}, "questions": {question_key: question}, "model": "english"}
            for report in reports
        ]
        start = perf_counter()
        with self._predict_lock:
            responses = self.router.predict_batch(requests)
        return BatchResult(
            [Decision.from_response(response, question, question_key) for response in responses],
            perf_counter() - start,
            "local batched inference",
            1,
        )


class RemoteLayaGateway:
    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("LAYA_BASE_URL must begin with http:// or https://")
        self.client = httpx.Client(
            timeout=60,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    def predict_batch(self, reports: list[str], question: dict = QUESTION, question_key: str = "queue") -> BatchResult:
        start = perf_counter()
        decisions = []
        for report in reports:
            response = self.client.post(
                self.base_url + "/v1/systemone",
                json={"state": {"report": report}, "questions": {question_key: question}, "model": "english"},
            )
            response.raise_for_status()
            decisions.append(Decision.from_response(response.json(), question, question_key))
        return BatchResult(decisions, perf_counter() - start, "remote sequential HTTP", len(reports))


class BatchHttpLayaGateway(RemoteLayaGateway):
    """Custom resident batch adapter, separate from laya-serve's single-state API."""

    def predict_batch(self, reports: list[str], question: dict = QUESTION, question_key: str = "queue") -> BatchResult:
        if not reports:
            return BatchResult([], 0, "remote batched HTTP", 0)
        start = perf_counter()
        response = self.client.post(
            self.base_url + "/v1/batch",
            json={"reports": reports, "question": question, "question_key": question_key},
        )
        response.raise_for_status()
        decisions = [Decision(**item) for item in response.json()["decisions"]]
        if len(decisions) != len(reports) or any(d.choice not in question["criteria"] for d in decisions):
            raise ValueError("Batch adapter returned invalid decisions")
        return BatchResult(
            decisions,
            perf_counter() - start,
            "remote batched HTTP",
            1,
        )


def make_gateway() -> LayaGateway:
    batch_url = os.getenv("LAYA_BATCH_BASE_URL", "").strip()
    if batch_url:
        return BatchHttpLayaGateway(batch_url, os.getenv("LAYA_API_KEY"))
    base_url = os.getenv("LAYA_BASE_URL", "").strip()
    if base_url:
        return RemoteLayaGateway(base_url, os.getenv("LAYA_API_KEY"))
    return LocalLayaGateway(os.getenv("LAYA_DEVICE", "cpu"))
