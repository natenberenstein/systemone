"""Resident batched HTTP adapter. Run: uvicorn batch_server:app --host 0.0.0.0 --port 8001"""

from dataclasses import asdict
import os
import secrets
from threading import Lock
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from demo.laya_gateway import LayaGateway, LocalLayaGateway


class BatchRequest(BaseModel):
    reports: list[str] = Field(min_length=1, max_length=64)
    question: dict[str, Any]
    question_key: str = "queue"


def create_app(gateway: LayaGateway | None = None) -> FastAPI:
    app = FastAPI(title="Laya batch adapter")
    resident = gateway
    init_lock = Lock()
    predict_lock = Lock()

    def resident_gateway() -> LayaGateway:
        nonlocal resident
        if resident is None:
            with init_lock:
                if resident is None:
                    resident = LocalLayaGateway(os.getenv("LAYA_DEVICE", "cpu"))
        return resident

    def check_auth(authorization: str | None) -> None:
        key = os.getenv("LAYA_API_KEY")
        if key and not secrets.compare_digest(authorization or "", f"Bearer {key}"):
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/health")
    def health(authorization: str | None = Header(default=None)) -> dict:
        check_auth(authorization)
        return {"status": "ok", "mode": "resident batch adapter"}

    @app.post("/v1/batch")
    def batch(body: BatchRequest, authorization: str | None = Header(default=None)) -> dict:
        check_auth(authorization)
        if any(not report or len(report) > 4000 for report in body.reports):
            raise HTTPException(status_code=422, detail="Each report must contain 1–4000 characters")
        if body.question.get("type") != "choice" or not isinstance(body.question.get("criteria"), dict):
            raise HTTPException(status_code=422, detail="Expected a choice question with criteria")
        if not 2 <= len(body.question["criteria"]) <= 8:
            raise HTTPException(status_code=422, detail="Expected 2–8 choices")
        with predict_lock:
            result = resident_gateway().predict_batch(body.reports, body.question, body.question_key)
        return {"decisions": [asdict(item) for item in result.decisions], "wall_seconds": result.wall_seconds}

    return app


app = create_app()
