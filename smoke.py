"""CLI smoke run of the actual Laya batch, with no OpenAI request."""

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).with_name(".env"))

from demo.data import SAMPLES
from demo.laya_gateway import make_gateway


batch = make_gateway().predict_batch([ticket.report for ticket in SAMPLES])
print(f"{batch.mode}: {len(batch.decisions)} tickets in {batch.wall_seconds:.2f}s; {batch.inference_calls} request(s)")
for ticket, decision in zip(SAMPLES, batch.decisions):
    print(f"{ticket.name:18} expected={ticket.expected:9} laya={decision.choice:9} p={decision.answer_confidence:.3f}")
