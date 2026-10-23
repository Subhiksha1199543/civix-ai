from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from decider import OPTIONS, QUESTION, REVIEW_THRESHOLD, decide
from generator import generate

app = FastAPI(title="Civix AI", description="Decision-support prototype for civic complaint triage")


class Complaint(BaseModel):
    complaint: str = Field(min_length=5, max_length=1500)
    location: str = Field(default="", max_length=200)


@app.post("/api/triage")
def triage(body: Complaint):
    decision = decide(body.complaint, body.location)           # 1) Decider 2B (policy pattern)
    plan = generate(body.complaint, body.location, decision["category"])  # 2) Bedrock
    return {"complaint": body.complaint, "location": body.location, "decision": decision, "plan": plan,
            "disclaimer": "Decision-support only. Nothing was dispatched or sent to any department."}


@app.get("/api/pattern")
def pattern():
    return {"question": QUESTION, "options": OPTIONS, "review_threshold": REVIEW_THRESHOLD}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
