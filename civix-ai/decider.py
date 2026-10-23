"""Decision layer: Strands Decider 2B (served by Ollama) using the policy pattern.

Typed decision question + fixed option set -> (category, confidence).
If the Decider server is unreachable, a clearly-labelled keyword fallback is used
(confidence is None in that case, never a fabricated number).
"""
import os
import re
import time

import httpx

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DECIDER_MODEL = os.getenv("DECIDER_MODEL", "strands-decider")

# --- Reusable policy pattern: question + option set + routing policy -------------
QUESTION = "Which urgency category should handle this report?"
OPTIONS = {
    "URGENT": "Immediate risk to people or property: injury risk, exposed wires, "
              "major water main break, collapsed road, near-accidents, fire hazard.",
    "NORMAL": "Real problem that needs repair within days: broken streetlight, "
              "small leak, overflowing bin, moderate road damage, no immediate danger.",
    "LOW_PRIORITY": "Cosmetic or minor issue with no safety impact: faded paint, "
                    "minor litter, small graffiti, suggestions.",
}
REVIEW_THRESHOLD = 0.60  # below this confidence -> human review


def route(category: str, confidence: float | None) -> dict:
    """Routing policy applied on top of the decision. Decider classifies; policy decides."""
    reasons = []
    if category == "URGENT":
        reasons.append("Urgent safety reports always need a human authority.")
    if confidence is None:
        reasons.append("No model confidence available (fallback mode).")
    elif confidence < REVIEW_THRESHOLD:
        reasons.append(f"Confidence {confidence:.0%} is below the {REVIEW_THRESHOLD:.0%} threshold.")
    return {"needs_human_review": bool(reasons), "review_reasons": reasons}


def _decider(state: str) -> dict:
    payload = {
        "model": DECIDER_MODEL,
        "state": state,
        "questions": {
            "urgency": {"type": "choice", "instructions": QUESTION, "criteria": OPTIONS}
        },
    }
    r = httpx.post(f"{OLLAMA_URL}/v1/systemone", json=payload, timeout=30)
    r.raise_for_status()
    ans = r.json()["answers"]["urgency"]
    choice = ans["choice"]
    probs = ans.get("probabilities") or {}
    return {
        "category": choice,
        "confidence": probs.get(choice),
        "probabilities": probs,
        "source": f"strands-decider-2b ({DECIDER_MODEL})",
    }


_URGENT = r"fall|fell|accident|injur|bleed|exposed|live wire|sparks?|fire|burst|flood|collaps|sinkhole|gas leak|danger|near(ly)?|unsafe|child|school"
_LOW = r"faded|graffiti|paint|cosmetic|minor|litter|suggest|bench|sticker"


def _fallback(state: str) -> dict:
    s = state.lower()
    cat = "URGENT" if re.search(_URGENT, s) else "LOW_PRIORITY" if re.search(_LOW, s) else "NORMAL"
    return {"category": cat, "confidence": None, "probabilities": {},
            "source": "keyword-fallback (Decider server not reachable)"}


def decide(complaint: str, location: str = "") -> dict:
    state = f"Civic complaint: {complaint.strip()}"
    if location.strip():
        state += f"\nLocation: {location.strip()}"
    t0 = time.perf_counter()
    try:
        out = _decider(state)
    except Exception as e:  # server down, model missing, schema change
        out = _fallback(state)
        out["error"] = type(e).__name__
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    out.update(route(out["category"], out["confidence"]))
    return out
