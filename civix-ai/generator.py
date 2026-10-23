"""Generative layer: Amazon Bedrock (Converse API) turns complaint + category into an action plan.
Falls back to a labelled template when Bedrock isn't configured, so the demo never breaks."""
import json
import os
import re

MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
REGION = os.getenv("AWS_REGION", "us-east-1")

DEPARTMENTS = {
    "pothole": "Roads & Highways Department",
    "streetlight": "Electrical / Street Lighting Department",
    "garbage": "Solid Waste Management Department",
    "water": "Water Supply & Drainage Board",
}

PROMPT = """You assist a municipal complaint-triage tool. A small decision model already classified the report.
Do NOT change the category. Write practical output for a human officer.

Complaint: {complaint}
Location: {location}
Assigned urgency: {category}

Return ONLY JSON with keys:
"department" (string), "explanation" (1-2 sentences why this urgency fits),
"next_steps" (list of 3-5 short imperative steps), "draft_message" (formal 4-6 line message to the department).
Do not claim the report was sent or that help was dispatched."""


def _department(text: str) -> str:
    t = text.lower()
    if re.search(r"pothole|road|crack|bridge", t): return DEPARTMENTS["pothole"]
    if re.search(r"light|lamp|wire|pole", t): return DEPARTMENTS["streetlight"]
    if re.search(r"garbage|trash|bin|waste|litter", t): return DEPARTMENTS["garbage"]
    if re.search(r"water|leak|pipe|drain|sewage|flood", t): return DEPARTMENTS["water"]
    return "Municipal Corporation - General Grievances"


def _template(complaint: str, location: str, category: str) -> dict:
    dept = _department(complaint)
    steps = {
        "URGENT": ["Escalate to a duty officer now", "Advise barricading or warning signage", "Schedule on-site inspection today", "Log follow-up check after repair"],
        "NORMAL": ["Register ticket with the department", "Schedule inspection within 3-5 days", "Notify citizen of ticket ID"],
        "LOW_PRIORITY": ["Add to routine maintenance queue", "Batch with nearby work orders", "Notify citizen when scheduled"],
    }[category]
    return {
        "department": dept,
        "explanation": f"Classified {category} based on the described impact. (Template output: Bedrock not configured.)",
        "next_steps": steps,
        "draft_message": (f"To: {dept}\nSubject: Citizen report - {category}\n\n"
                          f"A citizen reported: \"{complaint.strip()}\"" + (f" at {location.strip()}" if location.strip() else "") +
                          f".\nSuggested urgency: {category}. Please review and advise on action.\n\nRegards,\nCivix AI (draft for human review)"),
        "source": "template-fallback",
    }


def generate(complaint: str, location: str, category: str) -> dict:
    try:
        import boto3
        client = boto3.client("bedrock-runtime", region_name=REGION)
        resp = client.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": PROMPT.format(
                complaint=complaint, location=location or "not given", category=category)}]}],
            inferenceConfig={"maxTokens": 700, "temperature": 0.2},
        )
        text = resp["output"]["message"]["content"][0]["text"]
        data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        data["source"] = f"bedrock ({MODEL_ID})"
        return data
    except Exception as e:
        out = _template(complaint, location, category)
        out["error"] = type(e).__name__
        return out
