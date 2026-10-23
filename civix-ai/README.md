# Civix AI — Smart Civic Issue Agent

Decision-support prototype that triages citizen complaints (potholes, streetlights, garbage, leaks).
A small **decision model classifies**, a **generative model explains**.

- **Decision layer — Strands Decider 2B** (local, via Ollama `/v1/systemone`): `choice` question over `URGENT / NORMAL / LOW_PRIORITY`, returns category + per-option probabilities.
- **Generative layer — Amazon Bedrock** (Converse API): explanation, next steps, draft department message, conditioned on the assigned category.
- **Policy pattern**: typed question + fixed option set + routing policy (`decider.py`). Reusable for ticket triage / incident classification by swapping `QUESTION` and `OPTIONS`.
- **Human-in-the-loop**: URGENT or confidence < 60% => "Human review required". Nothing is dispatched or sent.

## Architecture
```
Complaint -> Decider 2B (category + confidence) -> routing policy -> Bedrock (plan + draft) -> UI
```

## Run
```bash
# 1. Decider 2B (Apple-silicon Mac, Ollama >= 0.40)
ollama pull strands-decider

# 2. App
pip install -r requirements.txt
export AWS_REGION=us-east-1                # Bedrock creds via standard AWS config
export BEDROCK_MODEL_ID=<model/inference-profile id enabled in your account>
uvicorn app:app --reload                   # http://localhost:8000

# 3. Evaluate the decision layer
python eval/run_eval.py
```
If Ollama or Bedrock is unreachable, the app falls back to a keyword classifier / template **and labels it in the UI** (confidence shows "n/a" in fallback mode, never a made-up number).

## Results (fill in from `python eval/run_eval.py` with Decider running)
- Accuracy on 18 labelled complaints: __/18
- Urgent recall: __/6
- Median decision latency: __ ms

## Limitations
Confidence is model certainty, not correctness. Eval set is small and hand-written; production use needs a larger labelled set, calibrated thresholds and human review.

## Links
- Source: <add repo URL>
- Demo: <add demo URL>

`#strands-decider-2b #policy`
