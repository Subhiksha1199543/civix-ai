"""Evaluate the decision layer on labelled complaints. Run with Ollama + strands-decider up for real numbers.
Usage: python eval/run_eval.py"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decider import REVIEW_THRESHOLD, decide  # noqa: E402

cases = json.loads((Path(__file__).parent / "cases.json").read_text())
rows = [(c, decide(c["text"])) for c in cases]

correct = sum(d["category"] == c["label"] for c, d in rows)
src = Counter(d["source"] for _, d in rows)
print(f"Source: {dict(src)}")
print(f"Accuracy: {correct}/{len(rows)} = {correct/len(rows):.0%}")
print(f"Median latency: {statistics.median(d['latency_ms'] for _, d in rows)} ms")

urgent = [(c, d) for c, d in rows if c["label"] == "URGENT"]
missed = [c["text"] for c, d in urgent if d["category"] != "URGENT"]
print(f"Urgent recall: {len(urgent)-len(missed)}/{len(urgent)}")
for t in missed:
    print("  MISSED URGENT:", t)

confs = [d["confidence"] for _, d in rows if d["confidence"] is not None]
if confs:
    lo = sum(c < REVIEW_THRESHOLD for c in confs)
    print(f"Routed to human review by low confidence (<{REVIEW_THRESHOLD}): {lo}/{len(confs)}")
    right = [d["confidence"] for c, d in rows if d["category"] == c["label"] and d["confidence"] is not None]
    wrong = [d["confidence"] for c, d in rows if d["category"] != c["label"] and d["confidence"] is not None]
    if right: print(f"Mean confidence when correct: {statistics.mean(right):.2f}")
    if wrong: print(f"Mean confidence when wrong:   {statistics.mean(wrong):.2f}")
print("\nWrong predictions:")
for c, d in rows:
    if d["category"] != c["label"]:
        print(f"  [{c['label']} -> {d['category']}] {c['text']}")
