"""RFT reward: fraction of reference root-cause identifiers present in the model's diagnosis.
ground_truth = JSON list of exact identifier strings from the live-verified diagnosis.
Mirrors the DEBUG benchmark's key-fact coverage metric. Small bonus for a structured diagnosis."""
import json
import re


def evaluate(model_output: str, ground_truth: str) -> float:
    try:
        facts = json.loads(ground_truth)
    except Exception:
        return 0.0
    if not facts:
        return 0.0
    text = re.sub(r"<\|?channel\|?>\w*\s*", " ", model_output or "")
    text = re.sub(r"\s+", " ", text.lower())
    hits = sum(1 for f in facts if re.sub(r"\s+", " ", str(f).lower()) in text)
    coverage = hits / len(facts)
    bonus = 0.1 if re.search(r"root cause|diagnosis", model_output or "", re.I) else 0.0
    return min(1.0, 0.9 * coverage + bonus)
