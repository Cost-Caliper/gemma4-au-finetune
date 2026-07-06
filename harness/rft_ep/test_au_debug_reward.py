"""RFT reward: coverage of reference root-cause identifiers in the model's diagnosis.

Mirrors the AU DEBUG benchmark metric. ground_truth per row = JSON list of exact
identifier strings from the live-verified diagnosis of that failure.
"""
import json
import re

from eval_protocol.models import EvaluateResult, EvaluationRow
from eval_protocol.pytest.default_single_turn_rollout_process import (
    SingleTurnRolloutProcessor,
)
from eval_protocol.pytest.evaluation_test import evaluation_test


@evaluation_test(
    input_dataset=["rft_debug_prompts.jsonl"],
    completion_params=[
        {"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b",
         "temperature": 0.8, "max_tokens": 1200}
    ],
    max_dataset_rows=8,
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=0.01,
    mode="pointwise",
)
def test_au_debug_reward(row: EvaluationRow) -> EvaluationRow:
    facts = []
    try:
        facts = json.loads(row.ground_truth or "[]")
    except Exception:
        pass
    assistant = [m for m in row.messages if m.role == "assistant"]
    out = assistant[-1].content if assistant else ""
    if not isinstance(out, str):
        out = ""
    text = re.sub(r"<\|?channel\|?>\w*\s*", " ", out)
    text = re.sub(r"\s+", " ", text.lower())
    hits = sum(1 for f in facts if re.sub(r"\s+", " ", str(f).lower()) in text)
    cov = hits / len(facts) if facts else 0.0
    bonus = 0.1 if re.search(r"root cause|diagnosis", out, re.I) else 0.0
    row.evaluation_result = EvaluateResult(
        score=min(1.0, 0.9 * cov + bonus),
        reason=f"{hits}/{len(facts)} reference identifiers present",
    )
    return row
