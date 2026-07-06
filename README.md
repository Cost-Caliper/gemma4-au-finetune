# Gemma-4 Agent University Fine-Tuning

Research program: can a small, cheap, open-weights model be fine-tuned on a
curated software-library corpus to match or beat a frontier model *on that
stack* — cheaply enough that an orchestrator can delegate library work to a
specialist and reserve the frontier model for review?

Base model: `google/gemma-4-26B-A4B-it` (MoE, ~4B active params, Apache-2.0).
Corpus: [Agent University](https://github.com/dennisonbertram/agent-university)
— ~90 live-tested curricula for AI/agent libraries and dev tools. Training:
LoRA supervised fine-tuning on Fireworks. Five published variants (v1–v5),
plus a sixth (v6a) that tested and disproved a further hypothesis.

**Read first:** [`docs/gemma4-specialist-research-report.md`](docs/gemma4-specialist-research-report.md)
— the full write-up: methodology, results, and conclusions.

- [`docs/gemma4-hypothesis-ledger.md`](docs/gemma4-hypothesis-ledger.md) — every
  experimental hypothesis registered before its run, with the verdict after.
- [`docs/glm-5-2-fine-tune-research-2026-07-03.md`](docs/glm-5-2-fine-tune-research-2026-07-03.md)
  — the full execution log (day-by-day, including the earlier GLM-5.2 research
  that led to the Gemma-4 pivot).

## Headline result

Every tuned variant beats Claude Sonnet 5 at building applications with these
specific libraries — ~52–54% correct-API-usage coverage vs Sonnet's honest
31.5% — at roughly 4–40× lower inference cost. Exact-fact recall plateaus at
~18–21% in the weights regardless of training recipe (retrieval is worth
+35–40 points to any model); the strongest positive result is v4's *emergent,
uncoached tool-use calibration* — it searches the corpus on 92–98% of
knowledge questions and 0% while coding, purely from tool-call training
traces.

## Published models

All 5 adapters + merged Q8 GGUFs are public on Hugging Face:
`dennisonb/gemma-4-26b-a4b-it-au-{v1..v5}-{adapter,gguf}`.

## Repo layout

- `harness/` — everything needed to reproduce the program:
  - `build_dataset.py`, `build_v4_tools.py`, `build_v5.py`, `build_v6a.py` —
    dataset builders for each training variant.
  - `run_eval.py`, `rag_eval.py`, `tool_eval.py`, `rescore_probes.py`,
    `post_judge.py`, `validate_recall.py` — the evaluation harness (key-fact
    coverage + Claude-judge scoring, with and without retrieval/tool use).
  - `build_code_bench.py` — the 40-task app-building benchmark.
  - `merge_adapter.py` — merges Fireworks' fused-MoE-LoRA layout into a
    standard HF checkpoint.
  - `modal_gemma.py` — Modal app: merge/GGUF-convert + serve endpoints.
  - `modal_hf_upload.py` — pushes adapters/GGUFs to Hugging Face.
  - `rft_ep/`, `rft_evaluator/` — the reinforcement-fine-tuning reward
    function and dataset (built, validated, blocked on Fireworks' Training
    API preview access as of 2026-07-05 — see the hypothesis ledger).
  - `train*.jsonl`, `eval_*.jsonl`, `results_*.jsonl`, `summary_*.json` —
    the actual training data and every recorded eval run.
- `docs/` — the research report, hypothesis ledger, and execution log.

Model weights and merged GGUFs are NOT in this repo (multi-GB binaries) —
they're public on Hugging Face; `harness/merge_adapter.py` + `modal_gemma.py`
reproduce them from source.
