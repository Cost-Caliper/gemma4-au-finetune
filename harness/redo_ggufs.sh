modal run modal_hf_upload.py --repo dennisonb/gemma-4-26b-a4b-it-au-v4-gguf --src /vol/tuned-v4-q8.gguf --readme-text '---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
tags:
- lora
- gemma-4
- agent-university
- code-specialist
---

# Agent University Gemma-4 specialist — v4

LoRA adapter for `google/gemma-4-26B-A4B-it`, fine-tuned on the Agent University corpus
(live-verified library curricula: gotchas, recipes, reference implementations, POC tasks).

- **Training:** rank 32, 7.5k tool-calling traces (au_search), $50 (Fireworks managed SFT)
- **Benchmark:** self-directed tool use: 92-98% call rate on knowledge, 0% on code; 49.6%/6.7 holdout with tools
  (40 held-out capstone tasks + 210 knowledge probes, judged vs live-verified references;
  Claude Sonnet 5 baseline: code 31.5%/3.82, knowledge 26.0/29.2%)
- **Format note:** trained with Fireworks `fw_lora_layout: fused_peft_3d_v1` — MoE expert LoRAs
  are fused 3D tensors. Standard PEFT loaders pick up attention/shared-MLP only; use
  `merge_adapter.py` from the agent-university repo (docs/plans/gemma4-au-finetune-harness/)
  for full-fidelity merging.
- **License:** Apache-2.0 weights subject to the Gemma Terms / Prohibited Use Policy.
- Companion `-gguf` repo carries the merged Q8_0 GGUF.
' 2>&1 | grep -E "UPLOADED|Error|error" | head -3
modal run modal_hf_upload.py --repo dennisonb/gemma-4-26b-a4b-it-au-v5-gguf --src /vol/tuned-v5-q8.gguf --readme-text '---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
tags:
- lora
- gemma-4
- agent-university
- code-specialist
---

# Agent University Gemma-4 specialist — v5

LoRA adapter for `google/gemma-4-26B-A4B-it`, fine-tuned on the Agent University corpus
(live-verified library curricula: gotchas, recipes, reference implementations, POC tasks).

- **Training:** rank 32, skill+debugging+tool traces, no memorization, $48 (Fireworks managed SFT)
- **Benchmark:** tool calibration 95-98%/7.5%; knowledge-via-tool 47.6%/6.76; code 47.1%/3.26
  (40 held-out capstone tasks + 210 knowledge probes, judged vs live-verified references;
  Claude Sonnet 5 baseline: code 31.5%/3.82, knowledge 26.0/29.2%)
- **Format note:** trained with Fireworks `fw_lora_layout: fused_peft_3d_v1` — MoE expert LoRAs
  are fused 3D tensors. Standard PEFT loaders pick up attention/shared-MLP only; use
  `merge_adapter.py` from the agent-university repo (docs/plans/gemma4-au-finetune-harness/)
  for full-fidelity merging.
- **License:** Apache-2.0 weights subject to the Gemma Terms / Prohibited Use Policy.
- Companion `-gguf` repo carries the merged Q8_0 GGUF.
' 2>&1 | grep -E "UPLOADED|Error|error" | head -3
