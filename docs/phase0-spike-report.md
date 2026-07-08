# Phase 0 GPU Serving Spike — gemma-4-26B-A4B-it on vLLM multi-LoRA (Modal A100-80GB)

Provenance: ran live on Modal (workspace `dennisonbertram`), GPU = A100-80GB, date = 2026-07-08.
Working dir: `.agent-university/finetune/spike/` (scripts: `serve_test.py`, `make_synth_lora.py`, `inspect_shapes.py`, `apply_patch.sh`).
Budget: hard cap ~2 GPU-hours. All GPU functions are short-lived `modal run` invocations (no deployments, no idle).

## Setup (CPU-only, no GPU cost) — DONE

- vLLM pin: **0.21.0** (same as Modal's official `vllm_inference.py` example for this exact model).
- Base image: `nvidia/cuda:12.9.0-devel-ubuntu22.04` + `uv_pip_install vllm==0.21.0`.
- Patched image: base + `patch -p1 --fuzz=5 < pr46772.diff` applied to site-packages.
  - PR vllm-project/vllm#46772 diff applied cleanly against 0.21.0 with offsets:
    - `gemma4.py` Hunk 1 @1369 (-4), Hunk 2 @1492 (+2), Hunk 3 @1720 (+2)
    - `gemma4_mm.py` Hunk 1 @1476 (-232)
  - Import-time verify: `Gemma4Model` / `Gemma4ForCausalLM` / `Gemma4ForConditionalGeneration` all have `get_expert_mapping` → `PATCH_VERIFY_OK`.
- Synthetic attention-only LoRA built on CPU → volume `/synth-lora-attn`:
  - rank 4, random bf16 weights, 120 tensors (~2.8 MB), q_proj/v_proj for all 30 layers.
  - Key format matches base checkpoint naming: `base_model.model.model.language_model.layers.{i}.self_attn.{q,v}_proj.lora_{A,B}.weight`.
  - Shapes from live inspection of `/vol/base` shards: hidden=2816, q_proj out=4096, v_proj out=2048.
- Shape inspection findings (live, from volume):
  - Base MoE layout per layer: `experts.down_proj [128, 2816, 704]`, `experts.gate_up_proj [128, 1408, 2816]`, `router.proj.weight [128, 2816]`.
  - `/adapter-v4` (Fireworks `fw_lora_layout: fused_peft_3d_v1`, r=32): attention LoRA weights are standard 2D
    (e.g. `q_proj.lora_A [32, 2816]`, `lora_B [4096, 32]`), but expert weights are fused 2D-projected 3D:
    `experts.lora_A [4096, 704]`, `experts.lora_B [2816, 4096]`, plus `experts.base_layer.lora_A [4096, 2816]` / `lora_B [1408, 4096]`.
    Note 4096 = 128 experts × r32? No — 128×32 = 4096. So the fused layout stacks all 128 experts × rank-32 into one dim.

## Harness incident log (for provenance)

- Attempt 1 of Test 1+2 (`modal run serve_test.py::test_base_and_tools`, app ap-gOwqE6xKIFN2I0y0AylpHm, 2026-07-08):
  client created the functions then logged "Stopping app - local entrypoint completed. Runner terminated." after 241s
  with ZERO function output — the function was never invoked/awaited (session also hit a credits outage).
  Fix: explicit `@app.local_entrypoint()` wrappers (`run_test1_2`, `run_test3_stock`, `run_test3_patched`) that block on
  `.remote()`, print the result JSON, and write `spike/results_*.json` locally; remote functions also print `RESULT_JSON:`
  inside the container so `modal app logs` retains results if the client dies.

## TEST 1 — BASE SERVES: **PASS**

Live on Modal A100-80GB, 2026-07-08, app `ap-oKnQ5Wl7cE7bHDpYEnD3uz` (attempt 2 via `run_test1_2`). Raw result: `spike/results_test1_2.json`.

- Command: `vllm serve /vol/base --served-model-name gemma4 --max-model-len 8192 --enforce-eager --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' --enable-auto-tool-choice --tool-call-parser gemma4`
- vLLM version: **0.21.0** (stock release, unpatched)
- Load time (process start → /health 200): **222.3 s** (~3.7 min; 53 GB BF16 from Modal Volume, `--enforce-eager`)
- Chat completion: HTTP 200, `"Two plus two equals four."` (temperature 0, max_tokens 100, 7 completion tokens)
- Tokens/s: **not representative** — 7 tokens / 8.62 s wall (~0.8 tok/s) because the FIRST request pays Triton JIT
  compilation of `fused_moe_kernel` (vLLM log: "JIT compilation during inference: fused_moe_kernel ... causes a latency spike").
  Also `--enforce-eager` disables CUDA graphs. No steady-state throughput measured (kept generation short to save budget).
- Total client wall for the whole run (spin-up + load + 2 requests + teardown): 267 s.

## TEST 2 — TOOL PARSER: **PASS**

Same server session as Test 1 (`--enable-auto-tool-choice --tool-call-parser gemma4`, vLLM 0.21.0 stock).

- Request: one OpenAI-format tool `au_search` (single required `query` string param), `tool_choice: "auto"`,
  prompt instructing the model to call the tool.
- Response: HTTP 200 with a **structured `tool_calls` array** — not raw `<|tool_call>` text. `content` was `null`. Exact shape:

```json
"tool_calls": [{
  "id": "chatcmpl-tool-b66958333a807889",
  "type": "function",
  "function": {"name": "au_search", "arguments": "{\"query\": \"gemma4 lora fine-tuning\"}"}
}]
```

- `function.arguments` is a JSON-encoded string (standard OpenAI shape). Tool name and argument matched the prompt.

## TEST 3 — LORA STARTUP

### 3a — stock vLLM 0.21.0 + `--enable-lora` (no adapters): **BUG REPRODUCED (startup fails, as predicted)**

Live on Modal A100-80GB, 2026-07-08, two runs of `run_test3_stock` (run 2 = app `ap-JNRVpG0pCE9Y08qwNwUhj4`).
Raw: `spike/results_test3_stock.json`.

- Command: base args + `--enable-lora --max-lora-rank 8 --max-loras 2` (no adapter paths).
- Both runs: vllm process **exited rc=1 at ~111 s**, during engine-core init, before /health ever succeeded.
- Root cause captured verbatim from the EngineCore traceback (run 2, targeted grep):

```
(EngineCore pid=22) ERROR ... [core.py:1140]     raise AttributeError(
(EngineCore pid=22) ERROR ... [core.py:1140] AttributeError: To support LoRA for MoE model, 'get_expert_mapping' must be implemented
(APIServer pid=5) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

- This is exactly the known upstream bug; fix is unmerged PR vllm-project/vllm#46772.
- Note (run 1 lesson): a blind 4 KB log tail only captures the APIServer wrapper traceback — the EngineCore root cause
  scrolls past. Grep the server log for `get_expert_mapping|AttributeError` instead.

### 3b — patched vLLM 0.21.0 (+#46772) + `--enable-lora`: **STARTUP PASS**; synthetic adapter run 1 FAIL (my shape bug)

Live on Modal A100-80GB, 2026-07-08, `run_test3_patched` run 1 (258 s client wall). Raw: `spike/results_test3_patched.json`.

- Patched build (same vllm 0.21.0 with the #46772 diff applied to site-packages) with `--enable-lora --max-lora-rank 32
  --max-loras 2` + `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`: **server starts and /health passes** (load 231.3 s).
  → The +18/−8-line PR fix is sufficient to unblock `--enable-lora` startup for this MoE model.
- Synthetic rank-4 q/v attention-only adapter, dynamic load via `POST /v1/load_lora_adapter`: **HTTP 500**
  `"Call to add_lora method failed: The size of tensor a (1024) must match the size of tensor b (2048) at non-singleton dimension 0"`.
  Diagnosis: MY synthetic adapter bug, not a vLLM bug — Gemma-4's `layer_types` mixes sliding_attention
  (v_proj out = 8 kv-heads × 256 = 2048) and full_attention layers (2 global kv-heads × 512 = 1024, per config
  `num_global_key_value_heads`/`global_head_dim`); I generated uniform 2048-shaped v_proj for all 30 layers.
  Confirmed by live shard inspection: full-attention layers (5, 11, 17, 23, 29) have q_proj `[8192, 2816]`,
  k_proj `[1024, 2816]`, and **no v_proj weight at all** (`attention_k_eq_v: true` — K/V shared on global layers).
  Important corollary: **vLLM strictly shape-checks adapters at load time** — a wrong-shape adapter is rejected with a 500,
  it does not load silently.

### 3b run 2 — synthetic 2D attention-only adapter (corrected shapes): **PASS** → TEST 3 OVERALL: **PASS**

Live on Modal A100-80GB, 2026-07-08, `run_test3_patched` run 2 (267 s client wall; server load 207.2 s).
Raw: `spike/results_test3_patched.json` (run 1 preserved as `results_test3_patched_run1.json`).

- Synthetic v2 (rank 4, random ×0.01 bf16, q_proj on all 30 layers with per-layer-type shapes, v_proj on the 25 sliding
  layers only — key structure mirrored from adapter-v4; built by `spike/make_synth_lora_v2.py` → `/vol/synth-lora-attn-v2`):
  - `POST /v1/load_lora_adapter {"lora_name": "synth-attn", ...}` → **HTTP 200** (`Loaded new LoRA adapter: name 'synth-attn'` in server log).
  - `POST /v1/chat/completions` with `model="synth-attn"` → **HTTP 200**, response `model` field = `synth-attn`,
    coherent text (tiny random deltas ⇒ near-base output; we were testing loading + routing, not quality).
  - Server log proves the LoRA execution path actually ran: Triton JIT-compiled `_lora_shrink_kernel`,
    `_lora_expand_kernel`, and `_fused_moe_lora_kernel` during this generation.

## TEST 4 — BONUS adapter-v4 (Fireworks `fused_peft_3d_v1`, r=32): loads **as-is** and generates; output-diff caveat

Both `run_test3_patched` sessions, patched build:

- `POST /v1/load_lora_adapter {"lora_name": "adapter-v4", "lora_path": "/vol/adapter-v4"}` → **HTTP 200**
  `"Success: LoRA adapter 'adapter-v4' added successfully."` in BOTH runs — as-is, no `is_3d_lora_weight` flag needed
  (that fallback branch never executed). No skip/ignore warnings around the load in the captured log window.
- Generation with `model="adapter-v4"` → **HTTP 200**, response `model` field = `adapter-v4`, coherent answer.
- **Caveat:** at temperature 0 on a generic prompt ("In one sentence, what does the au_search tool do?"), adapter-v4's
  output was byte-identical to the base model's output. Two readings, not distinguished by this spike:
  (a) weights applied but deltas too subtle to flip any temp-0 token on this prompt (plausible — v4 was tuned on AU
  tool-calling data), or (b) some/all modules silently not applied. Evidence FOR application: load-time shape checking is
  strict (see 3b run 1's 500), the fused-MoE-LoRA kernel exists and runs in this build, and the adapter loaded without any
  skip warnings. Definitive confirmation (e.g. a behavioral eval or a large-magnitude probe adapter) left out of scope to
  stay inside budget.

## Verdicts

| Test | Verdict |
|---|---|
| 1. Base serves on A100-80GB (vLLM 0.21.0, `--max-model-len 8192`) | **PASS** |
| 2. Tool parser returns structured `tool_calls` (`--tool-call-parser gemma4`) | **PASS** |
| 3. `--enable-lora`: stock **fails** with `get_expert_mapping` AttributeError (reproduced verbatim); patched (#46772, 26-line diff, applies with fuzz to 0.21.0 site-packages) **starts + dynamically loads + serves a plain 2D attention-only PEFT adapter** | **PASS (with patch)** |
| 4. Bonus: Fireworks fused-3D `/vol/adapter-v4` | **LOADS + GENERATES as-is on patched build** (weight-application not conclusively verified — see caveat) |

## What changes the plan

1. **The #46772 patch is mandatory but tiny and clean** — apply it to site-packages during Modal image build
   (`patch --fuzz=5`), verify with an import-time assert. No custom vLLM fork/build needed.
2. **Multi-LoRA serving of this MoE model is viable on one A100-80GB**: base + dynamic adapters, `--max-lora-rank 32`,
   `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, `POST /v1/load_lora_adapter`, route by `model=<adapter-name>`.
3. **Synthetic/hand-built adapters must respect Gemma-4's heterogeneous attention**: full-attention layers
   (every 6th: 5, 11, 17, 23, 29) have q=8192-out, k=1024-out, and NO v_proj. Mirror a real adapter's key set.
4. **adapter-v4 may be usable directly** (no conversion from the fused 3D layout) — but run a behavioral eval before
   trusting it; identical temp-0 outputs vs base on a generic prompt leave application unproven.
5. Load time ~207–231 s from Modal Volume with `--enforce-eager`; budget ~4 min of GPU per server-start when planning evals.
6. Modal client gotcha confirmed in a new form: `modal run file.py::gpu_function` can terminate without invoking the
   function; use an explicit `@app.local_entrypoint()` that blocks on `.remote()`, and print results inside the container.

## GPU-hours consumed: ~0.30 total (1070 s client wall on A100-80GB across 5 runs: 267 + 138 + 140 + 258 + 267 s), ~$0.75–1.00 at Modal A100-80GB rates — well under the 2 GPU-hour cap

## Teardown

All GPU work used ephemeral `modal run` apps that stopped at entrypoint completion (each log ends
"Stopping app - local entrypoint completed"). Verified post-spike with `modal app list` (2026-07-08): every
`au-gemma4-*` spike app from today shows **state=stopped, tasks=0**. The pre-existing `au-gemma4` deployed app
(2026-07-04, prior sessions, serverless llama.cpp endpoints) remains with 0 tasks — untouched by this spike, not billing.
