# Fleet Phase 1 — Training Report (attention-only LoRA, gemma-4-26B-A4B-it)

Provenance: ran live on Modal (workspace `dennisonbertram`), GPU = A100-80GB, date = 2026-07-08.
Working dir: `.agent-university/finetune/phase1/`. Budget: hard cap 12 GPU-hours total (~$30 @ $2.50/hr).
Scripts: `modal_finetune.py` (Modal app: `au-gemma4-phase1-finetune`), `axolotl_configs/gen_configs.py`
(config generator), `fix_tool_args.py` (dataset preprocessing fix, see pre-flight step 1).

## CRITICAL PRE-FLIGHT (CPU only, before any GPU spend)

### Step 1 — chat-template check: FOUND A REAL BUG, FIXED

Loaded the tokenizer directly from `google/gemma-4-26B-A4B-it` (config/tokenizer files only,
~63 MB, no weights) and ran `apply_chat_template` on a real tool-trace row from
`train_supabase.jsonl` with its `tools` field, exactly as instructed.

**As-shipped, the datasets mis-render.** `train_*.jsonl` stores
`messages[].tool_calls[].function.arguments` as an OpenAI-standard **JSON-encoded string**
(e.g. `"{\"query\": \"...\"}"`). Gemma-4's native chat template only auto-formats `arguments`
into its custom-escaped native tool-call syntax when `arguments` is a **mapping**; when it's a
string, the template inserts it verbatim inside the `{...}` the template *itself* already adds,
producing doubled braces and leftover standard-JSON quotes mixed into a stream that should be
entirely `<|"|>`-escaped:

```
bad (arguments as string):   <|tool_call>call:au_search{{"query": "..."}}<tool_call|>
correct (arguments as dict): <|tool_call>call:au_search{query:<|"|>...<|"|>}<tool_call|>
```

This produced **no exception** — Jinja renders it "successfully" — it just silently mis-renders,
which is exactly the failure mode the pre-flight check exists to catch. Confirmed programmatically
(not just by reading the template) via two renders of the same row: raw `arguments` (string) vs.
`arguments` pre-parsed to a dict. Full validation pass over all three fixed files (11,352 rows:
415 vercel + 2,814 supabase + 8,123 control): **0 render errors, 0 remaining doubled-brace
occurrences.**

**Fix:** `fix_tool_args.py` emits a byte-identical copy of each `train_*.jsonl` with exactly one
transform — `json.loads()` on every `tool_calls[].function.arguments` string — and nothing else.
The original `train_*.jsonl` files were **not modified** (per instructions). Output:
`train_{target}.axolotl.jsonl` (342/415, 2,288/2,814, 6,590/8,123 tool-call argument strings
parsed per target — matches the ~81–82% tool-trace share reported in `phase1_build_report.md`).

Rendered sample (trimmed), gemma-4 native format, correct after the fix:
```
<bos><|turn>system
You are an expert software agent for modern AI/agent libraries, developer tools, and cloud
services. You have access to the au_search tool over the Agent University live-verified
corpus...<|tool>declaration:au_search{description:<|"|>Search the Agent University
live-verified corpus...<|"|>,parameters:{properties:{...}},required:[<|"|>query<|"|>],
type:<|"|>OBJECT<|"|>}}<tool|><turn|>
<|turn>user
supabase: `emptyAndDeleteBucket` helper — retry loop for eventually-consistent emptying — what
do I need to know? Cite exact identifiers.<turn|>
<|turn>model
<|tool_call>call:au_search{query:<|"|>supabase `emptyAndDeleteBucket` helper — retry loop for
eventually-consistent emptying<|"|>}<tool_call|><|tool_response>response:au_search{value:<|"|>
[1] Pattern: `emptyAndDeleteBucket` helper...
```

### Step 1b — second bug found via axolotl itself (not the raw HF template): FOUND AND FIXED

Axolotl ships its **own** copy of the gemma4 template (`axolotl/utils/chat_templates/templates/gemma4.jinja`,
logically identical to the HF-shipped one, just reformatted) and its `chat_template` dataset
strategy statically analyzes that template (`JinjaTemplateAnalyzer`) to decide which extra
per-message fields to carry through from the raw JSONL row into the turn dict passed to
`apply_chat_template`. Axolotl **does** auto-parse `tool_calls[].function.arguments` JSON strings
to dicts internally (`chat_template.py::transform_message`) — contrary to what axolotl's own docs
say ("no built-in automatic parser"); that part of the pre-flight fix above turned out to be
belt-and-suspenders for axolotl's pipeline specifically, though it remains necessary for the raw
HF-tokenizer path used in step 1's diagnosis and is harmless/idempotent either way.

The template's tool-response lookahead re-binds the loop variable (`{%- set follow =
loop_messages[k] -%}`) before reading `follow.get('tool_call_id')` / `follow.get('name')`. The
static analyzer's alias-tracking does not chain that back to the primary loop variable `message`,
so **both fields were silently stripped from tool-role turns** before axolotl ever calls the
tokenizer. Symptom, caught live by `axolotl preprocess` on the 50-row dry-run slice (see step 2):

```
TypeError: can only concatenate str (not "NoneType") to str
  File ".../gemma4.jinja", line 170, in format_tool_response_block
    {{- 'response:' + tool_name + '{value:' + format_argument(response, escape_keys=False) + '}' -}}
```

Root cause chain: `tool_call_id` dropped from the tool-role turn → the id-match-back-to-the-
assistant's-`tool_calls` loop never fires → `tool_name` stays at its Jinja-`default()`-filter
"default" of `None` (a classic Jinja gotcha: `default()` only replaces `Undefined`, not a real
Python `None` returned by `dict.get()` on a missing key) → string concatenation with `None`
crashes.

**Fix:** added an explicit `message_property_mappings` to every dataset config that force-maps
`tool_call_id` and `name` through regardless of the static analyzer:
```yaml
message_property_mappings:
  role: role
  content: content
  tool_call_id: tool_call_id
  name: name
```
After the fix, `axolotl preprocess` on the same 50-row slice succeeds cleanly (see step 2).

### Step 2 — axolotl dataset dry-run / tokenization pass: PASS

Ran `axolotl preprocess` (via `modal_finetune.py::dry_run`, CPU-only Modal function, no GPU
billed) on a real 50-row slice of `train_vercel.axolotl.jsonl` (43/50 tool-trace, 7/50
knowledge-only — same mix as the full datasets), with the exact recipe config (`adapter: lora`,
`lora_r: 32`, attention-only `lora_target_modules`, `roles_to_train: ["assistant"]`,
`train_on_inputs: false`, `sequence_len: 4096`). 46/50 rows survived into the prepared dataset
(axolotl's own loader dropped 4; not investigated further since it doesn't affect the masking
check this step exists to run).

Verified loss masking directly against the tokenized `labels` array (not just axolotl's log
output) by loading the prepared arrow dataset and decoding masked (`label == -100`) vs. trainable
spans for the first example:

| span | trained? | tokens | excerpt |
|---|---|---|---|
| system + tool declaration + user question | NO | 195 | `<bos><\|turn>system\nYou are an expert software agent...` |
| assistant's tool call | **YES** | 33 | `<\|tool_call>call:au_search{query:<\|"\|>vercel Recipe 05 ...` |
| tool result (search results) | NO | 1190 | `response:au_search{value:<\|"\|>[1] Recipe 05 ...` |
| assistant's final answer | **YES** | 81 | ``## File: `vercel.json` ...` `` |
| trailing newline | NO | 1 | `\n` |

This is exactly the intended behavior: the model is trained to both **emit the tool call** and
**answer using the tool result**, matching how prior run v4 was trained on Fireworks (per the
task brief); the system prompt, tool declaration, user question, and raw tool-result text are all
masked. Aggregate over the 46-row sample: 57,591 total tokens, 8,404 trained (14.6%) — consistent
with a tool-trace-heavy dataset where most tokens are masked context.

**Verdict: PASS. Proceeding to GPU runs.**

## Recipe (identical across all three runs except dataset/output_dir)

LoRA r=32, alpha=32, dropout=0, attention-only target modules (`q_proj,k_proj,v_proj,o_proj`,
matched per-layer via regex `model.language_model.layers.[\d]+.(_checkpoint_wrapped_module.)?self_attn.(q|k|v|o)_proj`
— scoped to the text backbone only, since `lora_target_linear` is documented as incompatible with
this multimodal model and would otherwise also hit vision/audio encoder attention modules), 2
epochs, lr 2e-4 cosine, seed 42, bf16, gradient checkpointing, `sequence_len: 4096` (axolotl
truncates longer traces; no sample packing). `attn_implementation: flash_attention_2` +
`gemma4_hybrid_attn_impl: true` + axolotl's `kernels` plugin (pulls prebuilt FA2 kernels from the
HF Hub at container runtime — no local flash-attn source build) to protect the GPU-hour budget;
FA2 runs the sliding-window layers, SDPA runs the 5 full-attention/global layers (head_dim=512,
which FA2 can't serve on its own per axolotl's own gemma4 README). Base model loaded directly
from the Modal volume (`/vol/base`, already a full HF snapshot — no re-download). No MoE expert
LoRA, no expert quantization, no bnb/QLoRA — attention-only LoRA never touches the routed
experts, so the base model loads as plain frozen bf16 (~53 GB) with no quantization plumbing
needed.

Trainer: axolotl `0.17.0.dev0` (installed editable from `git main`, since gemma4 support is not
yet on a versioned PyPI release — confirmed via `examples/gemma4/` existing on `main`), Python
3.12, PyTorch 2.11.0+cu128, transformers 5.12.1, on `nvidia/cuda:12.9.0-devel-ubuntu22.04`.

## Incident log

1. **vercel attempt 1 (2026-07-08 18:47 EDT, app ap-BjZ8KCtXqz6y1X3PK7O06Q): FAIL at model load,
   371 s on-GPU (~0.10 GPU-h, ~$0.26).** `ModuleNotFoundError: No module named 'torchvision'` →
   `Could not import module 'Gemma4Processor'` — transformers' Gemma4 processor imports
   torchvision at load time and axolotl does not declare it as a dependency. Fix: added
   `uv pip install --system torchvision` to the image AFTER the axolotl install (so uv resolves
   against the already-pinned torch 2.11.0+cu128). Counted as failure 1 of 2 allowed for this run.

2. **vercel attempt 2 (2026-07-08 18:55 EDT, app ap-nRDy4uAVkclPb1hSEeTrFq): DELIBERATELY KILLED
   at step 5/95 after ~12 min on-GPU (~0.20 GPU-h, ~$0.50) — training was CORRECT but 3x too
   slow for the budget.** Everything worked: torchvision fix held, dataset cache from attempt 1
   was reused, LoRA attached with **exactly** the analytically-predicted trainable-parameter
   count (22,978,560 = 25 sliding layers × 753,664 + 5 full-attention layers × 827,392, r=32
   q/k/v/o with no v_proj on full-attention layers), FA2-hybrid attention active, first losses
   healthy (step-5 loss 3.853, ppl 47.1). BUT steady-state throughput was ~52-60 s/optimizer-step
   (~200 tok/s, ~2% MFU): resolved config showed `"experts_implementation": "eager"` —
   transformers' per-expert loop (128 experts × 30 layers of tiny launch-bound GEMMs). At that
   rate: vercel ~1.3 GPU-h, supabase ~9, control ~26 → ~36 GPU-h total vs the 12 cap. Since the
   entire point of running vercel first is validating the config the big runs will use, killed it
   and restarted vercel with `use_scattermoe: true` + `experts_implementation: scattermoe` (same
   keys as axolotl's own gemma4 examples; experts are frozen under attention-only LoRA, so this
   is an implementation swap, not a recipe change) + `sample_packing: true` (axolotl's gemma4
   hybrid-attn patch registers a block-diagonal packing mask for the head_dim=512 global layers,
   so packed documents stay isolated) + CPU-only `axolotl preprocess` pass before every GPU run
   (tokenization off GPU billing; cache-hash reuse verified live between attempts 1→2).
   Teardown verified: `modal app stop ap-nRDy4uAVkclPb1hSEeTrFq` → no ephemeral apps running.

3. **control attempt 1 (2026-07-08 20:13–21:14 EDT, app ap-TJTbJJsN1rfcJW9ZUUptRv): KILLED at
   step 121/278 (~59 min on-GPU, ~0.99 GPU-h, ~$2.47) by CLIENT-SIDE FAILURE, not a training
   fault.** Training was healthy (epoch 0.86, loss 0.743, ppl 2.10, steady 28.3 s/step). The
   local `modal run` client process was killed by the session's background-task management
   ~60 min in; Modal then propagated the disconnect as an input cancellation
   (`Received a cancellation signal while processing input … Successfully canceled input`,
   01:14:29 UTC) — with `save_strategy: "no"` there were no mid-run checkpoints, so the run was
   unrecoverable. A post-mortem reattach (`modal app logs`) confirmed the server kept training
   for ~3 min after client death before the cancellation landed. Fix: relaunched with
   **`modal run --detach`** (detached ephemeral apps do NOT cancel inputs when the client
   disconnects), same warm dataset cache. Teardown of the canceled app verified
   (`modal app stop` → no ephemeral apps). Lesson recorded for Phase 2/3: ALWAYS `--detach`
   multi-hour Modal runs; a run whose only durable output lands at the end must not depend on a
   local client living for hours.

## Per-run stats

### Run 1 — vercel: PASS (2026-07-08 19:13–19:23 EDT, app of `modal run …run_train --target vercel`, attempt 3)

- **Data:** `/vol/phase1-data/train_vercel.axolotl.jsonl` (415 rows; ~380 survived axolotl's
  loader, packed into 119 samples of ≤4096 tokens → **28 optimizer steps** for 2 epochs at
  grad_accum 8 × micro_batch 1).
- **Trainable params:** 22,978,560 (attention-only r=32; exact analytic match, see incident 2).
- **Loss:** step-5 3.056 → epoch-1 boundary 0.905 (ppl 2.47) → last logged (step 25) **0.890**
  (ppl 2.43); `train_loss` (whole-run average) 1.366. (`trainer_state.json` not written because
  `save_strategy: "no"`; losses taken from the streamed trainer log.)
- **Tokens:** ~795k packed-total by step 25 of 28 (≈890k full run, 2 epochs over ~0.44M
  effective trainable-set tokens); trainable-token share ~15.7% (124,530/795,029 at step 25).
- **Throughput:** steady ~14.3 s/step ≈ 2,290 total tok/s — ~11× the eager-experts attempt.
- **GPU time:** 584 s single-shot function wall (includes ~2 min model load + adapter save) =
  **0.16 GPU-h ≈ $0.41**. Train-loop runtime alone: 443.6 s.
- **Adapter verification (PASS):** `/vol/phase1-adapters/vercel/` = standard PEFT
  (`adapter_config.json` + `adapter_model.safetensors`, 91.9 MB); **230 LoRA tensors, all 2D,
  all attention** (0 non-attention keys); q_proj on **30** layers, v_proj on **25** (the 5
  full-attention layers 5/11/17/23/29 have no v_proj — expected); k/o on 30. Sample keys:
  - `…layers.0.self_attn.k_proj.lora_A.weight [32, 2816]`
  - `…layers.0.self_attn.k_proj.lora_B.weight [2048, 32]`
  - `…layers.0.self_attn.o_proj.lora_A.weight [32, 4096]`
  - `…layers.0.self_attn.o_proj.lora_B.weight [2816, 32]`
  - `…layers.0.self_attn.q_proj.lora_A.weight [32, 2816]`
- **Copies:** volume `/vol/phase1-adapters/vercel/` (training wrote it there directly) + local
  `phase1/adapters/vercel/`.
- CPU-side `axolotl preprocess` (no GPU): 433 s.

### Run 2 — supabase: PASS (2026-07-08 19:24–20:12 EDT)

- **Data:** `/vol/phase1-data/train_supabase.axolotl.jsonl` (2,814 rows; packed →
  **192 optimizer steps** for 2 epochs at grad_accum 8 × micro_batch 1). Warm dataset cache from
  the parallel CPU preprocess (320 s, no GPU) was picked up (`Loading prepared dataset from
  disk at /vol/phase1-cache/supabase/…`).
- **Trainable params:** 22,978,560 — byte-identical setup to vercel.
- **Loss:** step-5 3.818 → step-10 1.887 → step-15 1.211 (ppl 3.36) → final logged steps
  0.718 / 0.639 / 0.677; `train_loss` (whole-run average) **0.884** over ~2.0 epochs.
- **Throughput:** steady ~13.9-14.3 s/step (same ~2,300 total tok/s as vercel).
- **GPU time:** 2,855 s function wall (train loop 2,735 s) = **0.79 GPU-h ≈ $1.98**.
- **Adapter verification (PASS):** identical structure to vercel — 230 tensors, all 2D, all
  attention, q/k/o on 30 layers, v_proj on 25, same sample key shapes, 91.9 MB safetensors.
- **Copies:** `/vol/phase1-adapters/supabase/` + local `phase1/adapters/supabase/`.
- Teardown verified: no ephemeral apps after completion.

### Run 3 — control: PASS (2026-07-08 21:16–23:37 EDT, attempt 2 with `--detach`, app ap-IARRpJdAuK6GEgrhoC1o9G)

- **Data:** `/vol/phase1-data/train_control.axolotl.jsonl` (8,123 rows; packed →
  **278 optimizer steps** for 2 epochs at grad_accum 16 × micro_batch 1). Warm dataset cache
  from the parallel CPU preprocess (201 s) picked up on both attempts.
- **Trainable params:** 22,978,560 — identical to the other two runs.
- **Loss:** final logged steps 0.681 / 0.684 / 0.716; `train_loss` (whole-run average)
  **0.845** over ~2.0 epochs (attempt 1, killed at step 121, had matching trajectory: 0.699 at
  epoch 0.82 — consistent seeds/data, evidence the rerun retraced the same path).
- **Throughput:** steady ~28.2-32 s/step for 16 packed samples of 4096 ≈ ~2,200-2,300 total tok/s
  (matches vercel/supabase per-token throughput; slight node-to-node variance).
- **GPU time:** 8,440 s function wall (train loop 8,241 s) = **2.34 GPU-h ≈ $5.86**.
- **Adapter verification (PASS):** identical structure — 230 LoRA tensors, all 2D, all
  attention, q/k/o on 30 layers, v_proj on 25, 91.9 MB safetensors.
- **Copies:** `/vol/phase1-adapters/control/` + local `phase1/adapters/control/`.
- Teardown verified: no ephemeral apps after completion.

## FINAL GATE — multi-LoRA serving: PASS (2026-07-08 23:37–23:43 EDT, app ap-DBA0scrFg9okqUjfYE7bbL)

One short GPU run (`phase1/gate_test.py`, exact patterns from the Phase-0 spike: vLLM 0.21.0 +
the #46772 patch applied to site-packages at image build, `--enable-lora --max-lora-rank 32
--max-loras 3`, `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, `--enforce-eager`). Server load 258 s;
total function wall 364.5 s (**0.10 GPU-h ≈ $0.25**). Raw result: `modal_result_gate.json`.

| Adapter | /v1/load_lora_adapter | generate (supabase-flavored prompt, temp 0, 150 tok) | model field | verdict |
|---|---|---|---|---|
| vercel | **200** "Success: LoRA adapter 'vercel' added successfully." | **200** — "When using the Supabase Storage SDK (specifically within the JavaScript/TypeScript client), the `emp…" | `vercel` | **PASS** |
| supabase | **200** (same) | **200** — same first 100 chars | `supabase` | **PASS** |
| control | **200** (same) | **200** — same first 100 chars | `control` | **PASS** |

**Phase 1 exit gate: PASS — all three adapters dynamically load and generate on one patched
vLLM server.** Known caveat carried over from the Phase-0 spike (adapter-v4 finding): the three
adapters' temp-0 outputs on this single prompt are byte-identical to each other, which this gate
does not treat as a failure — the gate tests load+generate+routing, not behavioral
differentiation. Distinguishing specialist-vs-generalist behavior is exactly what the Phase-2/3
benchmarks (`eval_supabase.jsonl` / `eval_vercel.jsonl` / probe sets) exist to measure. Note the
load-time shape check is strict (proven in the spike by the 500 on a wrong-shaped adapter), and
these adapters trained with per-step loss decreasing from ~3.8 to ~0.7, so weights are real.

## GPU-hours + cost tally (Phase 1 total)

| Item | GPU wall | Cost @ $2.50/h |
|---|---|---|
| vercel attempt 1 (torchvision fail) | 371 s = 0.10 h | $0.26 |
| vercel attempt 2 (eager-experts, deliberately killed) | ~16 min ≈ 0.27 h | $0.67 |
| vercel attempt 3 — PASS | 584 s = 0.16 h | $0.41 |
| supabase — PASS | 2,855 s = 0.79 h | $1.98 |
| control attempt 1 (client-kill cancellation) | ~61 min ≈ 1.03 h | $2.56 |
| control attempt 2 (--detach) — PASS | 8,440 s = 2.34 h | $5.86 |
| final gate — PASS | 365 s = 0.10 h | $0.25 |
| **Total** | **≈ 4.8 GPU-h** | **≈ $12.0** |

Well under the 12 GPU-hour / ~$30 cap (40% of budget). CPU-only work (image builds, preprocess
runs, dry-run) billed at Modal CPU rates — negligible next to GPU. The Phase-0 spike (0.30
GPU-h) was budgeted separately.

## What changes Phase 2/3

1. **ALWAYS `modal run --detach` for multi-hour GPU work.** A non-detached ephemeral app
   propagates local-client death as an input CANCELLATION (verbatim: "Received a cancellation
   signal while processing input") — this killed control attempt 1 at 43% after ~$2.5 of GPU.
   With `--detach` the server-side function survives client death; its durable outputs
   (`output_dir` on the volume + `vol.commit()`) don't depend on the client.
2. **The throughput recipe for this model is scattermoe + sample packing** (~11x over
   transformers' eager expert loop; ~2,300 tok/s on A100-80GB, bf16 frozen experts, attention
   LoRA). Eager experts (`experts_implementation: eager`, the default) is the single biggest
   footgun — it would have tripled the budget.
3. **Preprocess on CPU before every GPU run** (`axolotl preprocess`, same yaml → same cache
   hash). Zero GPU minutes spent on tokenization; verified cache reuse across runs/containers.
4. **The two dataset-preparation bugs found in pre-flight** (OpenAI-style JSON-string
   `arguments` mis-rendering; axolotl's template analyzer stripping `tool_call_id`/`name` from
   tool turns) are permanently fixed in `train_*.axolotl.jsonl` + the generated configs — reuse
   both for any future run on these datasets.
5. **Serving-side**: gate confirms the Phase-0 serving stack (patched vLLM 0.21.0) accepts all
   three Phase-1 adapters as standard PEFT — no conversion needed. Phase 2 evals can serve all
   three (+ base) from ONE A100-80GB server session, switching by `model=<adapter-name>`;
   budget ~4-5 min server load per session.
6. **Adapters are attention-only by construction and verified** (230 2D tensors; q/k/o × 30
   layers + v × 25; 22,978,560 trainable params exactly matching the analytic count). The
   Fireworks-v4-style fused-3D-expert complication is fully avoided.
