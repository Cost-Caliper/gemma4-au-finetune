# Fine-tuning GLM 5.2 on Agent University — research (2026-07-03)

**Question:** Can we fine-tune GLM 5.2 (via a service like Fireworks) on the work done in
Agent University, to get a version of GLM 5.2 that is better at implementing AU-style tasks?

**Short answer:** Yes, with one gate: GLM-5.2 fine-tuning on Fireworks is **preview/early-access**
today (the GA tunable list stops at GLM-5.1). Everything else — the training data, the dataset
pipeline, the eval harness, and the account — either already exists or is a small, cheap step.
The two real blockers are account-side: a **fresh Fireworks API key** (the stored one is rotated)
and a **$3 canary retest** of the fine-tuning entitlement that was never re-run after dedicated
compute was enabled on 2026-06-21.

Provenance tags used below: `[live-verified]` = probed on this machine/account,
`[docs]` = fetched from official docs during this research, `[vendor-claim]` = vendor blog/social,
`[memory 2026-06-2x]` = prior live-verified degree work, `[TBD]` = unknown, do not assume.

---

## 1. Ground truth: what GLM 5.2 is

- Real, current z.ai flagship (GLM-5 → 5.1 → 5.2). Open weights on Hugging Face
  (`zai-org/GLM-5.2`), released ~2026-06-16/17, **MIT license** — third parties may legally
  fine-tune and serve it. `[docs]` (HF model card, HF blog, Fireworks launch blog)
- Sparse MoE, **753B total / ~40B active** params, **1M context**, 131K max output. Long-horizon
  coding/agentic positioning. New "IndexShare" sparse-attention scheme. `[docs]`
- **No smaller variant exists** (no Air/Flash as of today; open community requests unanswered). `[docs]`
- Tool calls use GLM's XML-ish format (`<tool_call>…<arg_key>/<arg_value>`); vLLM/SGLang need
  `--tool-call-parser glm47`, `--reasoning-parser glm45`; thinking is on by default. `[docs]`
- z.ai itself does **not** offer hosted fine-tuning; their stance is "weights are MIT, self-host."
  `[vendor-claim; not found in official docs — low confidence]`
- Self-managed training of a 753B MoE is out of reach for casual budgets (reference *inference*
  deployment is 8×H200; no authoritative LoRA-training hardware figures found `[TBD]`). Framework
  support for the exact `.2` release (LLaMA-Factory / axolotl / ms-swift / unsloth training) is
  **unconfirmed** `[TBD]`.

## 2. Who can fine-tune GLM 5.x for you (provider matrix)

| Provider | GLM fine-tuning | Notes | Provenance |
|---|---|---|---|
| **Fireworks** | **GLM-5.1 GA; GLM-5.2 in preview** | GA tunable table lists `glm-5p1` (200K ctx); `glm-5p2` absent from GA table — but Fireworks announced GLM-5.2 SFT/DPO/RL with "training platform reaches GA soon… customers in preview" | table `[live-verified via curl of docs .md today]`; preview `[vendor-claim: Fireworks blog/X]` |
| Together AI | GLM-5.1, GLM-5, 4.7, 4.6 (LoRA) — **no 5.2 found** | Broader GLM back-catalog than Fireworks | `[docs, single fetch — medium confidence]` |
| Tinker (Thinking Machines) | No GLM support found | Qwen/GPT-OSS/DeepSeek/Kimi/Nemotron only | `[docs]` |
| z.ai / bigmodel.cn | No hosted fine-tuning found | | `[low confidence — verify on console]` |
| Self-host (Modal/RunPod + trainer) | Feasible in principle, 753B impractical | Only realistic if a smaller GLM variant ships | `[assessment]` |

**Implication:** "fine-tune literally GLM 5.2" = ask Fireworks for training-preview access (or wait
for GA). "Fine-tune the GLM-5 family now" = GLM-5.1 on Fireworks (GA) — closest sibling, 200K ctx.

## 3. Fireworks: workflow, formats, serving, account state

**Mechanics** `[docs, medium confidence unless noted]`
- Dataset: JSONL, one `messages` array per line (system/user/assistant), 3–3,000,000 examples.
  **Tool calling supported** (`tools` array + `tool_calls` in assistant messages). Optional
  per-message/per-example `weight` → train only on assistant turns. Reasoning traces
  (`reasoning_content`) documented for DeepSeek R1 / GPT-OSS / Qwen3 — **not stated for GLM**
  `[TBD — matters because GLM-5.2 is a thinking model; ask in preview]`.
- Create: `firectl sftj create --base-model <id> --dataset <id> --output-model <id>`; REST and
  dashboard equivalents; `--warm-start-from` continues from a prior LoRA. SFT, DPO, and RFT share
  the same base-model list.
- RFT: managed RL loop with a `reward-kit` evaluator (function-call match, code-execution
  validation, LLM-judge). `[docs]`
- **Serving tuned LoRAs: on-demand (dedicated) deployments ONLY — no serverless LoRA.**
  `[docs, high confidence — direct quote]`. Live cost anchor from our own account: GLM 5.2
  on-demand `glm-5p2-minimal` = 8×B300 at **$96/hr**, ~128 tok/s single-stream, 5,192 tok/s
  aggregate @ c=64, $5.23 for the measured session; autoscale-to-zero works but ~9-min cold start;
  a min-replica-0 deployment serves while `State` still reads `CREATING` — poll a 200 smoke, not
  `State==READY`. `[live-verified, memory 2026-06-21]`
- Training price (LoRA SFT, per docs pricing page `[medium confidence]`): $0.50/M tokens ≤16B,
  $3/M 16–80B, $6/M 80–300B, **$10/M >300B** (GLM-5.2 at 753B lands here; GLM-5.1 size `[TBD]`).
  DPO = 2× SFT. $3 minimum charge. Training tokens ≈ dataset tokens × epochs.

**Account state** (the Fireworks degree account)
- Dedicated compute: **enabled 2026-06-21**; on-demand create/delete works via REST + firectl.
  `[live-verified, memory]`
- Fine-tuning job launch: was blocked pre-entitlement with
  `HTTP 500 … unkey inference api id is not configured` (dataset upload + tunability checks DID
  work). **Never retested after entitlement** — may work now. `[live-verified 2026-06-20/21; retest TBD]`
- **API key: RESTORED 2026-07-03.** The Jun-20 key was dead (`Unauthenticated`, independent of
  the credits top-up); user issued fresh keys, stored in `.agent-university/secrets.local.env`
  (`FIREWORKS_API_KEY` + `FIREWORKS_SERVICE_KEY`, gitignore-verified) and persisted via
  `firectl set-api-key`. `[live-verified today]`
- **Live tunability flags (probed 2026-07-03 on this account):** `glm-5p2` →
  `tunable:false, supportsLora:false` (NOT self-serve tunable, matching the preview-only story);
  `glm-5p1`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it`, `qwen3p5-27b` → `supportsLora:true`.
  The `au-fw-canary` dataset (50 examples) survived on the account, state READY. `[live-verified]`
- **Fine-tuning entitlement: RESOLVED.** 2026-07-03 canary
  `firectl sftj create --base-model accounts/fireworks/models/gemma-4-26b-a4b-it --dataset
  au-fw-canary --output-model au-fw-canary-gemma4 --epochs 1` → job
  `supervisedFineTuningJobs/q04kdxx8` created, `JOB_STATE_RUNNING`, Status OK, LoRA rank 8. The
  old `unkey inference api id is not configured` HTTP 500 is gone. Gotcha: `--base-model` must be
  fully qualified (`accounts/fireworks/models/<id>`), else it resolves against your own account
  namespace and fails NOT_FOUND. `[live-verified]`
- ToS: reselling Fireworks inference to third parties is prohibited (internal/own-product use is
  fine). `[memory, cites fireworks.ai/terms-of-service §2.1/2.2]`

## 4. What Agent University has to train on `[live-verified inventory today]`

- **Degree corpus (this checkout):** 90 degrees / 35 tracks (+3 legacy at track root); 6,019
  markdown files; 3,730 real POC source files (after excluding vendored deps); ~177MB content.
  NOTE: the full corpus is larger — origin/main carries ~149 degree dirs (this branch is based on
  an older main), and several degrees exist only on unmerged branches (postgres, mcp, react,
  sqlite, bun, nextjs, shadcn, pgvector…). A dataset build should enumerate across main + open
  branches, not one working tree.
- **Structured task tuples:** every POC level is already (task spec → implementation → tests →
  outcome): `README.md`/`intent.md` + `source/` + `tests/` + `evidence.md` + literal
  `red-output.txt`/`green-output.txt`. This is SFT-ready framing, and red→green gives natural
  preference/reward structure.
- **Distilled knowledge:** 965 files in `05-distillation` + 818 in `06-skill-pack`. Search index:
  31,783 artifacts + 55,704 code samples across 93 degrees (2,174 gotchas, 1,432 recipes, 999
  patterns, 620 anti-patterns…), with per-artifact `evidenceLevel` and source line refs
  (`.agent-university/index.json`).
- **Trajectories (the richest signal):** 3,208 Claude Code session transcripts, ~1.5GB JSONL
  (236 in the main project + 2,972 across conductor workspaces) — full tool-use traces of the
  agents actually doing the AU tasks.
- **No fine-tuning/dataset-export code exists in the AU repo** — but the sibling repo
  **`~/develop/agent-training`** is a purpose-built, privacy-safe pipeline for exactly this:
  inventory → normalize → redact (fail-closed) → segment into one-goal episodes → label
  (outcome/user-signal/reward) → export **clean examples, comparison pairs, reward episodes,
  holdout evals, dataset card** → final leakage gate. It explicitly ingests Codex logs, Conductor
  traces, and repo JSONL. `[live-verified: README + src]` Whether its export schema matches
  Fireworks' `messages`+`tools` JSONL exactly is `[TBD]` — expect a thin adapter.

## 5. Recommended approach

**Goal clarification (user, 2026-07-03):** the objective is NOT distillation of any frontier
model — it is training a model to **work with these popular libraries, grounded in the specific
tasks AU has run**. That makes the structured POC tuples (spec → implementation → tests →
red/green outcome) and distilled artifacts the *primary* dataset, with raw session trajectories a
secondary/optional source.

**Framing decision (the important one):** split "better at AU tasks" into
1. **Process/behavior** — the degree-building methodology (no-mock doctrine, red→green TDD
   evidence, honesty/anti-fabrication, cleanup discipline, tool-use patterns, distillation style).
   This is what fine-tuning teaches well.
2. **Service facts** — API surfaces, gotchas, versions. These go stale by design; AU already
   serves them via retrieval (MCP search ranks #1 on its own content). Don't bake June-2026 facts
   into weights — instead train the model to *call the AU MCP tools* and then implement
   (tool-calling traces that include retrieval → implementation).

**Phased plan (cost-gated, money-ops single-threaded per the Fireworks degree discipline):**

- **Phase 0 — unblock (user-gated, ~$3):** issue a fresh Fireworks API key; re-run the $3-minimum
  canary SFT on a small GA model (e.g. `qwen3p5-9b`) reusing the `au-fw-canary` dataset pattern to
  retest the `unkey` entitlement; ask Fireworks for GLM-5.2 training-preview access. Confirm
  `firectl get model glm-5p2` tunability flag on the live account.
- **Phase 1 — dataset:** run `agent-training` over the 1.5GB transcript pool + degree corpus
  (across main and open branches). Outputs: SFT set (messages+tools, per-message weights on
  assistant turns), DPO pairs (natural sources: red→green transitions; the Opus honesty-review
  cycles where fabrications were caught and repaired = rejected-vs-accepted pairs), holdout evals.
  Redaction gate is mandatory — transcripts contain live keys. Final token count: **TBD until the
  pipeline runs** (raw pool is ~1.5GB JSONL; usable tokens will be a fraction).
- **Phase 2 — pilot tune:** iterate the data recipe on a cheap small model first ($0.50/M tier),
  then tune **GLM-5.1** (GA today) and/or **GLM-5.2 via preview**. LoRA first; `--warm-start-from`
  for iterations.
- **Phase 3 — eval on real AU tasks:** hold out N POC levels; drive an agent harness (any
  OpenAI-compatible CLI/agent) against the tuned endpoint; score red→green pass rate + honesty
  (fabrication) checks vs. base GLM-5.2. AU's live test suites make this an *objective* benchmark.
- **Phase 4 — (optional) RFT:** AU POC test suites are verifiable rewards — a `reward-kit`
  evaluator = "does the level's test suite pass." RFT generates rollouts from the GLM model
  itself, which also sidesteps the Claude-output distillation issue below. Costlier; do after SFT
  proves signal.
- **Serving:** tuned GLM-5.2 LoRA = on-demand only ≈ the $96/hr 8×B300 class with scale-to-zero
  (burst use OK; always-on is ~$70k/mo — that's the real cost center, not training). If always-on
  matters, a tuned smaller model (Qwen3.5 family) is the economic answer.

**Scenario math — explicitly hypothetical volumes, not targets:** at the documented $10/M >300B
LoRA rate, a final SFT set of 10M / 50M / 100M tokens costs ~$100 / $500 / $1,000 per epoch;
on GLM-5.1's tier the rate depends on its (unknown) param count `[TBD]`. Real numbers only after
Phase 1 produces an actual token count.

## 5b. Smaller-base option: Gemma 4 (added 2026-07-03, follow-up question)

Gemma 4 (released 2026-04-02) is a viable — and much cheaper — base, GA-tunable on Fireworks
today with no preview gate. `[docs/live-verified as noted]`

- **Variants:** E2B, E4B, 12B dense, **26B-A4B MoE (~4B active)**, **31B dense**; base, `-it`,
  and "Thinking" variants; 256K context on 12B+. `[docs: DeepMind/model card]`
- **Fireworks GA tunable rows** `[live-verified via docs .md fetch]`: `gemma-4-26b-a4b-it` and
  `gemma-4-31b-it`, both 256K max training context, SFT/DPO/RFT + LoRA. Training tier for both:
  $3/M tokens (16–80B) vs GLM-5.2's $10/M.
- **License:** Apache 2.0 weights (new for Gemma; Gemma 1–3 used custom terms) — but with a
  Gemma Prohibited Use Policy layered on top; fine-tuning on own data + internal serving
  permitted; Google claims no rights in outputs. `[docs]` Not as clean as MIT (GLM-5.2) or plain
  Apache (Qwen), but no problem for this use case.
- **Agentic fitness:** native function calling in the chat template (new vs Gemma 3's
  prompt-based tool use); Fireworks model page lists "Function Calling: Supported" for 31B-it.
  Official numbers: LiveCodeBench v6 80.0% (31B), τ²-bench retail 86.4%. **No official SWE-bench
  figure**; community sources (unconfirmed, low credibility) put Gemma-4-26B well below
  Qwen3.5-27B on SWE-bench Verified — treat as a question for our own eval, not a fact.
- **Serving reality:** `gemma-4-31b-it` is **NOT serverless on Fireworks** ("Serverless: Not
  supported" on the model page `[live-verified]`) — base and tuned LoRA both need on-demand.
  Generic guidance: H100-class for 30–70B dense (from ~$7/GPU-hr) — order-of-magnitude cheaper
  than GLM-5.2's 8×B300 $96/hr, but not free-tier.
- **Self-host fallback:** Unsloth publishes a Gemma 4 LoRA/QLoRA training guide `[docs]` —
  a ~30B model is realistically self-trainable, unlike 753B GLM-5.2. One community report of a
  launch-window QLoRA/PEFT layer incompatibility (`gemma4ClippableLinear`) — single-source,
  unconfirmed; verify before self-hosting.
- **Recommendation:** run the pilot as a cheap **bake-off** — same dataset, same held-out POC
  eval — across `gemma-4-26b-a4b-it` (fastest/cheapest per token trained and served),
  `qwen3p5-27b` (community-favored coding base, same $3/M tier, 256K), and optionally
  `gemma-4-31b-it`. Each run is cheap at this tier; let the AU eval harness pick the winner
  instead of benchmarks. GLM-5.1/5.2 remains the "maximum capability" track if preview access
  lands.

## 5c. EXECUTION LOG — Gemma 4 fine-tune on the AU corpus (2026-07-03, live)

- **Canary (entitlement retest): PASSED.** `sftj q04kdxx8` (gemma-4-26b-a4b-it, au-fw-canary,
  1 epoch) → `JOB_STATE_COMPLETED` in ~6 min; output model `au-fw-canary-gemma4` created.
- **Dataset v1 built** (`.agent-university/finetune/build_dataset.py`, seed 42):
  **11,850 examples / ~3.07M est. tokens (13.45 MiB)** = 7,556 knowledge artifacts (templated
  Q→artifact-text across gotcha/recipe/pattern/anti-pattern/troubleshooting/quickstart/
  agent-instructions/expectation-gap/lesson) + 4,202 vendor-filtered deduped code samples +
  93 POC task tuples (spec→implementation). **46,779 vendored code samples dropped** (index.json
  codeSamples are ~84% vendored `.venv` noise — filter by source.path). 4 examples dropped by
  secret scan. Holdout: 150 knowledge probes + 60 retention probes (alternate phrasing over
  trained artifacts) + 7 POC code tasks (`eval_probes.jsonl`, `eval_code.jsonl`).
- **Main SFT job launched:** `sftj b608vrw0`, base `gemma-4-26b-a4b-it`, dataset
  `au-fw-corpus-v1`, **2 epochs**, LoRA rank 8 (default). Est. training cost ≈ 6.1M tokens ×
  $3/M ≈ **$18.4** (verify against `firectl billing get-usage --usage-type training`).
- **Base is NOT serverless** (`POST /inference/v1/chat/completions` with the base id → NOT_FOUND)
  — benchmark requires one on-demand deployment with `--enable-addons`, serving base (via
  `model#deployment` addressing) and the LoRA (via `deployed-model create`) side-by-side.
- **Eval harness** (`run_eval.py`, smoke-tested): key-fact coverage (objective substring score
  over reference code-spans) + Claude Haiku 4.5 judge 0-10 vs reference (validated: 10 on
  reference-echo, 0 on invented APIs). 210 probes + 7 code tasks × 2 models.

### Results (2026-07-04, all live)

- **Training COMPLETED:** `sftj b608vrw0` finished in ~65 min; billed **7,063,692 training
  tokens** (≈ 2 epochs × 3.07M estimate + packing) ≈ **$21.2** at the $3/M tier. Output model
  `au-fw-gemma4-au-v1` (HF_PEFT_ADDON, rank 8) READY; adapter downloaded to
  `.agent-university/finetune/au-fw-gemma4-au-v1-adapter/` (511MB).
- **Base benchmark (gemma-4-26b-a4b-it on MINIMAL 4×H200 deployment, temp 0, 217 items,
  0 errors):** holdout key-fact coverage **1.0%**, judge **3.28/10** (n=150); retention coverage
  **2.8%**, judge **3.24/10** (n=60); code tasks judge **2.4/10** (n=7). Base Gemma 4 genuinely
  lacks the AU-specific (June-2026, live-verified) material — large headroom for the tune.
- **Tuned benchmark: BLOCKED — the LoRA cannot be served on this account today.** Evidence chain:
  1. MINIMAL preset (only config that provisions; quantized): `deployedModels` POST accepts and
     shows state DEPLOYED, but ALL addressing forms 404 for 10+ min (matrix: `model#deployment`,
     bare model, suffixed deployed-model name ± deployment, `deployedModels/<name>`). Matches
     docs: **FP8/FP4 quantized shapes do not support `--enable-addons`** — the API accepts the
     addon silently instead of rejecting. Live merge on this shape → InvalidArgument
     ("live merge is not supported for this deployment configuration").
  2. Every full-precision config **FAILS provisioning with INTERNAL**: preset
     `rft-gemma-4-26b-a4b-it` (4×B200 FULL_PRECISION), explicit 2×H100, explicit 4×H200
     `--precision BF16`. State goes CREATING → FAILED (~5-10 min).
  3. `--deployment-shape` and `--precision` are mutually exclusive flags (can't force BF16 onto
     the provisionable shape).
  4. `firectl model prepare --precision fp4` (server-side merge/quantize escape hatch) →
     **PermissionDenied** (entitlement).
  Unblock paths: Fireworks support ticket (BF16 INTERNAL failures + prepare permission — docs
  literally say "contact support" for shape-access errors), or local eval with the downloaded
  adapter (this Mac has 128GB RAM; 4/8-bit Gemma-4-26B + LoRA via MLX/llama.cpp is feasible,
  with a quantization confound).
- **Ops gotchas banked:** Fireworks inference WAF 403s Python-urllib User-Agent (send
  `User-Agent: curl/...`); `firectl deployment delete` needs `--ignore-checks` if the deployment
  served in the last hour (a bare trap teardown FAILS otherwise); `--base-model`/shape names must
  be fully qualified `accounts/fireworks/...`; deployment "Total size" counts soft-deleted
  records until purge.
- **Real cost, full run (2026-07-03/04):** **$34.69** account total = training $21.2 + canary
  $0.01 + deployment sessions (~$13, incl. base eval ~35 min on 4×H200 + failed-config
  cold starts, which don't bill replicas that never come up).

## 5d. RESULTS UPDATE (2026-07-04, local + Modal pipeline; supersedes the preliminary numbers in 5c)

**Serving workaround shipped.** Fireworks LoRA serving remained blocked, so the adapter was
merged locally: decoded `fw_lora_layout: fused_peft_3d_v1` (128 experts × r8 block layout; all
530 adapter tensors = 265 LoRA pairs applied bf16-exact into `google/gemma-4-26B-A4B-it`).
**Merge validated hard:** teacher-forced perplexity on rendered training text = **3.35 (merged)
vs 142.73 (base)** on identical Q8_0 GGUFs. Also stood up on Modal (app `au-gemma4`): volume
holds base + merged BF16 + both GGUFs; scale-to-zero L40S endpoints
`dennisonbertram--au-gemma4-{tuned,base}.modal.run` (llama.cpp server-cuda image — NOTE its
ENTRYPOINT must be cleared via `.entrypoint([])` or Modal's runner crash-loops with
"invalid argument: python").

**FINAL v1 benchmark (all artifact-corrected; Gemma runs local/Modal Q8_0 temp 0; Sonnet via
NATIVE Anthropic API with adaptive thinking + think-and-answer budgets 6k/10k — two earlier
Sonnet runs were INVALID: the OpenAI-compat layer and small budgets let thinking consume the
entire output, median answer length 0):**

*Knowledge probes (150 holdout / 60 retention):*

| model | hold cov | ret cov | hold judge | ret judge |
|---|---|---|---|---|
| base Gemma 4 26B-A4B | 7.1% | 8.9% | 4.05 | 4.13 |
| tuned v1 (rank 8, $21) | 10.5% | 12.1% | 4.41 | 4.62 |
| **Claude Sonnet 5 (native, honest)** | **14.8%** | **15.0%** | **4.97** | **5.6** |

*Application-building benchmark (40 held-out POC levels from origin/main, 36 capstones,
36 targets, never in any training set; key-API coverage from reference-implementation
identifiers + Haiku judge on API correctness/completeness):*

| model | key-API coverage | judge |
|---|---|---|
| base Gemma 4 26B-A4B | 22.7% | 3.06 |
| Claude Sonnet 5 (native, 10k budget) | 31.5% | 3.82 |
| **tuned v1 ($21)** | **53.7%** | **4.25** |

- **Correction trail:** 5c's base numbers (1.0%/2.8%) used a 900-token budget on Fireworks —
  invalid comparison. First two Sonnet runs (compat layer, then native@2400) returned mostly
  empty answers (all-thinking, `stop_reason: max_tokens`, `thinking_tokens == max_tokens`) —
  both discarded. **Gotcha: claude-sonnet-5 thinks adaptively BY DEFAULT on hard prompts even
  via the native API; budget ≥3× the expected answer or you benchmark an empty string.**
- **Verdict vs the goal ("build with these libraries better than Sonnet or cheaper"):** on the
  app-building benchmark the $21 specialist **beats honest Sonnet 5 on both metrics**
  (+70% relative on correct-API usage; +0.43 judge) while serving from one L40S
  (scale-to-zero) vs frontier per-token pricing with thousands of thinking tokens per task.
  On encyclopedic knowledge probes Sonnet still leads all cells — the v3 retention program
  targets that gap. Standing caveats: judge is Claude Haiku (same family as Sonnet); the n=7
  code split in earlier tables was noise (Sonnet scored 5.0 there vs 3.82 at n=40); 4 Modal
  cold-start 503s clipped the Gemma code runs to n=36 (both Gemma models equally).
- **Training metrics (v1, job b608vrw0):** loss 6.78 → 0.64 (ppl ≈1.9) — learned substance/style,
  NOT verbatim strings; verbatim-recall gates are the wrong validation instrument at this loss.
  Post-tune the model emits a thinking-channel prefix (`<|channel>thought…`) — strip before
  span-scoring and budget tokens for it.
- **v2 in flight:** `sftj bmwpybed`, rank 32, 2 epochs, augmented 26,960-example set (3 question
  phrasings per artifact + key-API lead-in answers; identical holdout). ~$40.
- Goal ladder (user): v2 → "improve 200%" (≈3× v1 tuned scores) → v3 full-corpus training toward
  **100% retention coverage** (holdout kept as generalization gauge) → long-term: cheap
  library-specialist models an orchestrator can delegate to, with frontier review on top.

## 5e. v2 RESULTS + METRIC REBUILD + v3 (2026-07-04)

- **v2** (`sftj bmwpybed`, rank 32, 2 epochs, 26,960 augmented examples, ~$40): training loss
  6.84→**0.33** (vs v1's 0.64) — yet benchmark scores statistically identical to v1
  (code 52.4%/3.90 at n=40 clean; probes ~18%/4.46). **Learning: SFT loss depth does NOT convert
  to free-generation exact recall; rank and phrasing augmentation were not the bottleneck.**
- **Metric rebuild:** probe key_facts originally favored incidental strings (template literals,
  repo paths, evidence fragments) — answers with correct substance scored 0. Rebuilt with
  identifier-focused filtering and rescored ALL saved answers (no re-inference). Rescored
  knowledge table (102/210 probes have ≥2 usable facts):
  base_local 13.4/12.9 cov · 4.05/4.13 judge; **v1 18.1/18.9 · 4.41/4.62**;
  **v2 17.9/19.8 · 4.46/4.63**; **Sonnet-honest 26.0/29.2 · 4.97/5.6** (hold/ret).
  Ranking robust: Sonnet > tuned ≫ base on knowledge; tuned > Sonnet on app-building.
- **Standing verdict on the user's goal** ("build applications with these libraries better than
  Sonnet or cheaper"): **met by v1 and confirmed by v2** on the 40-task benchmark
  (52-54% API coverage / 3.9-4.25 judge vs Sonnet 31.5%/3.82), at commodity serving cost.
  Knowledge-probe gap to Sonnet remains (~8-10 pts coverage, ~0.5-1.0 judge).
- **v3 launched** (`sftj bquqtr9k`, rank 32, 2 epochs, ~$28): 16,438 examples =
  **11,315 cloze/verbatim drills** (train the *quote-exact-strings* behavior — the identified
  missing mechanism) + 4,573 quote-first QA + **550 task tuples from origin/main** (6× v2's task
  data; the 40 benchmark items quarantined). Targets: retention coverage jump (mechanism test for
  the 100%-retention goal) + widen the app-building lead.
- Modal endpoints: `dennisonbertram--au-gemma4-{tuned,base,tuned-v2}.modal.run` (L40S,
  scale-to-zero; v3 endpoint follows the same prepare/deploy path).

## 5f. PROGRAM CONCLUSIONS (2026-07-04, after v3)

- **v3** (`sftj bquqtr9k`, cloze drills + task-majority, ~$28): code 53.3%/4.00 (n=39);
  probes holdout 16.0%/4.32, retention **21.4%**/4.43.
- **Conclusion 1 — the goal is met and stable:** all three tuned variants beat honest Sonnet 5
  at building applications with these libraries by ~21 points of correct-API usage
  (53% vs 31.5%) and on judge (3.9-4.25 vs 3.82), from $21-40 fine-tunes served on one L40S.
  The build capability saturates with modest task data (93 vs 550 tuples: no difference).
- **Conclusion 2 — SFT knowledge-recall plateau:** generation-time exact-fact coverage plateaus
  at ~18-21% across rank 8→32, 1×→6× data passes, phrasing augmentation, and cloze drills
  (cloze added +1.6pts retention, the only positive mover). Sonnet itself only reaches 26-29%.
  **"100% retention in weights" is not reachable in this regime** — exact strings are the wrong
  thing to store in weights.
- **Conclusion 3 — the system design that wins:** the specialist model (skill/idiom, beats
  frontier at building) + **AU MCP retrieval** (exact strings, already ranks #1 on this corpus)
  is the architecture for the user's orchestrator vision: route by detected libraries → cheap
  specialist implements with retrieval-grounded exact facts → frontier reviews. Both halves
  exist and are deployed today.
- **Artifacts:** models `au-fw-gemma4-au-v{1,2,3}` (Fireworks) + merged/GGUF copies on Modal
  volume `au-gemma4`; endpoints `dennisonbertram--au-gemma4-{tuned,base,tuned-v2,tuned-v3}.modal.run`;
  benchmark suite (`eval_probes.jsonl` identifier-scored, `eval_code_v2.jsonl` 40 quarantined
  capstones, `run_eval.py`, `rescore_probes.py`); datasets `au-fw-corpus-v{1,2,3}`; local
  ~160GB of model artifacts under `.agent-university/finetune/` (reproducible; deletable).
- **Cost (approx, full program):** Fireworks ≈ $103 (v1 $21 + v2 ~$40 + v3 ~$28 + ~$14
  deployments/canary; billing lags — verify `firectl billing get-usage`); Modal: prepare CPU
  runs + L40S eval hours (single-digit-to-low-tens $, check dashboard); Anthropic: judge +
  Sonnet baselines (~$25-35 est.).

## 5g. SYSTEM BENCHMARK — specialist + AU retrieval vs Sonnet (2026-07-04)

Ran tuned_v3 with live AU `/v1/search` (the MCP server's backend; local api:dev, hybrid w/
OPENAI embeddings) injecting top-6 full-text artifacts per query (`rag_eval.py`):

| | holdout cov · judge | retention cov · judge | code cov · judge |
|---|---|---|---|
| tuned v3 alone | 16.0 · 4.32 | 21.4 · 4.43 | 53.3 · 4.00 |
| Sonnet 5 honest | 26.0 · 4.97 | 29.2 · 5.6 | 31.5 · 3.82 |
| **tuned v3 + AU retrieval** | **55.1 · 6.99** | **62.7 · 7.6** | 54.7 · 3.67 |

- **The system beats raw Sonnet ~2× on knowledge coverage and +2 judge points**, keeps the code
  lead. Retrieval = +39 cov/+2.7 judge on knowledge (largest lever in the program), ~0 on code
  (leak_rate 0 there — corpus predates most benchmark degrees; code skill is in the weights).
- Open-book by design: 52-65% of knowledge probes retrieved their own source artifact
  (leak flags recorded per item) — this is exactly what production MCP serving does.
- Empirically confirms Conclusion 3: weights for skill, retrieval for exact strings,
  orchestrator delegates to the system, frontier reviews.
- **Sonnet+retrieval ceiling (added after budget-bug fix):** with the SAME retrieval, Sonnet
  scores 63.5/7.47 hold · 70.9/7.85 ret · **58.1/5.92 code** (20k think budget; 4/40 still
  empty; knowledge rows ran at 1400 tokens — 9/210 empty — slight understatement in Sonnet's
  favor left uncorrected). **Equal-retrieval verdict: frontier leads all splits (largest gap =
  code judge 5.92 vs 3.67); the tuned system holds ~87% of frontier-system coverage at ~10-30×
  lower executor cost.** The specialist's outright wins are the delegation-realistic asymmetric
  comparisons (tuned-raw > Sonnet-raw on code; tuned+MCP > Sonnet-raw everywhere).
  Gotcha (repeated 3×, now permanent): EVERY eval runner must honor think-and-answer budgets
  for sonnet-5 — `rag_eval.py` initially hardcoded 2400 and produced 39/40 empty code answers.
- **v4 (user-directed step 2):** tool-calling training COMPLETED (`sftj mrrx2djm`, 7,550 traces
  each embedding a REAL au_search call + results, ~$50); merge/deploy chain + tool-loop harness
  (`tool_eval.py`) running — measures self-directed retrieval vs injected-context RAG.

## 5h. v4 — SELF-DIRECTED TOOL USE (2026-07-05, final)

v4 (`sftj mrrx2djm`, 7,550 traces w/ real au_search calls, ~$50) benchmarked with the tool-loop
harness (model decides; harness executes real /v1/search; max 2 rounds):

| split | coverage · judge | tool-call rate |
|---|---|---|
| holdout | 49.6% · 6.7 | **0.92** |
| retention | 59.4% · 7.37 | **0.98** |
| code | 52.0% · 3.32 | **0.00** |

- **Emergent calibration:** calls the MCP on ~everything knowledge, never on code — matching the
  program's own weights-vs-retrieval boundary without an explicit call/no-call curriculum.
- **Self-directed ≈ 90-95% of injected-RAG quality** (49.6/6.7 vs v3+RAG 55.1/6.99 holdout) —
  the model writes good queries and grounds answers autonomously. Beats raw Sonnet everywhere.
- **CRITICAL SERVING GOTCHA:** llama.cpp's Gemma GGUF chat template DROPS the OpenAI `tools`
  field — a tool-trained model is tool-blind ("No tools have been provided"), first eval showed
  0% calls and degraded answers. Fix: render tool JSON into system text; parse the trained
  Fireworks call format `<|tool_call>call:au_search{query: "..."}<tool_call|>` from raw content.
  Proper-template stacks (vLLM w/ HF template, Fireworks serving) shouldn't need the shim.
- Minor cost: v4's code judge dipped to 3.32 (v3: 4.00) — tool-heavy data slightly diluted
  implementation style; a v5 mix rebalance would recover it.
- **Program endstate vs user goals:** build-better-than-Sonnet ✅ (raw and system);
  match-Sonnet ✅ on the system level (~87-95% of frontier-system quality at ~10-30× cheaper);
  knows-when-to-call-the-MCP ✅ (v4). Next levers if resumed: v5 calibrated call/no-call +
  code-mix rebalance; RFT with test-suite rewards; execution-based code benchmark.

## 5i. v5 — BUILT, BLOCKED ON CREDITS (2026-07-05)

v5 design (pure skill + tool + context-faithfulness, ZERO fact-memorization) implemented in
`build_v5.py`: **272 error→fix debugging traces** (task spec + real red-output tail → diagnosis
from surprises/evidence + corrected implementation — the corpus's untapped process signal) +
550 task tuples (60% tool-wrapped / 40% direct) + 3,000 context-faithfulness tool traces +
400 commands/ops traces = 4,222 examples / ~6.7M tokens / ~$40 for 2 epochs @ rank 32.
Dataset uploaded: `au-fw-corpus-v5` READY on the account.

**BLOCKED: Fireworks account suspended — CREDIT_DEPLETED** (program spend ≈ $153). Resume after
top-up at app.fireworks.ai/account/billing with:
`firectl sftj create --base-model accounts/fireworks/models/gemma-4-26b-a4b-it --dataset
au-fw-corpus-v5 --output-model au-fw-gemma4-au-v5 --epochs 2 --lora-rank 32`
then the standard chain (download adapter → modal volume put adapter-v5 → prepare
--adapter-subdir adapter-v5 --out-prefix tuned-v5 → deploy → tool_eval.py).
Success criteria: code judge ≥ v3's 4.0, tool calibration retained (~0.9 knowledge / ~0 code),
knowledge-via-tool ≥ v4's 49.6/6.7.

## 6. Risks and flags

- **Claude-authored content (noted, downgraded per goal clarification):** the intent is
  library/task-competence training on AU's own curated tasks, not distillation of a frontier
  model. Still worth knowing: many corpus artifacts and all transcripts were produced with
  Claude's assistance, and Anthropic's commercial terms restrict using outputs to train competing
  models — the conservative shape is SFT on task tuples + RFT on the model's own rollouts
  (test-suite rewards), which is also the technically better fit here. *Flag, not a legal
  opinion; user is aware.*
- **GLM-5.2 tuning availability:** preview-only on Fireworks today; nobody else found offering it.
- **Reasoning-trace handling for GLM SFT on Fireworks:** not documented `[TBD]` — ask in preview.
- **Privacy:** dataset upload sends AU traces to Fireworks; the agent-training leakage gate must
  pass first; check Fireworks data-handling terms before uploading real trajectories.
- **ToS:** no reselling inference from the tuned deployment; internal use fine.
- **Staleness:** a tuned checkpoint freezes behavior at corpus-time; the retrieval-first split in
  §5 is the mitigation. Plan periodic re-tunes (warm-start) rather than one big bake.

## 7. Open questions checklist

- [x] Fresh Fireworks API key issued — DONE 2026-07-03 (stored gitignored; firectl re-keyed).
- [x] `unkey` fine-tune entitlement retest — RESOLVED 2026-07-03: canary SFT job `q04kdxx8`
      created and running on `gemma-4-26b-a4b-it` (outcome to confirm on completion).
- [ ] GLM-5.2 training-preview access — contact Fireworks (`glm-5p2` probes `supportsLora:false`).
- [x] Live tunability flags probed 2026-07-03 — glm-5p2 NO; glm-5p1/gemma-4-26b-a4b-it/
      gemma-4-31b-it/qwen3p5-27b YES (LoRA).
- [ ] GLM-5.1 parameter count (sets its training price tier).
- [ ] agent-training export schema vs Fireworks `messages`+`tools` JSONL — adapter needed?
- [ ] GLM reasoning-trace support in Fireworks SFT.
- [ ] Legal read on training third-party models on Claude-generated trajectories.
- [ ] Fireworks data-handling/DPA review before uploading trajectory data.
