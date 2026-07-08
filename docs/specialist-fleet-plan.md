# Specialist-Fleet Endpoint — Plan (registered 2026-07-08)

**Goal:** one Anthropic/OpenAI-compatible endpoint backed by a single resident
Gemma-4-26B-A4B base + a library of LoRA specialists selected per request + the
Agent University MCP server for retrieval. Proves the "orchestrator delegates
library work to a cheap specialist" service vision end to end.

**Hypothesis H11 (pre-registered, per ledger practice):** library-scoped LoRAs
beat the whole-corpus generalist (v4) on their own library's tasks, and
per-request adapter routing through one shared base serves them at generalist
cost. **Decision rule:** each specialist must beat v4 on its own library's
benchmark slice AND lose to the other specialist on the other library's slice
(the 2×2 must be diagonal). If specialists do NOT beat the generalist, the
honest product conclusion is ONE generalist adapter + retrieval — no fleet.

Everything in v1–v6a was a whole-corpus generalist; per-library specialization
has never been tested. That is the point of this program.

---

## Architecture

```
client (Anthropic SDK or OpenAI SDK)
   │
   ▼
LiteLLM proxy          ← /v1/chat/completions (+ /v1/messages later, see risk R3)
   │  model_name → hosted_vllm/<adapter> on one api_base
   │  MCP gateway: AU search registered once, attached per request
   ▼
vLLM (one GPU, persistent while active)
   base: google/gemma-4-26B-A4B-it (BF16, ~57.7GB)
   --enable-lora --lora-modules supabase=<path> mcp=<path> generalist-v4=<path>
   --enable-auto-tool-choice --tool-call-parser gemma4
```

Per-turn specialist selection = the request's `model` field. No KV-cache
surgery needed: chat APIs are stateless, so switching adapters between turns
is free by construction.

## Venue decisions (from two sourced research reports, 2026-07-08)

- **Training:** self-run **Axolotl on RunPod A100-80GB** (~$1.19–1.49/hr).
  Managed alternatives are dead ends today: Together lacks the 26B-A4B MoE in
  catalog; Predibase absorbed into Rubrik; OpenPipe sunset by CoreWeave;
  HF AutoTrain deprecated. Axolotl ships real `gemma4/26b-a4b-moe-*.yaml`
  examples targeting 1×80GB. Avoid Unsloth for now (open issue #4907 +
  a reported vLLM-load format mismatch — same failure shape as Fireworks).
- **Adapter format rule (the Fireworks lesson):** venue is chosen by OUTPUT
  format. First pass trains **attention-only LoRA (q/k/v/o_proj)** — PEFT's
  own shipped default for gemma4, a plain 2D adapter vLLM loads with no
  special flags. Expert-layer (3D-PEFT) LoRA only as a second pass behind a
  mandatory load-and-generate smoke test — vLLM docs warn a wrong 2D/3D
  declaration "silently produce[s] garbage outputs".
- **Serving:** single **A100-80GB**, RunPod (community ~$1.19/hr, secure
  ~$1.39/hr), stopped when idle. vLLM's own recipe confirms 1×80GB BF16 with
  `--max-model-len 32768`. Skip FP8 (no LoRA kernel benefit; open MoE-FP8-LoRA
  bugs). Skip 2×L40S (no NVLink; vLLM advises against TP on L40S).
  Modal serverless is the better venue only if traffic is truly bursty.
- **Gateway:** LiteLLM. Start with the OpenAI `/v1/chat/completions` shape;
  the Anthropic `/v1/messages`→hosted_vllm bridge has an open streaming bug
  (BerriAI/litellm#30043) — adopt it when fixed or shim it ourselves.

## The one hard blocker (and its size)

`vllm serve gemma-4-26B-A4B-it --enable-lora` currently fails on main:
`AttributeError: To support LoRA for MoE model, 'get_expert_mapping' must be
implemented`. Fix PRs are open and unmerged (checked live 2026-07-08):
vllm-project/vllm #43254, **#46772 (freshest, 2026-07-02: +18/−8 across
gemma4.py/gemma4_mm.py)**, #40584. Phase 0 cherry-picks #46772 onto the
current release. Track for upstream merge; drop the patch when it lands.
Open question the spike answers: whether the startup check blocks even
attention-only 2D adapters, or only expert-touching ones.

## Phases and gates

**Phase 0 — serving spike (~$10–20, half a day).** RunPod A100-80GB; stock
vLLM first (the fix may have merged), else + #46772 patch. Verify: base
serves; `gemma4` tool parser emits clean OpenAI-shape tool calls; attempt
loading existing Fireworks v4 adapter via the 3D path (bonus, not
load-bearing). Kill gate: gemma4+LoRA unfixable in practice → dense-model or
alternate-engine rethink before any training spend.

**Phase 1 — two library-scoped specialists (~$40–100 incl. iteration).**
Proposed libraries: **Supabase + MCP** (deep corpora: 18/20 live POCs; both
in the existing benchmark; maximally distinct domains). Datasets: filter the
v4 builder (`build_v4_tools.py`) by target. Attention-only LoRA, standard
PEFT export. Gate: both adapters load by name into the Phase 0 server and
generate sanely. Raw GPU cost per run is single-digit dollars (sourced
pricing; throughput estimate flagged, not measured).

**Phase 2 — gateway (~1 day, no new cost).** LiteLLM in front: each adapter a
`model_name`; AU MCP server registered in the MCP gateway. Gate: a stock
OpenAI-SDK client pointed at the proxy hits different specialists on
successive turns and a real corpus retrieval round-trips.

**Phase 3 — 2×2 benchmark (H11 verdict, ~$10 judge costs).** Slices:
{supabase-specialist, mcp-specialist, generalist-v4, base} × {supabase tasks,
mcp tasks} with the existing harness; plus latency/$ through the full gateway
path. Verdict recorded in the hypothesis ledger either way.

## Budget

Phases 0–3 one-time: **~$60–130**. Steady-state serving while active:
**~$1.19–1.39/hr** (≈$250–350/mo at 8h/day; ~$0 when stopped). Numbers are
from live pricing pages 2026-07-08; volatile.

## Risks

- **R1 (gating):** vLLM MoE-LoRA patch doesn't work even cherry-picked, or
  breaks attention-only adapters too → fall back to dense base or alternate
  engine; decided at Phase 0 gate, before training money.
- **R2:** attention-only LoRA underperforms the expert+attention Fireworks
  adapters on build skill (v2's capacity finding suggests not, but that is
  inference). Phase 3 measures it directly against v4.
- **R3:** LiteLLM Anthropic-bridge streaming bug — start OpenAI-shape only.
- **R4:** RunPod community-tier reliability — use secure tier if flaky.
