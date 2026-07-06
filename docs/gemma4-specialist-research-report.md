# Library-Specialist Models: Fine-tuning Gemma 4 on the Agent University Corpus

**Research report — 2026-07-03 → 2026-07-05 (v6a results pending, section 7)**
Companion documents: `glm-5-2-fine-tune-research-2026-07-03.md` (full execution log),
`gemma4-hypothesis-ledger.md` (per-run hypotheses and verdicts),
`gemma4-au-finetune-harness/` (all code, benchmarks, and per-run summaries).

---

## 1. Question and motivation

Can a small, cheap, open-weights model be fine-tuned on a curated corpus of library-specific
engineering knowledge to match or beat a frontier model *on that stack* — cheaply enough that an
orchestrator can delegate library work to specialists and reserve the frontier model for review?

The corpus: Agent University — ~90 curricula for AI/agent libraries and developer tools
(Supabase, MCP, Next.js, Cloudflare, Slack, sqlite, …), each with live-verified gotchas,
recipes, reference implementations, tests, and captured failure evidence. The base model:
`google/gemma-4-26B-A4B-it` (MoE, ~4B active parameters, Apache-2.0). Training: LoRA supervised
fine-tuning on Fireworks ($21–64/run). Comparison model: Claude Sonnet 5 with honest
think-and-answer token budgets.

## 2. Measurement

Three benchmark suites, all scored against live-verified references with two metrics
(objective key-identifier coverage + a Claude Haiku judge, 0–10):

| Suite | n | What it tests | Contamination control |
|---|---|---|---|
| BUILD | 40 | Implement an application-level task (mostly capstones) | Levels excluded from every training set |
| KNOW | 210 | Library facts (150 never-trained + 60 rephrased-trained) | Article-level exclusion |
| DEBUG | 25 | Diagnose a real captured failure (spec + failing output) | Same excluded levels as BUILD |

Measurement lessons that materially changed results (details §6): thinking models silently
return empty answers under tight token budgets; naive "key fact" extraction rewards noise;
n=7 code samples produced a reversed conclusion later overturned at n=40.

## 3. The experiments

Six training runs, each a registered hypothesis (full verdicts in the ledger):

| Run | Recipe | $ | Headline result |
|---|---|---|---|
| v1 | corpus Q→A + tasks (rank 8) | 21 | BUILD 53.7% vs Sonnet 31.5% — **specialist out-builds frontier** |
| v2 | rank 32 + 3× phrasings | 40 | No recall gain despite 2× lower loss — **loss ≠ recall** |
| v3 | + exact-string drills, 6× tasks | 28 | +1.6pts retention only — **facts don't belong in weights** |
| v4 | 7.5k tool-calling traces | 50 | **Emergent calibration**: searches corpus on 92–98% of fact questions, 0% while coding |
| v5 | debug traces + skills, no memorization | 48 | Flat everywhere incl. DEBUG — **one-shot imitation ≠ debugging skill** |
| v6a | v4 + tool-grounded debug traces | 64 | *pending — §7* |

## 4. Results

### 4.1 Building applications (the goal metric)

| model | correct-API coverage | judge |
|---|---|---|
| base Gemma 4 | 22.7% | 3.06 |
| Claude Sonnet 5 | 31.5% | 3.82 |
| **tuned (v1/v3/v4)** | **52–54%** | **3.9–4.25** |

Every tuned variant beats the frontier model at building with these libraries, by ~21 points of
correct-API usage. The capability saturates with ~100 task examples and is insensitive to
5 further data recipes.

### 4.2 Knowledge: weights vs retrieval

| configuration | never-trained cov · judge | rephrased cov · judge |
|---|---|---|
| tuned alone (best) | 18% · 4.4 | 21% · 4.6 |
| Sonnet alone | 26% · 5.0 | 29% · 5.6 |
| **tuned + AU search (injected)** | **55% · 7.0** | **63% · 7.6** |
| Sonnet + AU search | 63% · 7.5 | 71% · 7.9 |
| **tuned v4, self-directed tool use** | 50% · 6.7 | 59% · 7.4 |

Exact-fact recall in weights plateaus ~20% across every training recipe (Sonnet itself only
reaches ~27%). Retrieval is worth +35–40 points to *any* model. v4 achieves ~90–95% of
spoon-fed retrieval quality while deciding **on its own** when to search.

### 4.3 Debugging (added after v5)

| model | root-cause coverage | judge | searches on errors |
|---|---|---|---|
| base | 20.5% | 2.59 | — |
| v3 / v4 / v5 | 21–25% | 2.8–3.6 | 0–4% |
| Sonnet 5 | 35.2% | 3.80 | — |

The frontier leads. Tuned diagnoses are *directionally* competitive (v4 judge 3.6 vs 3.8) but
miss exact identifiers — and no variant thinks to search the corpus when shown an error, even
though many of these failures' gotchas are retrievable. v6a targets exactly this (§7).

### 4.4 Cost

Measured per build task: **~$0.01–0.03** (tuned, one $2/hr L40S) vs $0.13 (Sonnet) /
$0.22 (Opus 4.8) / $0.43 (Fable 5) at current API prices — **4–40× cheaper**, with training a
one-time $21–64. The recommended deployment (specialist executes + frontier reviews) lands
around $0.05/task with frontier-grade oversight.

## 5. Conclusions

1. **The specialist thesis holds.** A $21 fine-tune out-builds a frontier model on a specific,
   well-curated stack, at commodity serving cost. This result was stable across all six runs.
2. **Weights are for skill and idiom; retrieval is for facts.** Four escalating attempts to
   train exact-fact recall failed identically; retrieval closes the gap instantly and never
   goes stale as the corpus updates.
3. **Harness behavior is trainable — and calibrates itself.** The single strongest result:
   tool-use traces produced a model that knows *when* to look things up (fact questions yes,
   coding no) without that policy ever being stated.
4. **Imitation SFT has a measurable ceiling.** Six data recipes moved build quality ~0 after v1.
   Debugging didn't improve even when trained directly on debug examples. The evidence points
   to reward-based training (RFT) as the next axis — currently gated on Fireworks'
   Training API private preview.
5. **Measurement is half the work.** Three separate benchmark artifacts (empty thinking-model
   answers, noisy key facts, tiny-n reversals) each temporarily produced a wrong conclusion.
   The hypothesis ledger + pre-registered decision rules are now standing practice.

## 6. Deployment architecture (validated end-to-end)

Orchestrator detects the libraries in a task → routes to the specialist (one shared base on a
GPU; per-library LoRA adapters selected per request) → the specialist consults AU search on its
own → frontier model reviews the output. Every component exists and is benchmarked. Exact
cross-adapter KV-cache sharing is mathematically unavailable; per-adapter prefix caching plus
short specialist briefs make it unnecessary.

**Artifacts:** 10 public Hugging Face repos (`dennisonb/gemma-4-26b-a4b-it-au-v{1..5}-{adapter,gguf}`),
Modal serving endpoints (scale-to-zero L40S), the full eval suite, the fused-adapter merge
pipeline, and ~$250 of total training/eval spend across the program.

## 7. v6a — grounded debugging (resolved: disproven, with regression)

Registered hypothesis (ledger H9a): teaching the model to *search the corpus when shown an
error* — 544 tool-mediated debug traces on top of v4's corpus — lifts DEBUG coverage toward
Sonnet's 35%. Pass bar: ≥30% coverage AND ≥60% call-on-error rate.

**Result:** DEBUG 20.5% / 2.6 judge / **0% search-on-error** — both bars missed, base-model
level. Worse, BUILD regressed to 40.7% / 2.38 (v4: 52.0% / 3.32): the added traces diluted the
core skill without buying the new one.

**Why this is the program's most instructive failure:** the *identical* training mechanism —
multi-turn traces with real tool calls — produced near-perfect self-directed tool use on
knowledge questions (v4, 92–98%) and produced nothing on error-triggered tool use, even with
direct demonstrations. Knowledge questions resemble their training examples closely; a fresh
failing stack trace never matches a memorized one. The behavior appears to require **on-policy
learning** — the model practicing on its own rollouts with a reward — not imitation. This
sharpens conclusion §5.4 from "SFT has a ceiling" to "SFT transfers pattern-matched behaviors
only; novel-trigger behaviors need RFT."

**Final model ranking:** v4 is the program's flagship (best build score among tool-users,
calibrated tool use, best debug judge among tuned models). v6a will not be published.

## 8. Open items

- **v6b (RFT)**: reward function + dataset built and validated; blocked on Fireworks Training
  API private preview (one consolidated access request also covers GLM-5.2 tuning preview and
  the BF16 serving bug).
- **GLM-5.2 transfer (H10)**: deferred by decision until training venue is clear; the entire
  pipeline transfers unchanged.
- **Execution-based BUILD scoring**: replace judge-vs-reference with "do the tests pass" once
  sandboxed per-library services are practical.
