# Gemma-4 Specialist Program — Hypothesis Ledger

One entry per experimental turn: what we believed, how we tested it, what the data said.
Rule going forward: **every run gets its hypothesis registered here BEFORE launch.**

Benchmarks referenced: **BUILD** = 40 app-building tasks excluded from all training;
**KNOW** = 210 knowledge questions (150 excluded from training / 60 rephrased);
**DEBUG** = 25 diagnose-the-real-failure tasks from excluded levels (added 2026-07-05).
All scored against live-verified references; Claude Haiku judge + objective key-fact coverage.

---

## H1 — Fine-tuning teaches the corpus (v1, $21)
**Hypothesis:** LoRA SFT on the AU corpus makes Gemma-4-26B meaningfully better at AU tasks than base.
**Result:** BUILD 53.7%/4.25 vs base 22.7%/3.06; KNOW 18.1% vs 13.4%.
**Verdict: VALIDATED.** Single cheapest, largest capability jump of the program.

## H1b — Specialist can beat a frontier model on its home turf (v1)
**Hypothesis (implicit, surfaced by user goal):** a tuned specialist can out-build Claude Sonnet 5 on these libraries.
**Result:** BUILD 53.7% vs Sonnet 31.5% (honest thinking-budget baseline).
**Verdict: VALIDATED** — and stable across v1/v2/v3 (52–54%).

## H2 — Recall is capacity/phrasing-limited (v2, $40)
**Hypothesis:** rank 32 (4× capacity) + 3× question phrasings will lift exact-fact recall.
**Result:** KNOW 17.9/19.8% ≈ v1's 18.1/18.9%, despite training loss halving (0.64→0.33).
**Verdict: DISPROVEN.** Key learning: training-loss depth does not convert to free-generation recall.

## H3 — Verbatim recall is a trainable behavior (v3, $28)
**Hypothesis:** cloze/exact-string drills (11.3k) teach the model to quote exact identifiers.
**Result:** retention +1.6pts (19.8→21.4%); holdout flat.
**Verdict: MOSTLY DISPROVEN.** Facts don't belong in weights; retrieval's job.

## H3b — More task data lifts build quality (v3)
**Hypothesis:** 6× task tuples (93→550) improves BUILD.
**Result:** 53.3% ≈ v1's 53.7%.
**Verdict: DISPROVEN.** Build skill saturates with ~100 task examples at this model scale.

## H4 — Specialist + retrieval beats raw frontier (RAG test)
**Hypothesis:** weights-for-skill + corpus retrieval outperforms a frontier model answering from memory.
**Result:** KNOW 55.1/62.7% cov · ~7.0-7.6 judge vs raw Sonnet 26.0/29.2% · 4.97/5.6.
**Verdict: VALIDATED (~2×).** Largest single quality lever in the program (+39 cov, +2.7 judge).

## H4b — With equal retrieval, specialist ≈ frontier
**Result:** Sonnet+same-retrieval leads all splits (63.5/70.9/58.1); specialist system ≈ 87–95% of it at ~10–30× lower executor cost.
**Verdict: PARTIAL.** Parity not reached; the gap is priced exactly for a frontier review pass.

## H5 — Tool use (incl. WHEN to call) is trainable (v4, $50)
**Hypothesis:** multi-turn traces with real `au_search` calls teach both the call format and the judgment of when to call.
**Result:** 92–98% call rate on knowledge, 0% on code — emergent, uncoached calibration; self-directed quality ≈ 90–95% of spoon-fed retrieval.
**Verdict: VALIDATED — strongest positive result of the program.**

## H6 — Skill-shaped data (debug traces, no memorization) lifts code quality (v5, $48)
**Hypothesis:** error→diagnosis→fix traces + task-majority mix recovers code judge to ≥4.0.
**Result on BUILD:** 47.1%/3.26 — target missed; ≈ v4.
**Verdict: DISPROVEN on the one-shot build benchmark; PENDING on DEBUG (H8), the benchmark it was actually trained for.**

## H7 — The build benchmark measures the skills we train (meta)
**Hypothesis (implicit until v5):** one-shot spec→implementation covers "coding skill."
**Result:** v5's trained skill (debugging) is invisible to it.
**Verdict: DISPROVEN → DEBUG benchmark created (25 uncontaminated diagnose-the-failure tasks).**

## H-M1 — Measurement: thinking models need think+answer budgets (meta)
**Hypothesis (after 3 failed Sonnet runs):** sonnet-5 adaptively thinks even via the native API; tight `max_tokens` yields empty answers that score as zeros.
**Result:** `thinking_tokens == max_tokens`, `stop_reason: max_tokens`; at 10–20k budgets answers are real.
**Verdict: VALIDATED.** All frontier baselines re-run honestly; early "v1 beats Sonnet on knowledge" claim retracted.

## H-C1 — Cost: specialist ~10–40× cheaper per task (economics)
**Result:** measured ~$0.01–0.03/task on one L40S vs $0.13–0.43 (Sonnet→Fable) at current API prices.
**Verdict: VALIDATED** (with "1/100th" reachable only at high batch utilization).

---

## OPEN — pre-registered

## H8 — v5's debug training shows on a debugging benchmark (resolved 2026-07-05)
**Hypothesis:** on DEBUG, v5 > v3/v4 by a clear margin; secondary: v5 calls `au_search` during diagnosis.
**Result:** v5 21.6%/3.16 (tool rate 4%) ≈ v4 21.0%/3.60 (0%) ≈ v3 24.6%/2.75; Sonnet 35.2%/3.8.
**Verdict: DISPROVEN** — flat even on its home-turf task; one-shot debug-trace imitation doesn't
produce debugging skill. Secondary also disproven: tool-trained models don't reach for the corpus
on errors (0–4%), despite relevant gotchas being retrievable.
**Diagnostic nuance:** v4's judge (3.60) ≈ Sonnet (3.80) — diagnoses are directionally right but
miss exact identifiers; the deficit is grounding, not reasoning style.

## H9 — v6 (registered 2026-07-05; NOT YET LAUNCHED — user go/no-go)
**Two-armed hypothesis, cheap arm first:**
- **v6a (SFT, ~$50):** restructuring debug data as multi-step tool-mediated traces
  (error → au_search the failure → grounded diagnosis quoting retrieved identifiers) + explicit
  call-on-error triggering will lift DEBUG coverage toward Sonnet's 35% by closing the grounding
  gap H8 exposed. Foundation: v4's corpus (proven calibration) — per user direction.
- **v6b (RFT):** managed reinforcement fine-tuning (reward = root-cause key facts matched /
  tests pass) raises the ceiling SFT cannot, per H2/H3/H6/H8's collective evidence.
**Decision rule for v6a:** DEBUG coverage ≥ 30% AND call-on-error rate ≥ 60% → SFT grounding
works, iterate; else → all future training dollars go to RFT.

**RESOLVED 2026-07-05 — v6a DISPROVEN on both bars, with regression:**
- DEBUG: 20.5% coverage / 2.6 judge / **0% search-on-error** (bars: ≥30% / ≥60%) — base-model level.
- BUILD regression: 40.7% / 2.38 (v4: 52.0% / 3.32) — the debug traces *diluted* build skill.
- Even 544 direct demonstrations of call-on-error did not transfer to eval time, while the
  identical training mechanism produced 92–98% call rates on knowledge questions (v4).
  Call-on-error appears to need on-policy learning (RFT), not imitation.
- **Program conclusion: the SFT arm is closed. v4 is the flagship. Next training dollar → RFT
  (v6b, blocked on Fireworks Training API preview).**
- (Run details: `sftj xx55f1xo`, 8,094 examples = v4's 7,550 + 544 tool-grounded debug traces,
  rank 32, 2 epochs, $64.)
- **v6b BLOCKED on preview access** — evaluator `au-fw-debug-reward-v2` built on Eval Protocol,
  validated ACTIVE; 264-prompt reward dataset uploaded. Legacy RFT rejects gemma4
  ("use RLOR trainer"); the RLOR/Training API is **private preview** (fireworks.ai/contact-training).
  Warm-start from v4 also rejected (legacy-trainer adapters incompatible with trainer v2).
  All artifacts ready to fire the moment access lands.

## H10 — GLM-5.2 transfer (registered 2026-07-05, user-directed)
**Hypothesis:** the entire pipeline (datasets, tool traces, benchmarks, merge/serve/eval harness)
transfers to GLM-5.2 and yields a strictly stronger specialist (bigger model, 1M ctx, native
thinking). **Gate:** GLM-5.2 fine-tuning is preview-only everywhere (`supportsLora:false` probed
on-account). One consolidated Fireworks access request unlocks H9b AND H10: (1) Training API /
RLOR preview, (2) GLM-5.2 fine-tuning preview, (3) the BF16 Gemma serving INTERNAL failures.
**Interim option:** GLM-5.1 is GA-tunable today — the v4/v6a recipe can run on it unchanged
(price tier TBD by its param count).
**Status 2026-07-05: DEFERRED by user** — wait for clarity on training venue before any GLM-5.2
work. Program continues on Gemma toward a consolidated research report.
