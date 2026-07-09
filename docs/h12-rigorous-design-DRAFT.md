# H12-R — Pre-registered factorial evaluation (DRAFT for methods review)

Status: DRAFT v1, 2026-07-09. Not yet executed. This document will be revised
after external methods critique, then frozen BEFORE any cell is run.

## Background (what exists, what is claimed)

Prior results this experiment must reconcile:
- P1 (no-tools, old 40-task bench): expert+attention LoRA (v4, Fireworks) 52.0%
  key-API coverage vs base 22.7%; Claude Sonnet 5 (honest budgets) 31.5%.
- P2 (tools/RAG): tuned+retrieval ≈ 87–95% of Sonnet+retrieval at ~10–40×
  lower serving cost. **base+tools was never measured** — attribution of the
  economics claim to *training* (vs mere tool access) is unknown.
- P3 (H11, new 30-task library benches, all arms WITH tools): supabase/vercel
  specialists ≈ whole-corpus generalist ≈ base (43–45% / 36–41%); judge-score
  edge only for the generalist (+0.3–0.5 vs base). Confounds: new benches,
  attention-only adapters, no no-tools cells.

## Research questions (each with a pre-registered contrast)

- RQ1 Tool substitution: does self-directed retrieval close the fine-tuning
  gap? Contrast: (base,tools−base,none) vs (adapter,tools−adapter,none), old
  bench primary.
- RQ2 Specialization: in-domain specialist vs generalist, tools on and off,
  new benches. (H11 re-test with proper controls.)
- RQ3 Frontier fraction & economics (PRIMARY per sponsor): every Gemma arm
  reported as %-of-Sonnet on the same cells, with measured $-per-task. The
  headline table is X% of Sonnet at Y× cost, ± tools.
- RQ4 Adapter placement: does the old expert+attention adapter (v4) beat
  attention-only adapters under identical serving/eval?

## Design

Factors (full crossing where budget allows):
- ARM (7): sonnet-5 · base · base+generalist(control adapter, 8,123-ex) ·
  base+supabase-spec (2,814-ex) · base+vercel-spec (415-ex) ·
  base+v4-merged (expert+attn, served from merged weights) ·
  base+placebo-adapter (NEW, see controls)
- TOOLS (2): none · self-directed au_search (identical loop, schema, round
  cap=2, and search backend for every arm including Sonnet)
- SUITE (3): old-40 (all targets) · new-supabase-15 · new-vercel-15
  (expand to ~30/domain if feasible from already-quarantined levels only)

Not all 42 cells are equally informative; pre-registered priority order and
the drop rule are listed in §Budget.

## Controls

- C1 Generalist adapter = specialization control (exists).
- C2 Cross-domain specialist cells = domain control (specialist out-of-domain).
- C3 **Placebo adapter (new, ~$2)**: identical recipe/size trained on a
  disjoint unrelated slice of the corpus (e.g. cloudflare-only, ~similar
  example count to vercel-spec) — controls for "any LoRA changes behavior".
- C4 **Data-budget confound**: generalist (8,123 ex) vs specialists (2,814 /
  415 ex) differ in dataset size AND scope. Mitigation options (pick after
  review): (a) train a size-matched generalist (2,814-ex random subsample,
  ~$3), or (b) report scope and size effects as jointly identified only.
- C5 Serving-stack control: all Gemma arms on the SAME vLLM server/params;
  v4-merged is a separate endpoint (llama.cpp) — its serving stack is a
  confound vs the vLLM arms, acknowledged; mitigate by also running base on
  llama.cpp for a bridge cell if time allows.
- C6 Same decoding everywhere: temperature 0, same max_tokens per suite,
  same stop conditions. Sonnet: adaptive thinking with ≥8k budget, stop_reason
  audited per item (never score a truncated/empty answer without flagging).

## Measures

- Primary: key-fact coverage (objective, identifier-filtered, existing
  extractor, frozen before runs).
- Secondary: LLM-judge 0–10 (claude-haiku, FROZEN version string), blinded to
  arm, answer order randomized; 20% double-judge for intra-judge reliability
  (report agreement); judge prompt frozen pre-run.
- Tertiary: tool_call_rate, rounds used, tokens in/out, measured $/task
  (GPU-seconds amortized for Gemma arms; token pricing for Sonnet).

## Statistics

- Item-paired design: identical items across all arms → per-item paired
  differences.
- 95% CIs via paired bootstrap (10k resamples) on coverage and judge.
- Pre-registered smallest effect size of interest: 5pp coverage / 0.5 judge.
- Familywise: Holm correction over the 6 primary contrasts (listed in RQs).
- No post-hoc subgroup claims without "exploratory" label.

## Threats to validity (acknowledged)

- n=15/cell on new suites is underpowered for <10pp effects even paired;
  expansion to 30 (from already-quarantined levels ONLY — no retraining, no
  new leakage surface) is a pre-registered improvement if authoring passes
  the same holdout checks.
- Single training seed per adapter (cost); noted.
- Judge circularity: haiku judging answers produced partly from
  Anthropic-authored corpus text; mitigated by objective primary metric.
- Old-bench references were built and scored under the original harness;
  re-running old cells re-uses the same extractor for comparability.
- Quarantine hygiene: phase1 path-based verification (0/651 overlaps) is
  inherited; contamination via distilled artifacts quoting quarantined POCs
  exists at the margin (documented in phase1 report) and affects all arms
  equally.

## Budget & execution

- GPU (Modal, patched vLLM 0.21.0 + #46772; scattermoe lessons applied):
  ≤10 GPU-h ≈ $25. Placebo (+ optional size-matched generalist) training:
  ≤$6.
- Anthropic: Sonnet arm + judge ≤$35, with per-cell metering; reuse the
  existing old-bench no-tools Sonnet results if and only if decoding params
  match the frozen protocol (else re-run).
- Cell priority (drop from the bottom if caps bind): all base/adapters cells
  on all suites → sonnet+tools old+new → sonnet no-tools new → placebo cells
  on new suites → v4 bridge cell → probes (exploratory).
- Implementation: Sonnet subagents; every cell writes results+summary
  immediately (crash-safe); all endpoints torn down at end; verdict doc with
  the full matrix, contrasts, CIs, and per-leg falsification outcomes.
