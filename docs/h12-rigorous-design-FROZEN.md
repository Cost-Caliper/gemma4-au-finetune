# H12-R v2 — FROZEN pre-registered design (2026-07-09)

Revision of the v1 draft after external methods review
(`h12-methods-review-codex.md`, GPT-5.5 xhigh). Every BLOCKING finding is
addressed below; MAJOR findings addressed or explicitly accepted as
limitations. This document is FROZEN before any cell runs; deviations must be
logged in the run report as protocol deviations.

## Changes from v1 (mapped to review findings)

- B1 → explicit confirmatory contrast table (§Contrasts), minimum required
  cell matrix non-optional; drop rule can only drop exploratory cells.
- B2 → content-level leakage controls: the eval-time search index EXCLUDES all
  artifacts/code samples sourced from any benchmark level's directory (path-
  prefix filter on source.path, applied to a copied index; hash recorded).
  Retrieval provenance logged per answer (artifact IDs + source paths);
  pre-registered rule: any answer whose retrieved set includes a benchmark
  level's own content is flagged and excluded from confirmatory scoring.
- B3 → fixed-retrieval cells added (same injected top-k context for every arm
  in the cell, model-independent query = the task title+first spec line);
  self-directed tool effects are reported as tool-policy+content, never as
  pure retrieval.
- B4 → NEW size-matched generalist adapter (gen-2814: 2,814-example seed-42
  subsample of the control recipe, identical hyperparameters) trained (~$3);
  RQ2 confirmatory contrast uses it. Vercel specialization DEMOTED to
  exploratory (415 examples cannot support a confirmatory claim). Placebo
  adapter DROPPED (per review: not a clean placebo; funds go to fresh Sonnet).
- B5 → v4 served from its MERGED HF checkpoint in the SAME vLLM stack and
  version as all other Gemma arms (sequential sessions on the same GPU
  class). If the merged HF checkpoint is unavailable, RQ4 is demoted to
  exploratory with mandatory llama.cpp bridge cells — not silently kept.
- B6 → ALL Sonnet confirmatory cells are fresh runs under this protocol
  (exact model ID recorded; no historical reuse). Economics defined narrowly:
  marginal inference cost under this harness (GPU-seconds amortized at the
  provider hourly rate vs token-metered API cost), explicitly NOT production
  TCO. X reported separately for coverage and judge, as paired per-item
  ratios with CIs.
- M1 → unit of analysis = item (per-item identifier-recall averaged within
  item); intention-to-treat: failures/truncations score 0 with a flag.
- M2 → primary inference = exact paired permutation tests (sign-flip on item
  differences); bootstrap CIs reported as descriptive only.
- M3 → SESOI: ≥10pp identifier-recall or ≥1.0 judge points. 5pp effects are
  descriptive.
- M4 → ONE confirmatory family = the 6 named contrasts below, Holm-adjusted;
  one primary outcome each (identifier recall). Everything else exploratory.
- M5 → "key-fact coverage" renamed **identifier recall**; identifier
  PRECISION (invalid/irrelevant identifiers per answer) reported alongside;
  no "quality" claims from recall alone.
- M6 → judge hardening: answers stripped of tool logs/metadata, wrapped in a
  neutral template, item-bundled and order-randomized; ALL confirmatory cells
  double-judged (claude-haiku + claude-sonnet-5 judge prompts, both frozen);
  report mean absolute disagreement; judge is secondary outcome only.
- M7 → all confirmatory conclusions within-suite; no pooled cross-suite
  averages; %-of-Sonnet computed within suite and condition.
- M9 → single seed/artifact limitation stated: claims attach to the trained
  artifacts, not the recipes.
- M10 → arms interleaved per item; frozen search-index hash; one retry
  (network-class errors only) then ITT-0; server/model/prompt/index hashes
  logged per cell.

## Arms

A1 sonnet-5 (exact ID at runtime, adaptive thinking, max_tokens 8192, stop_reason audited)
A2 base gemma-4-26B-A4B (vLLM 0.21.0 + #46772 patch)
A3 base + control-8123 adapter (existing)
A4 base + gen-2814 adapter (NEW, size-matched to A5)
A5 base + supabase-spec adapter (existing)
A6 merged-v4 (expert+attention, same vLLM stack)
(Exploratory only: vercel-spec.)

## Conditions

T0 none · T1 self-directed au_search (identical schema/loop/2-round cap for
all arms incl. Sonnet; leakage-filtered index) · T2 fixed-retrieval (same
injected top-3 context per item for every arm; only on cells listed below).

## Suites

S-old40 (primary; existing eval_code_v2.jsonl, frozen extractor)
S-supa30 (expand phase1 eval_supabase from 15→~30 items authored ONLY from
already-quarantined levels, same schema+extractor, holdout checks re-run;
if expansion fails checks, run at n=15 and report the power limitation)
(Exploratory: S-verc15.)

## Confirmatory contrasts (the ONLY confirmatory family; Holm over 6)

K1 RQ1 tool lift:        A2/T1 − A2/T0 on S-old40
K2 RQ1 training-on-top:  A3/T1 − A2/T1 on S-old40
K3 RQ2 specialization:   A5/T1 − A4/T1 on S-supa30
K4 RQ4 placement:        A6/T1 − A2/T1 on S-old40
K5 RQ1 content-held:     A3/T2 − A2/T2 on S-old40
K6 RQ3 non-inferiority:  A2/T1 ≥ 0.90 × A1/T1 on S-old40 (paired ratio,
                         one-sided; the sponsor economics claim)
Primary outcome for all: item-level identifier recall. Direction and SESOI
as in M3. Required cells: A1/T1, A2/{T0,T1,T2}, A3/{T1,T2}, A4/T1, A5/T1,
A6/T1 on S-old40; A4/T1+A5/T1 on S-supa30. NON-DROPPABLE.

## Exploratory (run if budget remains, labeled exploratory)

A1/T0 fresh; A1/T2; vercel-spec cells; S-verc15 anything; probes; judge-only
analyses; cross-suite descriptives.

## Budget

Training gen-2814 ~$3 · GPU eval ≤10 h ≈ $25 · Anthropic (Sonnet arm fresh,
~40-70 gens + double-judging) ≤ $35. Hard stop at caps; if caps bind, cut
exploratory cells only (confirmatory cells are protected — if a confirmatory
cell cannot run, its contrast is reported as NOT RUN, never substituted).

## Execution

Sonnet subagents implement; every cell writes results+summary immediately;
provenance (model IDs, index hash, patch hash, prompts) recorded in
h12/PROVENANCE.json; final verdict = per-contrast table (estimate, permutation
p, Holm-adjusted, CI, SESOI verdict) + exploratory appendix + limitations.
