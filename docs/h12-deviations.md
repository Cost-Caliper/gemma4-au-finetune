# H12-R Protocol Deviations Log

Per DESIGN.md: "The design is frozen: you implement it exactly; if reality forces a deviation,
LOG it here and continue — never silently substitute."

## D1 — M10 "arms interleaved per item" implemented at cell granularity, not item granularity

DESIGN M10 says "arms interleaved per item." The harness (`h12/run_cells.py`) instead
randomizes the ORDER of whole (arm, condition, suite) CELLS within a suite (fixed seed) and
runs each cell's items concurrently within that cell, rather than literally interleaving
individual item requests across arms in one combined schedule. Rationale: `run_arm_vllm.py` /
`run_arm_sonnet.py` each own their own concurrency pool per cell; building a single
cross-arm/cross-condition item-level scheduler was judged not worth the added complexity given
that the underlying concern M10 exists to address -- server/version drift correlating with arm
identity -- is otherwise fully addressed: every Gemma cell in a given `--set` run executes
against ONE continuous, unchanging vLLM deployment (same server process, same loaded adapters,
same index hash) for the whole suite, so there is no server-version confound to interleave
away. Cell order is still randomized so no time-of-day block effect can align with arm identity.
Logged here per the frozen-design instruction; not treated as blocking.

## D2 — 4/40 S-old40 items have empty `key_facts` (pre-existing benchmark property, not new)

`eval_code_v2.jsonl` (frozen, inherited from prior phases) has 4 items (indices 3, 12, 18, 32)
where `key_apis()` extracted zero identifiers from the reference at benchmark-build time. This
is a property of the frozen benchmark file itself, not something H12-R introduced. Per-item
identifier_recall is undefined (`None`, not `0.0`) for these 4 items for EVERY arm equally (it
is not a per-arm failure), so `h12/stats.py` excludes them from paired-diff pairing (reports
`n_excluded_no_key_facts`) rather than forcing a misleading `0.0` that would inflate `n` without
adding signal. Effective `n` for S-old40 confirmatory contrasts is ≤36, not 40 — reported
explicitly per contrast in `h12/stats_results.json`, not silently absorbed into a "n=40" claim.

## D3 — B2's path-prefix filter (implemented EXACTLY as DESIGN specifies) has a confirmed
##      residual leakage vector; logged as a limitation, not silently patched around

DESIGN B2's literal mechanism is: "path-prefix filter on source.path, applied to a copied
index." `h12/leakage_filtered_index.json` implements exactly that — every artifact/codeSample
whose `source.path` is under an excluded level's `03-pocs/<level>/` directory is removed. This
was built exactly as specified and verified (0 leaked paths under the 70 excluded `03-pocs/`
dirs; see PROVENANCE.json).

**Empirically confirmed gap (live test against the running local API, before any eval spend):**
a search for terms specific to the quarantined supabase capstone level
(`supabase/degrees/01-supabase-database-and-rls/03-pocs/L-capstone-combined-system`) returns,
as its TOP results, detailed solution content from OTHER files in the SAME degree that are NOT
under `03-pocs/` and therefore survive the path-prefix filter untouched:
- `supabase/degrees/01-supabase-database-and-rls/06-skill-pack/curriculum.md` (stage summary)
- `supabase/degrees/01-supabase-database-and-rls/04-logs/live-evidence-ledger.md` (test-by-test
  live evidence, including the exact migration filename and behavioral-test IDs)
- `supabase/degrees/01-supabase-database-and-rls/06-skill-pack/labs/lab-Lcapstone-teams.md` (a
  full walkthrough lab reproducing the forward-migration SQL, including the exact
  `au_test_get_admin_team_ids()` security-definer function body)

This is exactly the residual channel `phase1_build_report.md` already named for TRAINING data
("distilled 05-distillation artifacts quote a few key lines from the POCs they were distilled
from... cannot be removed without also quarantining derived artifacts") and exactly what
`METHODS-REVIEW.md` B2 warned needed "exact hashes plus text-similarity/identifier-overlap
checks," not path-only exclusion, to fully close. DESIGN.md (frozen, written AFTER the methods
review) nonetheless specifies path-prefix as the exact mechanism -- so this was implemented as
literally specified; the gap is in the frozen spec's chosen mechanism, not in the
implementation of it. 57 of the corpus's degree directories contain at least one excluded
level, so this is a broad, not isolated, residual channel.

**What this means for interpretation, not for the confirmatory scoring rule:** the frozen
pre-registered flag ("any answer whose retrieved set includes a benchmark level's OWN content
is flagged and excluded from confirmatory scoring") is unchanged and still correctly fires zero
times by construction (the level's own `03-pocs/` content is fully removed from the index, so it
literally cannot be retrieved). What it does NOT catch is same-degree derived/distilled content
describing the quarantined level's solution. `h12/rag_index.py` additionally logs a
`same_degree_broader_flag` diagnostic (NOT part of confirmatory scoring, reported only as an
exploratory transparency measure) on every T1/T2 retrieval so the final verdict can honestly
report how often this broader channel was actually exercised, rather than letting a technically-
correct "0 leakage flags" summary imply the retrieval index has no solution-adjacent content at
all for these items.

## D4 — A1 (Sonnet) max_tokens raised 8192 → 24000 after live pilot showed empty-answer truncation

DESIGN A1 spec: "adaptive thinking, max_tokens 8192, stop_reason audited". Live pilot (6 items,
T0, S-old40, exact model `claude-sonnet-5`) at 8192: 2/6 items (33%) returned stop_reason
`max_tokens` with a COMPLETELY EMPTY text answer — adaptive thinking consumed the entire 8192
budget before emitting any answer text on code-build items with large multi-file references.
This is a harness-parameter artifact, not a capability measurement, and it matters most for K6
(the only confirmatory contrast involving A1): an artificially truncated A1 denominator biases
the non-inferiority claim toward "supported" — the wrong direction for an honest sponsor
economics claim. Retested at 16000: still 2/6 max_tokens (one with 8.4k chars of usable partial
text, one still empty). Final setting 24000 with client timeout 480s: 1/6 max_tokens (the
sqlite long-term-memory capstone, still an empty answer → scores ITT-0 per the pre-registered
rule; the flag rides on the record). All other A1 parameters exactly as frozen: exact model ID
recorded, adaptive thinking (no effort override), same au_search tool schema + 2-round cap,
stop_reason audited per item. The harness calls the raw HTTP API (h12/common.py::post), not the
SDK, so the SDK's "stream above ~16K" client-timeout guidance is addressed directly via the
explicit 480s read timeout rather than streaming. Residual truncations at 24000 are accepted
as-is and scored ITT-0 with flags — no further budget chasing (a model that cannot finish
within 3x the pre-registered token budget on an item is a reportable observation, not a harness
bug). Anthropic cost impact of the raise is bounded by output actually generated, not by the
ceiling; metered in h12/spend.md.

## D5 — "2-round cap" operationalized as cap-then-forced-answer (both harnesses), plus
##      parallel-tool-call handling; found via live smoke tests before any confirmatory spend

Two harness-semantics clarifications of the frozen "identical schema/loop/2-round cap for all
arms" language, both discovered by live smoke tests (2 items each) and fixed BEFORE any
confirmatory cell ran:

1. **Parallel tool calls.** Sonnet issues PARALLEL au_search calls (multiple `tool_use` blocks
   in one assistant turn); the Anthropic API hard-rejects (400) any continuation that does not
   return a `tool_result` for EVERY `tool_use` id (verified live). The loop now executes every
   query in the turn and returns all results in one user message. A "round" for the 2-round cap
   = one assistant tool-calling TURN (however many parallel calls it contains) — the only
   reading compatible with the API contract. The Gemma harness mirrors the same
   all-queries-per-turn handling for exact loop parity.

2. **Cap semantics.** After the 2nd round's results, Sonnet (natural agentic behavior) kept
   requesting a 3rd round; the naive implementation scored that ITT-0
   ("tool_call_cap_exceeded") — a harness-imposed zero on K6's DENOMINATOR, biasing K6 toward
   "non-inferiority supported", the wrong direction for an honest sponsor claim. The cap now
   means: at most 2 retrieval rounds, then the model MUST answer — on the final round the
   harness appends "Now give your final answer using the information you have." and sets
   tool_choice {"type":"none"}. This mirrors the Gemma harness's continuation text ("Now give
   your final answer."), already the proven phase3 pattern, so the nudge is identical across
   arms rather than Sonnet-specific.
