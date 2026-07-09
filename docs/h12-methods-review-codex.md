# H12-R Methods Review

Verdict: do not run this design as written. The draft has the right instinct--paired cells, frozen metrics, explicit controls--but the confirmatory estimands are still not identified. The largest problems are not "more n would be nice"; they are confounding of training, retrieval, benchmark leakage, serving stack, and economics.

## What Is Fine

Temperature-zero decoding, identical nominal tool schema, frozen extractor/prompt, paired item reuse, and pre-run freezing are reasonable implementation constraints. Reporting tool calls, rounds, tokens, and cost is also useful. Those pieces do not rescue the causal claims below.

## BLOCKING Findings

### B1. The confirmatory contrasts are not actually specified

The RQs say "each with a pre-registered contrast," but only RQ1 gives a formula, and even that says "adapter" without specifying which adapter or whether the estimand is any adapter, best adapter, v4, generalist, or domain specialist. RQ2 has no contrast formula. RQ3 is a ratio-reporting exercise, not a contrast. RQ4 asks about v4 versus attention-only adapters but is not tied to a suite, tool setting, or serving-stack adjustment. The statistics section then refers to "the 6 primary contrasts" although the RQ section does not list six.

Why this undermines the RQs: RQ1 can be answered several incompatible ways; RQ2 can be made significant by choosing a domain, tool setting, or comparator after seeing results; RQ3 has no inferential target; RQ4 is undefined once v4 is served differently. The current drop rule makes this worse by allowing controls needed for identification to be dropped while still preserving the headline RQs.

Cheapest fix: add a one-page contrast table before any run: contrast ID, RQ, suite, arms, tool setting, exact estimand, direction, SESOI, multiplicity family, and required cells. Make the minimum cell matrix non-optional. If a required control is dropped, the corresponding RQ is dropped or explicitly demoted to exploratory.

### B2. Leakage/quarantine is under-controlled, especially for tool arms

The adapters were trained on traces from the same curated corpus that generated the benchmarks, and the tool uses an Agent University search backend. The draft relies on path-based overlap checks and says marginal contamination "affects all arms equally." That is not credible. Contamination affects trained adapters, base+tools, adapter+tools, and Sonnet+tools differently because they differ in memorization, search-query behavior, and ability to exploit retrieved snippets.

Why this undermines the RQs: RQ1 can mistake benchmark-answer retrieval for "tool substitution." RQ2 can mistake same-corpus memorization for specialization. RQ3 can overstate cheap-model value if the cheap model is effectively evaluated open-book on its own training ecosystem. RQ4 can confuse v4 placement with old-corpus overlap.

Cheapest fix: before evaluation, freeze and publish a contamination audit at the item/reference/search-document level, not just path level. Exclude quarantined tasks, reference implementations, solution explanations, and near-duplicate distilled artifacts from the search index and training traces using exact hashes plus text-similarity/identifier-overlap checks. Log retrieved document IDs per answer. If a task retrieves quarantined or solution-derived material, score it separately or exclude it by pre-registered rule.

### B3. The "tools" treatment confounds retrieval content with learned tool policy

The tools condition is self-directed `au_search` with the same schema and round cap for all arms. But the adapters were trained on tool-calling conversation traces, so adapter+tools is not just "same model plus retrieval"; it is also "model trained to use this tool loop." Base and Sonnet may be worse or better at the tool protocol for reasons unrelated to library knowledge.

Why this undermines the RQs: RQ1 asks whether retrieval closes the fine-tuning gap, but the contrast mixes three mechanisms: access to documents, ability to formulate good searches, and training-induced tool-format compliance. RQ3's economics claim can similarly be driven by a cheap model being tuned to the harness rather than being a generally cheaper substitute.

Cheapest fix: add a fixed-retrieval control for the primary RQ1 cells. For each item, retrieve top-k context once with a model-independent query recipe, inject the same context into base, the primary adapter, and Sonnet, and compare that to self-directed tools. This can be limited to the old-40 primary suite and the few primary arms, so it costs little relative to full factorial expansion. At minimum, report retrieval recall, query failures, and tool-format failures by arm and do not interpret self-directed tool effects as pure retrieval effects.

### B4. RQ2 specialization is not identified because data size and domain scope are inseparable

The generalist has 8,123 examples; supabase has 2,814; vercel has 415. The generalist also likely contains in-domain or adjacent examples while the specialists differ dramatically from each other in data volume. The draft flags this, but option (b), "report scope and size effects as jointly identified only," means RQ2 has not been answered.

Why this undermines RQ2: any specialist/generalist difference can be due to domain scope, total examples, library difficulty, train/test similarity, or sample size. Cross-domain specialist cells help, but they do not separate these mechanisms, especially with the 415-example Vercel adapter.

Cheapest fix: spend the optional training budget on size-matched generalists, at least 2,814 examples and 415 examples, and make those cells required for RQ2. If budget cannot cover both, restrict RQ2 to the Supabase comparison and explicitly demote Vercel specialization to exploratory. Drop the placebo adapter before dropping size controls; the placebo is less important for the stated RQs.

### B5. RQ4 is invalid unless v4 uses the same serving stack

The design asks whether expert+attention v4 beats attention-only adapters "under identical serving/eval," but v4 is served from merged weights on llama.cpp while the other Gemma arms are on vLLM. The draft acknowledges this and makes a base-on-llama bridge optional. Optional bridging is not enough.

Why this undermines RQ4: serving stack can change numerics, context handling, stop behavior, tool formatting, latency, truncation, and decoding edge cases. A base bridge estimates only one stack effect; it does not prove the stack effect is constant across adapters, tools, suites, or merged-vs-LoRA execution.

Cheapest fix: serve v4 through the same vLLM path if technically possible. If not, make the base-on-llama bridge mandatory for every suite/tool setting used in an RQ4 comparison and weaken the RQ4 claim to "v4 plus serving stack" unless the bridge shows negligible differences. Do not leave this in the drop tail.

### B6. The primary economics claim is not a controlled estimand

RQ3 says "X% of Sonnet at Yx cost," but the design does not define whether X is based on coverage, judge score, or a composite; whether ratios are per suite or pooled; whether uncertainty is on the ratio; whether Sonnet is a fresh paired run; or which costs are included. Reusing old no-tools Sonnet results if decoding matches is especially unsafe for a primary sponsor claim because model snapshots, system prompts, pricing, and harness details can drift.

Why this undermines RQ3: a stale or differently prompted Sonnet denominator changes both X and Y. A generous Sonnet thinking budget can make Sonnet look unnecessarily expensive. Excluding retrieval, indexing, idle GPU time, batching, retries, engineering, and latency makes the serving-cost ratio a narrow lab marginal cost, not an economic claim.

Cheapest fix: rerun Sonnet for all primary RQ3 cells under the frozen protocol and exact model version; do not use historical Sonnet data for confirmatory ratios. Define X separately for coverage and judge score, with paired ratio CIs. Define Y as marginal inference cost under the experimental harness, and label broader production economics as out of scope unless retrieval cost, GPU utilization, prompt caching, batching, retries, and latency are included.

## MAJOR Findings

### M1. The paired design is useful, but the unit of analysis is under-specified

"Identical items across all arms" supports paired item contrasts within a suite. It does not justify pooling old and new suites, fact-level bootstrapping, or reuse of historical cells. If key-fact coverage is computed over all identifiers globally, items with many identifiers dominate and the apparent sample size becomes fake. If it is averaged per item, the item is the unit and n is 15, 30, or 40.

Cheapest fix: define the item-level score as the primary unit; average item scores within suite; use paired item differences for all confirmatory contrasts; never treat key identifiers as independent observations. Predefine handling for missing, truncated, tool-failed, and timeout answers, preferably intention-to-treat with score 0 plus a failure flag.

### M2. The bootstrap plan is too optimistic at n=15 and still fragile at n=30

A 10k paired bootstrap does not create information. With 15 items, percentile bootstrap intervals for bounded, discrete, skewed coverage scores can have poor coverage and many ties. Even at 30, CIs will be wide for realistic effects, and bootstrap validity depends on the items being exchangeable draws from a target population that is not defined.

Cheapest fix: keep the paired bootstrap as descriptive, but make paired randomization/permutation intervals or sign/randomization tests the primary inferential check for differences. Report all item-level paired differences and the number of items favoring each arm. State that inference generalizes only to the authored benchmark population unless item sampling is made random from a defined frame.

### M3. The SESOI is not justified and is likely below measurement resolution

Five percentage points of key coverage can be one identifier on a small task set and may be below scorer noise or benchmark-authoring noise. A 0.5 judge-point SESOI is also likely near the noise floor of a single LLM judge, especially with only 20% duplicate judging. The design is simultaneously underpowered for <10pp effects and pre-registers 5pp as meaningful.

Cheapest fix: justify SESOI from utility: e.g., ">=10pp coverage or >=1.0 judge point is practically meaningful at n=15; 5pp is descriptive only unless n is expanded and reliability supports it." For RQ3, use a non-inferiority or cost-effectiveness margin directly tied to the sponsor claim instead of an arbitrary 5pp.

### M4. Holm over "6 primary contrasts" is not a multiple-comparisons plan

The design has many possible comparisons: base vs each adapter, tools vs none, specialist vs generalist in each domain, cross-domain specialists, Gemma vs Sonnet ratios, v4 vs attention-only, coverage and judge outcomes, old and new suites. Holm over six unnamed contrasts leaves too many researcher degrees of freedom.

Cheapest fix: define one confirmatory family with exactly named contrasts and one primary outcome per contrast. Use adjusted p-values or adjusted CIs consistently. Put all other arms, domains, metrics, and pooled summaries in a descriptive/exploratory table.

### M5. Key-identifier coverage is not a sufficient primary measure of build quality

Identifier coverage rewards mentioning the right API names, not producing a working implementation. It misses call order, argument correctness, error handling, types, security, version compatibility, integration logic, and whether the answer is executable. It can be gamed by listing many identifiers, and retrieval-heavy answers may get high recall while hallucinating invalid or irrelevant APIs. It is also vulnerable to ceiling effects when retrieved docs contain the target identifiers and floor effects on tasks whose solution requires conceptual glue rather than named APIs.

Cheapest fix: keep coverage as primary only if paired with a pre-registered precision/invalid-identifier penalty and a small executable or human-blind audit. At minimum report recall, precision, extra invalid identifiers, and "critical missing identifier" rates per item. Do not call key coverage "objective quality"; call it identifier recall.

### M6. The LLM-judge plan is not blinded or reliable enough for quality claims

Answer order randomization does little if the judge sees stylistic fingerprints, tool citations, formatting artifacts, or Sonnet-like prose. "20% double-judge for intra-judge reliability" is not a serious reliability design: if it is the same model with deterministic settings, it mostly checks determinism; if stochastic, the sample is tiny. Agreement is also the wrong summary for a 0-10 scale unless defined.

Cheapest fix: strip metadata, normalize answer wrappers, hide tool logs unless they are part of the scored answer, and judge answers in randomized bundles by item. Double-judge all confirmatory cells or at least all items in the primary contrast with a second independent judge prompt/model or a blind human audit. Report ICC or mean absolute disagreement, not just "agreement."

### M7. Cross-benchmark comparability is not established

The old-40 suite, new-supabase-15, and new-vercel-15 differ in target libraries, task construction, reference implementations, extractor behavior, and probably difficulty. Reusing the old extractor helps old-suite continuity; it does not make old and new suites commensurable. Percent-of-Sonnet can be compared within a suite, but raw coverage averages cannot be interpreted as one common scale across suites.

Cheapest fix: pre-register all primary conclusions within-suite. If a headline average is required, compute a suite-stratified estimate with equal suite weights and show per-suite results first. Do not use old-vs-new changes to explain prior discrepancies unless the same arms are rerun on both suites under the same protocol.

### M8. The placebo adapter is not a clean placebo

A Cloudflare-only adapter trained with the same recipe is not "no treatment"; it is an out-of-domain domain adapter that may teach tool format, response style, or general corpus habits. It controls for "some unrelated training," not for "any LoRA changes behavior."

Cheapest fix: rename it an unrelated-domain negative control. If a true placebo is needed, train on shuffled prompt-response pairs, response-style-only traces, or no-op synthetic traces with the same token budget. Otherwise drop it in favor of the size-matched generalist controls.

### M9. Single training seed and single decoding pass limit attribution

One seed per adapter means adapter differences include seed noise. One deterministic generation per item means inference instability is unmeasured. This is acceptable for a budgeted pilot, but not for strong claims that one adapter recipe beats another.

Cheapest fix: state that adapter-level variance is unestimated. For the smallest primary subset, run two decoding repeats or one alternate seed only if budget remains; otherwise avoid claims about recipes and talk about trained artifacts.

### M10. Run-order, retry, and truncation rules are incomplete

The design audits Sonnet stop reasons but does not define retry policy, API/server failures, search failures, malformed tool calls, missing outputs, or whether truncated answers score zero. If arms are run in blocks, server drift, backend changes, and cache effects can align with arm.

Cheapest fix: randomize/interleave item-arm execution order within suite, freeze the search index, define one retry policy, and score unresolved failures by intention-to-treat. Log server version, model checksum, prompt hash, search-index hash, and retrieved document IDs.

## MINOR Findings

- Freeze the exact Sonnet and Haiku model IDs, not only "Sonnet 5" and a Haiku version string.
- Define max tokens by suite before running and report truncation rates by arm.
- Report latency distributions alongside dollars; a cheap model that is much slower or less available is not economically equivalent.
- Report both mean and median item scores because small n and skewed tasks make means fragile.
- Publish raw prompts, answers, extracted identifiers, judge rubrics, and scorer outputs so failures can be audited.

## Required Changes Before Running

1. Freeze an exact confirmatory contrast table and minimum required cell matrix. No optional dropping of cells needed for an RQ.
2. Run and document a content-level leakage audit for training traces, benchmark references, and the search index; exclude quarantined/solution-derived material and log retrieval provenance.
3. Add the missing identification controls: fixed-retrieval primary cells for RQ1, size-matched generalist controls for RQ2, and same-stack v4 serving or mandatory bridge cells for RQ4.
4. Replace the statistics plan with item-level paired estimands, exact/randomization inference for small n, justified SESOIs, named multiplicity families, ratio CIs for RQ3, and explicit missing-data rules.
5. Harden measurement and economics: add identifier precision/invalid-ID reporting, a credible judge reliability/blinding plan, fresh paired Sonnet runs for primary cells, and a narrow cost definition that does not overclaim production economics.
