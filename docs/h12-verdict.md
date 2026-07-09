# H12-R Verdict (2026-07-09, frozen protocol of h12/DESIGN.md)

All six pre-registered confirmatory contrasts ran (K4 after two documented
serving failures were fixed: wrong endpoint routing, then a missing chat
template on the merged-v4 checkpoint — both logged; the failed runs were
discarded and the cell rerun cleanly, 40/40 real answers).

## Cell means — item-level identifier recall, S-old40 (n=40/cell)

| arm | T0 none | T1 self-directed tools | T2 fixed retrieval |
|---|---|---|---|
| A1 sonnet-5 | (pilot only, n=6) | **57.5** | — |
| A2 base | 54.5 | 55.3 | 51.1 |
| A3 generalist-8123 | — | 52.9 | 51.5 |
| A4 gen-2814 (size-matched) | — | 52.3 | — |
| A5 supabase-spec | — | 52.5 | — |
| A6 v4-merged (expert+attn) | — | 55.1 | — |

S-supa30 (n=29): A4 44.6 · A5 43.4.

## Confirmatory contrasts (Holm over 6; SESOI 10pp; permutation inference)

| K | contrast | est (pp) | p (perm) | Holm | verdict |
|---|---|---|---|---|---|
| K1 | tools lift base (A2/T1−A2/T0) | +0.81 | .75 | 1.0 | NULL (CI −2.7..+4.8) |
| K2 | training on top of tools (A3−A2, T1) | −2.43 | .26 | 1.0 | NULL (CI −6.3..+0.8) |
| K3 | specialization, size-matched (A5−A4, supa30) | −1.15 | .64 | 1.0 | NULL (CI −4.3..+2.3) |
| K4 | expert+attn placement (A6−A2, T1) | −0.23 | 1.0 | 1.0 | NULL |
| K5 | training, content held fixed (A3−A2, T2) | +0.35 | .88 | 1.0 | NULL |
| K6 | base+tools ≥ 90% of Sonnet+tools | ratio 0.911 | — | — | POINT ESTIMATE MEETS 0.90; NOT statistically confirmed at n=40 |

(The stats.py `sesoi_verdict` string "LARGE BUT NOT SIGNIFICANT" is a labeling
bug — all K1–K5 estimates are SMALL and not significant; the CIs above are the
authoritative statement.)

## What this means

1. **No training effect survives the unified protocol.** Generalist,
   size-matched generalist, library specialist, and the historical
   expert+attention v4 all land within ±2.5pp of the untrained base
   (all p ≥ .26). The prior program's headline gap (tuned 52% vs base 22.7%)
   **does not replicate under this protocol**: base alone scores 54.5 here.
   The historical gap was protocol-dependent (old runs: Q8-quantized llama.cpp
   serving, different prompting/params), not a training effect on these tasks.
2. **No tool effect on S-old40 for base (K1 null).** Under this protocol the
   corpus search neither helped nor hurt on the old suite. With K5 also null,
   neither retrieval content nor trained tool policy moved identifier recall.
3. **Base ≈ 91% of Sonnet** (K6 point estimate 0.911) at ~10–40× lower
   serving cost, but the non-inferiority bound at 0.90 is not statistically
   confirmed with n=40 — descriptively supportive, formally unresolved.
4. Judge (secondary): double-judged; mean absolute inter-judge disagreement
   ≈1.5 points on 0–10 — judge scores are noisy; no confirmatory claims from
   them.

## Limitations (pre-registered + logged deviations)

Single seed per adapter; n=40/29 per cell (underpowered below ~8–10pp);
identifier recall ≠ working code; D3 residual leakage vector in the eval
search index (degree-level summary docs outside level dirs); lexical-only
search backend in eval (prod uses hybrid); v4-merged uses base's chat
template (merged checkpoint shipped none). All raw data, judged files,
provenance hashes, and deviations in h12/.

## Program implication

The consistent, rigorous result across H11 and H12-R: **for these benchmarks,
neither LoRA fine-tuning (any recipe tested) nor retrieval measurably beats
the raw Gemma-4 base model, which itself sits at ~91% of Sonnet-with-tools on
identifier recall at a fraction of the cost.** The earlier program's
tuned-vs-base and tools-vs-no-tools gaps were artifacts of non-unified
protocols. Product-wise: the cheapest viable system on this evidence is the
untrained base model; corpus freshness still lives in the MCP server for
tasks that actually require post-cutoff facts (this suite largely does not —
a benchmark-construction insight for the next iteration: build tasks whose
answers CANNOT be in base weights).
