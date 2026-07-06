#!/usr/bin/env python3
"""Benchmark a model WITH Agent University retrieval (the MCP server's backend, /v1/search).

For each eval item: query AU search, join full artifact text from index.json, prepend as
reference notes, then ask the model. Scores identically to run_eval (coverage + judge).
Logs retrieved ids and a leakage flag (did retrieval return the item's own reference artifact).

Usage: rag_eval.py --model label=model_string [--code-only] [--judge]
Env: EVAL_API_BASE (model endpoint), AU_API (default http://127.0.0.1:4123), FIREWORKS_API_KEY,
     ANTHROPIC_API_KEY, EVAL_CODE_FILE
"""
import json, os, re, sys, argparse
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as E

HERE = os.path.dirname(os.path.abspath(__file__))
AU_API = os.environ.get("AU_API", "http://127.0.0.1:4123")
TOPK = int(os.environ.get("RAG_TOPK", 6))
DOC_CAP = 2200

print("loading index.json for full-text join...", file=sys.stderr)
_idx = json.load(open(os.path.join(HERE, "..", "index.json")))
FULLTEXT = {a["id"]: a for a in _idx["artifacts"]}
del _idx

def search(query, limit=TOPK):
    r = E.post(f"{AU_API}/v1/search",
               {"query": query[:400], "limit": limit},
               {"Authorization": "Bearer dev", "Content-Type": "application/json"})
    if "_error" in r:
        return []
    rs = r.get("results", [])
    if isinstance(rs, dict):
        rs = rs.get("results", [])
    return rs

def build_notes(results):
    parts, ids = [], []
    for i, r in enumerate(results, 1):
        aid = r.get("artifactId")
        ids.append(aid)
        art = FULLTEXT.get(aid)
        body = (art["text"][:DOC_CAP] if art else str(r.get("snippet", ""))[:DOC_CAP])
        parts.append(f"[{i}] {r.get('title','')} ({r.get('target','')}, {r.get('artifactType','')}, "
                     f"evidence: {r.get('evidenceLevel','?')})\n{body}")
    return "\n\n".join(parts), ids

def rag_question(item, kind):
    if kind == "code":
        q_search = f"{item['target']} {item['question'][:300]}"
    else:
        q_search = item["question"]
    results = search(q_search)
    notes, ids = build_notes(results)
    prompt = (f"Reference notes from the Agent University live-verified corpus:\n\n{notes}\n\n"
              f"----\nUsing these notes where they are relevant (they may not all apply), "
              f"answer precisely with exact identifiers:\n\n{item['question']}")
    return prompt, ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--code-only", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--concurrency", type=int, default=2)
    args = ap.parse_args()
    label, model = args.model.split("=", 1)

    probes = [] if args.code_only else [json.loads(l) for l in open(os.path.join(HERE, "eval_probes.jsonl"))]
    code_file = os.environ.get("EVAL_CODE_FILE", "eval_code_v2.jsonl")
    code = [json.loads(l) for l in open(os.path.join(HERE, code_file))]
    items = [("probe", p) for p in probes] + [("code", c) for c in code]

    def run_one(kind, item):
        prompt, ids = rag_question(item, kind)
        max_t = (int(os.environ.get("EVAL_MAX_CODE", 2400)) if kind == "code"
                 else int(os.environ.get("EVAL_MAX_PROBE", 1400)))
        ans, err = E.fw_chat(model, prompt, max_t)
        own = item.get("id")
        rec = {"kind": kind, "split": item.get("split", "code"),
               "target": item.get("target"), "id": item.get("id") or item.get("level"),
               "retrieved": ids, "leak_own_artifact": bool(own and own in ids),
               "error": err}
        if ans is not None:
            rec["answer"] = ans
            rec["coverage"] = E.coverage(ans, item.get("key_facts", []))
            if args.judge:
                rec["judge"] = E.judge(item["question"], item["reference"], ans, kind == "code")
        return rec

    recs = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(run_one, k, i) for k, i in items]
        for n, f in enumerate(futs, 1):
            recs.append(f.result())
            if n % 25 == 0:
                print(f"{n}/{len(futs)}", file=sys.stderr)

    with open(os.path.join(HERE, f"results_{label}.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out = {}
    for split in ("holdout", "retention", "code"):
        sel = [r for r in recs if r["split"] == split and not r["error"]]
        if not sel:
            continue
        covs = [r["coverage"] for r in sel if r.get("coverage") is not None]
        js = [float(r["judge"]) for r in sel if isinstance(r.get("judge"), (int, float, str))
              and str(r.get("judge")).replace(".", "").isdigit()]
        out[split] = {"n": len(sel),
                      "errors": sum(1 for r in recs if r["split"] == split and r["error"]),
                      "key_fact_coverage": round(sum(covs)/len(covs), 4) if covs else None,
                      "judge_mean": round(sum(js)/len(js), 2) if js else None,
                      "leak_rate": round(sum(1 for r in sel if r.get("leak_own_artifact"))/len(sel), 3)}
    json.dump({label: out}, open(os.path.join(HERE, f"summary_{label}.json"), "w"), indent=2)
    print(json.dumps({label: out}, indent=2))

if __name__ == "__main__":
    main()
