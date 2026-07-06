#!/usr/bin/env python3
"""Tool-loop benchmark: the model may CALL au_search itself; harness executes real /v1/search.

Flow per item: question (+tools) -> if model emits a tool call (OpenAI field OR raw-text
patterns), run the real search, feed result back, get final answer -> score like run_eval.
Records tool_called / n_rounds per item.

Usage: tool_eval.py --model label=model_string [--code-only] [--judge] [--concurrency N]
Env: EVAL_API_BASE (model), AU_API, FIREWORKS_API_KEY, ANTHROPIC_API_KEY, EVAL_CODE_FILE
"""
import json, os, re, sys, argparse
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as E
import rag_eval as R  # reuse search + tool_result formatting (loads index once)

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_ROUNDS = 2

TOOLS = [{"type": "function", "function": {
    "name": "au_search",
    "description": ("Search the Agent University live-verified corpus (gotchas, recipes, patterns, "
                    "reference implementations for 90+ libraries/services)."),
    "parameters": {"type": "object",
                   "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                   "required": ["query"]}}}]

SYSTEM = ("You are an expert software agent for modern AI/agent libraries, developer tools, and "
          "cloud services. You have access to the au_search tool over the Agent University "
          "live-verified corpus; use it to ground answers in exact, evidence-backed identifiers. "
          "You implement tasks against real services with runnable code and never invent APIs."
          "\n\nAvailable tools:\n" + json.dumps([{"type": "function", "function": {
              "name": "au_search",
              "description": ("Search the Agent University live-verified corpus (gotchas, recipes, "
                              "patterns, reference implementations for 90+ libraries/services)."),
              "parameters": {"type": "object",
                             "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                             "required": ["query"]}}}])
          + "\n\nTo call a tool, emit a tool_call with JSON arguments.")

def chat_raw(model, messages, max_tokens):
    payload = {"model": model, "temperature": 0.0, "max_tokens": max_tokens,
               "messages": messages, "tools": TOOLS}
    r = E.post(f"{E.API_BASE}/chat/completions", payload,
               {"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}",
                "Content-Type": "application/json"})
    if "_error" in r:
        return None, r["_error"]
    try:
        return r["choices"][0]["message"], None
    except (KeyError, IndexError):
        return None, json.dumps(r)[:300]

def extract_tool_call(msg):
    """Return query string or None. Handles OpenAI tool_calls field and raw-text formats."""
    if not msg:
        return None
    tcs = msg.get("tool_calls") or []
    for tc in tcs:
        try:
            if tc["function"]["name"] == "au_search":
                return json.loads(tc["function"]["arguments"]).get("query")
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
    text = msg.get("content") or ""
    if "au_search" in text:
        m = (re.search(r"<\|tool_call>call:au_search\{query\s*:\s*\"([^\"]{3,300})\"", text)
             or re.search(r'"query"\s*:\s*"([^"]{3,300})"', text)
             or re.search(r"query\s*:\s*\"([^\"]{3,300})\"", text)
             or re.search(r"au_search\((?:query=)?[\"']([^\"']{3,300})[\"']", text)
             or re.search(r"<arg_value>([^<]{3,300})</arg_value>", text))
        if m:
            return m.group(1)
    return None

def run_item(model, kind, item, do_judge):
    max_t = 2400 if kind == "code" else 1400
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": item["question"]}]
    rounds, queries = 0, []
    msg, err = chat_raw(model, messages, max_t)
    while msg and rounds < MAX_ROUNDS:
        q = extract_tool_call(msg)
        if not q:
            break
        rounds += 1
        queries.append(q)
        results = R.search(q)
        notes, _ = R.build_notes(results)
        # flattened tool exchange (robust across chat templates)
        messages.append({"role": "assistant", "content": (msg.get("content") or "") +
                         f"\n[called au_search(query={q!r})]"})
        messages.append({"role": "user", "content":
                         f"au_search results:\n\n{notes}\n\nNow give your final answer."})
        msg, err = chat_raw(model, messages, max_t)
    ans = (msg or {}).get("content")
    rec = {"kind": kind, "split": item.get("split", "code"),
           "target": item.get("target"), "id": item.get("id") or item.get("level"),
           "tool_called": rounds > 0, "rounds": rounds, "queries": queries, "error": err}
    if ans:
        rec["answer"] = ans
        rec["coverage"] = E.coverage(ans, item.get("key_facts", []))
        if do_judge:
            rec["judge"] = E.judge(item["question"], item["reference"], ans, kind == "code")
    return rec

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

    recs = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(run_item, model, k, i, args.judge) for k, i in items]
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
        js = [float(r["judge"]) for r in sel
              if isinstance(r.get("judge"), (int, float)) or
              (isinstance(r.get("judge"), str) and r["judge"].replace(".", "").isdigit())]
        out[split] = {"n": len(sel),
                      "errors": sum(1 for r in recs if r["split"] == split and r["error"]),
                      "key_fact_coverage": round(sum(covs)/len(covs), 4) if covs else None,
                      "judge_mean": round(sum(js)/len(js), 2) if js else None,
                      "tool_call_rate": round(sum(1 for r in sel if r["tool_called"])/len(sel), 3)}
    json.dump({label: out}, open(os.path.join(HERE, f"summary_{label}.json"), "w"), indent=2)
    print(json.dumps({label: out}, indent=2))

if __name__ == "__main__":
    main()
