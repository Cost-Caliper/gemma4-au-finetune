#!/usr/bin/env python3
"""Benchmark base vs tuned Gemma 4 on held-out AU probes and code tasks.

Usage:
  python3 run_eval.py --model base='accounts/fireworks/models/gemma-4-26b-a4b-it#accounts/dennison-bertram/deployments/XXX' \
                      --model tuned='accounts/dennison-bertram/models/au-fw-gemma4-au-v1' \
                      [--probes N] [--judge]

Scores:
  key_fact_coverage — fraction of reference `code-span` facts present in the answer (objective)
  judge_score       — Claude Haiku rubric 0-10 vs reference (--judge; needs ANTHROPIC_API_KEY)
"""
import json, os, re, sys, time, argparse, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
FW_KEY = os.environ["FIREWORKS_API_KEY"]
ANTH_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

SYSTEM = ("You are an expert software agent for modern AI/agent libraries, "
          "developer tools, and cloud services. You implement tasks against real "
          "services with runnable code, cite exact API surfaces, and answer from "
          "hands-on evidence — you flag pitfalls honestly and never invent APIs.")

def post(url, payload, headers, retries=5):
    headers = {"User-Agent": "curl/8.7.1", "Accept": "*/*", **headers}
    body = json.dumps(payload).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except Exception as e:
            code = getattr(e, "code", None)
            if i == retries - 1:
                return {"_error": f"{type(e).__name__} {code}: {e}"}
            time.sleep(3 * (i + 1) if code in (429, 500, 502, 503, None) else 1)

API_BASE = os.environ.get("EVAL_API_BASE", "https://api.fireworks.ai/inference/v1")

def fw_chat(model, user, max_tokens):
    if os.environ.get("EVAL_NATIVE_ANTHROPIC"):
        r = post("https://api.anthropic.com/v1/messages",
                 {"model": model, "max_tokens": max_tokens, "system": SYSTEM,
                  "messages": [{"role": "user", "content": user}]},
                 {"x-api-key": ANTH_KEY, "anthropic-version": "2023-06-01",
                  "Content-Type": "application/json"})
        if "_error" in r:
            return None, r["_error"]
        try:
            return "".join(b.get("text", "") for b in r["content"]), None
        except (KeyError, TypeError):
            return None, json.dumps(r)[:300]
    payload = {"model": model, "temperature": 0.0, "max_tokens": max_tokens,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": user}]}
    if os.environ.get("EVAL_NO_TEMP"):
        payload.pop("temperature")
    r = post(f"{API_BASE}/chat/completions", payload,
             {"Authorization": f"Bearer {FW_KEY}", "Content-Type": "application/json"})
    if "_error" in r:
        return None, r["_error"]
    try:
        return r["choices"][0]["message"]["content"], None
    except (KeyError, IndexError):
        return None, json.dumps(r)[:300]

def strip_channels(t):
    return re.sub(r"<\|?channel\|?>\w*\s*", " ", t)

def coverage(answer, facts):
    if not facts:
        return None
    a = re.sub(r"\s+", " ", strip_channels(answer).lower())
    hit = sum(1 for f in facts if re.sub(r"\s+", " ", f.lower()) in a)
    return hit / len(facts)

def judge(question, reference, answer, is_code):
    rubric = ("Rate the CANDIDATE answer 0-10 for factual/API agreement with the REFERENCE "
              "(10 = same key facts/API usage, 0 = contradicts or invents APIs). "
              "The reference is ground truth from live-verified testing. "
              'Respond ONLY with JSON: {"score": <0-10>, "why": "<one sentence>"}')
    if is_code:
        rubric = ("Rate the CANDIDATE implementation 0-10 vs the REFERENCE for correct use of the "
                  "library/service APIs and task completeness (10 = equivalent API usage and covers the task; "
                  "0 = wrong/invented APIs or misses the task). Style differences don't matter. "
                  'Respond ONLY with JSON: {"score": <0-10>, "why": "<one sentence>"}')
    r = post("https://api.anthropic.com/v1/messages",
             {"model": "claude-haiku-4-5-20251001", "max_tokens": 200,
              "messages": [{"role": "user", "content":
                  f"{rubric}\n\nQUESTION:\n{question[:4000]}\n\nREFERENCE:\n{reference[:6000]}\n\nCANDIDATE:\n{strip_channels(answer)[:6000]}"}]},
             {"x-api-key": ANTH_KEY, "anthropic-version": "2023-06-01",
              "Content-Type": "application/json"})
    try:
        txt = r["content"][0]["text"]
        return json.loads(re.search(r"\{.*\}", txt, re.S).group(0))["score"]
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="label=model_string")
    ap.add_argument("--probes", type=int, default=0, help="limit probe count (0=all)")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    models = dict(m.split("=", 1) for m in args.model)

    probes = [json.loads(l) for l in open(os.path.join(HERE, "eval_probes.jsonl"))]
    code_file = os.environ.get("EVAL_CODE_FILE", "eval_code.jsonl")
    code = [json.loads(l) for l in open(os.path.join(HERE, code_file))]
    if args.probes:
        probes = probes[:args.probes]
    if os.environ.get("EVAL_CODE_ONLY"):
        probes = []

    items = ([("probe", p) for p in probes] + [("code", c) for c in code])
    results = {label: [] for label in models}

    def run_one(label, model, kind, item):
        max_t = (int(os.environ.get("EVAL_MAX_CODE", 2400)) if kind == "code"
                 else int(os.environ.get("EVAL_MAX_PROBE", 1400)))
        ans, err = fw_chat(model, item["question"], max_t)
        rec = {"kind": kind, "split": item.get("split", "code"),
               "target": item.get("target"), "id": item.get("id") or item.get("level"),
               "error": err}
        if ans is not None:
            rec["answer"] = ans
            rec["coverage"] = coverage(ans, item.get("key_facts", []))
            if args.judge:
                rec["judge"] = judge(item["question"], item["reference"], ans, kind == "code")
        return label, rec

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(run_one, label, model, kind, item)
                for label, model in models.items() for kind, item in items]
        done = 0
        for f in futs:
            label, rec = f.result()
            results[label].append(rec)
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(futs)} done", file=sys.stderr)

    out = {}
    for label, recs in results.items():
        with open(os.path.join(HERE, f"results_{label}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summ = {}
        for split in ("holdout", "retention", "code"):
            sel = [r for r in recs if r["split"] == split and not r["error"]]
            covs = [r["coverage"] for r in sel if r.get("coverage") is not None]
            js = [r["judge"] for r in sel if r.get("judge") is not None]
            summ[split] = {"n": len(sel),
                           "errors": sum(1 for r in recs if r["split"] == split and r["error"]),
                           "key_fact_coverage": round(sum(covs) / len(covs), 4) if covs else None,
                           "judge_mean": round(sum(js) / len(js), 2) if js else None}
        out[label] = summ
    with open(os.path.join(HERE, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
