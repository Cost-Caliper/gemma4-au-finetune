#!/usr/bin/env python3
"""v6a: v4's tool corpus + GROUNDED debugging traces.

New debug shape (targets H8's two measured gaps — identifier grounding, call-on-error):
  user(spec + real failing output) -> assistant CALLS au_search(error-derived query)
  -> tool(REAL results) -> assistant(diagnosis that opens with exact identifiers + fix).
Both single-shot and multi-turn (implement -> "it fails" -> search -> diagnose) shapes.
Benchmark levels (40) excluded everywhere, as always.
"""
import json, os, re, random, glob, sys, importlib.util, urllib.request
from concurrent.futures import ThreadPoolExecutor

random.seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = "/private/tmp/claude-501/-Users-dennison-develop-agent-university/962af201-c8e0-4761-9533-4f901bed9e7e/scratchpad/au-main"
AU_API = "http://127.0.0.1:4123"

spec = importlib.util.spec_from_file_location("bcb", os.path.join(HERE, "build_code_bench.py"))
bcb = importlib.util.module_from_spec(spec)
_so = sys.stdout; sys.stdout = open(os.devnull, "w"); spec.loader.exec_module(bcb); sys.stdout = _so

v4 = [json.loads(l) for l in open(os.path.join(HERE, "train_v4.jsonl"))]
TOOLS = next(ex["tools"] for ex in v4 if "tools" in ex)
SYSTEM = v4[0]["messages"][0]["content"]

_idx = json.load(open(os.path.join(HERE, "..", "index.json")))
FULLTEXT = {a["id"]: a for a in _idx["artifacts"]}
del _idx

def search(q):
    body = json.dumps({"query": q[:400], "limit": 4}).encode()
    req = urllib.request.Request(f"{AU_API}/v1/search", data=body,
                                 headers={"Authorization": "Bearer dev",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            rs = json.loads(r.read()).get("results", [])
            return rs if isinstance(rs, list) else rs.get("results", [])
    except Exception:
        return []

def notes(rs):
    ps = []
    for i, r in enumerate(rs, 1):
        art = FULLTEXT.get(r.get("artifactId"))
        body = (art["text"][:1800] if art else str(r.get("snippet", ""))[:1800])
        ps.append(f"[{i}] {r.get('title','')} ({r.get('target','')}, {r.get('artifactType','')})\n{body}")
    return "\n\n".join(ps) or "No results."

BAD = re.compile(r"(\$\{|03-pocs/|04-logs/|\.md$|new Date|toISOString)")
def facts_of(text):
    spans = [s.strip() for s in re.findall(r"`([^`\n]{3,50})`", text)]
    out, seen = [], set()
    for s in spans:
        if s.lower() in seen or BAD.search(s) or len(s.split()) > 2 or not re.search(r"[A-Za-z]", s):
            continue
        seen.add(s.lower()); out.append(s)
    return out[:6]

def read(p, cap):
    try:
        with open(p, errors="replace") as f:
            return f.read()[:cap]
    except OSError:
        return ""

def error_query(target, red_tail):
    # last non-empty error-looking line + target
    lines = [l.strip() for l in red_tail.splitlines() if l.strip()]
    err = ""
    for l in reversed(lines):
        if re.search(r"error|fail|exception|assert|denied|refus|timeout|not found|invalid", l, re.I):
            err = l[:120]; break
    if not err and lines:
        err = lines[-1][:120]
    return f"{target} {err}"

bench = set((i["target"], i["level"]) for i in map(json.loads, open(os.path.join(HERE, "eval_code_v2.jsonl"))))

cands = []
for red in sorted(glob.glob(os.path.join(MAIN, "*", "degrees", "*", "03-pocs", "L*", "red-output.txt"))):
    lv = os.path.dirname(red)
    parts = lv[len(MAIN) + 1:].split(os.sep)
    target, level = parts[0], parts[-1]
    if (target, level) in bench:
        continue
    t = bcb.build_tuple(lv, target)
    if not t:
        continue
    q_spec, impl, _ = t
    red_txt = read(red, 20000)
    red_tail = red_txt[-1500:]
    diag = read(os.path.join(lv, "surprises.md"), 2500) or read(os.path.join(lv, "evidence.md"), 2500)
    if len(red_tail.strip()) < 80 or len(diag.strip()) < 100:
        continue
    cands.append((target, q_spec, impl, red_tail, diag))

def make_traces(c):
    target, q_spec, impl, red_tail, diag = c
    s_query = error_query(target, red_tail)
    rs = search(s_query)
    ns = notes(rs)
    fs = facts_of(diag)
    lead = ("**Root cause identifiers:** " + ", ".join(f"`{x}`" for x in fs) + "\n\n") if len(fs) >= 2 else ""
    spec_text = q_spec.split("\n\nImplement this")[0]
    diagnosis = (f"{lead}## Diagnosis (grounded in corpus evidence)\n\n{diag}\n\n"
                 f"## Corrected implementation\n\n{impl[:8000]}")
    call = {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function",
            "function": {"name": "au_search", "arguments": json.dumps({"query": s_query})}}]}
    tool = {"role": "tool", "tool_call_id": "call_1", "content": ns}
    single = {"tools": TOOLS, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{spec_text}\n\nMy attempt FAILS with:\n```\n{red_tail}\n```\n"
                                    f"Diagnose the root cause and fix it."},
        call, tool,
        {"role": "assistant", "content": diagnosis}]}
    multi = {"tools": TOOLS, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": spec_text + "\n\nImplement this against the real service (no mocks)."},
        {"role": "assistant", "content": impl[:8000]},
        {"role": "user", "content": f"It fails with:\n```\n{red_tail}\n```\nWhat's wrong?"},
        call, tool,
        {"role": "assistant", "content": f"{lead}## Diagnosis (grounded in corpus evidence)\n\n{diag}"}]}
    return [single, multi]

with ThreadPoolExecutor(max_workers=8) as ex:
    debug_traces = [t for pair in ex.map(make_traces, cands) for t in pair]

out = v4 + debug_traces
random.shuffle(out)
toks = sum(len(m.get("content") or json.dumps(m.get("tool_calls", ""))) // 4
           for e in out for m in e["messages"])
with open(os.path.join(HERE, "train_v6a.jsonl"), "w") as f:
    for e in out:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
print(json.dumps({"v4_carried": len(v4), "debug_traces": len(debug_traces),
                  "total": len(out), "est_tokens": toks,
                  "cost_2ep": round(toks / 1e6 * 3 * 2, 2)}))
