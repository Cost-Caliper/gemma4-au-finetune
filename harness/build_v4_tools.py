#!/usr/bin/env python3
"""Build v4 tool-calling dataset: teach the model to call the AU MCP search tool.

Each trace: user question -> assistant au_search tool_call (realistic query) ->
tool result (REAL /v1/search output, full text joined) -> assistant grounded final answer.
Standard OpenAI tools/tool_calls format (Fireworks-supported). ~20% of knowledge items are
emitted as direct answers (no tool call) so the model doesn't over-call.
Holdout probes + the 40 benchmark code tasks stay excluded.
"""
import json, os, re, random, sys, glob
from concurrent.futures import ThreadPoolExecutor
import urllib.request

random.seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))
AU_API = os.environ.get("AU_API", "http://127.0.0.1:4123")
TOPK = 4
DOC_CAP = 1800

SYSTEM = ("You are an expert software agent for modern AI/agent libraries, developer tools, and "
          "cloud services. You have access to the au_search tool over the Agent University "
          "live-verified corpus; use it to ground answers in exact, evidence-backed identifiers. "
          "You implement tasks against real services with runnable code and never invent APIs.")

TOOLS = [{"type": "function", "function": {
    "name": "au_search",
    "description": ("Search the Agent University live-verified corpus (gotchas, recipes, patterns, "
                    "reference implementations for 90+ libraries/services). Returns top matching artifacts."),
    "parameters": {"type": "object",
                   "properties": {"query": {"type": "string", "description": "search query"},
                                  "limit": {"type": "integer"}},
                   "required": ["query"]}}}]

_idx = json.load(open(os.path.join(HERE, "..", "index.json")))
FULLTEXT = {a["id"]: a for a in _idx["artifacts"]}

def search(query):
    body = json.dumps({"query": query[:400], "limit": TOPK}).encode()
    req = urllib.request.Request(f"{AU_API}/v1/search", data=body,
                                 headers={"Authorization": "Bearer dev",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            rs = json.loads(r.read()).get("results", [])
            return rs if isinstance(rs, list) else rs.get("results", [])
    except Exception:
        return []

def tool_result_text(results):
    parts = []
    for i, r in enumerate(results, 1):
        art = FULLTEXT.get(r.get("artifactId"))
        body = (art["text"][:DOC_CAP] if art else str(r.get("snippet", ""))[:DOC_CAP])
        parts.append(f"[{i}] {r.get('title','')} ({r.get('target','')}, {r.get('artifactType','')}, "
                     f"evidence: {r.get('evidenceLevel','?')})\n{body}")
    return "\n\n".join(parts) if parts else "No results."

BAD = re.compile(r"(\$\{|03-pocs/|04-logs/|\.md$|new Date|toISOString)")
def facts_of(text):
    spans = [s.strip() for s in re.findall(r"`([^`\n]{3,50})`", text)]
    out, seen = [], set()
    for s in spans:
        if s.lower() in seen or BAD.search(s) or len(s.split()) > 2 or not re.search(r"[A-Za-z]", s):
            continue
        seen.add(s.lower()); out.append(s)
    return out[:6]

def clean_title(t):
    t = t.split(" / ")[0]
    return re.sub(r"^(Gotcha|Recipe|Pattern|Anti-pattern|Lesson|Troubleshooting|Quickstart)\s*[:—-]\s*",
                  "", t, flags=re.I).strip()[:160]

def trace(question, s_query, answer, with_tool=True):
    if not with_tool:
        return {"tools": TOOLS,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": question},
                             {"role": "assistant", "content": answer}]}
    results = search(s_query)
    return {"tools": TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
                {"role": "assistant", "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "au_search",
                                 "arguments": json.dumps({"query": s_query})}}]},
                {"role": "tool", "tool_call_id": "call_1",
                 "content": tool_result_text(results)},
                {"role": "assistant", "content": answer}]}

holdout_ids = set(p["id"] for p in map(json.loads, open(os.path.join(HERE, "eval_probes.jsonl")))
                  if p["split"] == "holdout")
KEEP = {"gotcha", "recipe", "pattern", "anti_pattern", "troubleshooting", "quickstart",
        "agent_instructions", "expectation_gap", "lesson"}
arts = [a for a in FULLTEXT.values()
        if a["artifactType"] in KEEP and len(a.get("text", "")) >= 120 and a["id"] not in holdout_ids]
random.shuffle(arts)
arts = arts[:7000]  # cap dataset scale

def make_knowledge(a):
    title = clean_title(a.get("title", ""))
    if not title:
        return None
    text = a["text"][:10000]
    fs = facts_of(text)
    lead = ("Exact identifiers: " + ", ".join(f"`{x}`" for x in fs) + "\n\n") if len(fs) >= 2 else ""
    q = f"{a['target']}: {title} — what do I need to know? Cite exact identifiers."
    s_query = f"{a['target']} {title}"
    return trace(q, s_query, lead + text, with_tool=(random.random() > 0.2))

with ThreadPoolExecutor(max_workers=8) as ex:
    out = [t for t in ex.map(make_knowledge, arts) if t]
n_k = len(out)

# task tuples with a tool call for implementation guidance
bench = set((i["target"], i["level"]) for i in map(json.loads, open(os.path.join(HERE, "eval_code_v2.jsonl"))))
import importlib.util
spec = importlib.util.spec_from_file_location("bcb", os.path.join(HERE, "build_code_bench.py"))
bcb = importlib.util.module_from_spec(spec)
_stdout = sys.stdout; sys.stdout = open(os.devnull, "w")
spec.loader.exec_module(bcb)
sys.stdout = _stdout
MAIN = "/private/tmp/claude-501/-Users-dennison-develop-agent-university/962af201-c8e0-4761-9533-4f901bed9e7e/scratchpad/au-main"
tasks = []
for lv in sorted(glob.glob(os.path.join(MAIN, "*", "degrees", "*", "03-pocs", "L*"))):
    if not os.path.isdir(lv):
        continue
    parts = lv[len(MAIN) + 1:].split(os.sep)
    target, level = parts[0], parts[-1]
    if (target, level) in bench:
        continue
    t = bcb.build_tuple(lv, target)
    if t:
        tasks.append((target, level, t))

def make_task(item):
    target, level, (q, ans, _) = item
    s_query = f"{target} {level.replace('-', ' ')} implementation gotchas recipe"
    return trace(q, s_query, ans, with_tool=True)

with ThreadPoolExecutor(max_workers=8) as ex:
    task_traces = list(ex.map(make_task, tasks))
out.extend(task_traces)

random.shuffle(out)
toks = sum(len(m.get("content") or json.dumps(m.get("tool_calls", ""))) // 4
           for ex_ in out for m in ex_["messages"])
with open(os.path.join(HERE, "train_v4.jsonl"), "w") as f:
    for ex_ in out:
        f.write(json.dumps(ex_, ensure_ascii=False) + "\n")
print(json.dumps({"knowledge": n_k, "tasks": len(task_traces), "total": len(out),
                  "est_tokens": toks, "cost_2ep": round(toks / 1e6 * 3 * 2, 2)}))
