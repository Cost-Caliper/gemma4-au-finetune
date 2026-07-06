#!/usr/bin/env python3
"""Behavioral validation of the merged model: ask N questions the model was
TRAINED on (verbatim user turns from train.jsonl); a correct merge should show
strong key-fact recall (base ~0.03). Exits 0 on PASS, 2 on FAIL."""
import json, os, random, re, sys, urllib.request

BASE_URL = os.environ.get("EVAL_API_BASE", "http://127.0.0.1:8080/v1")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
THRESHOLD = 0.25

SYSTEM = ("You are an expert software agent for modern AI/agent libraries, "
          "developer tools, and cloud services. You implement tasks against real "
          "services with runnable code, cite exact API surfaces, and answer from "
          "hands-on evidence — you flag pitfalls honestly and never invent APIs.")

def chat(user):
    body = json.dumps({"model": "local", "temperature": 0.0, "max_tokens": 500,
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "curl/8.7.1"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]

def key_facts(text, question):
    spans = [s.strip() for s in re.findall(r"`([^`\n]{3,60})`", text)]
    out, seen = [], set()
    for s in spans:
        if s.lower() not in seen and s not in question:
            seen.add(s.lower()); out.append(s)
    return sorted(out, key=len, reverse=True)[:8]

random.seed(99)
lines = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.jsonl")).readlines()
random.shuffle(lines)
picked, covs = 0, []
for l in lines:
    ex = json.loads(l)
    q = ex["messages"][1]["content"]
    ref = ex["messages"][2]["content"]
    facts = key_facts(ref, q)
    if len(facts) < 4 or len(q) > 600:
        continue
    picked += 1
    ans = chat(q)
    a = re.sub(r"\s+", " ", ans.lower())
    cov = sum(1 for f in facts if re.sub(r"\s+", " ", f.lower()) in a) / len(facts)
    covs.append(cov)
    print(f"[{picked}/{N}] cov={cov:.2f}  q={q[:80]!r}")
    if picked >= N:
        break
mean = sum(covs) / len(covs)
print(f"RECALL_MEAN={mean:.3f} (threshold {THRESHOLD}, Fireworks base measured 0.028)")
sys.exit(0 if mean >= THRESHOLD else 2)
