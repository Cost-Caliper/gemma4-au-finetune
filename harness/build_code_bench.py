#!/usr/bin/env python3
"""Build the expanded code benchmark (eval_code_v2.jsonl) from origin/main POC levels.

Selection: levels present in the origin/main worktree but NOT in the local working tree's
training tuples (clean holdout for v1/v2). Prefers diversity across targets; includes capstones.
Each item: task spec -> reference implementation + key_apis (identifiers extracted from
reference code) for objective coverage scoring alongside the judge.
"""
import json, os, re, glob, random, sys

random.seed(4242)
MAIN = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "/private/tmp/claude-501/-Users-dennison-develop-agent-university/962af201-c8e0-4761-9533-4f901bed9e7e/scratchpad/au-main")
LOCAL_REPO = "/Users/dennison/develop/agent-university"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_code_v2.jsonl")

VENDOR = re.compile(r"(\.venv|site-packages|node_modules|__pycache__|/dist/|\.git/)")
SRC_EXT = {".ts", ".tsx", ".js", ".mjs", ".py", ".sql", ".sh", ".toml", ".yaml", ".yml", ".json"}
SECRETS = re.compile(
    r"(sk-[A-Za-z0-9_-]{20,}|fw_[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{30,}"
    r"|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|eyJ[A-Za-z0-9_-]{30,}\.[A-Za-z0-9_-]{10,}\."
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----|Bearer [A-Za-z0-9._~+/=-]{25,})")

def read(p, cap):
    try:
        with open(p, errors="replace") as f:
            return f.read()[:cap]
    except OSError:
        return ""

def build_tuple(lv, target):
    spec = ""
    for f in ("README.md", "intent.md"):
        p = os.path.join(lv, f)
        if os.path.exists(p):
            spec += read(p, 6000) + "\n"
    if len(spec.strip()) < 200:
        return None
    files, body, total = [], [], 0
    for p in sorted(glob.glob(os.path.join(lv, "source", "**", "*"), recursive=True)):
        if not os.path.isfile(p) or VENDOR.search(p):
            continue
        if os.path.splitext(p)[1] not in SRC_EXT or os.path.getsize(p) > 20000:
            continue
        files.append(p)
    for p in files:
        c = read(p, 14000)
        if total + len(c) > 14000:
            continue
        total += len(c)
        rel = os.path.relpath(p, lv)
        body.append(f"### {rel}\n```{os.path.splitext(p)[1].lstrip('.')}\n{c}\n```")
    if total < 500:
        return None
    level = os.path.basename(lv)
    q = (f"Task from the {target} curriculum (POC level {level}):\n\n{spec.strip()}\n\n"
         f"Implement this against the real service (no mocks). Provide the key source files with paths.")
    ans = "\n\n".join(body)
    if SECRETS.search(q) or SECRETS.search(ans):
        return None
    return q, ans, level

def key_apis(code_text):
    """Identifiers that indicate correct library usage: imports, dotted calls, quoted endpoints."""
    facts = set()
    for m in re.findall(r"(?:from|import)\s+['\"]?([@\w./-]{4,60})['\"]?", code_text):
        if not m.startswith((".", "/")):
            facts.add(m)
    for m in re.findall(r"\b([a-zA-Z_]\w{2,}\.[a-zA-Z_]\w{2,})\(", code_text):
        facts.add(m)
    for m in re.findall(r"['\"](/v\d[\w/.-]{2,50})['\"]", code_text):
        facts.add(m)
    GENERIC = re.compile(
        r"^(console|self|this|sys|os|json|Math|JSON|assert|expect|test|describe|it|res|req|resp"
        r"|process|path|fs|http|url|Object|Array|String|Number|Promise|Date|Buffer|util|node)\b"
        r"|\.(map|filter|includes|match|toLowerCase|toUpperCase|push|join|split|slice|forEach"
        r"|strictEqual|deepEqual|equal|ok|length|trim|replace|find|some|every|keys|values"
        r"|entries|get|set|has|add|log|error|stringify|parse|toString|catch|then|finally)$")
    facts = {f for f in facts if not GENERIC.search(f)}
    return sorted(facts, key=len, reverse=True)[:12]

# levels already used in local training (exclude their (target,level) pairs)
trained = set()
for lv in glob.glob(os.path.join(LOCAL_REPO, "*", "degrees", "*", "03-pocs", "L*")):
    parts = lv[len(LOCAL_REPO) + 1:].split(os.sep)
    trained.add((parts[0], parts[-1]))

cands = []
for lv in sorted(glob.glob(os.path.join(MAIN, "*", "degrees", "*", "03-pocs", "L*"))):
    if not os.path.isdir(lv):
        continue
    parts = lv[len(MAIN) + 1:].split(os.sep)
    target, level = parts[0], parts[-1]
    if (target, level) in trained:
        continue
    t = build_tuple(lv, target)
    if t:
        cands.append((target, level, t))

# diversity: max 2 per target, prefer capstones + mid levels, cap 40
random.shuffle(cands)
by_target = {}
picked = []
for target, level, t in sorted(cands, key=lambda x: ("capstone" not in x[1], random.random())):
    if by_target.get(target, 0) >= 2 or len(picked) >= 40:
        continue
    by_target[target] = by_target.get(target, 0) + 1
    picked.append((target, level, t))

with open(OUT, "w") as f:
    for target, level, (q, ans, _) in picked:
        f.write(json.dumps({"target": target, "level": level, "question": q,
                            "reference": ans, "key_facts": key_apis(ans),
                            "split": "code"}, ensure_ascii=False) + "\n")
print(json.dumps({"candidates": len(cands), "picked": len(picked),
                  "targets": len(by_target),
                  "capstones": sum(1 for _, l, _ in picked if "capstone" in l)}))
