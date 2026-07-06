#!/usr/bin/env python3
"""Build Fireworks SFT dataset from the Agent University corpus.

Sources:
  1. .agent-university/index.json  -> knowledge artifacts (gotcha/recipe/pattern/...)
                                      + code samples (vendor-filtered)
  2. working-tree degree dirs      -> POC task tuples (spec -> implementation)

Outputs (in .agent-university/finetune/):
  train.jsonl       Fireworks messages-format training set
  eval_probes.jsonl held-out knowledge probes (+ retention probes) with key_facts
  eval_code.jsonl   held-out POC code tasks with reference implementations
  stats.json        counts, token estimates, secret-scan drops
"""
import json, os, re, random, hashlib, glob, sys

random.seed(42)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .agent-university
REPO = os.path.dirname(ROOT)
OUT = os.path.join(ROOT, "finetune")

SYSTEM = ("You are an expert software agent for modern AI/agent libraries, "
          "developer tools, and cloud services. You implement tasks against real "
          "services with runnable code, cite exact API surfaces, and answer from "
          "hands-on evidence — you flag pitfalls honestly and never invent APIs.")

VENDOR = re.compile(r"(\.venv|site-packages|node_modules|__pycache__|/dist/|\.git/)")
SECRETS = re.compile(
    r"(sk-[A-Za-z0-9_-]{20,}|fw_[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{30,}"
    r"|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|eyJ[A-Za-z0-9_-]{30,}\.[A-Za-z0-9_-]{10,}\."
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----|Bearer [A-Za-z0-9._~+/=-]{25,})")

Q_TMPL = {
    "gotcha": "I'm building with {t}. What's the gotcha here: {title}? How do I avoid it?",
    "recipe": "Using {t}, show me how to accomplish this: {title}. Give working code where relevant.",
    "pattern": "What's a proven pattern for '{title}' when working with {t}?",
    "anti_pattern": "When using {t}, what's the anti-pattern around '{title}', and what should I do instead?",
    "troubleshooting": "I'm hitting a problem with {t}: {title}. How do I diagnose and fix it?",
    "quickstart": "Give me a practical quickstart for {t}: {title}.",
    "agent_instructions": "You're an agent about to work with {t}. What operating instructions matter for: {title}?",
    "expectation_gap": "Where does {t} behave differently than the docs or common assumptions suggest, regarding: {title}?",
    "lesson": "What did hands-on work with {t} teach about: {title}?",
}
# alternate phrasing used for retention probes (never used in training)
Q_TMPL_ALT = {
    "gotcha": "Working with {t} — is there a known pitfall related to: {title}? What's the workaround?",
    "recipe": "How would you implement '{title}' with {t}? Include the important code.",
    "pattern": "Recommend an approach for '{title}' in {t} and explain why it works.",
    "anti_pattern": "What mistake do people make around '{title}' with {t}? What's the right way?",
    "troubleshooting": "Debugging {t}: symptoms match '{title}'. Walk me through the fix.",
    "quickstart": "What are the first steps to get {t} working for: {title}?",
    "agent_instructions": "Before automating against {t}, what rules should an agent follow for: {title}?",
    "expectation_gap": "Any surprises vs. documented behavior in {t} concerning: {title}?",
    "lesson": "From real usage of {t}, what's the takeaway on: {title}?",
}
CODE_LANGS = {"typescript", "ts", "tsx", "javascript", "js", "python", "py",
              "bash", "sh", "sql", "json", "jsonc", "yaml", "toml"}
SRC_EXT = {".ts", ".tsx", ".js", ".mjs", ".py", ".sql", ".sh", ".toml", ".yaml", ".yml", ".json"}

def clean_title(t):
    t = t.split(" / ")[0]
    t = re.sub(r"^(Gotcha|Recipe|Pattern|Anti-pattern|Lesson|Troubleshooting|Quickstart)\s*[:—-]\s*", "", t, flags=re.I)
    return t.strip()[:160]

def toks(s):  # rough token estimate
    return len(s) // 4

def key_facts(text, question):
    spans = re.findall(r"`([^`\n]{3,60})`", text)
    seen, out = set(), []
    for s in spans:
        s = s.strip()
        if s.lower() in seen or s in question or len(s) < 3:
            continue
        seen.add(s.lower()); out.append(s)
    out.sort(key=len, reverse=True)
    return out[:8]

def scan_ok(*parts):
    return not any(SECRETS.search(p) for p in parts)

def main():
    with open(os.path.join(ROOT, "index.json")) as f:
        d = json.load(f)

    stats = {"dropped_secrets": 0, "dropped_vendor_cs": 0}
    train, probes, retention_pool = [], [], []

    # ---- 1. knowledge artifacts ----
    arts = [a for a in d["artifacts"] if a["artifactType"] in Q_TMPL and len(a.get("text", "")) >= 120]
    random.shuffle(arts)
    holdout = set(a["id"] for a in arts[:150])          # never trained; eval probes
    stats["knowledge_total"] = len(arts)
    for a in arts:
        title = clean_title(a.get("title", ""))
        if not title:
            continue
        q = Q_TMPL[a["artifactType"]].format(t=a["target"], title=title)
        ans = a["text"][:12000]
        if not scan_ok(q, ans):
            stats["dropped_secrets"] += 1; continue
        rec = {"id": a["id"], "type": a["artifactType"], "target": a["target"],
               "question": q, "reference": ans, "key_facts": key_facts(ans, q)}
        if a["id"] in holdout:
            rec["split"] = "holdout"
            probes.append(rec)
        else:
            train.append({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": q},
                {"role": "assistant", "content": ans}]})
            if len(rec["key_facts"]) >= 3:
                retention_pool.append((a, rec))

    # retention probes: alternate phrasing over trained artifacts
    random.shuffle(retention_pool)
    for a, rec in retention_pool[:60]:
        q2 = Q_TMPL_ALT[a["artifactType"]].format(t=a["target"], title=clean_title(a["title"]))
        probes.append({**rec, "question": q2, "split": "retention",
                       "key_facts": key_facts(rec["reference"], q2)})

    # ---- 2. code samples (vendor-filtered) ----
    cs_seen = set()
    cs_kept = 0
    for c in d["codeSamples"]:
        src = str(c.get("source", {}))
        if VENDOR.search(src):
            stats["dropped_vendor_cs"] += 1; continue
        code = c.get("code", "")
        if c.get("language") not in CODE_LANGS or not (200 <= len(code) <= 4000):
            continue
        h = hashlib.sha1(code.encode()).hexdigest()
        if h in cs_seen:
            continue
        cs_seen.add(h)
        title = clean_title(c.get("title", "")) or "the documented usage"
        q = f"Using {c['target']} ({c.get('language')}), write the code for: {title}."
        ans = f"```{c.get('language')}\n{code}\n```"
        if not scan_ok(q, ans):
            stats["dropped_secrets"] += 1; continue
        train.append({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant", "content": ans}]})
        cs_kept += 1
        if cs_kept >= 6000:
            break
    stats["code_samples_kept"] = cs_kept

    # ---- 3. POC task tuples from working tree ----
    poc_train, poc_eval = 0, []
    level_dirs = {}
    for lv in glob.glob(os.path.join(REPO, "*", "degrees", "*", "03-pocs", "L*")):
        if not os.path.isdir(lv):
            continue
        parts = lv.split(os.sep)
        target = parts[-4].split("/")[0] if "degrees" not in parts[-4] else parts[-5]
        target = lv[len(REPO) + 1:].split(os.sep)[0]
        level_dirs.setdefault(target, []).append(lv)

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
        files = []
        for p in sorted(glob.glob(os.path.join(lv, "source", "**", "*"), recursive=True)):
            if not os.path.isfile(p) or VENDOR.search(p):
                continue
            if os.path.splitext(p)[1] not in SRC_EXT or os.path.getsize(p) > 20000:
                continue
            files.append(p)
        if not files:
            return None
        body, total = [], 0
        for p in files:
            c = read(p, 14000)
            if total + len(c) > 14000:
                continue
            total += len(c)
            rel = os.path.relpath(p, lv)
            lang = os.path.splitext(p)[1].lstrip(".")
            body.append(f"### {rel}\n```{lang}\n{c}\n```")
        if total < 300:
            return None
        level = os.path.basename(lv)
        q = (f"Task from the {target} curriculum (POC level {level}):\n\n{spec.strip()}\n\n"
             f"Implement this against the real service (no mocks). Provide the key source files with paths.")
        ans = "\n\n".join(body)
        if not scan_ok(q, ans):
            stats["dropped_secrets"] += 1
            return None
        return q, ans, level

    eval_targets = []
    for target, lvs in sorted(level_dirs.items()):
        built = []
        for lv in sorted(lvs):
            t = build_tuple(lv, target)
            if t:
                built.append((lv, t))
        hold = None
        mids = [(lv, t) for lv, t in built
                if "capstone" not in lv and not lv.rstrip("/").endswith("L0")]
        if len(built) >= 3 and mids and len(eval_targets) < 15:
            hold = random.choice(mids)[0]
            eval_targets.append(target)
        for lv, (q, ans, level) in built:
            if lv == hold:
                poc_eval.append({"target": target, "level": level, "question": q, "reference": ans})
            else:
                train.append({"messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": ans}]})
                poc_train += 1
    stats["poc_train"] = poc_train
    stats["poc_eval"] = len(poc_eval)

    # ---- write ----
    random.shuffle(train)
    total_toks = sum(toks(m["content"]) for ex in train for m in ex["messages"])
    stats.update(train_examples=len(train), probe_count=len(probes),
                 est_train_tokens=total_toks,
                 est_cost_1epoch_usd=round(total_toks / 1e6 * 3.0, 2))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "train.jsonl"), "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "eval_probes.jsonl"), "w") as f:
        for p in probes:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "eval_code.jsonl"), "w") as f:
        for p in poc_eval:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
