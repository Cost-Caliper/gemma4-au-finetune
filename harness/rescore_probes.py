#!/usr/bin/env python3
"""Rebuild probe key_facts with identifier-focused filtering, then rescore coverage on all
saved results_*.jsonl (probe splits only; answers unchanged; judges unchanged)."""
import json, os, re, glob, sys

HERE = os.path.dirname(os.path.abspath(__file__))

BAD = re.compile(r"(\$\{|03-pocs/|04-logs/|\.md$|new Date|toISOString|^Open this URL"
                 r"|^https?://localhost|\s\S+\s\S+\s)")  # template lits, repo paths, >3-word sentences

def good_facts(text, question):
    spans = [s.strip() for s in re.findall(r"`([^`\n]{3,50})`", text)]
    out, seen = [], set()
    for s in spans:
        if s.lower() in seen or s in question:
            continue
        if BAD.search(s) or len(s.split()) > 2:
            continue
        # identifier-ish: contains alnum and at least one of . _ - / ( ) = or is CAPS flag
        if not re.search(r"[A-Za-z]", s):
            continue
        seen.add(s.lower()); out.append(s)
    out.sort(key=len, reverse=True)
    return out[:8]

def strip_channels(t):
    return re.sub(r"<\|?channel\|?>\w*\s*", " ", t)

def cov(ans, facts):
    if not facts:
        return None
    a = re.sub(r"\s+", " ", strip_channels(ans).lower())
    return sum(1 for f in facts if re.sub(r"\s+", " ", f.lower()) in a) / len(facts)

# rebuild probe facts
probes = [json.loads(l) for l in open(os.path.join(HERE, "eval_probes.jsonl"))]
for p in probes:
    p["key_facts"] = good_facts(p["reference"], p["question"])
with open(os.path.join(HERE, "eval_probes.jsonl"), "w") as f:
    for p in probes:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
usable = sum(1 for p in probes if len(p["key_facts"]) >= 2)
print(f"probes with >=2 identifier facts: {usable}/{len(probes)}")

by_key = {}
for p in probes:
    by_key.setdefault(p["id"], {})[p["split"]] = p

rows = {}
for path in glob.glob(os.path.join(HERE, "results_*.jsonl")):
    label = os.path.basename(path)[8:-6]
    recs = [json.loads(l) for l in open(path)]
    changed = False
    for r in recs:
        if r.get("kind") != "probe" or not r.get("answer"):
            continue
        p = by_key.get(r["id"], {}).get(r["split"])
        if not p:
            continue
        r["coverage"] = cov(r["answer"], p["key_facts"])
        changed = True
    if not changed:
        continue
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = {}
    for split in ("holdout", "retention"):
        sel = [r for r in recs if r["split"] == split and not r.get("error")]
        covs = [r["coverage"] for r in sel if r.get("coverage") is not None]
        js = []
        for r in sel:
            try:
                if r.get("judge") is not None:
                    js.append(float(r["judge"]))
            except (TypeError, ValueError):
                pass
        if sel:
            summ[split] = {"n": len(sel),
                           "cov": round(sum(covs)/len(covs), 4) if covs else None,
                           "judge": round(sum(js)/len(js), 2) if js else None}
    if summ:
        rows[label] = summ
print(json.dumps(rows, indent=1))
