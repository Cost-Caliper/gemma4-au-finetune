#!/usr/bin/env python3
"""Fill missing judge scores in saved results_<label>.jsonl files (offline re-judge),
then rewrite summary_<label>.json. Usage: post_judge.py <label> [<label>...]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as E

HERE = os.path.dirname(os.path.abspath(__file__))
probes = {p.get("id"): p for p in map(json.loads, open(os.path.join(HERE, "eval_probes.jsonl")))}
codes = {c.get("level"): c for c in map(json.loads, open(os.path.join(HERE, "eval_code.jsonl")))}

for label in sys.argv[1:]:
    path = os.path.join(HERE, f"results_{label}.jsonl")
    recs = [json.loads(l) for l in open(path)]
    filled = 0
    for r in recs:
        if r.get("judge") is not None or r.get("error") or "answer" not in r:
            continue
        ref = probes.get(r["id"]) if r["kind"] == "probe" else codes.get(r["id"])
        if not ref:
            continue
        s = E.judge(ref["question"], ref["reference"], r["answer"], r["kind"] == "code")
        if s is not None:
            r["judge"] = s; filled += 1
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = {}
    for split in ("holdout", "retention", "code"):
        sel = [r for r in recs if r["split"] == split and not r.get("error")]
        covs = [r["coverage"] for r in sel if r.get("coverage") is not None]
        js = []
        for r in sel:
            try:
                if r.get("judge") is not None:
                    js.append(float(r["judge"]))
            except (TypeError, ValueError):
                pass
        summ[split] = {"n": len(sel),
                       "errors": sum(1 for r in recs if r["split"] == split and r.get("error")),
                       "key_fact_coverage": round(sum(covs)/len(covs), 4) if covs else None,
                       "judge_mean": round(sum(js)/len(js), 2) if js else None,
                       "judged_n": len(js)}
    json.dump({label: summ}, open(os.path.join(HERE, f"summary_{label}.json"), "w"), indent=2)
    print(label, "filled", filled, "->", json.dumps(summ))
