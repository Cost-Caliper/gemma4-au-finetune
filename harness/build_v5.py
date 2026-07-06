#!/usr/bin/env python3
"""v5 dataset: SKILL + TOOL + CONTEXT-FAITHFULNESS. No fact-memorization.

Components:
  1. error->fix process traces: task spec + real red-output tail -> diagnosis (surprises/evidence)
     + fixed source files. NEW skill signal (debugging/iteration).
  2. task tuples (spec -> implementation), 60% wrapped in an au_search tool trace, 40% direct.
  3. context-faithfulness: subsample of v4 tool traces (call -> real results -> grounded
     quote-first answer). Teaches faithful grounding + tool use, not recall.
  4. commands/ops traces (small): spec -> exact commands to run/verify.
Holdouts excluded everywhere: 40 code-bench levels + 150 knowledge probe artifacts (v4 file
already excluded them).
"""
import json, os, re, random, glob, sys, importlib.util

random.seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = "/private/tmp/claude-501/-Users-dennison-develop-agent-university/962af201-c8e0-4761-9533-4f901bed9e7e/scratchpad/au-main"

SYSTEM = ("You are an expert software agent for modern AI/agent libraries, developer tools, and "
          "cloud services. You have access to the au_search tool over the Agent University "
          "live-verified corpus; use it to ground answers in exact, evidence-backed identifiers. "
          "You implement tasks against real services with runnable code and never invent APIs.")

TOOLS = json.loads(open(os.path.join(HERE, "train_v4.jsonl")).readline())["tools"]

SECRETS = re.compile(
    r"(sk-[A-Za-z0-9_-]{20,}|fw_[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}|eyJ[A-Za-z0-9_-]{30,}\.[A-Za-z0-9_-]{10,}\."
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----|Bearer [A-Za-z0-9._~+/=-]{25,})")

def read(p, cap):
    try:
        with open(p, errors="replace") as f:
            return f.read()[:cap]
    except OSError:
        return ""

# task-tuple builder (suppress its import-time main)
spec = importlib.util.spec_from_file_location("bcb", os.path.join(HERE, "build_code_bench.py"))
bcb = importlib.util.module_from_spec(spec)
_so = sys.stdout; sys.stdout = open(os.devnull, "w")
spec.loader.exec_module(bcb)
sys.stdout = _so

bench = set((i["target"], i["level"]) for i in map(json.loads, open(os.path.join(HERE, "eval_code_v2.jsonl"))))

out = []
def emit(messages, with_tools=True):
    for m in messages:
        c = m.get("content") or ""
        if SECRETS.search(c):
            return False
    ex = {"messages": messages}
    if with_tools:
        ex["tools"] = TOOLS
    out.append(ex)
    return True

# ---- 1. error->fix traces ----
n_fix = 0
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
    red_tail = red_txt[-1500:] if len(red_txt) > 1500 else red_txt
    diag = read(os.path.join(lv, "surprises.md"), 2500) or read(os.path.join(lv, "evidence.md"), 2500)
    if len(red_tail.strip()) < 80 or len(diag.strip()) < 100:
        continue
    spec_text = q_spec.split("\n\nImplement this")[0]
    user = (f"{spec_text}\n\nMy current attempt FAILS with this output:\n```\n{red_tail}\n```\n\n"
            f"Diagnose the root cause and give me the corrected implementation.")
    ans = (f"## Diagnosis (from hands-on evidence)\n\n{diag}\n\n## Corrected implementation\n\n{impl}")
    if emit([{"role": "system", "content": SYSTEM},
             {"role": "user", "content": user},
             {"role": "assistant", "content": ans}]):
        n_fix += 1

# ---- 2. task tuples: 60% tool-wrapped, 40% direct ----
v4_task_traces = []
v4_knowledge_traces = []
for l in open(os.path.join(HERE, "train_v4.jsonl")):
    ex = json.loads(l)
    roles = [m["role"] for m in ex["messages"]]
    user_c = ex["messages"][1]["content"]
    if user_c.startswith("Task from the "):
        v4_task_traces.append(ex)
    elif "tool" in roles:
        v4_knowledge_traces.append(ex)

n_task_tool = n_task_direct = 0
for ex in v4_task_traces:
    if random.random() < 0.6:
        out.append(ex); n_task_tool += 1
    else:
        ms = ex["messages"]
        direct = [ms[0], ms[1], ms[-1]]
        out.append({"tools": TOOLS, "messages": direct}); n_task_direct += 1

# ---- 3. context-faithfulness subset (tool -> grounded answer) ----
random.shuffle(v4_knowledge_traces)
faith = v4_knowledge_traces[:3000]
out.extend(faith)

# ---- 4. commands/ops traces ----
n_cmd = 0
cmd_files = sorted(glob.glob(os.path.join(MAIN, "*", "degrees", "*", "03-pocs", "L*", "commands.md")))
random.shuffle(cmd_files)
for cf in cmd_files:
    if n_cmd >= 400:
        break
    lv = os.path.dirname(cf)
    parts = lv[len(MAIN) + 1:].split(os.sep)
    target, level = parts[0], parts[-1]
    if (target, level) in bench:
        continue
    spec_text = read(os.path.join(lv, "README.md"), 2000) or read(os.path.join(lv, "intent.md"), 2000)
    cmds = read(cf, 4000)
    if len(spec_text.strip()) < 150 or len(cmds.strip()) < 150:
        continue
    user = (f"Task from the {target} curriculum (POC level {level}):\n\n{spec_text.strip()}\n\n"
            f"What are the exact commands to set up, run, and verify this against the real service?")
    if emit([{"role": "system", "content": SYSTEM},
             {"role": "user", "content": user},
             {"role": "assistant", "content": cmds}]):
        n_cmd += 1

random.shuffle(out)
toks = 0
for ex in out:
    for m in ex["messages"]:
        toks += len(m.get("content") or json.dumps(m.get("tool_calls", ""))) // 4
with open(os.path.join(HERE, "train_v5.jsonl"), "w") as f:
    for ex in out:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
print(json.dumps({"error_fix": n_fix, "task_tool": n_task_tool, "task_direct": n_task_direct,
                  "faithfulness": len(faith), "commands": n_cmd, "total": len(out),
                  "est_tokens": toks, "cost_2ep": round(toks / 1e6 * 3 * 2, 2)}))
