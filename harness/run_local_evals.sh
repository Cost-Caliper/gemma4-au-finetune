#!/bin/bash
# Final dual eval on local Q8 GGUFs (tuned then base). Assumes ggufs exist.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
ANTH=$(grep '^ANTHROPIC_API_KEY=' ../secrets.local.env | cut -d= -f2-)
SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

serve() {
  llama-server -m "$1" --port 8080 --ctx-size 16384 -ngl 999 --jinja --parallel 2 >server-"$2".log 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 60); do
    curl -s http://127.0.0.1:8080/health | grep -q '"ok"' && return 0
    sleep 5
  done
  echo "EVALS_FAIL: server $2 never healthy"; return 1
}

echo "=== TUNED eval ($(date +%H:%M:%S)) ==="
serve tuned-q8.gguf tuned2 || exit 1
EVAL_API_BASE=http://127.0.0.1:8080/v1 FIREWORKS_API_KEY=unused ANTHROPIC_API_KEY="$ANTH" \
  $PY run_eval.py --model tuned_local=local --judge --concurrency 2 2>tuned-local-progress.log
mv summary.json summary_tuned_local.json
kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""; sleep 5

echo "=== BASE eval ($(date +%H:%M:%S)) ==="
serve base-q8.gguf base2 || exit 1
EVAL_API_BASE=http://127.0.0.1:8080/v1 FIREWORKS_API_KEY=unused ANTHROPIC_API_KEY="$ANTH" \
  $PY run_eval.py --model base_local=local --judge --concurrency 2 2>base-local-progress.log
mv summary.json summary_base_local.json
kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""

$PY - <<'EOF'
import json
out={}
for n in ("summary_tuned_local","summary_base_local","summary_base"):
    try: out[n]=json.load(open(f"{n}.json"))
    except FileNotFoundError: pass
json.dump(out, open("summary_final.json","w"), indent=2)
print(json.dumps(out, indent=2))
EOF
echo "EVALS_DONE"
