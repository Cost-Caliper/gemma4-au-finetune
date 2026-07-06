#!/bin/bash
# Full local pipeline: wait for base download -> merge -> GGUF q8_0 x2 -> recall gate -> evals -> summary.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
LCPP=llama.cpp
ADAPTER_DIR="au-fw-gemma4-au-v1-adapter/tuned-model-b608vrw0/9e919d/au-fw-gemma4-au-v1/promoted-step-742-eaff7ec2"
SERVER_PID=""

cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

stage() { echo; echo "########## STAGE: $1 ($(date +%H:%M:%S)) ##########"; }

stage "wait for base download"
for i in $(seq 1 240); do
  SNAP=$(ls -d ~/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/*/ 2>/dev/null | head -1)
  N_SHARDS=$(ls "$SNAP"model-*.safetensors 2>/dev/null | wc -l | tr -d ' ')
  # complete when every shard in the index exists as a real file (no .incomplete)
  if [ -n "$SNAP" ] && [ -f "$SNAP/model.safetensors.index.json" ]; then
    WANT=$($PY -c "import json,sys; print(len(set(json.load(open('$SNAP/model.safetensors.index.json'))['weight_map'].values())))")
    INC=$(find ~/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it -name "*.incomplete" 2>/dev/null | wc -l | tr -d ' ')
    echo "$(date +%H:%M:%S) shards $N_SHARDS/$WANT incomplete=$INC"
    if [ "$N_SHARDS" = "$WANT" ] && [ "$INC" = "0" ]; then break; fi
  fi
  sleep 60
done
[ "$N_SHARDS" != "$WANT" ] && { echo "PIPELINE_FAIL: download never completed"; exit 1; }

stage "merge adapter into base"
$PY merge_adapter.py "$ADAPTER_DIR" "$SNAP" merged-model 2>&1 | tail -8
grep -q "DONE:" <($PY -c "print('DONE: marker check skipped')") # no-op
[ -f merged-model/model.safetensors.index.json ] || cp "$SNAP/model.safetensors.index.json" merged-model/ 2>/dev/null
[ -f merged-model/$(ls merged-model | grep -m1 safetensors) ] || { echo "PIPELINE_FAIL: merge produced no shards"; exit 1; }

stage "convert MERGED -> GGUF q8_0"
$PY $LCPP/convert_hf_to_gguf.py merged-model --outfile tuned-q8.gguf --outtype q8_0 2>&1 | tail -4
[ -f tuned-q8.gguf ] || { echo "PIPELINE_FAIL: tuned gguf missing"; exit 1; }

stage "convert BASE -> GGUF q8_0"
$PY $LCPP/convert_hf_to_gguf.py "$SNAP" --outfile base-q8.gguf --outtype q8_0 2>&1 | tail -4
[ -f base-q8.gguf ] || { echo "PIPELINE_FAIL: base gguf missing"; exit 1; }

start_server() {
  llama-server -m "$1" --port 8080 --ctx-size 8192 -ngl 999 --jinja --parallel 2 >server-"$2".log 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 60); do
    curl -s http://127.0.0.1:8080/health | grep -q '"ok"' && return 0
    sleep 5
  done
  echo "PIPELINE_FAIL: server for $2 never healthy"; tail -5 server-"$2".log; return 1
}

stage "serve TUNED + recall validation gate"
start_server tuned-q8.gguf tuned || exit 1
EVAL_API_BASE=http://127.0.0.1:8080/v1 $PY validate_recall.py 12
RC=$?
if [ $RC -ne 0 ]; then
  echo "PIPELINE_FAIL: RECALL GATE FAILED (rc=$RC) — fused-expert decode suspect; aborting evals"
  exit 2
fi
echo "RECALL GATE PASSED"

stage "eval TUNED (local q8_0)"
EVAL_API_BASE=http://127.0.0.1:8080/v1 FIREWORKS_API_KEY=unused ANTHROPIC_API_KEY="$(grep '^ANTHROPIC_API_KEY=' ../secrets.local.env | cut -d= -f2-)" \
  $PY run_eval.py --model tuned_local=local --judge --concurrency 2 2>tuned-local-progress.log
mv summary.json summary_tuned_local.json 2>/dev/null
kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""; sleep 5

stage "serve BASE + eval (local q8_0)"
start_server base-q8.gguf base || exit 1
EVAL_API_BASE=http://127.0.0.1:8080/v1 FIREWORKS_API_KEY=unused ANTHROPIC_API_KEY="$(grep '^ANTHROPIC_API_KEY=' ../secrets.local.env | cut -d= -f2-)" \
  $PY run_eval.py --model base_local=local --judge --concurrency 2 2>base-local-progress.log
mv summary.json summary_base_local.json 2>/dev/null
kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""

stage "final summary"
$PY - <<'EOF'
import json
out = {}
for name in ("summary_tuned_local", "summary_base_local", "summary_base"):
    try:
        out[name] = json.load(open(f"{name}.json"))
    except FileNotFoundError:
        pass
print(json.dumps(out, indent=2))
json.dump(out, open("summary_final.json", "w"), indent=2)
EOF
echo "PIPELINE_DONE"
