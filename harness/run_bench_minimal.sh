#!/bin/bash
# Multi-LoRA on the known-good MINIMAL H200 shape, with full addressing matrix -> eval -> ALWAYS tear down.
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ../secrets.local.env; set +a

ACCT=dennison-bertram
BASE_MODEL="accounts/fireworks/models/gemma-4-26b-a4b-it"
TUNED="accounts/$ACCT/models/au-fw-gemma4-au-v1"
SHAPE="accounts/fireworks/deploymentShapes/gemma4-26b-a4b-it"
DEP_ID=""

teardown() {
  if [ -n "$DEP_ID" ]; then
    echo "=== TEARDOWN: deleting $DEP_ID ==="
    firectl deployment delete "$DEP_ID" --ignore-checks -a "$ACCT" --api-key "$FIREWORKS_API_KEY" 2>&1 | grep -vE "updates|version|upgrade" | head -5
    sleep 10
    firectl deployment get "$DEP_ID" -a "$ACCT" --api-key "$FIREWORKS_API_KEY" 2>&1 | grep -E "^State:" | head -2
  fi
}
trap teardown EXIT

smoke() {  # $1 = model address; echoes HTTP code
  curl -s -o /tmp/fw-s.json -w "%{http_code}" -X POST \
    "https://api.fireworks.ai/inference/v1/chat/completions" \
    -H "Authorization: Bearer $FIREWORKS_API_KEY" -H "Content-Type: application/json" \
    -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: pong\"}],\"max_tokens\":10}"
}

echo "=== 1. create MINIMAL deployment with addons ==="
firectl deployment create "$BASE_MODEL" --deployment-shape "$SHAPE" --enable-addons \
  --min-replica-count 1 --max-replica-count 1 \
  --display-name au-fw-eval-min2 -a "$ACCT" --api-key "$FIREWORKS_API_KEY" \
  2>&1 | grep -vE "updates|version|upgrade" | tee min2-deploy.log | grep -E "Name:|State:|Status:"
DEP_ID=$(grep -oE "deployments/[a-z0-9]+" min2-deploy.log | head -1 | cut -d/ -f2)
echo "deployment id: $DEP_ID"
[ -z "$DEP_ID" ] && { echo "CREATE FAILED"; exit 1; }
DEP="accounts/$ACCT/deployments/$DEP_ID"
BASE_ADDR="$BASE_MODEL#$DEP"

echo "=== 2. wait for base ==="
OK=0
for i in $(seq 1 40); do
  CODE=$(smoke "$BASE_ADDR"); echo "$(date +%H:%M:%S) base HTTP $CODE"
  [ "$CODE" = "200" ] && { OK=1; break; }
  S=$(firectl deployment get "$DEP_ID" -a "$ACCT" --api-key "$FIREWORKS_API_KEY" 2>/dev/null | grep -E "^State:" | awk '{print $2}')
  case "$S" in *FAILED*) echo "DEPLOYMENT FAILED"; exit 1;; esac
  sleep 30
done
[ "$OK" != "1" ] && { echo "BASE NEVER SERVED"; exit 1; }

echo "=== 3. load LoRA ==="
RESP=$(curl -s -X POST "https://api.fireworks.ai/v1/accounts/$ACCT/deployedModels" \
  -H "Authorization: Bearer $FIREWORKS_API_KEY" -H "Content-Type: application/json" \
  -d "{\"model\":\"$TUNED\",\"deployment\":\"$DEP\"}")
echo "$RESP" | head -c 400; echo
DM_NAME=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('name','').split('/')[-1])" 2>/dev/null || true)
echo "deployed-model name: $DM_NAME"

echo "=== 4. addressing matrix until one serves ==="
CANDS=("$TUNED#$DEP" "$TUNED")
[ -n "$DM_NAME" ] && CANDS+=("accounts/$ACCT/models/$DM_NAME#$DEP" "accounts/$ACCT/models/$DM_NAME" "accounts/$ACCT/deployedModels/$DM_NAME")
TUNED_ADDR=""
for i in $(seq 1 30); do
  for C in "${CANDS[@]}"; do
    CODE=$(smoke "$C")
    echo "$(date +%H:%M:%S) HTTP $CODE <- $C"
    [ "$CODE" = "200" ] && { TUNED_ADDR="$C"; break 2; }
  done
  sleep 20
done
if [ -z "$TUNED_ADDR" ]; then head -c 300 /tmp/fw-s.json; echo; echo "TUNED NEVER SERVED"; exit 1; fi
echo "TUNED SERVING at: $TUNED_ADDR"

echo "=== 5. sanity: tuned answers differ from base? ==="
Q='{"model":"MODEL","messages":[{"role":"system","content":"You are an expert software agent for modern AI/agent libraries, developer tools, and cloud services. You implement tasks against real services with runnable code, cite exact API surfaces, and answer from hands-on evidence — you flag pitfalls honestly and never invent APIs."},{"role":"user","content":"I am building with supabase. What is the gotcha here: Broadcasts on private channels do NOT persist to realtime.messages? How do I avoid it?"}],"max_tokens":150,"temperature":0}'
for M in "$BASE_ADDR" "$TUNED_ADDR"; do
  echo "--- $M"
  curl -s -X POST "https://api.fireworks.ai/inference/v1/chat/completions" \
    -H "Authorization: Bearer $FIREWORKS_API_KEY" -H "Content-Type: application/json" \
    -d "${Q/MODEL/$M}" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'][:250])" 2>/dev/null
done

echo "=== 6. eval BOTH models ==="
python3 run_eval.py --model base_min="$BASE_ADDR" --model tuned="$TUNED_ADDR" \
  --judge --concurrency 6 2>min2-eval-progress.log
RC=$?
echo "tuned eval exit: $RC"
exit $RC
