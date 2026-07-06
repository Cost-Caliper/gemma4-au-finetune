#!/bin/bash
# Phase 2 (v2): BF16 multi-LoRA deployment serving BOTH base and tuned -> eval both -> ALWAYS tear down.
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ../secrets.local.env; set +a

ACCT=dennison-bertram
BASE_MODEL="accounts/fireworks/models/gemma-4-26b-a4b-it"
TUNED="accounts/$ACCT/models/au-fw-gemma4-au-v1"
SHAPE="accounts/fireworks/deploymentShapes/rft-gemma-4-26b-a4b-it"   # 4xB200 FULL_PRECISION (BF16 -> addons supported)
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

echo "=== 1. create BF16 deployment with addons ==="
firectl deployment create "$BASE_MODEL" --deployment-shape "$SHAPE" --enable-addons \
  --min-replica-count 1 --max-replica-count 1 \
  --display-name au-fw-eval-bf16 -a "$ACCT" --api-key "$FIREWORKS_API_KEY" \
  2>&1 | grep -vE "updates|version|upgrade" | tee tuned-deploy.log | grep -E "Name:|State:|Status:|Failed|error"
DEP_ID=$(grep -oE "deployments/[a-z0-9]+" tuned-deploy.log | head -1 | cut -d/ -f2)
echo "deployment id: $DEP_ID"
[ -z "$DEP_ID" ] && { echo "CREATE FAILED"; exit 1; }
DEP="accounts/$ACCT/deployments/$DEP_ID"
BASE_ADDR="$BASE_MODEL#$DEP"
TUNED_ADDR="$TUNED#$DEP"

echo "=== 2. wait for base to serve ==="
OK=0
for i in $(seq 1 60); do
  CODE=$(curl -s -o /tmp/fw-b.json -w "%{http_code}" -X POST \
    "https://api.fireworks.ai/inference/v1/chat/completions" \
    -H "Authorization: Bearer $FIREWORKS_API_KEY" -H "Content-Type: application/json" \
    -d "{\"model\":\"$BASE_ADDR\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: pong\"}],\"max_tokens\":10}")
  echo "$(date +%H:%M:%S) base smoke HTTP $CODE"
  [ "$CODE" = "200" ] && { OK=1; break; }
  S=$(firectl deployment get "$DEP_ID" -a "$ACCT" --api-key "$FIREWORKS_API_KEY" 2>/dev/null | grep -E "^State:" | awk '{print $2}')
  echo "  state=$S"
  case "$S" in *FAILED*) echo "DEPLOYMENT FAILED"; exit 1;; esac
  sleep 30
done
[ "$OK" != "1" ] && { echo "BASE NEVER SERVED"; exit 1; }

echo "=== 3. load LoRA addon ==="
curl -s -X POST "https://api.fireworks.ai/v1/accounts/$ACCT/deployedModels" \
  -H "Authorization: Bearer $FIREWORKS_API_KEY" -H "Content-Type: application/json" \
  -d "{\"model\":\"$TUNED\",\"deployment\":\"$DEP\"}" | head -c 300; echo

echo "=== 4. wait for tuned to serve ==="
OK=0
for i in $(seq 1 40); do
  CODE=$(curl -s -o /tmp/fw-t.json -w "%{http_code}" -X POST \
    "https://api.fireworks.ai/inference/v1/chat/completions" \
    -H "Authorization: Bearer $FIREWORKS_API_KEY" -H "Content-Type: application/json" \
    -d "{\"model\":\"$TUNED_ADDR\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: pong\"}],\"max_tokens\":10}")
  echo "$(date +%H:%M:%S) tuned smoke HTTP $CODE"
  [ "$CODE" = "200" ] && { OK=1; break; }
  sleep 15
done
head -c 200 /tmp/fw-t.json; echo
[ "$OK" != "1" ] && { echo "TUNED NEVER SERVED"; exit 1; }

echo "=== 5. eval BOTH models on same BF16 deployment ==="
python3 run_eval.py --model base_bf16="$BASE_ADDR" --model tuned="$TUNED_ADDR" \
  --judge --concurrency 6 2>tuned-eval-progress.log
RC=$?
echo "tuned eval exit: $RC"
exit $RC
