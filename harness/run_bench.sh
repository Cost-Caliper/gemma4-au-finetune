#!/bin/bash
# Benchmark base vs tuned Gemma 4 on the eval deployment, then ALWAYS tear down.
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ../secrets.local.env; set +a

ACCT=dennison-bertram
DEP_ID=$(cat deployment.id)
DEP="accounts/$ACCT/deployments/$DEP_ID"
BASE="accounts/fireworks/models/gemma-4-26b-a4b-it#$DEP"
TUNED="accounts/$ACCT/models/au-fw-gemma4-au-v1"

teardown() {
  echo "=== TEARDOWN: deleting $DEP ==="
  firectl deployment delete "$DEP_ID" -a "$ACCT" --api-key "$FIREWORKS_API_KEY" 2>&1 | grep -vE "updates|version|upgrade" | head -5
  sleep 10
  echo "=== deployment list after delete ==="
  firectl deployment list -a "$ACCT" --api-key "$FIREWORKS_API_KEY" 2>&1 | grep -vE "updates|version|upgrade" | head -10
}
trap teardown EXIT

echo "=== 1. load LoRA addon onto deployment ==="
for i in $(seq 1 10); do
  RESP=$(curl -s -X POST "https://api.fireworks.ai/v1/accounts/$ACCT/deployedModels" \
    -H "Authorization: Bearer $FIREWORKS_API_KEY" -H "Content-Type: application/json" \
    -d "{\"model\":\"$TUNED\",\"deployment\":\"$DEP\"}")
  echo "$RESP" | head -c 300; echo
  echo "$RESP" | grep -q '"state"' && break
  echo "$RESP" | grep -qi "already" && break
  sleep 20
done

echo "=== 2. wait for tuned model to serve (200 smoke) ==="
TUNED_ADDR="$TUNED"
for i in $(seq 1 40); do
  CODE=$(curl -s -o /tmp/fw-tuned-smoke.json -w "%{http_code}" -X POST \
    "https://api.fireworks.ai/inference/v1/chat/completions" \
    -H "Authorization: Bearer $FIREWORKS_API_KEY" -H "Content-Type: application/json" \
    -d "{\"model\":\"$TUNED_ADDR\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: pong\"}],\"max_tokens\":10}")
  echo "$(date +%H:%M:%S) tuned smoke HTTP $CODE"
  if [ "$CODE" = "200" ]; then break; fi
  if [ "$i" = "20" ]; then
    # fallback addressing: model#deployment
    TUNED_ADDR="$TUNED#$DEP"
    echo "switching tuned addressing to $TUNED_ADDR"
  fi
  sleep 15
done
head -c 200 /tmp/fw-tuned-smoke.json; echo
if [ "$CODE" != "200" ]; then echo "TUNED MODEL NEVER SERVED — aborting eval"; exit 1; fi

echo "=== 3. run eval (both models, all probes + code, with judge) ==="
python3 run_eval.py \
  --model base="$BASE" \
  --model tuned="$TUNED_ADDR" \
  --judge --concurrency 4 2>eval-progress.log
EVAL_RC=$?
echo "eval exit: $EVAL_RC"
exit $EVAL_RC
