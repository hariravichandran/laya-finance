#!/usr/bin/env bash
# Wait for the running GPU fine-tune to finish, train the next version on the current data, then
# evaluate both checkpoints on the two eval sets. Results land in results/ and the run logs in
# ~/models/laya-email/<version>/.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/../../../.venv/bin/python"
GPU_PY="$HOME/venvs/laya-rocm/bin/python"
DATA="$HOME/data/laya-email"
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# wait for the previous run to write its summary; a process check would match this shell
while [ ! -f "$HOME/models/laya-email/folders-v2/run_summary.json" ]; do sleep 60; done

V=${VERSION:-folders-v3}
EPOCHS=${EPOCHS:-2}
mkdir -p "$HOME/models/laya-email/$V"
"$GPU_PY" "$HERE/train_gpu.py" --base convaiinnovations/laya \
  --data "$DATA/synthetic_emails.jsonl" --data "$DATA/synthetic_all_folders.jsonl" \
  --epochs "$EPOCHS" --out "$HOME/models/laya-email/$V" > "$HOME/models/laya-email/$V/train.log" 2>&1

for name in folders-v2 "$V"; do
  M="$HOME/models/laya-email/$name"
  [ -f "$M/model.safetensors" ] || continue
  "$PY" "$HERE/evaluate.py" --model "$M" --out "$HERE/results/${name}_six_team.json"
  "$PY" "$HERE/evaluate.py" --model "$M" --eval "$HERE/eval_folders.jsonl" --preset detailed-12 \
    --out "$HERE/results/${name}_detailed12.json"
done
echo CHAIN_DONE
