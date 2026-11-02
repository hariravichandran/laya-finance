#!/usr/bin/env bash
# Score each model on the same balanced real-email subset, one after another (memory limit).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/venvs/laya-rocm314/bin/python"
H="$HOME/data/laya-email/public/public_holdout_balanced.jsonl"
export HSA_OVERRIDE_GFX_VERSION=10.3.0
for m in convaiinnovations/laya "$HOME/models/laya-email/folders-v4" "$HOME/models/laya-email/public-v7"; do
  n=$(basename "$m")
  "$PY" "$HERE/score_public.py" --model "$m" --holdout "$H" \
    --out "$HERE/results/balanced_$n.json" --dump "$HOME/data/laya-email/dump_balanced_$n.jsonl"
done
echo SCORE_ALL_DONE
