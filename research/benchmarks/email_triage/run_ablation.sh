#!/usr/bin/env bash
# Two six-team controls, one setting changed each, to find what makes the category head collapse.
#   soft-ce:    the soft cross-entropy loss alone, no policy-gradient term
#   no-shuffle: RLCD loss with option order left as listed (no per-epoch permutation)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_PY="$HOME/venvs/laya-rocm314/bin/python"
DATA="$HOME/data/laya-email"
export HSA_OVERRIDE_GFX_VERSION=10.3.0
run() {
  local name=$1; shift
  mkdir -p "$HOME/models/laya-email/$name"
  "$GPU_PY" "$HERE/train_gpu.py" --base convaiinnovations/laya --data "$DATA/synthetic_emails.jsonl" \
    --epochs 2 --out "$HOME/models/laya-email/$name" "$@" > "$HOME/models/laya-email/$name/train.log" 2>&1
}
run abl-soft-ce --loss soft-ce
run abl-no-shuffle --no-shuffle
echo ABLATION_DONE
