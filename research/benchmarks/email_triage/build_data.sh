#!/usr/bin/env bash
# Regenerate every synthetic training file from the current generator.
# Output goes to $DATA_DIR (default ~/data/laya-email), outside the repo: the files are large and
# fully determined by the seeds below, so they are rebuilt rather than committed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HERE/../../../.venv/bin/python}"
DATA_DIR="${DATA_DIR:-$HOME/data/laya-email}"
mkdir -p "$DATA_DIR"

# six default teams (the original six-team set)
"$PY" "$HERE/generate_synthetic.py" --n 1200 --seed 7 --out "$DATA_DIR/synthetic_emails.jsonl"

# one file per folder set, then the combined file used for training
for preset in minimal-5 standard-8 detailed-12; do
  "$PY" "$HERE/generate_synthetic.py" --n 2500 --seed 11 --preset "$preset" \
    --out "$DATA_DIR/synthetic_$preset.jsonl"
done
cat "$DATA_DIR/synthetic_minimal-5.jsonl" "$DATA_DIR/synthetic_standard-8.jsonl" \
    "$DATA_DIR/synthetic_detailed-12.jsonl" > "$DATA_DIR/synthetic_all_folders.jsonl"

wc -l "$DATA_DIR"/*.jsonl
