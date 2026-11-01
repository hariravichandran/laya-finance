"""Fine-tune the email sorter on the GPU (ROCm or CUDA) and save the checkpoint.

Concatenates the training JSONL files, runs `laya.train.finetune` from a base checkpoint, and
writes the checkpoint plus a run summary next to it. The summary records the exact data files,
row counts and config, so a checkpoint can always be traced back to what made it.

On the AMD Radeon integrated GPU (gfx1035), set HSA_OVERRIDE_GFX_VERSION=10.3.0 before running.
The ROCm torch build lives in ~/venvs/laya-rocm; see README.md in this folder.

Usage:
  HSA_OVERRIDE_GFX_VERSION=10.3.0 ~/venvs/laya-rocm/bin/python train_gpu.py \
      --base convaiinnovations/laya --data synthetic_emails.jsonl --data synthetic_all_folders.jsonl \
      --epochs 3 --out ~/models/laya-email/folders-v1
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from huggingface_hub import snapshot_download  # noqa: E402

from laya.train import TrainConfig, finetune  # noqa: E402


def resolve_base(base):
    """A local checkpoint directory, or a Hub id downloaded to the cache."""
    if os.path.isdir(base):
        return base
    return snapshot_download(base)


def concatenate(paths, dest):
    rows = 0
    with open(dest, "w", encoding="utf-8") as out:
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        rows += 1
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="Hub id or local checkpoint directory")
    parser.add_argument("--data", action="append", required=True, help="JSONL training file (repeat)")
    parser.add_argument("--out", required=True, help="output checkpoint directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--head-max-len", type=int, default=128)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda", help="cuda (also ROCm), cpu, or auto")
    parser.add_argument("--loss", default="rlcd", choices=("rlcd", "soft-ce"),
                        help="rlcd: policy gradient plus soft cross-entropy; soft-ce: soft cross-entropy only")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="do not permute choice options during training")
    args = parser.parse_args()

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    base = resolve_base(args.base)
    cfg = TrainConfig(epochs=args.epochs, max_len=args.max_len, head_max_len=args.head_max_len,
                      micro_batch=args.micro_batch, grad_accum=args.grad_accum, seed=args.seed,
                      shuffle_options=() if args.no_shuffle else ("choice",),
                      loss=args.loss, log_every=100)

    with tempfile.TemporaryDirectory() as tmp:
        merged = os.path.join(tmp, "train.jsonl")
        rows = concatenate([os.path.expanduser(p) for p in args.data], merged)
        print("training rows %d from %d file(s)" % (rows, len(args.data)), flush=True)
        summary = finetune(merged, base, out, cfg, device=args.device)
        # keep a copy of the exact training rows that produced this checkpoint
        shutil.copy(merged, os.path.join(out, "train_rows.jsonl"))

    record = {
        "base": args.base,
        "data": [os.path.basename(p) for p in args.data],
        "rows": rows,
        "config": asdict(cfg),
        "device": args.device,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "train_items": summary["train_items"],
        "calibration_items": summary["calibration_items"],
        "epoch_loss": summary["epoch_loss"],
        "temperature": summary["temperature"],
    }
    with open(os.path.join(out, "run_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
