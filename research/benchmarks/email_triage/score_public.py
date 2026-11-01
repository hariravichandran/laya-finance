"""Score the yes/no flags of a checkpoint on the real-email holdout from `build_public.py`.

Each holdout row has `spam` and/or `phish` labels (None where the corpus does not label that flag).
Accuracy is over the labelled rows only. Phishing precision and recall are reported separately,
since a missed lure and a wrongly flagged real email have different costs.

Usage:
  python score_public.py --model ~/models/laya-email/public-v7 \
      --holdout ~/data/laya-email/public/public_holdout.jsonl
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import laya  # noqa: E402
import evaluate as E  # noqa: E402
from laya.presets import email_questions  # noqa: E402

CHUNK = 50
FLAG_QUESTIONS = {k: v for k, v in email_questions().items() if k in ("is_spam", "is_phishing")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--out", default=None)
    # CPU by default: scoring the holdout on the ROCm iGPU grew to 15 GB of host memory,
    # apparently from the allocator holding blocks across variable-length batches
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dump", default=None,
                        help="write per-email probabilities (JSONL) for threshold calibration")
    args = parser.parse_args()

    with open(os.path.expanduser(args.holdout), encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    agent = laya.load(os.path.expanduser(args.model), device=args.device)
    # chunked: one predict_batch over the whole holdout exhausted host memory
    answers = []
    for start in range(0, len(rows), CHUNK):
        answers += E.predict_all(agent, rows[start:start + CHUNK], FLAG_QUESTIONS)

    spam_pairs = [(r["spam"], a["is_spam"]["noul"] >= 0.5) for r, a in zip(rows, answers) if r["spam"] is not None]
    phish_pairs = [(r["phish"], a["is_phishing"]["noul"] >= 0.5) for r, a in zip(rows, answers)
                   if r["phish"] is not None]
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            for r, a in zip(rows, answers):
                fh.write(json.dumps({"corpus": r["corpus"], "spam": r["spam"], "phish": r["phish"],
                                     "p_spam": a["is_spam"]["noul"], "p_phish": a["is_phishing"]["noul"]})
                         + "\n")
    tp = sum(1 for y, p in phish_pairs if y and p)
    fp = sum(1 for y, p in phish_pairs if not y and p)
    fn = sum(1 for y, p in phish_pairs if y and not p)
    metrics = {
        "spam_acc": round(sum(int(y) == int(p) for y, p in spam_pairs) / max(1, len(spam_pairs)), 4),
        "spam_n": len(spam_pairs),
        "phishing_acc": round(sum(int(y) == int(p) for y, p in phish_pairs) / max(1, len(phish_pairs)), 4),
        "phishing_precision": round(tp / max(1, tp + fp), 4),
        "phishing_recall": round(tp / max(1, tp + fn), 4),
        "phishing_n": len(phish_pairs),
    }
    print(json.dumps(metrics, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"model": args.model, "holdout": os.path.basename(args.holdout), "metrics": metrics},
                      fh, indent=2)


if __name__ == "__main__":
    main()
