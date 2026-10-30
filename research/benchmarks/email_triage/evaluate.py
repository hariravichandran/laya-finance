"""Score a Laya checkpoint on the hand-labelled email set.

Runs `laya.email_questions()` over `eval_set.jsonl` and reports, per question: category accuracy,
the is_spam and is_phishing accuracy (with phishing recall, the number that matters most), the
urgency exact and within-one accuracy, and needs_reply accuracy. The threshold for a yes/no is
0.5 on P(yes), the same rule the preset's verdict uses.

Usage:
  python evaluate.py --model hravi/laya-finance
  python evaluate.py --model /path/to/finetuned --out results.json
"""

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

import laya  # noqa: E402
from laya.email import email_state  # noqa: E402
from laya.presets import email_questions  # noqa: E402

import folders as FO  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load_eval(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def predict_all(agent, rows, questions):
    states = [email_state(r["subject"], r["body"], r["sender"]) for r in rows]
    return [out["answers"] for out in agent.predict_batch(states, questions)]


def score(rows, answers):
    n = len(rows)
    cat = sum(a["category"]["choice"].lower() == r["team"].lower() for r, a in zip(rows, answers))
    spam = sum((a["is_spam"]["noul"] >= 0.5) == bool(r["spam"]) for r, a in zip(rows, answers))
    phish_hits = [(a["is_phishing"]["noul"] >= 0.5) for r, a in zip(rows, answers) if r["phish"]]
    phish_fp = [(a["is_phishing"]["noul"] >= 0.5) for r, a in zip(rows, answers) if not r["phish"]]
    urg_pred = [max(a["urgency"]["probabilities"], key=a["urgency"]["probabilities"].get) for a in answers]
    urg_exact = sum(int(p) == r["urg"] for p, r in zip(urg_pred, rows))
    urg_within = sum(abs(int(p) - r["urg"]) <= 1 for p, r in zip(urg_pred, rows))
    reply = sum((a["needs_reply"]["noul"] >= 0.5) == bool(r["reply"]) for r, a in zip(rows, answers))
    return {
        "n": n,
        "category_acc": round(cat / n, 3),
        "is_spam_acc": round(spam / n, 3),
        "is_phishing_recall": round(sum(phish_hits) / max(1, len(phish_hits)), 3),
        "is_phishing_false_positive_rate": round(sum(phish_fp) / max(1, len(phish_fp)), 3),
        "urgency_exact_acc": round(urg_exact / n, 3),
        "urgency_within_one_acc": round(urg_within / n, 3),
        "needs_reply_acc": round(reply / n, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hub id or local checkpoint directory")
    parser.add_argument("--eval", default=os.path.join(HERE, "eval_set.jsonl"))
    parser.add_argument("--out", default=None, help="write the metrics as JSON to this path")
    parser.add_argument("--preset", choices=sorted(FO.PRESETS), default=None,
                        help="score against this folder set's questions; default is the six default teams")
    args = parser.parse_args()

    rows = load_eval(args.eval)
    agent = laya.load(args.model)
    if args.preset:
        questions = FO.questions_for(FO.preset(args.preset), email_questions)
    else:
        questions = email_questions()
    answers = predict_all(agent, rows, questions)
    metrics = score(rows, answers)
    print(json.dumps(metrics, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"model": args.model, "metrics": metrics}, fh, indent=2)


if __name__ == "__main__":
    main()
