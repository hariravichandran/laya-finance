"""Score a two-model pipeline: one model for the category, another for the yes/no flags.

The category head is the weakest part of every fine-tune so far (it collapses onto one folder), while
the flags (phishing, spam, urgency, reply) are where the fine-tunes help. This takes the category from
one checkpoint and the flags from another, and scores the result the same way `evaluate.py` does.

Usage:
  python combine_eval.py --category-model convaiinnovations/laya \
      --flag-model ~/models/laya-email/folders-v4 --preset detailed-12
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

import laya  # noqa: E402
import evaluate as E  # noqa: E402
import folders as FO  # noqa: E402
from laya.presets import email_questions  # noqa: E402

FLAG_FIELDS = ("is_spam", "is_phishing", "urgency", "needs_reply")


def combined(category_answers, flag_answers):
    """Category from the first model, the flag fields from the second."""
    out = []
    for cat, flags in zip(category_answers, flag_answers):
        merged = dict(flags)
        merged["category"] = cat["category"]
        out.append(merged)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category-model", required=True)
    parser.add_argument("--flag-model", required=True)
    parser.add_argument("--eval", default=os.path.join(HERE, "eval_set.jsonl"))
    parser.add_argument("--preset", choices=sorted(FO.PRESETS), default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = E.load_eval(args.eval)
    questions = FO.questions_for(FO.preset(args.preset), email_questions) if args.preset else email_questions()
    cat_answers = E.predict_all(laya.load(args.category_model), rows, questions)
    flag_answers = E.predict_all(laya.load(args.flag_model), rows, questions)
    metrics = E.score(rows, combined(cat_answers, flag_answers))
    print(json.dumps(metrics, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"category_model": args.category_model, "flag_model": args.flag_model,
                       "metrics": metrics}, fh, indent=2)


if __name__ == "__main__":
    main()
