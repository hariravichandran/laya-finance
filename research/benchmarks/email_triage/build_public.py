"""Public labelled email corpora, converted for the flag heads of the email sorter.

Two corpora, both downloaded from Hugging Face and kept outside the repo:

  - SetFit/enron_spam (`enron_spam_data.csv`): Enron messages labelled spam or ham. Licence is not
    stated on the card; the underlying Enron corpus is a public release. Spam gives is_spam; ham is
    real business mail, so it also gives is_phishing = no.
  - zefang-liu/phishing-email-dataset (`Phishing_Email.csv`): a copy of the Kaggle phishing-email
    dataset, licence LGPL-3.0. Phishing gives is_phishing; the safe mail has no spam label, so it
    gives nothing for is_spam.

These rows carry no folder label, so they only train the yes/no questions. A row's `gold` names only
the questions it has a label for; the training code skips the rest. The category head is not trained
on them, and that is deliberate: the category head is trained on the labelled synthetic data.

Each corpus is split 90/10 with a fixed seed. The 10% is a real-email holdout for the flags, which is
a far larger test than the 30 hand-written emails.

Usage:
  python build_public.py --data-dir ~/data/public --out-dir ~/data/laya-email/public \
      --train-per-corpus 5000
"""

import argparse
import csv
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from laya.email import email_state  # noqa: E402
from laya.presets import email_questions  # noqa: E402

csv.field_size_limit(10 ** 9)

LABEL_YES = 0.95
LABEL_NO = 0.05


def noul_gold(yes):
    """P(yes) as the `{false, true}` distribution the training code reads (`p[1]` is P(true))."""
    y = LABEL_YES if yes else LABEL_NO
    return {"probabilities": {"false": 1 - y, "true": y}}


def enron_rows(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            body = (r.get("Message") or "").strip()
            if not body:
                continue
            spam = (r.get("Spam/Ham") or "").strip().lower() == "spam"
            gold = {"is_spam": noul_gold(spam)}
            if not spam:
                gold["is_phishing"] = noul_gold(False)   # ham is real business mail, not a lure
            yield {"subject": (r.get("Subject") or "").strip(), "body": body, "gold": gold, "corpus": "enron"}


def phishing_rows(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            body = (r.get("Email Text") or "").strip()
            kind = (r.get("Email Type") or "").strip()
            if not body or kind not in ("Phishing Email", "Safe Email"):
                continue
            gold = {"is_phishing": noul_gold(kind == "Phishing Email")}
            yield {"subject": "", "body": body, "gold": gold, "corpus": "phishing"}


def dedupe(rows):
    seen, out = set(), []
    for row in rows:
        key = row["body"][:500]
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def split(rows, seed, holdout_frac=0.1):
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    cut = int(len(rows) * (1 - holdout_frac))
    return rows[:cut], rows[cut:]


def to_training_row(row, questions):
    state = email_state(row["subject"], row["body"], None)
    return {"state": state, "questions": questions, "gold": row["gold"], "corpus": row["corpus"]}


def to_holdout_row(row):
    """The same email, labelled the way evaluate.py reads a row: flags only, with the raw sentence."""
    gold = row["gold"]
    return {"subject": row["subject"], "body": row["body"], "sender": "", "corpus": row["corpus"],
            "spam": int(gold["is_spam"]["probabilities"]["true"] > 0.5) if "is_spam" in gold else None,
            "phish": int(gold["is_phishing"]["probabilities"]["true"] > 0.5) if "is_phishing" in gold else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-per-corpus", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    data = os.path.expanduser(args.data_dir)
    out = os.path.expanduser(args.out_dir)
    os.makedirs(out, exist_ok=True)

    corpora = {
        "enron": dedupe(enron_rows(os.path.join(data, "enron_spam_data.csv"))),
        "phishing": dedupe(phishing_rows(os.path.join(data, "Phishing_Email.csv"))),
    }
    questions = email_questions()
    train_rows, holdout_rows = [], []
    for name, rows in corpus_items(corpora):
        train, holdout = split(rows, args.seed)
        print("%s: %d unique, %d train pool, %d holdout" % (name, len(rows), len(train), len(holdout)))
        train_rows += [to_training_row(r, questions) for r in train[:args.train_per_corpus]]
        holdout_rows += [to_holdout_row(r) for r in holdout]

    random.Random(args.seed).shuffle(train_rows)
    with open(os.path.join(out, "public_train.jsonl"), "w", encoding="utf-8") as fh:
        for row in train_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "public_holdout.jsonl"), "w", encoding="utf-8") as fh:
        for row in holdout_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("train rows %d, holdout rows %d -> %s" % (len(train_rows), len(holdout_rows), out))


def corpus_items(corpora):
    for name, rows in corpora.items():
        yield name, rows


if __name__ == "__main__":
    main()
