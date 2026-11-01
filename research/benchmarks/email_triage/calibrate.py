"""Pick the spam and phishing thresholds that fix Outlook's two kinds of error.

Outlook's own filter makes two mistakes the sorter has to undo:

  - false positives: legitimate mail that Outlook sends to Junk (the sorter rescues it)
  - false negatives: spam and phishing that Outlook leaves in the Inbox (the sorter moves it)

Each has its own threshold on the model's probability:

  move_to_junk  p_spam (or p_phish) at or above this moves a message from the Inbox to Junk.
                Set high enough that a legitimate message is almost never moved there.
  rescue_below  p_spam (or p_phish) below this rescues a message from Junk to its folder.
                Set low enough that spam is almost never rescued.

Thresholds are chosen on a labelled dump from `score_public.py --dump`, which must not be the
same rows the model trained on. Each target is a precision or false-rate limit, reported with the
count it was measured on, so a threshold resting on a few emails is visible.

Usage:
  python calibrate.py dump.jsonl --out thresholds.json \
      --max-legit-to-junk 0.02 --max-spam-rescued 0.02
"""

import argparse
import json


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def choose_move(scores, labels, max_false_rate):
    """Lowest threshold t where the share of legitimate rows at or above t is within the limit.

    Legitimate rows are `label == 0`. Among the rows flagged at t, the legitimate share is the false
    rate for moving mail to Junk. The lowest such t catches the most true positives.
    """
    pairs = sorted(zip(scores, labels), reverse=True)
    best = None
    legit_above = 0
    for i, (score, label) in enumerate(pairs):
        if label == 0:
            legit_above += 1
        flagged = i + 1
        false_rate = legit_above / flagged
        if false_rate <= max_false_rate:
            best = (score, flagged, false_rate)
    return best


def choose_rescue(scores, labels, max_spam_rate):
    """Highest threshold t where the share of positive rows below t is within the limit.

    Rescue candidates are rows with score below t. Among them, the spam share is the rate of spam that
    would wrongly leave Junk. The highest such t rescues the most legitimate mail.
    """
    pairs = sorted(zip(scores, labels))
    best = None
    spam_below = 0
    for i, (score, label) in enumerate(pairs):
        if label == 1:
            spam_below += 1
        rescued = i + 1
        spam_rate = spam_below / rescued
        if spam_rate <= max_spam_rate:
            best = (score, rescued, spam_rate)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-legit-to-junk", type=float, default=0.02)
    parser.add_argument("--max-spam-rescued", type=float, default=0.02)
    args = parser.parse_args()

    rows = load(args.dump)
    result = {}

    spam_rows = [r for r in rows if r["spam"] is not None]
    if spam_rows:
        s = [r["p_spam"] for r in spam_rows]
        y = [int(r["spam"]) for r in spam_rows]
        move = choose_move(s, y, args.max_legit_to_junk)
        rescue = choose_rescue(s, y, args.max_spam_rescued)
        result["spam"] = {
            "n": len(spam_rows),
            "move_to_junk_at": move[0] if move else None,
            "move_flagged_n": move[1] if move else 0,
            "move_legit_share": round(move[2], 4) if move else None,
            "rescue_below": rescue[0] if rescue else None,
            "rescue_candidates_n": rescue[1] if rescue else 0,
            "rescue_spam_share": round(rescue[2], 4) if rescue else None,
        }

    phish_rows = [r for r in rows if r["phish"] is not None]
    if phish_rows:
        s = [r["p_phish"] for r in phish_rows]
        y = [int(r["phish"]) for r in phish_rows]
        move = choose_move(s, y, args.max_legit_to_junk)
        rescue = choose_rescue(s, y, args.max_spam_rescued)
        result["phishing"] = {
            "n": len(phish_rows),
            "move_to_junk_at": move[0] if move else None,
            "move_flagged_n": move[1] if move else 0,
            "move_legit_share": round(move[2], 4) if move else None,
            "rescue_below": rescue[0] if rescue else None,
            "rescue_candidates_n": rescue[1] if rescue else 0,
            "rescue_spam_share": round(rescue[2], 4) if rescue else None,
        }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
