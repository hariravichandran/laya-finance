"""Decide where each email goes, for the Inbox and for Junk.

The policy, in order:

  1. Flagged mail stays where it is, on every pass. The person flagged it on purpose.
  2. A message is sorted on its first pass. After that it is re-sorted only once it is at least
     `resort_after` old (14 days by default), and only up to `max_passes` in total (2 by default).
     A message that was relevant and has gone stale can fall to Other on the second pass,
     without the sorter churning the same mail every day.
  3. Mail in Junk is rescued into its folder only when the model is confident it is not spam
     (spam probability below the folder set's `rescue_below`). Phishing never leaves Junk on the
     model's word. Anything the model is unsure of stays in Junk, where the person can see it.
  4. Mail in the Inbox is filed into its folder. Spam goes to Junk, and phishing goes to Junk too,
     so neither reaches the Inbox.

Outlook's Focused and Other are a separate axis: a folder named `Other` is filed as Other, and
everything else as Focused, because most mail a person needs is in Focused.

A user's rules (see `email_rules.py`) run before the policy, so a person's own rule can move
mail the model would have sent elsewhere.

The state file records, per message id, when it was last sorted and how many passes it has had.
Decisions are plain functions of the answers, so they can be checked offline. Moving real mail
needs a mail API: Outlook's Focused/Other is the `inferenceClassification` property and the
flag is `flag.flagStatus` in Microsoft Graph. That call is not made here.

Usage:
  python sorter.py decide --folders folders.json --state state.json < batch.jsonl
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import email_rules as R  # noqa: E402
import folders as FO  # noqa: E402

INBOX = "Inbox"
JUNK = "Junk"
FOCUSED = "focused"
OTHER = "other"
RESORT_AFTER = timedelta(days=14)
MAX_PASSES = 2


def decide(answers, flagged=False, source=INBOX, folder_set=None):
    """Destination for one email from its answers.

    `source` is where the email is now, `Inbox` or `Junk`. `folder_set` is the folder set's config
    (see `folders.py`); when it is `None` the default `standard-8` set is used.
    """
    folder_set = folder_set or FO.preset("standard-8")
    rescue_below = folder_set.get("rescue_below", 0.2)
    category = answers["category"]["choice"]
    p_phish = answers["is_phishing"]["noul"]
    p_spam = answers["is_spam"]["noul"]

    if flagged:
        return {"folder": source, "move": False, "classification": FOCUSED, "reason": "flagged"}

    if source == JUNK:
        if p_phish >= 0.5:
            return {"folder": JUNK, "move": False, "classification": OTHER, "reason": "phishing stays in junk"}
        if p_spam >= 0.5:
            return {"folder": JUNK, "move": False, "classification": OTHER, "reason": "spam stays in junk"}
        if p_spam < rescue_below:
            return {"folder": category, "move": True, "classification": _classification(category),
                    "reason": "rescued from junk"}
        return {"folder": JUNK, "move": False, "classification": OTHER, "reason": "unsure, left in junk"}

    if p_phish >= 0.5:
        return {"folder": JUNK, "move": True, "classification": OTHER, "reason": "phishing to junk"}
    if p_spam >= 0.5:
        return {"folder": JUNK, "move": True, "classification": OTHER, "reason": "spam to junk"}
    return {"folder": category, "move": True, "classification": _classification(category), "reason": "default"}


def _classification(category):
    return OTHER if category == FO.FALLBACK else FOCUSED


def _due(record, now, resort_after, max_passes):
    """Whether a message with this state record should be sorted now."""
    if record is None:
        return True
    if record["passes"] >= max_passes:
        return False
    return now - datetime.fromisoformat(record["sorted_at"]) >= resort_after


def sort_batch(emails, model_answers, rule_set, state, folder_set=None, now=None,
               resort_after=RESORT_AFTER, max_passes=MAX_PASSES):
    """Sort a batch. `emails` is a list of dicts with `id`, `state`, `folder` (where the email is
    now: `Inbox` or `Junk`) and `flagged`. `model_answers` is the aligned list of answers from the
    model, and `state` maps message id to `{"sorted_at": iso, "passes": n}` and is updated in place.

    Messages that are not due come back as `skipped`, with the reason. Each decision carries the
    `pass` number it was made on, so a re-sort is visible in the output.
    """
    now = now or datetime.now(timezone.utc)
    folder_set = folder_set or FO.preset("standard-8")
    rules = rule_set_for(rule_set, folder_set)
    decisions = []
    for email, answers in zip(emails, model_answers):
        record = state.get(email["id"])
        if email.get("flagged"):
            decisions.append({"id": email["id"], "skipped": True, "reason": "flagged"})
            continue
        if not _due(record, now, resort_after, max_passes):
            reason = "not due" if record["passes"] < max_passes else "pass cap reached"
            decisions.append({"id": email["id"], "skipped": True, "reason": reason})
            continue
        answers, applied = R.apply_rules(email["state"], answers, rules)
        decision = decide(answers, flagged=email.get("flagged", False),
                          source=email.get("folder", INBOX), folder_set=folder_set)
        passes = (record["passes"] if record else 0) + 1
        state[email["id"]] = {"sorted_at": now.isoformat(), "passes": passes}
        decision.update(id=email["id"], rules=applied, **{"pass": passes})
        decisions.append(decision)
    return decisions


def rule_set_for(rule_set, folder_set):
    """The rules as they apply to this folder set. A rule's folder is matched to the set's folder
    name without regard to case (`billing` is `Billing`), and rewritten to the exact name. A rule
    naming a folder that is not in the set is dropped, and the drop is reported on stderr, so it is
    never a silent miss."""
    by_lower = {n.lower(): n for n in FO.names(folder_set)}
    keep = []
    for rule in rule_set.get("rules", []):
        category = rule.get("set", {}).get("category")
        if category is None:
            keep.append(rule)
        elif category.lower() in by_lower:
            keep.append(dict(rule, set=dict(rule["set"], category=by_lower[category.lower()])))
        else:
            print("skipped rule %s: no folder named %r in this folder set" % (rule.get("id"), category),
                  file=sys.stderr)
    return {"name": rule_set.get("name"), "rules": keep}


def load_state(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["messages"]


def save_state(path, state):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"messages": state}, fh, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    dec = sub.add_parser("decide", help="print a decision per email for a JSONL batch on stdin")
    dec.add_argument("--folders", required=True, help="folder set JSON (see folders.py)")
    dec.add_argument("--state", required=True, help="state file (created if missing)")
    args = parser.parse_args()
    emails = [json.loads(line) for line in sys.stdin if line.strip()]
    state = load_state(args.state)
    folder_set = FO.load(args.folders)
    problems = FO.validate(folder_set)
    if problems:
        parser.error("folder set: " + "; ".join(problems))
    rule_set = R.load(os.path.join(HERE, "rules", "default.json"))
    model_answers = [e["answers"] for e in emails]
    for d in sort_batch(emails, model_answers, rule_set, state, folder_set=folder_set):
        print(json.dumps(d))
    save_state(args.state, state)


if __name__ == "__main__":
    main()
