"""Turn what the person does with their mail into labels for the sorter.

The mail integration writes one event per message the person touched. An event is a JSON line:

  {"id": "m1", "from": "x@acme.example", "subject": "...", "body": "...",
   "team": "billing",           # optional: the team the person put it in, when they moved it there
   "sorted_to": "other",        # where the sorter put it
   "folder_before": "junk",     # where it was when the person last saw it
   "folder_after": "inbox",     # where it is now
   "view_seconds": 42,          # time the message was open, 0 if never opened
   "replied": false, "flagged": false}

Signals, strongest first:

  - moved out of Junk to Focused/Inbox: not spam (strong; a correction, weight 1.0)
  - moved into Junk from Focused/Inbox: spam (strong; a correction, weight 1.0)
  - replied to or flagged: the person needed it (strong; not spam, and needs a reply if replied)

Weak signals are logged for review and never trained on:

  - read in Junk for a while, then left there: the person looked and kept it junk, or they read
    it and did not get round to moving it. Ambiguous, so it is a note, not a label.
  - opened in Focused for under 2 seconds and never acted on: a glance is not a judgement.

Dwell time alone is noisy: a long open can be a phone call, and a short one can be a skim of a
message already known. So dwell time only decides whether a Junk read is worth a note.

Usage:
  python feedback.py events.jsonl --labels labels.jsonl --rules-out suggested_rules.jsonl
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import email_rules as R  # noqa: E402

JUNK = "junk"
FOCUSED_FOLDERS = ("inbox", "focused")
LONG_READ_SECONDS = 10
GLANCE_SECONDS = 2
MIN_SUPPORT = 3


def _base(event):
    return {"from": event.get("from", ""), "subject": event.get("subject", ""), "body": event.get("body", "")}


def derive(event):
    """A label for one event, or `None` when it carries no strong signal.

    A label is the email plus `spam`, `needs_reply` and `weight`, and `team` when the event says
    where the email belongs. `corrections_to_rows` uses `team` to build a training row.
    """
    before, after = event.get("folder_before", ""), event.get("folder_after", "")
    team = {"team": event["team"]} if event.get("team") else {}
    if before == JUNK and after in FOCUSED_FOLDERS:
        return dict(_base(event), spam=0, needs_reply=int(bool(event.get("replied"))), weight=1.0,
                    why="rescued from junk", **team)
    if before in FOCUSED_FOLDERS and after == JUNK:
        return dict(_base(event), spam=1, needs_reply=0, weight=1.0, why="moved to junk", **team)
    if event.get("replied") or event.get("flagged"):
        return dict(_base(event), spam=0, needs_reply=int(bool(event.get("replied"))), weight=1.0,
                    why="replied or flagged", **team)
    return None


def weak_note(event):
    """A note for an event with no strong signal, for review. Never a label."""
    after = event.get("folder_after", "")
    seconds = event.get("view_seconds", 0)
    if after == JUNK and seconds >= LONG_READ_SECONDS:
        return {"id": event.get("id"), "from": event.get("from", ""), "why": "read in junk, not rescued"}
    if after in FOCUSED_FOLDERS and 0 < seconds < GLANCE_SECONDS and not event.get("replied") \
            and not event.get("flagged"):
        return {"id": event.get("id"), "from": event.get("from", ""), "why": "glanced, not acted on"}
    return None


def summarise(events):
    """Labels, weak notes and rule suggestions from a list of events.

    A rule is suggested when the person rescues mail from one sender domain into the same team
    at least `MIN_SUPPORT` times. The team has to come from the event, so a rescue without a team
    gives no rule; guessing one would put the sorter's mistake into a rule.
    """
    labels, notes, rescues = [], [], []
    for event in events:
        label = derive(event)
        if label is not None:
            labels.append(label)
            if label["why"] == "rescued from junk" and label.get("team"):
                rescues.append({"from": label["from"], "team": label["team"]})
            continue    # a strong signal is not also a weak note
        note = weak_note(event)
        if note is not None:
            notes.append(note)
    return {"labels": labels, "weak": notes, "suggestions": R.propose(rescues, min_support=MIN_SUPPORT)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("events", help="JSONL of events from the mail integration")
    parser.add_argument("--labels", required=True, help="where to write the labels (JSONL)")
    parser.add_argument("--rules-out", required=True, help="where to write suggested rules (JSONL)")
    args = parser.parse_args()
    with open(args.events, encoding="utf-8") as fh:
        events = [json.loads(line) for line in fh if line.strip()]
    result = summarise(events)
    with open(args.labels, "w", encoding="utf-8") as fh:
        for label in result["labels"]:
            fh.write(json.dumps(label, ensure_ascii=False) + "\n")
    with open(args.rules_out, "w", encoding="utf-8") as fh:
        for rule in result["suggestions"]:
            fh.write(json.dumps(rule) + "\n")
    print("events %d, labels %d, weak notes %d, suggested rules %d"
          % (len(events), len(result["labels"]), len(result["weak"]), len(result["suggestions"])))


if __name__ == "__main__":
    main()
