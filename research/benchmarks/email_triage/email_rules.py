"""User rules for the email sorter: a starting rule set that a user can edit, and corrections that
turn into new rules or training rows.

Rules run after the model. Each rule has a `when` (every listed condition must hold; a list inside
one condition means any of its values) and a `set` (the answers it overrides). Later rules win.
A rule only touches the fields it names, and every applied rule is reported, so a verdict can
always be traced to the rule that made it.

Safety guard: a rule that clears `is_phishing` for a message the model flags at 0.9 or more is
skipped unless it says `"force": true`. A keyword rule should not be able to wave through a
credential lure by accident.

Rule file shape (see rules/default.json):

  {"name": "...", "rules": [{"id": "...", "when": {...}, "set": {...}, "force": false}]}

  when: sender_domain, sender_exact, subject_contains, body_contains
  set:  category (team name), is_spam, is_phishing, urgency (0-2), needs_reply (bool)

Usage:
  python email_rules.py check rules/default.json        # validate a rule file
  python email_rules.py propose corrections.jsonl       # suggest rules from corrections
"""

import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from generate_synthetic import labels  # noqa: E402

TEAMS = ("billing", "technical", "sales", "security", "hr", "other")
WHEN_KEYS = ("sender_domain", "sender_exact", "subject_contains", "body_contains")
SET_KEYS = ("category", "is_spam", "is_phishing", "urgency", "needs_reply")
PHISH_GUARD = 0.9


def _values(v):
    return [v] if isinstance(v, str) else list(v)


def validate(rule_set, categories=TEAMS):
    """Return a list of problems in a rule set; an empty list means it is valid.
    `categories` is the folder set the rules may file into."""
    problems = []
    seen = set()
    for i, rule in enumerate(rule_set.get("rules", [])):
        where = rule.get("id", "rule %d" % i)
        if not rule.get("id"):
            problems.append("rule %d has no id" % i)
        elif rule["id"] in seen:
            problems.append("duplicate id %r" % rule["id"])
        seen.add(rule.get("id"))
        when, sets = rule.get("when", {}), rule.get("set", {})
        if not when:
            problems.append("%s: empty 'when' would match every email" % where)
        for key in when:
            if key not in WHEN_KEYS:
                problems.append("%s: unknown condition %r" % (where, key))
        for key in sets:
            if key not in SET_KEYS:
                problems.append("%s: unknown field %r" % (where, key))
        if "category" in sets and sets["category"] not in categories:
            problems.append("%s: category %r is not a folder" % (where, sets["category"]))
        if "urgency" in sets and sets["urgency"] not in (0, 1, 2):
            problems.append("%s: urgency must be 0, 1 or 2" % where)
        if not sets:
            problems.append("%s: rule sets nothing" % where)
    return problems


def _sender(state):
    return (state.get("from") or "").strip().lower()


def _matches(when, state):
    sender = _sender(state)
    domain = sender.split("@")[-1] if "@" in sender else ""
    subject = (state.get("subject") or "").lower()
    body = (state.get("body") or "").lower()
    checks = {
        "sender_domain": lambda vals: any(domain == v.lower() or domain.endswith("." + v.lower())
                                          for v in _values(vals)),
        "sender_exact": lambda vals: sender in [v.lower() for v in _values(vals)],
        "subject_contains": lambda vals: any(v.lower() in subject for v in _values(vals)),
        "body_contains": lambda vals: any(v.lower() in body for v in _values(vals)),
    }
    return all(checks[key](vals) for key, vals in when.items())


def _urgency_answer(level):
    return {"score": float(level), "probabilities": {str(i): 1.0 if i == level else 0.0 for i in range(3)}}


def apply_rules(state, answers, rule_set):
    """Apply the rules to a model's answers. Returns `(answers, applied_rule_ids)`.

    `answers` is what `Agent.predict` returns under `result["answers"]`. It is copied, not
    mutated, so the model's own output stays available for comparison.
    """
    out = json.loads(json.dumps(answers))
    applied = []
    for rule in rule_set.get("rules", []):
        if not _matches(rule.get("when", {}), state):
            continue
        sets = rule.get("set", {})
        if ("is_phishing" in sets and sets["is_phishing"] is False and not rule.get("force")
                and out["is_phishing"]["noul"] >= PHISH_GUARD):
            continue
        if "category" in sets:
            out["category"]["choice"] = sets["category"]
            probs = {t: 0.02 for t in out["category"]["probabilities"]}
            probs[sets["category"]] = 0.9
            out["category"]["probabilities"] = probs
        if "is_spam" in sets:
            out["is_spam"]["noul"] = 0.95 if sets["is_spam"] else 0.05
        if "is_phishing" in sets:
            out["is_phishing"]["noul"] = 0.95 if sets["is_phishing"] else 0.05
        if "urgency" in sets:
            out["urgency"] = _urgency_answer(sets["urgency"])
        if "needs_reply" in sets:
            out["needs_reply"]["noul"] = 0.9 if sets["needs_reply"] else 0.1
        applied.append(rule.get("id"))
    return out, applied


def merge(base, user):
    """A user's rule set on top of the starter set. A user rule with the same id replaces the
    starter rule in place; new ids are appended, so later rules still win."""
    by_id = {r["id"]: i for i, r in enumerate(base.get("rules", []))}
    rules = list(base.get("rules", []))
    for rule in user.get("rules", []):
        if rule["id"] in by_id:
            rules[by_id[rule["id"]]] = rule
        else:
            rules.append(rule)
    return {"name": user.get("name", base.get("name")), "rules": rules}


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def propose(corrections, min_support=3, categories=TEAMS):
    """Suggest rules from a user's corrections.

    A correction is `{"from": ..., "subject": ..., "team": ...}` where `team` is the team the
    user says the email belongs to. When one sender domain is corrected to the same team at least
    `min_support` times, that is a candidate rule. Suggestions are returned, not saved: a person
    reviews them before they go in a rule file.
    """
    per_domain = Counter()
    for c in corrections:
        sender = (c.get("from") or "").lower()
        if "@" not in sender or c.get("team") not in categories:
            continue
        per_domain[(sender.split("@")[-1], c["team"])] += 1
    return [
        {"id": "suggested-%s-%s" % (domain.replace(".", "-"), team),
         "when": {"sender_domain": domain}, "set": {"category": team}, "support": n}
        for (domain, team), n in per_domain.most_common() if n >= min_support
    ]


def corrections_to_rows(corrections, questions):
    """Corrections as training rows in the `{state, questions, gold}` schema, so a user's own
    decisions can be folded into fine-tuning. A correction gives the team; the other fields are
    taken from the email's current answers when the caller supplies them, otherwise they are
    neutral (not spam, not phishing, urgency 0, no reply)."""
    rows = []
    for c in corrections:
        if c.get("team") not in TEAMS:
            continue
        intent = {
            "team": c["team"],
            "spam": int(c.get("spam", 0)),
            "phish": int(c.get("phish", 0)),
            "urg": int(c.get("urgency", 0)),
            "reply": int(c.get("needs_reply", 0)),
        }
        state = {"subject": c.get("subject", ""), "body": c.get("body", ""), "from": c.get("from", "")}
        rows.append({"state": state, "questions": questions, "gold": labels(intent)})
    return rows


def _check_cli(path):
    problems = validate(load(path))
    if problems:
        for p in problems:
            print("ERROR", p)
        return 1
    print("ok:", path)
    return 0


def _propose_cli(path):
    with open(path, encoding="utf-8") as fh:
        corrections = [json.loads(line) for line in fh if line.strip()]
    for rule in propose(corrections):
        print(json.dumps(rule))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("check", help="validate a rule file")
    check.add_argument("path")
    prop = sub.add_parser("propose", help="suggest rules from corrections (JSONL)")
    prop.add_argument("path")
    args = parser.parse_args()
    return _check_cli(args.path) if args.cmd == "check" else _propose_cli(args.path)


if __name__ == "__main__":
    sys.exit(main())
