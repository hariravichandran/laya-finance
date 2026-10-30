"""Checks for the sorting policy and folder sets. Run: python test_sorter.py"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import email_rules as R  # noqa: E402
import folders as FO  # noqa: E402
import sorter as S  # noqa: E402

STANDARD = FO.preset("standard-8")
FOLDERS = FO.names(STANDARD)


def answers(category="Other", spam=0.05, phish=0.05):
    cats = {name: 0.02 for name in FOLDERS}
    cats[category] = 0.9
    return {
        "category": {"choice": category, "probabilities": cats},
        "is_spam": {"noul": spam},
        "is_phishing": {"noul": phish},
        "urgency": {"score": 0.0, "probabilities": {"0": 1.0, "1": 0.0, "2": 0.0}},
        "needs_reply": {"noul": 0.1},
    }


def state(subject="", body="", sender=""):
    return {"subject": subject, "body": body, "from": sender}


FAILED = []


def check(name, cond):
    if not cond:
        FAILED.append(name)


T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
DAY = timedelta(days=1)
default_rules = R.load(os.path.join(HERE, "rules", "default.json"))

# --- Inbox: ordinary mail is filed into its folder and is Focused
d = S.decide(answers(category="Billing"), source=S.INBOX, folder_set=STANDARD)
check("inbox mail filed to its folder", d["folder"] == "Billing" and d["move"])
check("inbox mail is focused", d["classification"] == S.FOCUSED)

# --- Inbox: spam and phishing go to Junk, never to the Inbox
d = S.decide(answers(category="Sales", spam=0.9), source=S.INBOX, folder_set=STANDARD)
check("inbox spam goes to junk", d["folder"] == S.JUNK and d["move"])
d = S.decide(answers(category="Security", phish=0.9, spam=0.9), source=S.INBOX, folder_set=STANDARD)
check("inbox phishing goes to junk", d["folder"] == S.JUNK and d["move"] and d["reason"] == "phishing to junk")

# --- Junk: confidently not spam is rescued into its folder
d = S.decide(answers(category="Billing", spam=0.02), source=S.JUNK, folder_set=STANDARD)
check("junk rescued when confident not spam", d["folder"] == "Billing" and d["move"]
      and d["reason"] == "rescued from junk")

# --- Junk: unsure mail stays in Junk, for the person to see
d = S.decide(answers(category="Billing", spam=0.3), source=S.JUNK, folder_set=STANDARD)
check("unsure junk stays in junk", d["folder"] == S.JUNK and d["move"] is False)

# --- Junk: spam and phishing never leave Junk on the model's word
d = S.decide(answers(category="Billing", spam=0.9), source=S.JUNK, folder_set=STANDARD)
check("spam stays in junk", d["folder"] == S.JUNK and d["move"] is False)
d = S.decide(answers(category="Billing", phish=0.9, spam=0.02), source=S.JUNK, folder_set=STANDARD)
check("phishing stays in junk", d["folder"] == S.JUNK and d["move"] is False)

# --- flagged mail stays exactly where it is, in either folder
d = S.decide(answers(category="Sales", spam=0.99), flagged=True, source=S.INBOX, folder_set=STANDARD)
check("flagged inbox mail stays", d["folder"] == S.INBOX and d["move"] is False)
d = S.decide(answers(category="Billing", spam=0.01), flagged=True, source=S.JUNK, folder_set=STANDARD)
check("flagged junk mail stays in junk", d["folder"] == S.JUNK and d["move"] is False)

# --- the Other folder is the Other classification; everything else is Focused
d = S.decide(answers(category="Other"), source=S.INBOX, folder_set=STANDARD)
check("Other folder is Other classification", d["classification"] == S.OTHER)

# --- each email is sorted once: the next pass is not due until 14 days have passed
emails = [{"id": "m1", "state": state(subject="invoice 9"), "folder": S.INBOX, "flagged": False}]
st = {}
first = S.sort_batch(emails, [answers(category="Billing")], default_rules, st, folder_set=STANDARD, now=T0)
check("first pass sorts", first[0]["folder"] == "Billing" and first[0]["pass"] == 1)
again = S.sort_batch(emails, [answers(category="Sales")], default_rules, st, folder_set=STANDARD, now=T0 + DAY)
check("not re-sorted before 14 days", again[0]["skipped"] and again[0]["reason"] == "not due")

# --- after 14 days it is re-sorted, and two passes is the cap
later = S.sort_batch(emails, [answers(category="Other", spam=0.9)], default_rules, st,
                     folder_set=STANDARD, now=T0 + 14 * DAY)
check("re-sorted after 14 days", later[0]["pass"] == 2 and later[0]["folder"] == S.JUNK)
capped = S.sort_batch(emails, [answers(category="Sales")], default_rules, st, folder_set=STANDARD, now=T0 + 28 * DAY)
check("pass cap holds", capped[0]["skipped"] and capped[0]["reason"] == "pass cap reached")

# --- flagged mail is never re-sorted
flag_mail = [{"id": "f1", "state": state(subject="x"), "folder": S.INBOX, "flagged": True}]
fst = {"f1": {"sorted_at": T0.isoformat(), "passes": 1}}
out = S.sort_batch(flag_mail, [answers(spam=0.99)], default_rules, fst, folder_set=STANDARD, now=T0 + 30 * DAY)
check("flagged never re-sorted", out[0]["skipped"] is True)

# --- a user rule runs before the policy; rules match folder names without regard to case
emails = [{"id": "m2", "state": state(subject="hello", sender="ap@acme.example"), "folder": S.INBOX, "flagged": False}]
rule_set = {"rules": [{"id": "acme", "when": {"sender_domain": "acme.example"}, "set": {"category": "billing"}}]}
decided = S.sort_batch(emails, [answers(category="Sales")], rule_set, {}, folder_set=STANDARD, now=T0)
check("user rule applied before policy", decided[0]["folder"] == "Billing" and decided[0]["rules"] == ["acme"])

# --- a rule for a folder that is not in the set is dropped, not silently applied
narrow = FO.preset("minimal-5")
kept = S.rule_set_for(default_rules, narrow)["rules"]
check("rules for missing folders are dropped",
      all(r.get("set", {}).get("category", "Work") in FO.names(narrow) for r in kept))
check("starter rules map to the folder set", any(r["id"] == "billing-keywords" for r in kept))

# --- folder sets: 5 to 15 folders, unique names, must include Other
check("standard-8 valid", FO.validate(STANDARD) == [])
check("minimal-5 valid", FO.validate(FO.preset("minimal-5")) == [])
check("detailed-12 valid", FO.validate(FO.preset("detailed-12")) == [])
check("too few folders flagged", any("need 5 to 15" in p for p in FO.validate(
    {"folders": [{"name": "Other", "description": "x"}]})))
check("too many folders flagged", any("need 5 to 15" in p for p in FO.validate(
    {"folders": [{"name": "F%d" % i, "description": "x"} for i in range(15)] + [{"name": "Other", "description": "x"}]})))
check("missing Other flagged", any("Other" in p for p in FO.validate(
    {"folders": [{"name": "F%d" % i, "description": "x"} for i in range(5)]})))
check("duplicate names flagged", any("unique" in p for p in FO.validate(
    {"folders": [{"name": "A", "description": "x"}] * 4 + [{"name": "Other", "description": "x"}]})))

# --- the gold labels of the hand-labelled set, routed from Inbox, keep most mail in named folders
with open(os.path.join(HERE, "eval_set.jsonl"), encoding="utf-8") as fh:
    eval_rows = [json.loads(line) for line in fh if line.strip()]
routed = [S.decide(answers(category=r["team"].capitalize() if r["team"] != "hr" else "HR",
                           spam=0.95 if r["spam"] else 0.05, phish=0.95 if r["phish"] else 0.05),
                   source=S.INBOX, folder_set=STANDARD) for r in eval_rows]
focused = sum(d["classification"] == S.FOCUSED for d in routed)
print("gold-label routing on eval set: %d/%d focused" % (focused, len(routed)))
check("majority focused on gold labels", focused * 2 > len(routed))

if FAILED:
    print("FAILED:")
    for name in FAILED:
        print("  -", name)
    sys.exit(1)
print("all sorter checks passed")
