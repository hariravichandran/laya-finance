"""Checks for the email rule layer. Run: python test_email_rules.py"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import email_rules as R  # noqa: E402
from generate_synthetic import labels  # noqa: E402


def answers(category="other", spam=0.05, phish=0.05, urgency=0, reply=0.1):
    return {
        "category": {"choice": category, "probabilities": {t: (0.9 if t == category else 0.02) for t in R.TEAMS}},
        "is_spam": {"noul": spam},
        "is_phishing": {"noul": phish},
        "urgency": {"score": float(urgency), "probabilities": {"0": 0.8, "1": 0.1, "2": 0.1}},
        "needs_reply": {"noul": reply},
    }


def state(subject="", body="", sender=""):
    return {"subject": subject, "body": body, "from": sender}


FAILED = []


def check(name, cond):
    if not cond:
        FAILED.append(name)


default = R.load(os.path.join(HERE, "rules", "default.json"))
example = R.load(os.path.join(HERE, "rules", "users", "example.json"))

# starter and example rule files are valid
check("default validates", R.validate(default) == [])
check("example validates", R.validate(example) == [])

# keyword rule sets the team and leaves the other fields alone
out, applied = R.apply_rules(state(subject="Duplicate charge on invoice 42"), answers(), default)
check("billing keyword sets team", out["category"]["choice"] == "billing")
check("billing keyword reported", "billing-keywords" in applied)
check("billing keyword leaves phishing alone", out["is_phishing"]["noul"] == 0.05)

# the model's own answers are not mutated
model = answers()
R.apply_rules(state(subject="invoice"), model, default)
check("input answers untouched", model["category"]["choice"] == "other")

# phishing guard: a rule cannot clear phishing the model is confident about...
unforced = {"rules": [{"id": "t", "when": {"sender_domain": "vendor.example"}, "set": {"is_phishing": False}}]}
out, applied = R.apply_rules(state(sender="x@vendor.example"), answers(phish=0.95), unforced)
check("guard blocks clearing confident phishing", out["is_phishing"]["noul"] == 0.95 and applied == [])

# ...unless it is forced (the example user file does this for its trusted vendor)
out, applied = R.apply_rules(state(sender="x@vendor.example"), answers(phish=0.95), example)
check("forced clear goes through", out["is_phishing"]["noul"] == 0.05 and "our-vendor-is-trusted" in applied)

# a lure rule can always raise phishing, even when the model is unsure
out, applied = R.apply_rules(state(body="Please buy gift cards today"), answers(phish=0.1), default)
check("lure rule raises phishing", out["is_phishing"]["noul"] == 0.95)

# domain matching: subdomains match, lookalike domains do not
rs = {"rules": [{"id": "d", "when": {"sender_domain": "bank.example"}, "set": {"category": "billing"}}]}
check("subdomain matches", R.apply_rules(state(sender="a@mail.bank.example"), answers(), rs)[1] == ["d"])
check("lookalike does not match", R.apply_rules(state(sender="a@bank.example.evil.test"), answers(), rs)[1] == [])

# later rules win
rs = {"rules": [{"id": "a", "when": {"body_contains": "x"}, "set": {"category": "billing"}},
                {"id": "b", "when": {"body_contains": "x"}, "set": {"category": "hr"}}]}
check("later rule wins", R.apply_rules(state(body="x"), answers(), rs)[0]["category"]["choice"] == "hr")

# urgency is a one-hot answer
out, _ = R.apply_rules(state(subject="cannot log in"), answers(), default)
check("urgency set to 2", out["urgency"]["score"] == 2.0 and out["urgency"]["probabilities"]["2"] == 1.0)

# validation catches real mistakes
bad = {"rules": [{"id": "x", "when": {"sender": "a"}, "set": {"category": "sales-ish", "urgency": 5}},
                 {"id": "x", "when": {}, "set": {}}]}
problems = R.validate(bad)
check("unknown condition flagged", any("unknown condition" in p for p in problems))
check("bad folder flagged", any("not a folder" in p for p in problems))
check("bad urgency flagged", any("urgency must be" in p for p in problems))
check("duplicate id flagged", any("duplicate id" in p for p in problems))
check("empty when flagged", any("empty 'when'" in p for p in problems))

# merge: user rule with the same id replaces the starter rule; new ids append
merged = R.merge(default, example)
ids = [r["id"] for r in merged["rules"]]
check("merge replaces same id", ids.count("billing-keywords") == 1)
check("merge appends new ids", "our-recruiter" in ids and ids[-1] == "our-recruiter")
check("merged set validates", R.validate(merged) == [])

# propose: a domain corrected to the same team enough times becomes a suggestion
corr = [{"from": "ap@acme.example", "team": "billing"}] * 3 + [{"from": "x@other.example", "team": "sales"}]
sugg = R.propose(corr, min_support=3)
check("propose finds recurring domain", len(sugg) == 1 and sugg[0]["when"]["sender_domain"] == "acme.example")
check("propose ignores one-offs", all(s["when"]["sender_domain"] != "other.example" for s in sugg))

# corrections become training rows in the fine-tuning schema, with the team as the gold label
rows = R.corrections_to_rows([{"from": "a@b.example", "subject": "s", "body": "b", "team": "hr"}],
                             {"category": {"type": "choice", "instructions": "i", "criteria": {}}})
check("correction row has state/questions/gold", set(rows[0]) == {"state", "questions", "gold"})
check("correction gold names the team", rows[0]["gold"]["category"]["probabilities"]["HR"] == 0.9)
check("labels helper matches generator", labels({"team": "hr", "spam": 0, "phish": 0, "urg": 1, "reply": 1})
      ["urgency"]["probabilities"]["1"] == 0.8)

if FAILED:
    print("FAILED:")
    for name in FAILED:
        print("  -", name)
    sys.exit(1)
print("all email rule checks passed")
