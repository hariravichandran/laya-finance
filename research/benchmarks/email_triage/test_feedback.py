"""Checks for the feedback signals. Run: python test_feedback.py"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import feedback as F  # noqa: E402

FAILED = []


def check(name, cond):
    if not cond:
        FAILED.append(name)


def ev(sender="a@acme.example", **kw):
    base = {"id": "m", "from": sender, "subject": "s", "body": "b", "view_seconds": 0,
            "replied": False, "flagged": False}
    base.update(kw)
    return base


# a rescue from Junk is a strong not-spam label
label = F.derive(ev(folder_before="junk", folder_after="inbox", team="billing"))
check("rescue is not spam", label is not None and label["spam"] == 0 and label["weight"] == 1.0)
check("rescue keeps the team", label.get("team") == "billing")

# a move into Junk is a strong spam label
label = F.derive(ev(folder_before="inbox", folder_after="junk"))
check("move to junk is spam", label is not None and label["spam"] == 1)

# replying or flagging is a strong not-spam label, and a reply sets needs_reply
label = F.derive(ev(folder_before="inbox", folder_after="inbox", replied=True))
check("reply is not spam and needs a reply", label["spam"] == 0 and label["needs_reply"] == 1)
check("flag is a not-spam label", F.derive(ev(folder_before="inbox", folder_after="inbox", flagged=True))["spam"] == 0)

# a long read in Junk that is left there is a note, not a label
e = ev(folder_before="junk", folder_after="junk", view_seconds=45)
check("long junk read is not a label", F.derive(e) is None)
check("long junk read is a note", F.weak_note(e) is not None)

# a glance in Focused is a note, not a label
e = ev(folder_before="inbox", folder_after="inbox", view_seconds=1)
check("glance is not a label", F.derive(e) is None)
check("glance is a note", F.weak_note(e)["why"] == "glanced, not acted on")

# a plain event with no signal gives nothing
check("no signal, no label", F.derive(ev(folder_before="inbox", folder_after="inbox")) is None)

# rule suggestions need repeated rescues from one domain into the same team, with the team given
events = [ev(sender="ap@acme.example", folder_before="junk", folder_after="inbox", team="billing")] * 3
events += [ev(sender="x@other.example", folder_before="junk", folder_after="inbox", team="sales")]
events += [ev(sender="ap@acme.example", folder_before="junk", folder_after="inbox")]  # no team: ignored
result = F.summarise(events)
check("suggests a rule for a recurring rescue", len(result["suggestions"]) == 1)
check("suggestion targets the domain and team",
      result["suggestions"][0]["when"]["sender_domain"] == "acme.example"
      and result["suggestions"][0]["set"]["category"] == "billing")
check("one-off rescue gives no rule", all(s["when"]["sender_domain"] != "other.example"
                                          for s in result["suggestions"]))

# labels and notes are reported separately
mixed = [ev(folder_before="junk", folder_after="junk", view_seconds=60), ev(folder_before="junk", folder_after="inbox")]
result = F.summarise(mixed)
check("labels exclude long junk reads", len(result["labels"]) == 1)
check("notes hold long junk reads", len(result["weak"]) == 1)

if FAILED:
    print("FAILED:")
    for name in FAILED:
        print("  -", name)
    sys.exit(1)
print("all feedback checks passed")
