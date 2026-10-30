"""Synthetic labelled emails for fine-tuning Laya as an email sorter.

Each row is one email, rendered through `laya.email_state` (the same cleaning inference applies)
and labelled for the five questions in `laya.presets.email_questions()`:

  category     six team queues, gold = 0.9 on the true team and 0.02 on each other
  is_spam      unsolicited or bulk marketing, gold = P(yes)
  is_phishing  credential, payment or fraud lure, gold = P(yes)
  urgency      score 0..2, gold = 0.8 on the true level and 0.1 on each neighbour
  needs_reply  whether the sender expects an answer, gold = P(yes)

`noul` probabilities are P(yes) (`p[1]` is P(true) in `laya/agent.py`), so a row carries
`{"false": 1 - y, "true": y}`.

The labels are the template's intent, not a human's, so the held-out set in `eval_set.jsonl`
is the only measure of whether the model learned anything transferable.

Usage:
  python generate_synthetic.py --n 1200 --seed 7 --out synthetic_emails.jsonl
"""

import argparse
import json
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from laya.email import email_state  # noqa: E402
from laya.presets import email_questions  # noqa: E402

import folders as FO  # noqa: E402

FIRST = ["Dana", "Priya", "Marco", "Yuki", "Omar", "Lena", "Tomasz", "Ana", "Kwame", "Sofia", "Ravi", "Hana"]
LAST = ["Reyes", "Shah", "Rossi", "Tanaka", "Haddad", "Berg", "Nowak", "Silva", "Mensah", "Costa", "Iyer", "Sato"]
COMPANIES = ["Acme", "Northwind", "Globex", "Initech", "Contoso", "Hooli", "Umbrella", "Stark Labs"]
PRODUCTS = ["the Pro plan", "the analytics add-on", "the enterprise seats", "the API tier", "the support bundle"]

# Each intent: team, spam, phishing, urgency level, needs_reply, subjects, bodies.
# Bodies use {name} {order} {amount} {date} {product} {co} and may hold several sentences.
INTENTS = [
    # billing
    dict(team="billing", spam=0, phish=0, urg=2, reply=1,
         subj=["Duplicate charge on invoice {order}", "Charged twice?", "Refund needed"],
         body=["Hi team, we were charged {amount} twice for {product} on {date}. Please reverse the duplicate "
               "and confirm once it is done. Order {order}."]),
    dict(team="billing", spam=0, phish=0, urg=0, reply=1,
         subj=["Copy of invoice {order}", "Invoice request", "Need last quarter's invoices"],
         body=["Could you send me a copy of invoice {order} for our records? Our accountant needs it for the "
               "month-end close."]),
    dict(team="billing", spam=0, phish=0, urg=2, reply=1,
         subj=["Payment declined", "Card failed on renewal"],
         body=["Our card was declined when {product} renewed on {date}. The subscription will lapse on {date} "
               "unless the payment goes through. Can you check the card on file?"]),
    dict(team="billing", spam=0, phish=0, urg=1, reply=1,
         subj=["Where is my refund?", "Refund status {order}"],
         body=["You said the refund for order {order} would take five business days. It has been three weeks. "
               "What is the status?"]),
    # technical
    dict(team="technical", spam=0, phish=0, urg=2, reply=1,
         subj=["Cannot log in since this morning", "Login broken for our whole team"],
         body=["Nobody on our team can log in since about 8am. The page loads and then shows a blank screen. "
               "Is there an outage? We have a client demo at noon."]),
    dict(team="technical", spam=0, phish=0, urg=2, reply=1,
         subj=["API returning 500 errors", "Integration down in production"],
         body=["Since the last deploy every call to /v2/orders returns HTTP 500. Production is affected and our "
               "customers cannot check out. Logs attached."]),
    dict(team="technical", spam=0, phish=0, urg=1, reply=1,
         subj=["Export button does nothing", "Bug: CSV export"],
         body=["When I click Export on the reports page, nothing downloads. It worked last week. "
               "I am on Chrome 129 on Windows."]),
    dict(team="technical", spam=0, phish=0, urg=0, reply=1,
         subj=["How do I set up SSO?", "Question about webhooks"],
         body=["We are planning to roll out SSO next quarter. Is there a guide for configuring SAML with "
               "Okta? No rush."]),
    # sales
    dict(team="sales", spam=0, phish=0, urg=1, reply=1,
         subj=["Demo request", "Could we see a demo next week?"],
         body=["I am the head of operations at {co}. We are evaluating tools for our warehouse team and would like "
               "a 30 minute demo next week."]),
    dict(team="sales", spam=0, phish=0, urg=0, reply=1,
         subj=["Pricing for 50 seats", "Quote request"],
         body=["Could you send pricing for 50 seats of {product}? We are comparing three vendors this month."]),
    dict(team="sales", spam=0, phish=0, urg=1, reply=1,
         subj=["Renewal talks", "Our contract ends in March"],
         body=["Our contract with {co} ends in March. Before we renew, we want to discuss whether we can add "
               "{product} at a discount."]),
    # hr
    dict(team="hr", spam=0, phish=0, urg=1, reply=1,
         subj=["Leave request for {date}", "Time off next week"],
         body=["I would like to request leave from {date} to the following Friday for a family event. "
               "Please let me know if you need anything else."]),
    dict(team="hr", spam=0, phish=0, urg=1, reply=1,
         subj=["Payslip looks wrong", "Overtime not on payroll"],
         body=["My payslip for last month is missing the overtime I logged, about {amount} gross. Could someone "
               "check the timesheet against payroll?"]),
    dict(team="hr", spam=0, phish=0, urg=0, reply=1,
         subj=["Application for the analyst role", "Job application"],
         body=["I am applying for the analyst position posted last week. My CV is attached and I would welcome "
               "a conversation at your convenience."]),
    dict(team="hr", spam=0, phish=0, urg=2, reply=1,
         subj=["Benefits enrollment closes Friday", "Action needed: enrol by {date}"],
         body=["Benefits enrolment closes on {date}. If you have not chosen a plan, you will be default enrolled "
               "in the basic plan. Please confirm your choice."]),
    # security (legitimate reports)
    dict(team="security", spam=0, phish=0, urg=2, reply=1,
         subj=["I think my account was accessed", "Suspicious sign-in on my account"],
         body=["I got a password reset email I did not request, and there is a login from a city I have never been "
               "to. Can you check whether my account is compromised?"]),
    dict(team="security", spam=0, phish=0, urg=1, reply=1,
         subj=["Vulnerability report", "Found a security issue in your login page"],
         body=["While testing the login flow I noticed the session token is returned in the URL. "
               "I can share the steps privately."]),
    # phishing lures (security queue, phishing = yes)
    dict(team="security", spam=1, phish=1, urg=2, reply=0,
         subj=["Urgent: your account has been locked", "Action required: unusual sign-in"],
         body=["We detected unusual activity and locked your account. Verify your details within 24 hours at "
               "http://secure-{co_low}-verify.{tld}/login or the account will be closed."]),
    dict(team="security", spam=1, phish=1, urg=2, reply=0,
         subj=["Wire transfer needed today", "Quick favour before the meeting"],
         body=["I am in a meeting and cannot talk. I need you to buy four gift cards today and send me the codes. "
               "Reply with the numbers as soon as you can."]),
    dict(team="security", spam=1, phish=1, urg=1, reply=0,
         subj=["Updated bank details for {co} invoices", "Change of payment account"],
         body=["Please note our bank details have changed. Send all future payments for {co} to the new account "
               "below and disregard the old one."]),
    dict(team="security", spam=1, phish=1, urg=1, reply=0,
         subj=["Your mailbox storage is full", "Password expires today"],
         body=["Your password expires today. Click here to keep your current password and avoid losing access "
               "to your mailbox."]),
    # spam and bulk marketing
    dict(team="other", spam=1, phish=0, urg=0, reply=0,
         subj=["50% off everything this weekend only!", "Flash sale ends tonight"],
         body=["Huge savings across the store this weekend. Use code SAVE50 at checkout. Unsubscribe any time."]),
    dict(team="other", spam=1, phish=0, urg=0, reply=0,
         subj=["You have been selected for a prize", "Congratulations winner"],
         body=["You have been selected to receive a reward. Claim it before the offer expires."]),
    dict(team="other", spam=1, phish=0, urg=0, reply=0,
         subj=["Acme Weekly: what shipped this month", "Product newsletter {date}"],
         body=["Here is what is new: a faster export, a new dashboard widget and two integration fixes. "
               "You are receiving this because you subscribed."]),
    dict(team="other", spam=1, phish=0, urg=0, reply=0,
         subj=["Boost your leads with our list", "Email list for {co} prospects"],
         body=["We sell verified B2B contact lists at a low price. Reply to get a sample of 10,000 contacts."]),
    # other
    dict(team="other", spam=0, phish=0, urg=0, reply=0,
         subj=["Thank you", "Thanks for the great event"],
         body=["Just wanted to say thank you for the event on {date}. The sessions were excellent."]),
    dict(team="other", spam=0, phish=0, urg=0, reply=0,
         subj=["Out of office", "Automatic reply"],
         body=["I am out of the office until {date} with limited access to email. For urgent matters contact "
               "the front desk."]),
    dict(team="other", spam=0, phish=0, urg=0, reply=1,
         subj=["Feedback on the new dashboard", "Some thoughts"],
         body=["I like the new dashboard overall. The filters are hard to find on a small screen. "
               "Would you be open to a short call?"]),
    dict(team="other", spam=0, phish=0, urg=1, reply=1,
         subj=["Meeting moved to {date}", "Can we reschedule?"],
         body=["Can we move our sync to {date}? I have a conflict on the original slot and want to make sure the "
               "agenda is ready."]),
    # ---- custom folders: each team is a candidate list, so a folder set without the first name
    # files the mail under the next one it has, and otherwise under Other
    dict(team="notifications", spam=0, phish=0, urg=1, reply=0,
         subj=["Build {order} failed", "Your backup did not complete", "Scheduled maintenance tonight"],
         body=["The nightly job {order} failed at 03:12 with exit code 2. No action has been taken automatically."]),
    dict(team="notifications", spam=0, phish=0, urg=0, reply=0,
         subj=["Your weekly activity summary", "Monthly usage report"],
         body=["Here is your usage summary for the past week. No action is required."]),
    dict(team="notifications", spam=0, phish=0, urg=1, reply=0,
         subj=["Password changed successfully", "New sign-in to your account"],
         body=["Your password was changed on {date}. If you did not do this, contact support from the app."]),
    dict(team="personal", spam=0, phish=0, urg=0, reply=1,
         subj=["Dinner on Sunday?", "Photos from the weekend", "Happy birthday!"],
         body=["Are we still on for dinner on Sunday? I can bring the dessert if you want. Let me know."]),
    dict(team="personal", spam=0, phish=0, urg=0, reply=1,
         subj=["Mum's birthday", "Visiting in {date}"],
         body=["We are coming to visit on {date} and staying two nights. Can you check the spare room is free?"]),
    dict(team="travel", spam=0, phish=0, urg=2, reply=1,
         subj=["Flight {order} has been cancelled", "Gate change for your flight"],
         body=["Your flight {order} on {date} has been cancelled. Please confirm the rebooking option you prefer."]),
    dict(team="travel", spam=0, phish=0, urg=0, reply=0,
         subj=["Your itinerary for {date}", "Booking confirmation {order}"],
         body=["Your hotel is confirmed for {date}. Check-in opens at 3pm and the reservation number is {order}."]),
    dict(team="receipts", spam=0, phish=0, urg=0, reply=0,
         subj=["Receipt for your order {order}", "Thanks for your purchase"],
         body=["Thanks for your order. Total paid: {amount}. This email is your receipt for the purchase."]),
    dict(team="receipts", spam=0, phish=0, urg=0, reply=0,
         subj=["Your order has shipped", "Order {order} confirmed"],
         body=["Your order {order} has shipped and should arrive by {date}. Tracking details are below."]),
    dict(team="legal", spam=0, phish=0, urg=2, reply=1,
         subj=["Data access request", "Subject access request from a customer"],
         body=["Under data protection rules we need a response to this access request within 30 days. "
               "Please confirm who will handle it."]),
    dict(team="legal", spam=0, phish=0, urg=1, reply=1,
         subj=["Redline on the NDA", "Comments on the master services agreement"],
         body=["Attached are our comments on the NDA. The main point is the term of confidentiality in clause 4."]),
    dict(team="renewals", spam=0, phish=0, urg=1, reply=1,
         subj=["Subscription renews on {date}", "Your licence expires soon"],
         body=["Your licence for {product} renews on {date}. Reply if you want to change the seat count before then."]),
    dict(team="renewals", spam=0, phish=0, urg=2, reply=1,
         subj=["Contract expires Friday", "Renewal price increase notice"],
         body=["Our contract for {product} expires on {date}. The renewal price increases by 12 percent unless we act."]),
    dict(team="support", spam=0, phish=0, urg=1, reply=1,
         subj=["How do I reset a user's MFA?", "Question about my ticket"],
         body=["A colleague has lost their phone and cannot get past multi-factor login. How do we reset it?"]),
    dict(team="support", spam=0, phish=0, urg=0, reply=1,
         subj=["Thanks for resolving ticket {order}", "Follow-up on my support request"],
         body=["Thanks for the help with ticket {order}. One more question: can the export include archived records?"]),
    dict(team=["work", "technical"], spam=0, phish=0, urg=1, reply=1,
         subj=["Review of the migration plan", "Action items from the planning meeting"],
         body=["Please review the migration plan before Thursday. I have added the open questions at the bottom."]),
    dict(team=["clients", "sales"], spam=0, phish=0, urg=1, reply=1,
         subj=["Follow-up from our call", "Proposal for {co}"],
         body=["Thanks for the call today. I have attached the proposal for {co} with the revised scope."]),
    dict(team=["admin", "hr"], spam=0, phish=0, urg=1, reply=1,
         subj=["Office access card", "Room booking for Thursday"],
         body=["My access card stopped working this morning. Can facilities reissue it before the team offsite?"]),
    dict(team=["billing", "receipts"], spam=0, phish=0, urg=1, reply=1,
         subj=["Please confirm the purchase order", "PO {order} needs approval"],
         body=["Purchase order {order} for {amount} is waiting for your approval before the supplier can ship."]),
]

SIGNOFFS = ["Best regards,\n{first} {last}", "Thanks,\n{first}", "Kind regards,\n{first} {last}\n{co}", "Cheers,\n{first}",
            ""]
QUOTES = ["On Mon, {date}, {first} {last} <{first_low}@{co_low}.example> wrote:\n> earlier message text here",
          "-----Original Message-----\nFrom: {first} {last}\nSent: {date}", ""]
DISCLAIMERS = ["This email is confidential and intended solely for the named addressee.",
               "Sent from my iPhone", ""]
SENDER_TLDS = ["example", "test", "invalid"]


def fill(text, rng):
    first, last = rng.choice(FIRST), rng.choice(LAST)
    co = rng.choice(COMPANIES)
    return text.format(
        name=first, first=first, last=last,
        order="#%d" % rng.randint(1000, 99999),
        amount="$%d.%02d" % (rng.randint(20, 9000), rng.randint(0, 99)),
        date=rng.choice(["Monday", "Friday", "March 14", "12 Oct", "the 3rd", "next Tuesday"]),
        product=rng.choice(PRODUCTS), co=co, co_low=co.lower().replace(" ", ""),
        first_low=first.lower(), tld=rng.choice(["tk", "xyz", "top"]),
    )


def render(intent, rng):
    """One email: subject, body (with optional history, signature and footer), sender."""
    first, last = rng.choice(FIRST), rng.choice(LAST)
    subject = fill(rng.choice(intent["subj"]), rng)
    body = fill(rng.choice(intent["body"]), rng)
    if rng.random() < 0.5:
        body = "Hi team,\n\n" + body
    if rng.random() < 0.3:
        body += "\n\n" + fill(rng.choice(["Also, could you cc my manager on the reply?",
                                          "Let me know if you need more details."]), rng)
    signoff = fill(rng.choice(SIGNOFFS), rng).replace("{first}", first).replace("{last}", last)
    if signoff:
        body += "\n\n" + signoff
    quote = fill(rng.choice(QUOTES), rng)
    if quote:
        body += "\n\n" + quote
    footer = rng.choice(DISCLAIMERS)
    if footer:
        body += "\n\n" + footer
    sender = "%s.%s@%s.%s" % (first.lower(), last.lower(), rng.choice(COMPANIES).lower().replace(" ", ""),
                              rng.choice(SENDER_TLDS))
    return subject, body, sender


DEFAULT_CATEGORIES = ["Billing", "Technical", "Sales", "Security", "HR", "Other"]


# When a folder set lacks a default team, the closest folder it does have takes the mail instead
FALLBACKS = {"technical": ["work"], "sales": ["clients"], "hr": ["admin"]}


def resolve_team(team, categories):
    """The folder an intent files into: the first of its candidate teams that the folder set has,
    matched without regard to case. An intent with no folder in the set falls back to Other, which
    every folder set must have."""
    candidates = list(team) if isinstance(team, (list, tuple)) else [team] + FALLBACKS.get(team, [])
    by_lower = {c.lower(): c for c in categories}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return "Other"


def labels(intent, categories=DEFAULT_CATEGORIES):
    """Gold distributions for the five questions of `email_questions()` over `categories`."""
    # 0.9 on the team and the remaining 0.1 shared out, so the distribution sums to 1 at any size
    residue = 0.1 / (len(categories) - 1)
    cat = {c: residue for c in categories}
    cat[resolve_team(intent["team"], categories)] = 0.9
    level = intent["urg"]
    # Each neighbour on the scale gets 0.1; a neighbour that falls off the scale returns its 0.1 to the
    # true level, so the distribution always sums to 1.
    urg = {str(i): 0.0 for i in range(3)}
    urg[str(level)] = 0.8
    for neighbour in (level - 1, level + 1):
        if 0 <= neighbour <= 2:
            urg[str(neighbour)] = 0.1
        else:
            urg[str(level)] += 0.1
    return {
        "category": {"probabilities": cat},
        "is_spam": {"probabilities": {"false": 1 - (0.95 if intent["spam"] else 0.05),
                                      "true": 0.95 if intent["spam"] else 0.05}},
        "is_phishing": {"probabilities": {"false": 1 - (0.95 if intent["phish"] else 0.05),
                                          "true": 0.95 if intent["phish"] else 0.05}},
        "urgency": {"probabilities": urg},
        "needs_reply": {"probabilities": {"false": 1 - (0.9 if intent["reply"] else 0.1),
                                          "true": 0.9 if intent["reply"] else 0.1}},
    }


# Phishing is the costliest mistake to miss, so its intents are drawn twice as often
PHISH_INTENTS = [i for i in INTENTS if i["phish"]]
SAMPLED_INTENTS = INTENTS + PHISH_INTENTS


def _sampleable(intent, categories):
    """An intent a folder set can teach: it files into one of the set's own folders, or it is
    genuinely other mail. Intents with no folder in the set are left out. Labelling them Other with
    high confidence taught the model Other as a default, and it then sent most mail there."""
    return intent["team"] == "other" or resolve_team(intent["team"], categories) != "Other"


def generate(n, seed, folder_set=None):
    """`n` labelled emails for one folder set. With no folder set, the six default teams.

    The questions carry the folder set's own criteria, so the model is trained on the same choice
    text it will see at inference. An intent with no folder in the set is labelled Other.
    """
    rng = random.Random(seed)
    if folder_set is None:
        categories = DEFAULT_CATEGORIES
        questions = email_questions()
    else:
        categories = [f["name"] for f in folder_set["folders"]]
        questions = email_questions(categories={f["name"]: f["description"] for f in folder_set["folders"]})
    rows = []
    for _ in range(n):
        intent = rng.choice([i for i in SAMPLED_INTENTS if _sampleable(i, categories)])
        subject, body, sender = render(intent, rng)
        rows.append({
            "state": email_state(subject, body, sender),
            "questions": questions,
            "gold": labels(intent, categories),
            "team": resolve_team(intent["team"], categories),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--preset", choices=sorted(FO.PRESETS), default=None,
                        help="folder set to label for; default is the six default teams")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    folder_set = FO.preset(args.preset) if args.preset else None
    rows = generate(args.n, args.seed, folder_set)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote %d rows to %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
