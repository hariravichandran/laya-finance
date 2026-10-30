"""The folders the sorter files mail into: 5 to 15 of them, chosen on first run and editable later.

A folder set is a JSON file:

  {"name": "my-folders",
   "folders": [{"name": "Billing", "description": "invoices, payments, refunds"}, ...],
   "rescue_below": 0.2}

- `folders` is the list the model chooses between, and its names are the model's choice labels.
  Between 5 and 15, unique, and must include `Other`, the place for mail that fits nowhere else.
- `description` is the criteria text the model sees for that choice. Keep it short and concrete;
  it is the main lever a person has on what lands where.
- `rescue_below` is the spam probability under which a message in Junk is taken out of it. Mail
  the model is less sure about than this stays in Junk. Phishing never leaves Junk on the model's
  word alone.

Presets give a starting point for each size, so the first run is one choice, not a blank page.

Usage:
  python folders.py presets                              # list the presets
  python folders.py setup --preset standard-8 --out f.json
  python folders.py setup --interactive --out f.json     # type your own names
  python folders.py check f.json
"""

import argparse
import json
import sys

MIN_FOLDERS, MAX_FOLDERS = 5, 15
FALLBACK = "Other"

PRESETS = {
    "minimal-5": [
        ("Billing", "invoices, payments, refunds"),
        ("Work", "colleagues, projects, meetings, technical problems"),
        ("Clients", "customers, prospects, pricing and demos"),
        ("Admin", "HR, leave, payroll, benefits, legal"),
        (FALLBACK, "none of the above"),
    ],
    "standard-8": [
        ("Billing", "invoices, payments, refunds"),
        ("Technical", "bugs, outages, integrations, support requests"),
        ("Sales", "pricing, demos, new purchases, renewals"),
        ("Security", "account access, verification, suspicious sign-ins"),
        ("HR", "hiring, leave, payroll, benefits"),
        ("Notifications", "automatic alerts and system messages"),
        ("Personal", "friends, family, personal accounts"),
        (FALLBACK, "none of the above"),
    ],
    "detailed-12": [
        ("Billing", "invoices, payments, refunds"),
        ("Technical", "bugs, outages, integrations"),
        ("Support", "customer questions and requests for help"),
        ("Sales", "pricing, demos, new purchases"),
        ("Renewals", "contracts, subscriptions and renewal dates"),
        ("Security", "account access, verification, phishing"),
        ("HR", "hiring, leave, payroll, benefits"),
        ("Legal", "contracts, compliance, data requests"),
        ("Travel", "bookings, itineraries, check-ins"),
        ("Receipts", "order confirmations and purchase receipts"),
        ("Notifications", "automatic alerts and system messages"),
        (FALLBACK, "none of the above"),
    ],
}


def _folder_list(pairs):
    return [{"name": name, "description": desc} for name, desc in pairs]


def validate(config):
    """Problems with a folder set; an empty list means it is usable."""
    problems = []
    folders = config.get("folders", [])
    names = [f.get("name", "") for f in folders]
    if not MIN_FOLDERS <= len(folders) <= MAX_FOLDERS:
        problems.append("need %d to %d folders, got %d" % (MIN_FOLDERS, MAX_FOLDERS, len(folders)))
    if len(set(names)) != len(names):
        problems.append("folder names must be unique")
    if any(not n.strip() for n in names):
        problems.append("folder names cannot be blank")
    if FALLBACK not in names:
        problems.append("the set must include %r for mail that fits nowhere else" % FALLBACK)
    rescue = config.get("rescue_below", 0.2)
    if not isinstance(rescue, (int, float)) or not 0 <= rescue <= 0.5:
        problems.append("rescue_below must be between 0 and 0.5")
    return problems


def names(config):
    return [f["name"] for f in config["folders"]]


def questions_for(config, email_questions):
    """The preset's questions, with the category choice set to this folder set."""
    return email_questions(categories={f["name"]: f["description"] for f in config["folders"]})


def preset(name):
    if name not in PRESETS:
        raise ValueError("unknown preset %r; choose one of %s" % (name, ", ".join(PRESETS)))
    return {"name": name, "folders": _folder_list(PRESETS[name]), "rescue_below": 0.2}


def interactive(out):
    """Ask for folder names one at a time. Blank line finishes; the Other folder is added last."""
    print("Name between %d and %d folders. A blank line finishes." % (MIN_FOLDERS, MAX_FOLDERS))
    folders = []
    while len(folders) < MAX_FOLDERS:
        name = input("folder %d name: " % (len(folders) + 1)).strip()
        if not name:
            break
        desc = input("  what goes in %s? (short): " % name).strip() or name
        folders.append((name, desc))
    if FALLBACK not in [n for n, _ in folders]:
        folders.append((FALLBACK, "none of the above"))
    config = {"name": "custom", "folders": _folder_list(folders), "rescue_below": 0.2}
    problems = validate(config)
    if problems:
        for p in problems:
            print("not saved:", p)
        return 1
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    print("saved %d folders to %s" % (len(folders), out))
    return 0


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("presets", help="list the presets")
    setup = sub.add_parser("setup", help="write a folder set")
    setup.add_argument("--preset", choices=sorted(PRESETS))
    setup.add_argument("--interactive", action="store_true")
    setup.add_argument("--out", required=True)
    check = sub.add_parser("check", help="validate a folder set")
    check.add_argument("path")
    args = parser.parse_args()

    if args.cmd == "presets":
        for name, pairs in PRESETS.items():
            print("%s (%d): %s" % (name, len(pairs), ", ".join(n for n, _ in pairs)))
        return 0
    if args.cmd == "check":
        problems = validate(load(args.path))
        for p in problems:
            print("ERROR", p)
        return 1 if problems else 0
    if args.interactive:
        return interactive(args.out)
    if not args.preset:
        parser.error("choose --preset or --interactive")
    config = preset(args.preset)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    print("wrote %s (%d folders)" % (args.out, len(config["folders"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
