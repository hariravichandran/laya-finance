"""Calibrate the sorter's thresholds on a user's own mail, from an export.

The mail never leaves the machine: the tool reads an mbox file or a folder of .eml files, and every
step writes to a local JSONL file. Four steps:

  import    read an mbox (one per Outlook/Thunderbird folder) or a directory of .eml files. The folder
            name is the mail's current folder, and each message gets a stable id from its Message-ID.
  label     go through a sample and mark each message legitimate, spam, or phishing. Labels are
            written to a separate file keyed by id, so re-importing an export does not lose them.
  score     run a checkpoint over the labelled messages and dump the spam and phishing probabilities.
  calibrate pick the move and rescue thresholds from the labels (see calibrate.py) and write them
            into a folder set's `thresholds` block.

Usage:
  python mail_calibrate.py import ~/exports/outlook.mbox --out mail.jsonl
  python mail_calibrate.py label mail.jsonl --labels labels.jsonl --sample 150
  python mail_calibrate.py score mail.jsonl --labels labels.jsonl --model ~/models/laya-email/v7 --out dump.jsonl
  python mail_calibrate.py calibrate dump.jsonl --folders folders.json --out folders.json
"""

import argparse
import email
import email.policy
import hashlib
import json
import mailbox
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import calibrate as C  # noqa: E402

LABELS = ("legit", "spam", "phish")
MAX_BODY = 4000


def _text(msg):
    """The plain-text body, or the text of an HTML part with the tags left out."""
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        return ""
    body = part.get_content() if part.get_content_maintype() == "text" else ""
    if part.get_content_subtype() == "html":
        import re
        body = re.sub(r"<[^>]+>", " ", body)
    return " ".join(body.split())[:MAX_BODY]


def _message_id(msg, fallback):
    raw = (msg.get("Message-ID") or "").strip() or fallback
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:16]


def _row(msg, folder, fallback):
    return {
        "id": _message_id(msg, fallback),
        "folder": folder,
        "from": str(msg.get("From", "")).strip(),
        "subject": str(msg.get("Subject", "")).strip(),
        "body": _text(msg),
    }


def read_export(path):
    """(folder, message) pairs from an mbox file, or from a directory of .eml files."""
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.lower().endswith(".eml"):
                with open(os.path.join(path, name), "rb") as fh:
                    yield os.path.basename(path), email.message_from_binary_file(fh, policy=email.policy.default)
    else:
        folder = os.path.splitext(os.path.basename(path))[0]
        for msg in mailbox.mbox(path):
            yield folder, email.message_from_bytes(msg.as_bytes(), policy=email.policy.default)


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_import(args):
    rows, seen = [], set()
    for i, (folder, msg) in enumerate(read_export(args.export)):
        row = _row(msg, folder, fallback="%s-%d" % (args.export, i))
        if row["id"] not in seen:
            seen.add(row["id"])
            rows.append(row)
    write_jsonl(args.out, rows)
    print("imported %d messages from %s -> %s" % (len(rows), args.export, args.out))


def cmd_label(args):
    """Ask for a label on a random sample of messages not yet labelled. Keys: l legit, s spam,
    p phishing, k skip, q quit. Only what you enter is written, so you can stop at any time."""
    mail = load_jsonl(args.mail)
    labels = {r["id"]: r["label"] for r in load_jsonl(args.labels)}
    todo = [m for m in mail if m["id"] not in labels]
    random.Random(args.seed).shuffle(todo)
    done = 0
    for m in todo[:args.sample]:
        print("\n[%s] from: %s\nsubject: %s\n%s" % (m["folder"], m["from"], m["subject"], m["body"][:300]))
        key = input("label (l=legit s=spam p=phishing k=skip q=quit): ").strip().lower()
        if key == "q":
            break
        if key in ("l", "s", "p"):
            labels[m["id"]] = {"l": "legit", "s": "spam", "p": "phish"}[key]
            done += 1
            write_jsonl(args.labels, [{"id": k, "label": v} for k, v in labels.items()])
    print("labelled %d; %d labels in %s" % (done, len(labels), args.labels))


def cmd_score(args):
    """Score the labelled messages with a checkpoint and write the probabilities calibrate needs."""
    import laya
    from laya.presets import email_questions
    from laya.email import email_state

    mail = load_jsonl(args.mail)
    labels = {r["id"]: r["label"] for r in load_jsonl(args.labels)}
    labelled = [m for m in mail if m["id"] in labels]
    questions = {k: v for k, v in email_questions().items() if k in ("is_spam", "is_phishing")}
    agent = laya.load(os.path.expanduser(args.model), device=args.device)
    rows = []
    for start in range(0, len(labelled), args.chunk):
        chunk = labelled[start:start + args.chunk]
        states = [email_state(m["subject"], m["body"], m["from"]) for m in chunk]
        for m, out in zip(chunk, agent.predict_batch(states, questions)):
            answers = out["answers"]
            label = labels[m["id"]]
            rows.append({"id": m["id"], "folder": m["folder"], "spam": int(label == "spam"),
                         "phish": int(label == "phish"),
                         "p_spam": answers["is_spam"]["noul"], "p_phish": answers["is_phishing"]["noul"]})
    write_jsonl(args.out, rows)
    print("scored %d labelled messages -> %s" % (len(rows), args.out))


def cmd_calibrate(args):
    """Write the thresholds into the folder set. Spam and phishing have separate thresholds."""
    with open(os.path.expanduser(args.folders), encoding="utf-8") as fh:
        folder_set = json.load(fh)
    rows = load_jsonl(args.dump)
    thresholds = {}
    for flag, key in (("spam", "p_spam"), ("phish", "p_phish")):
        # phishing is not legitimate mail, so it stays out of the spam calibration
        sub = [r for r in rows if not (flag == "spam" and r.get("phish"))]
        if not sub:
            continue
        s = [r[key] for r in sub]
        y = [int(r[flag]) for r in sub]
        move = C.choose_move(s, y, args.max_legit_to_junk)
        rescue = C.choose_rescue(s, y, args.max_spam_rescued)
        if move:
            thresholds["%s_move_to_junk" % ("spam" if flag == "spam" else "phishing")] = round(move[0], 4)
        # only spam sets rescue: phishing never leaves Junk on the model's word alone
        if rescue and flag == "spam":
            folder_set["rescue_below"] = round(rescue[0], 4)
        print("%s: n=%d move_at=%s (flagged %d) rescue_below=%s (candidates %d)" % (
            flag, len(sub), move and round(move[0], 4), move[1] if move else 0,
            rescue and round(rescue[0], 4), rescue[1] if rescue else 0))
    folder_set["thresholds"] = thresholds
    with open(os.path.expanduser(args.out), "w", encoding="utf-8") as fh:
        json.dump(folder_set, fh, indent=2)
    print("wrote thresholds %s -> %s" % (thresholds, args.out))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("import")
    p.add_argument("export")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_import)

    p = sub.add_parser("label")
    p.add_argument("mail")
    p.add_argument("--labels", required=True)
    p.add_argument("--sample", type=int, default=150)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(fn=cmd_label)

    p = sub.add_parser("score")
    p.add_argument("mail")
    p.add_argument("--labels", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--chunk", type=int, default=50)
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("calibrate")
    p.add_argument("dump")
    p.add_argument("--folders", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-legit-to-junk", type=float, default=0.02)
    p.add_argument("--max-spam-rescued", type=float, default=0.02)
    p.set_defaults(fn=cmd_calibrate)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
