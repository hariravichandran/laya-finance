# Email sorter on Laya

A fine-tuned Laya that sorts mail into a folder set of 5 to 15 folders, flags phishing and spam, and
works out urgency and whether a reply is expected. A rule layer on top handles things the model
should not decide alone, and a policy turns the answers into moves for the Inbox and Junk.

## Layout

| file | what it does |
|---|---|
| `generate_synthetic.py` | synthetic labelled emails for the six default teams or any folder set |
| `build_data.sh` | regenerates every training file from the current generator (not committed) |
| `folders.py` | folder sets: presets (`minimal-5`, `standard-8`, `detailed-12`), setup, validation |
| `email_rules.py` | user rules, the phishing guard, rule suggestions from corrections |
| `sorter.py` | the policy: Inbox and Junk, rescue from Junk, sort once, re-sort after 14 days |
| `feedback.py` | turns view and move events into labels and rule suggestions |
| `train_gpu.py` | fine-tune on the GPU and save the checkpoint with a run summary |
| `evaluate.py` | score a checkpoint on a hand-labelled set, for the six teams or a folder set |
| `eval_set.jsonl` | 30 hand-written emails for the six default teams |
| `eval_folders.jsonl` | 24 hand-written emails for the custom folders |
| `results/` | evaluation output for each checkpoint, as JSON |
| `rules/default.json` | starter rules; `rules/users/example.json` is a per-user override |
| `test_*.py` | checks for the rules, the sorter and the feedback signals |
| `build_public.py` | public Enron spam and phishing-email corpora as flag training data and holdout |
| `score_public.py`, `score_all.sh` | flag scores and probability dumps on the real-email holdout |
| `calibrate.py` | move and rescue thresholds from labelled probabilities, at a limit on error rates |
| `mail_calibrate.py` | calibrate on the user's own mail export: import, label, score, calibrate |
| `combine_eval.py` | category from one checkpoint, flags from another |
| `run_ablation.sh`, `run_chain.sh` | the loss and shuffling ablations, and the chained training runs |

## Running

Tests (CPU venv is enough):

    ../../../.venv/bin/python test_sorter.py
    ../../../.venv/bin/python test_email_rules.py
    ../../../.venv/bin/python test_feedback.py

Data, then training on the GPU:

    bash build_data.sh
    HSA_OVERRIDE_GFX_VERSION=10.3.0 ~/venvs/laya-rocm/bin/python train_gpu.py \
        --base convaiinnovations/laya \
        --data ~/data/laya-email/synthetic_emails.jsonl \
        --data ~/data/laya-email/synthetic_all_folders.jsonl \
        --epochs 3 --out ~/models/laya-email/folders-v1

Evaluation:

    ../../../.venv/bin/python evaluate.py --model ~/models/laya-email/folders-v1
    ../../../.venv/bin/python evaluate.py --model ~/models/laya-email/folders-v1 \
        --eval eval_folders.jsonl --preset detailed-12

### ROCm setup (AMD Radeon integrated GPU, gfx1035)

The project venv has a CPU-only torch. The GPU needs the ROCm wheel, in a separate venv:

    python3 -m venv ~/venvs/laya-rocm
    ~/venvs/laya-rocm/bin/pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
    ~/venvs/laya-rocm/bin/pip install "transformers==5.19.0" "safetensors==0.8.0" \
        "huggingface_hub==1.33.0" "tokenizers==0.23.2" "numpy==2.5.3" "PyYAML==6.0.3" \
        "regex==2026.9.29" "requests==2.34.2" "tqdm==4.70.1" "filelock==3.32.3"

Set `HSA_OVERRIDE_GFX_VERSION=10.3.0` for this GPU. Without it the device is not usable.

## Results

Scores are on hand-labelled sets, so they are small: 30 emails for the six teams, 24 for the
custom folders. A difference of one email is 3 to 4 percentage points.

Six-team set, base versus fine-tuned:

| model | category | spam | phishing recall | phishing FP | urgency exact | needs reply |
|---|---|---|---|---|---|---|
| base (`convaiinnovations/laya`) | 0.633 | 0.833 | 0.667 | 0.167 | 0.367 | 0.433 |
| fine-tuned, 2 epochs, six-team data | 0.700 | 0.967 | 0.500 | 0.000 | 0.700 | 0.933 |
| fine-tuned, mixed folder data, 1 epoch, CPU | 0.633 | 0.867 | 0.000 | 0.000 | 0.667 | 0.933 |

Custom-folder set (detailed-12):

| model | category | spam | urgency exact | needs reply |
|---|---|---|---|---|
| base | 0.375 | 0.875 | 0.417 | 0.458 |
| fine-tuned, 2 epochs, six-team data | 0.417 | 0.958 | 0.708 | 0.792 |
| fine-tuned, mixed folder data, 1 epoch, CPU | 0.417 | 0.958 | 0.708 | 0.833 |

The phishing recall on the custom-folder set is not reported: that set has no phishing positives.

The mixed-data CPU run lost phishing recall entirely. More epochs on the full data, on the GPU, is
the next step; see `results/` for the raw numbers.

## Real-email results (balanced holdout)

Measured on 800 real emails: 400 Enron spam/ham and 400 from the Kaggle phishing-email corpus,
drawn from a shuffled 10% holdout that no model trained on (`build_public.py`, `score_all.sh`).

| model | spam accuracy | phishing precision | phishing recall |
|---|---|---|---|
| base (`convaiinnovations/laya`) | 0.995 | 0.980 | 0.980 |
| folders-v4 (six-team + folder-set data) | 0.963 | 0.989 | 0.603 |
| public-v7 (six-team + public Enron and phishing data) | 0.990 | 0.986 | 0.973 |

Held-out threshold check (`calibrate.py`, fitted on half the dumps, tested on the other half):
- base: spam move at 0.61 (99% of spam caught), phishing move at 0.69 (94% caught)
- public-v7: spam move at 0.07 (100% caught), phishing move at 0.94 (95% caught)
- neither model moved a legitimate email to Junk on the test half. The halves have about 40
  positives each, so the recall figures are noisy.

**Correction.** Earlier results in this README and in the commit history reported phishing recall
of 0.0 for the base model and for v7, and concluded the base model never flags real phishing. Those
numbers came from `public_holdout_1500.jsonl`, which was not shuffled and therefore held only Enron
mail, so the phishing figures had no phishing positives. They are invalid. The balanced table above
replaces them. The six-team and custom-folder eval tables above are unaffected.

The base model is already strong on real phishing, so the fine-tunes have not yet beaten it on the
flags. The practical route is the base model with thresholds calibrated on the user's own mail.

## Calibrating on the user's own mail

`mail_calibrate.py` runs four steps, all local:

    python mail_calibrate.py import ~/exports/outlook.mbox --out mail.jsonl
    python mail_calibrate.py label mail.jsonl --labels labels.jsonl --sample 150
    python mail_calibrate.py score mail.jsonl --labels labels.jsonl --model <checkpoint> --out dump.jsonl
    python mail_calibrate.py calibrate dump.jsonl --folders folders.json --out folders.json

The calibrated thresholds go into the folder set's `thresholds` block, and `sorter.py` reads them.
Legitimate mail is rescued from Junk below the spam rescue threshold, and spam moves out of the Inbox
above the spam move threshold. The limits are `--max-legit-to-junk` and `--max-spam-rescued`, both 2%
by default.

## Status

- Working: folder sets, Inbox and Junk policy, rules, feedback signals, sort-once and 14-day re-sort.
- Not connected: Outlook. Nothing calls Microsoft Graph. Focused/Other and flags map to
  `inferenceClassification` and `flag.flagStatus`, unverified.
- Open: the fine-tuned models have not beaten the base model on the real-email flags. The next
  real gain should come from labelled mail from the user, not more synthetic data.
