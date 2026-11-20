# Laya-Finance

**A fast financial-text classifier built on [Laya](https://github.com/NandhaKishorM/laya).** One forward pass (3 to 33 ms per item on an AMD GPU, see [Full benchmarks](#full-benchmarks)) answers multiple-choice questions about financial text: sentiment, topic, central-bank stance, headline direction, finance terminology and trading approach.

[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-hravi%2Flaya--finance-blue)](https://huggingface.co/hravi/laya-finance)
[![Original Laya](https://img.shields.io/badge/Original-NandhaKishorM%2Flaya-lightgrey)](https://github.com/NandhaKishorM/laya)
[![Use](https://img.shields.io/badge/use-educational%20only-orange)](#intended-use)

Weights: https://huggingface.co/hravi/laya-finance

## Highlights

- **Beats ProsusAI/FinBERT on all three sentiment sets**: Financial PhraseBank 0.920 vs 0.893, Twitter 0.895 vs 0.725, FiQA 0.731 vs 0.472.
- **Large gains over original Laya** on every task (table below): topic +38.4 points, trading approach +34.4, terminology +27.3, central-bank stance +20.3, headline direction +19.0.
- **Does more than FinBERT**: FinBERT only handles sentiment; this model also covers five other financial text tasks with one 421M-parameter encoder.
- **About 60 to 600 times faster than a zero-shot 20B LLM** (3 to 33 ms vs about 2 s per item), and ahead of `gpt-oss-20b` on 6 of the 8 benchmark rows.
- **A good first-pass gate**: confident answers are much more accurate than average (for example, headline direction 0.991 on the 80% most confident items), so uncertain items can be passed to a larger model. See [First-pass gate](#using-it-as-a-first-pass-gate).
- **Honest limits**: the zero-shot LLM is still clearly better on FiQA and slightly better on central-bank stance.

## Improvement over original Laya

Held-out test accuracy; the original Laya is the same model used zero-shot, without financial tuning.

| Task | Original Laya | Laya-Finance | Change |
|---|---|---|---|
| Sentiment, PhraseBank | 0.879 | **0.920** | +4.1 |
| Sentiment, Twitter | 0.767 | **0.895** | +12.8 |
| Sentiment, FiQA | 0.528 | **0.731** | +20.3 |
| Topic | 0.477 | **0.861** | +38.4 |
| Headline direction | 0.758 | **0.948** | +19.0 |
| Central-bank stance | 0.458 | **0.661** | +20.3 |
| Finance terminology (domain) | 0.604 | **0.877** | +27.3 |
| Trading approach (style) | 0.442 | **0.786** | +34.4 |

## Full benchmarks

Held-out test accuracy, fp32, one forward pass per item, on the full test set of each task. The zero-shot LLM takes about 2 s per item (1.7 to 2.4 s measured). Each row's source file is listed under [Sources](#sources).

| Task | ProsusAI/finbert | Original Laya (zero-shot) | Laya-Finance | gpt-oss-20b (zero-shot) |
|---|---|---|---|---|
| Sentiment, PhraseBank | 0.893 | 0.879 | **0.920** | 0.819 |
| Sentiment, Twitter | 0.725 | 0.767 | **0.895** | 0.745 |
| Sentiment, FiQA | 0.472 | 0.528 | 0.731 | **0.834** |
| Topic | n/a | 0.477 | **0.861** | 0.677 |
| Headline direction | n/a | 0.758 | **0.948** | 0.792 |
| Central-bank stance | n/a | 0.458 | 0.661 | **0.683** |
| Finance terminology (domain) | n/a | 0.604 | **0.877** | 0.834 |
| Trading approach (style) | n/a | 0.442 | **0.786** | 0.728 |

Notes:

- On PhraseBank both models were trained on the same training split, so that row compares them on the same data; FinBERT's score may partly reflect its own training. The Twitter and FiQA gains are partly from in-domain training data FinBERT never saw.
- The zero-shot LLM is clearly better on FiQA and slightly better on central-bank stance. On terminology and trading-approach macro-F1, Laya-Finance is level with or slightly ahead of it (0.810 vs 0.805 and 0.690 vs 0.670).
- The gpt-oss-20b column was rerun on the full test sets on 2026-11-16. The earlier run used random 250-item samples for five tasks; those results are kept in `finance/results/superseded/`. Re-running the same items moved FiQA from 0.824 to 0.834, so expect about one point of run-to-run variation for the LLM.
- Central-bank stance: the LLM is now ahead by 2.2 points (0.683 vs 0.661) on 496 items. That gap is within noise.
- Timing is a controlled benchmark (see [Sources](#sources)), median of three repeats over 512 items per task (full set for terminology and trading style). On an AMD Radeon 8060S GPU: 3 to 28 ms per item at batch size 32, and 24 to 33 ms at batch size 1. On an AMD Ryzen AI MAX+ 395 CPU, 64 items per task at batch size 32: 41 to 395 ms per item. Re-measure before quoting timing on other hardware.
- Terminology and trading-approach test labels were produced and cross-checked by strong language models, not human annotators, so treat those two rows as indicative.
- FinBERT only supports sentiment, so other cells are marked n/a.

## Using it as a first-pass gate

Laya-Finance is fast enough to screen every item, and its confidence is informative: the more confident it is, the more often it is right. A practical pattern is to accept confident answers and send the rest to a larger model or a person. Measured on the held-out test sets, ranking items from most to least confident:

| Task | All items | Accuracy of the 80% most confident | Share of items answerable at 95% precision |
|---|---|---|---|
| Headline direction | 0.948 | 0.991 | 99.6% |
| Sentiment, PhraseBank | 0.920 | 0.972 | 90% |
| Sentiment, Twitter | 0.895 | 0.957 | 83% |
| Topic | 0.861 | 0.936 | 74% |
| Finance terminology | 0.877 | 0.946 | 79% |
| Trading approach (style) | 0.786 | 0.849 | 48% |
| Sentiment, FiQA | 0.731 | 0.799 | 27% |
| Central-bank stance | 0.661 | 0.717 | 18% |

Read this as a best case: the confidence cut-off that gives each figure was chosen on the test set itself, so in practice choose it on held-out data. Confidence separates good from bad answers well for headline direction, short-text sentiment, topic and terminology, and poorly for FiQA and central-bank stance, where most items should go to a larger model. The combined pipeline with a larger model has not been evaluated here. Nuanced or long inputs (FiQA-style news, long documents, summaries, reasoning) are better handled by a larger model.

## Usage

```python
import laya

agent = laya.load("hravi/laya-finance")  # downloads the weights from Hugging Face
q = {"type": "choice",
     "instructions": "What is the sentiment of this financial text toward the stock price or company outlook?",
     "criteria": {"bearish": "negative for the stock price or outlook",
                  "neutral": "no clear price impact, or purely factual",
                  "bullish": "positive for the stock price or outlook"}}
out = agent.predict_batch(["Operating profit rose to EUR 13.1 mn from EUR 8.7 mn."], {"a": q})
print(out[0]["answers"]["a"]["probabilities"])
```

Keep the `criteria` order fixed: it defines the label index. The model answers any multiple-choice question about a financial text, but it was tuned on the tasks above and works best with those label sets.

## Intended use

**Educational and research purposes only. Not for commercial use.** The restriction comes from the terms of some of the training data. The original Laya code and weights remain under Apache-2.0.

## Limitations

- English text only; short passages (a few hundred words at most).
- Classifies text. It makes no claim to improve trading returns, and nothing here is financial advice.
- The zero-shot LLM is well ahead on FiQA (0.834 vs 0.731) and slightly ahead on central-bank stance (0.683 vs 0.661).
- The rarest trading-approach classes have only 4 to 10 test items each, so their scores are noisy.
- Scores come from one training run, so differences of about one point are within noise.

## Sources

Result files are in [`finance/results/`](finance/results/):

| Row | File |
|---|---|
| Laya-Finance (all tasks) | `laya_mt_j.json` (also the source of the timing range) |
| Original Laya (zero-shot) | `laya_zeroshot.json` |
| ProsusAI/finbert | `finbert.json` (sentiment only) |
| gpt-oss-20b, PhraseBank, Twitter, FiQA, topic, headline, central-bank stance | `llm_gptoss20b_full.json`; run command and log in `logs/` |
| gpt-oss-20b, terminology, trading approach | `llm_gptoss20b_new.json` |
| Superseded gpt-oss-20b run on 250-item samples | `superseded/llm_gptoss20b_250sample.json` (not cited) |

The training and evaluation log for the Laya-Finance run is in `logs/pipeline_j.log`. It records a ROCm (AMD GPU) torch build, which is the timing device. Timing results are in `finance/results/latency/` (`latency_gpu_512.json`, `latency_cpu_64.json`), produced by `bench_latency.py`. The GPU idle state was not recorded because `rocm-smi` is not installed on the benchmark machine.

## Origin

Derived from the original [Laya](https://github.com/NandhaKishorM/laya) (ModernBERT-large decision encoder, Apache-2.0), fine-tuned on public financial datasets and a private corpus. This repository is a fork of Laya; for the original model, documentation and code, see https://github.com/NandhaKishorM/laya. See [LICENSE](LICENSE).
