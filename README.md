# Laya-Finance

**A fast financial-text classifier built on [Laya](https://github.com/NandhaKishorM/laya).** One forward pass (5 to 32 ms per item on an integrated GPU) answers multiple-choice questions about financial text: sentiment, topic, central-bank stance, headline direction, finance terminology and trading approach.

[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-hravi%2Flaya--finance-blue)](https://huggingface.co/hravi/laya-finance)
[![Original Laya](https://img.shields.io/badge/Original-NandhaKishorM%2Flaya-lightgrey)](https://github.com/NandhaKishorM/laya)
[![Use](https://img.shields.io/badge/use-educational%20only-orange)](#intended-use)

Weights: https://huggingface.co/hravi/laya-finance

## Highlights

- **Beats ProsusAI/FinBERT on all three sentiment sets**: Financial PhraseBank 0.922 vs 0.893, Twitter 0.884 vs 0.725, FiQA 0.731 vs 0.472.
- **Large gains over original Laya** on every task (table below): topic +38.4 points, trading approach +30.8, headline direction +19.2, central-bank stance +20.9, terminology +24.1.
- **Does more than FinBERT**: FinBERT only handles sentiment; this model also covers five other financial text tasks with one 421M-parameter encoder.
- **About 100x faster than a zero-shot 20B LLM** (milliseconds vs 2 to 3 s per item), and ahead of `gpt-oss-20b` on 6 of the 8 benchmark rows.
- **Honest limits**: the zero-shot LLM is still better on FiQA, slightly better on central-bank stance, and ahead on trading-approach and terminology macro-F1.

## Improvement over original Laya

Held-out test accuracy; the original Laya is the same model used zero-shot, without financial tuning.

| Task | Original Laya | Laya-Finance | Change |
|---|---|---|---|
| Sentiment, PhraseBank | 0.879 | **0.922** | +4.3 |
| Sentiment, Twitter | 0.767 | **0.884** | +11.7 |
| Sentiment, FiQA | 0.528 | **0.731** | +20.3 |
| Topic | 0.477 | **0.861** | +38.4 |
| Headline direction | 0.758 | **0.950** | +19.2 |
| Central-bank stance | 0.458 | **0.667** | +20.9 |
| Finance terminology (domain) | 0.604 | **0.845** | +24.1 |
| Trading approach (style) | 0.442 | **0.750** | +30.8 |

## Full benchmarks

Held-out test accuracy, fp32, one forward pass per item. The zero-shot LLM takes roughly 2 to 3 s per item.

| Task | ProsusAI/finbert | Original Laya (zero-shot) | Laya-Finance | gpt-oss-20b (zero-shot) |
|---|---|---|---|---|
| Sentiment, PhraseBank | 0.893 | 0.879 | **0.922** | 0.812 |
| Sentiment, Twitter | 0.725 | 0.767 | **0.884** | 0.740 |
| Sentiment, FiQA | 0.472 | 0.528 | 0.731 | **0.824** |
| Topic | n/a | 0.477 | **0.861** | 0.652 |
| Headline direction | n/a | 0.758 | **0.950** | 0.784 |
| Central-bank stance | n/a | 0.458 | 0.667 | **0.680** |
| Finance terminology (domain) | n/a | 0.604 | **0.845** | 0.834 |
| Trading approach (style) | n/a | 0.442 | **0.750** | 0.728 |

Notes:

- PhraseBank is the fair FinBERT comparison, since FinBERT was trained on it. The Twitter and FiQA gains are partly from in-domain training data FinBERT never saw.
- On trading-approach macro-F1 the zero-shot LLM is ahead (0.670 vs 0.628), because Laya-Finance is weaker on the rarest classes. On terminology the LLM is also ahead on macro-F1 (0.805 vs 0.765), while Laya-Finance is ahead on accuracy.
- Terminology and trading-approach test labels were produced and cross-checked by strong language models, not human annotators, so treat those two rows as indicative.
- FinBERT only supports sentiment, so other cells are marked n/a.

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
- Scores come from one training run, so differences of about one point are within noise.
- Trading-approach accuracy is 0.750 here versus 0.781 for the first checkpoint, with macro-F1 about the same (0.628 vs 0.627); the rarest classes have only 4 to 10 test items each, so their scores are noisy.

## Origin

Derived from the original [Laya](https://github.com/NandhaKishorM/laya) (ModernBERT-large decision encoder, Apache-2.0), fine-tuned on public financial datasets and a private corpus. This repository is a fork of Laya; for the original model, documentation and code, see https://github.com/NandhaKishorM/laya. See [LICENSE](LICENSE).
