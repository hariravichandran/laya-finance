# Laya-Finance

A fine-tuned version of [Laya](https://github.com/NandhaKishorM/laya) (ModernBERT-large decision encoder) for financial text decisions: sentiment, topic, central-bank stance, headline direction, finance terminology and trading-approach classification.

Weights: https://huggingface.co/hravi/laya-finance

Derived from the original Laya (Apache-2.0). Fine-tuned on public financial datasets and a private corpus. Same license as the original. This is a preliminary checkpoint; an improved one is planned.

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

Keep the `criteria` order fixed: it defines the label index. The model answers any multiple-choice question about a financial text, but it was tuned on the tasks below, and works best with the label sets used there.

## Benchmarks

Held-out test accuracy, fp32, one forward pass per item. Laya-Finance runs at roughly 5 to 32 ms per item on an integrated GPU; the zero-shot LLM takes roughly 2 to 3 s per item.

| Task | ProsusAI/finbert | Laya (original, zero-shot) | Laya-Finance | gpt-oss-20b (zero-shot) |
|---|---|---|---|---|
| Sentiment, PhraseBank | 0.893 | 0.879 | **0.905** | 0.812 |
| Sentiment, Twitter | 0.725 | 0.767 | **0.881** | 0.740 |
| Sentiment, FiQA | 0.472 | 0.528 | 0.674 | **0.824** |
| Topic | n/a | 0.477 | **0.853** | 0.652 |
| Headline direction | n/a | 0.758 | **0.948** | 0.784 |
| Central-bank stance | n/a | 0.458 | 0.653 | **0.680** |
| Finance terminology (domain) | n/a | 0.604 | 0.824 | **0.834** |
| Trading approach (style) | n/a | 0.442 | **0.781** | 0.728 |

Notes:

- Laya-Finance beats FinBERT on all three sentiment sets. PhraseBank is the fair comparison, since FinBERT was trained on it. The Twitter and FiQA gains are partly from in-domain training data FinBERT never saw.
- The zero-shot LLM is better on FiQA and slightly better on central-bank stance and terminology accuracy. On trading-approach macro-F1 it is also ahead (0.670 vs 0.627), because Laya-Finance is weaker on the rarest classes.
- Terminology and trading-approach test labels were produced and cross-checked by strong language models, not human annotators, so treat those two rows as indicative.
- FinBERT only supports sentiment, so other cells are marked n/a.

## Limitations

- English text only; short passages (a few hundred words at most).
- Classifies text. It makes no claim to improve trading returns, and nothing here is financial advice.
- Scores on the tasks above come from one training run, so differences of about one point are within noise.
