# Laya-Finance

A fine-tuned version of [Laya](https://github.com/NandhaKishorM/laya) (ModernBERT-large decision encoder) for financial text decisions such as sentiment, topic and stance.

Derived from the original Laya (Apache-2.0). Weights: coming soon.

## Usage

```python
import laya

agent = laya.load("path/to/laya-finance")
q = {"type": "choice",
     "instructions": "What is the sentiment of this financial text toward the stock price or company outlook?",
     "criteria": {"bearish": "negative for the stock price or outlook",
                  "neutral": "no clear price impact, or purely factual",
                  "bullish": "positive for the stock price or outlook"}}
out = agent.predict_batch(["Operating profit rose to EUR 13.1 mn from EUR 8.7 mn."], {"a": q})
print(out[0]["answers"]["a"]["probabilities"])
```

Keep the `criteria` order fixed: it defines the label index.

## Benchmarks

Held-out accuracy on public sentiment test sets (fp32, about 8-13 ms per item on an iGPU).

| Model | PhraseBank | Twitter | FiQA |
|---|---|---|---|
| ProsusAI/finbert | 0.893 | 0.725 | 0.472 |
| Laya (original, zero-shot) | 0.879 | 0.767 | 0.528 |
| Laya-Finance | **0.909** | 0.899 | 0.736 |
| 20B open LLM, zero-shot (about 200x slower) | 0.812 | 0.740 | **0.824** |

PhraseBank is the fair comparison with FinBERT, which was trained on it. The Twitter and FiQA gains are partly from in-domain training data that FinBERT never saw, and the zero-shot LLM still wins on FiQA. Results on further tasks will be added with the final checkpoint.
