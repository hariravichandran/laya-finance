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
