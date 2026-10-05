# Laya-Finance

Laya (ModernBERT-large decision encoder) fine-tuned for financial text decisions:
sentiment, topic, central-bank stance, headline direction and finance terminology,
in a single forward pass (about 8-13 ms per item on an iGPU, fp32).

Trained on public financial datasets plus a private corpus. The training recipe,
data and labels are not published.

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

## Results (held-out accuracy)

| Model | PhraseBank | Twitter | FiQA |
|---|---|---|---|
| ProsusAI/finbert | 0.893 | 0.725 | 0.472 |
| Laya zero-shot | 0.879 | 0.767 | 0.528 |
| Laya-Finance (sentiment fine-tune) | **0.909** | 0.899 | 0.736 |
| gpt-oss:20b zero-shot (about 200x slower) | 0.812 | 0.740 | **0.824** |

PhraseBank is the fair comparison with FinBERT. The Twitter and FiQA gains are partly
in-domain training. A zero-shot LLM still wins on FiQA. Multi-task results will be added
with the final checkpoint.

## Limits

- Not a trading signal. No return or Sharpe benefit is claimed or measured.
- Weights are not yet uploaded.
- Source datasets carry their own licenses (Financial PhraseBank is CC BY-NC-SA), which may restrict commercial use of the weights.
