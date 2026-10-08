# Using laya-finance (instructions for AI agents)

`hravi/laya-finance` is a fine-tuned Laya encoder that classifies short financial texts. It runs locally in about 5-30 ms per item, with no API calls. Use it instead of an LLM when the task is one of the label sets below. Educational and research use only; not for commercial use; not financial advice.

## Quick start

```python
import laya
agent = laya.load("hravi/laya-finance")   # first call downloads ~0.8 GB from Hugging Face
out = agent.predict_batch([text1, text2], {"a": question})   # texts × questions, batch freely
ans = out[0]["answers"]["a"]
```

Verify an install with `python finance/smoke_test.py` (about 1 minute on CPU; exit code 0 means all four checks passed). Set `CUDA_VISIBLE_DEVICES=""` to force CPU.

## Question formats

- `choice`: `{"type": "choice", "instructions": str, "criteria": {label: description, ...}}`. Result: `ans["choice"]` and `ans["probabilities"]` (dict label -> prob). **Keep the criteria dict exactly as below, same labels and order; the order defines the label index the model was trained on.**
- `noul` (yes/no): `{"type": "noul", "instructions": str}`. Result: `ans["noul"]` is the probability the answer is **no** (low = yes). Treat < 0.5 as yes.

## Supported tasks (use `tasks.json` verbatim)

| Task | Instructions | Labels (in order) |
|---|---|---|
| sentiment | What is the sentiment of this financial text toward the stock price or company outlook? | bearish, neutral, bullish |
| fomc | What is the monetary policy stance of this central bank text? | dovish, hawkish, neutral |
| headline | Does the news headline talk about price? (`noul`) | n/a |
| topic | What type of financial news event is this? | analyst_update, central_banks, company_news, treasuries_debt, dividend, earnings, energy_oil, financials, currencies, general_opinion, metals, ipo, legal_regulation, mna, macro, markets, politics, personnel, stock_commentary, stock_movement |
| term_domain | Which area of finance does this passage's terminology belong to? | equity_valuation, fixed_income, derivatives, portfolio_theory, risk_management, corporate_finance, accounting_reporting, macro_monetary, market_structure, technical_analysis, quant_methods, behavioral_finance, banking_credit, other |
| trading_style | Which kind of trading approach does this passage mainly describe? | trend_following, mean_reversion, breakout_momentum, options_volatility, spreads_arbitrage, seasonality_cycles, risk_position_sizing, system_validation, indicator_construction, commentary_education |

All six question definitions with the exact `criteria` strings are in [`tasks.json`](tasks.json). Load and use them directly:

```python
import json
Q = json.load(open("finance/tasks.json"))
out = agent.predict_batch(texts, {"style": Q["trading_style"]})
```

## Rules of thumb

- English only; a few hundred words at most per item (longer text is truncated).
- Use probabilities, not just the argmax. Below ~0.6 top probability, treat the label as uncertain and route to a human or a larger model.
- Weakest areas (see benchmarks in the root README): fomc stance (~0.65 accuracy), the rare trading styles (spreads_arbitrage, mean_reversion, seasonality_cycles), and FiQA-style sentiment. Do not rely on it unsupervised there.
- It labels text. It does not predict prices or returns; do not present its output as a trading signal or advice.
- Batch many texts in one `predict_batch` call; the model loads once per process, so keep the `agent` object alive.

## Testing and reporting

1. Run `python finance/smoke_test.py`.
2. For your own use case, hand-label 50-100 items and compare; report accuracy and the confusion on the classes you care about.
3. If results are poor, report the task, a few failing texts, and the probabilities, rather than changing the criteria wording (the model is sensitive to it).
