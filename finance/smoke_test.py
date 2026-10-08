#!/usr/bin/env python3
"""Smoke test for hravi/laya-finance: loads the model and checks one example per task.

  python finance/smoke_test.py [path_or_hf_id]     # default: hravi/laya-finance
"""
import sys

import laya

TASKS = {
    "sentiment": ({
        "type": "choice",
        "instructions": "What is the sentiment of this financial text toward the stock price or company outlook?",
        "criteria": {"bearish": "negative for the stock price or outlook",
                     "neutral": "no clear price impact, or purely factual",
                     "bullish": "positive for the stock price or outlook"}},
        "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn.", "bullish"),
    "fomc": ({
        "type": "choice",
        "instructions": "What is the monetary policy stance of this central bank text?",
        "criteria": {"dovish": "suggests easing or accommodative policy",
                     "hawkish": "suggests tightening or restrictive policy",
                     "neutral": "no clear policy direction"}},
        "The Committee judged that further increases in the policy rate would be needed to bring inflation back to target.",
        "hawkish"),
    "headline_price": ({
        "type": "noul", "instructions": "Does the news headline talk about price?"},
        "gold futures end the week with 2% gain", "yes"),
    "term_domain": ({
        "type": "choice",
        "instructions": "Which area of finance does this passage's terminology belong to?",
        "criteria": {"equity_valuation": "valuing stocks, equity research, fundamentals, multiples",
                     "fixed_income": "bonds, yields, duration, credit spreads, term structure",
                     "derivatives": "options, futures, swaps, pricing and greeks",
                     "portfolio_theory": "portfolio construction, CAPM, factors, allocation, performance measurement",
                     "risk_management": "VaR, drawdown, hedging, position sizing, tail risk",
                     "corporate_finance": "capital structure, M&A, budgeting, dividends, financing",
                     "accounting_reporting": "financial statements, GAAP/IFRS, earnings quality, audit",
                     "macro_monetary": "central banks, inflation, rates, GDP, economic cycles",
                     "market_structure": "trading venues, order types, liquidity, execution, microstructure",
                     "technical_analysis": "chart patterns, indicators, trend and momentum rules, backtested trading systems",
                     "quant_methods": "statistics, econometrics, machine learning, time series, optimization",
                     "behavioral_finance": "biases, sentiment, investor psychology",
                     "banking_credit": "lending, bank regulation, credit risk, insurance",
                     "other": "not clearly finance terminology"}},
        "Duration measures a bond's price sensitivity to a change in yield, and convexity refines that estimate.",
        "fixed_income"),
}


def main():
    agent = laya.load(sys.argv[1] if len(sys.argv) > 1 else "hravi/laya-finance")
    bad = 0
    for name, (q, text, want) in TASKS.items():
        ans = agent.predict_batch([text], {"a": q})[0]["answers"]["a"]
        if q["type"] == "choice":
            got, p = ans["choice"], ans["probabilities"]
        else:
            p = ans["noul"]
            got = "yes" if p < 0.5 else "no"  # noul is the probability the question is NOT satisfied
        ok = got == want
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: got={got} want={want} probs={p}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
