"""
Rules-Based Multi-Factor Equity Portfolio

Monthly rebalanced, equal-weighted portfolio constructed from a fixed universe
of large-cap U.S. equities using momentum, valuation, profitability, growth,
and risk factors. Performance is evaluated relative to SPY using both gross
and net-of-transaction-cost returns.

This implementation is intended as a methodological prototype and not as a
live trading strategy.
"""

from __future__ import annotations

import warnings
from typing import Dict, List
import os
import sys
import subprocess

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ==========================
# CONFIG
# ==========================

TICKERS: List[str] = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","BRK-B","LLY","AVGO",
    "JPM","TSLA","V","XOM","UNH","MA","COST","HD","PG","JNJ",
    "ORCL","MRK","ABBV","CVX","NFLX","KO","CRM","BAC","WMT","PEP",
    "AMD","ADBE","TMO","MCD","QCOM","NKE","LIN","DIS","CSCO","ABT",
    "ACN","VZ","TXN","DHR","INTC","NEE","PM","UPS","MS","AMGN"
]

START_DATE = "2016-01-01"
END_DATE = None

TOP_N = 30
REBALANCE = "M"

MOMENTUM_LOOKBACK_DAYS = 252
VOL_LOOKBACK_DAYS = 252

WEIGHTS: Dict[str, float] = {
    "value_pe": 0.20,
    "profit_roe": 0.20,
    "growth_rev": 0.20,
    "risk_vol": 0.20,
    "risk_de": 0.20,
}

BENCHMARK = "SPY"
TC_BPS_PER_100_TURNOVER = 10.0

SAVE_HOLDINGS_CHANGES_XLSX = "holdings_changes.xlsx"
SAVE_EQUITY_CURVES_CSV = "equity_curves.csv"

# ==========================
# SYSTEM OPEN HELPER
# ==========================

def open_file(path: str):
    try:
        if sys.platform.startswith("darwin"):
            subprocess.call(("open", path))
        elif os.name == "nt":
            os.startfile(path)
        elif os.name == "posix":
            subprocess.call(("xdg-open", path))
    except Exception as e:
        print(f"Could not open {path}: {e}")

# ==========================
# HELPERS
# ==========================

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def zscore(s: pd.Series):
    return (s - s.mean()) / s.std() if s.std() != 0 else s * 0

def winsorize(s, lo=0.01, hi=0.99):
    return s.clip(s.quantile(lo), s.quantile(hi))

def download_prices(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    return df["Close"]

# ==========================
# FACTORS
# ==========================

def compute_price_factors(prices):
    rets = prices.pct_change()
    momentum = prices / prices.shift(MOMENTUM_LOOKBACK_DAYS) - 1
    vol = rets.rolling(VOL_LOOKBACK_DAYS).std() * np.sqrt(252)
    return momentum, vol

def fetch_fundamentals(tickers):
    rows = []
    for t in tickers:
        info = yf.Ticker(t).info or {}
        rows.append({
            "ticker": t,
            "trailingPE": safe_float(info.get("trailingPE")),
            "ROE": safe_float(info.get("returnOnEquity")),
            "revenueGrowth": safe_float(info.get("revenueGrowth")),
            "debtToEquity": safe_float(info.get("debtToEquity")),
        })
    return pd.DataFrame(rows).set_index("ticker")

def make_scores(momentum_row, vol_row, fundamentals):
    df = fundamentals.copy()
    df["risk_vol"] = vol_row
    df["mom_12m"] = momentum_row

    for c in df.columns:
        df[c] = winsorize(df[c])

    score = (
        -zscore(df["trailingPE"]) * WEIGHTS["value_pe"] +
         zscore(df["ROE"]) * WEIGHTS["profit_roe"] +
         zscore(df["revenueGrowth"]) * WEIGHTS["growth_rev"] +
        -zscore(df["risk_vol"]) * WEIGHTS["risk_vol"] +
        -zscore(df["debtToEquity"]) * WEIGHTS["risk_de"]
    )

    df["score"] = score
    return df

# ==========================
# BACKTEST
# ==========================

def backtest(prices, fundamentals, bench_px):
    momentum, vol = compute_price_factors(prices)
    rebal_dates = prices.resample(REBALANCE).last().index

    gross, net = [], []
    holdings_log = []
    prev = None

    for d0, d1 in zip(rebal_dates[:-1], rebal_dates[1:]):
        scores = make_scores(momentum.loc[d0], vol.loc[d0], fundamentals)
        picks = scores.sort_values("score", ascending=False).head(TOP_N).index.tolist()

        added = picks if prev is None else sorted(set(picks) - set(prev))
        removed = [] if prev is None else sorted(set(prev) - set(picks))
        stayed = [] if prev is None else sorted(set(prev) & set(picks))

        holdings_log.append({
            "rebalance_date": d0.date(),
            "added": ", ".join(added),
            "removed": ", ".join(removed),
            "stayed": ", ".join(stayed),
            "held": ", ".join(sorted(picks)),
        })

        prev = picks

        period = prices.loc[d0:d1, picks].pct_change().dropna()
        if period.empty:
            continue

        g = period.mean(axis=1)
        n = g.copy()
        if removed:
            n.iloc[0] -= (TC_BPS_PER_100_TURNOVER / 10000) * (len(removed) / TOP_N)

        gross.append(g)
        net.append(n)

    gross = pd.concat(gross)
    net = pd.concat(net)
    bench = bench_px.pct_change().reindex(gross.index)

    return gross, net, bench, pd.DataFrame(holdings_log)

# ==========================
# PLOT
# ==========================

def plot_equity_curves(gross, net, bench):
    plt.figure(figsize=(10, 6))
    plt.plot((1 + gross).cumprod(), label="Portfolio Gross", linewidth=2)
    plt.plot((1 + net).cumprod(), label="Portfolio Net", linestyle="--", linewidth=2)
    plt.plot((1 + bench).cumprod(), label="SPY", linewidth=1.8)
    plt.title("Equity Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==========================
# MAIN
# ==========================

def main():
    prices = download_prices(TICKERS, START_DATE, END_DATE)
    fundamentals = fetch_fundamentals(TICKERS)
    bench_px = download_prices([BENCHMARK], START_DATE, END_DATE)[BENCHMARK]

    gross, net, bench, holdings = backtest(prices, fundamentals, bench_px)

    holdings.to_excel(SAVE_HOLDINGS_CHANGES_XLSX, index=False)
    open_file(SAVE_HOLDINGS_CHANGES_XLSX)

    equity = pd.DataFrame({
        "Portfolio_Gross": (1 + gross).cumprod(),
        "Portfolio_Net": (1 + net).cumprod(),
        "Benchmark": (1 + bench).cumprod()
    })
    equity.to_csv(SAVE_EQUITY_CURVES_CSV)
    open_file(SAVE_EQUITY_CURVES_CSV)

    plot_equity_curves(gross, net, bench)

if __name__ == "__main__":
    main()
