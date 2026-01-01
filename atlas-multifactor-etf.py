"""
Rules-Based Multi-Factor Equity Portfolio (with VIX Regime Sensitivity)

Monthly rebalanced, equal-weighted portfolio constructed from a fixed universe
of large-cap U.S. equities using valuation, profitability, growth, and risk factors
(+ momentum used as a tie-breaker).

Performance is evaluated relative to SPY using gross and net-of-transaction-cost returns.
Additionally, results are split into LOW vs HIGH volatility regimes using VIX.

This implementation is intended as a methodological prototype and not as a live trading strategy.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional
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
TC_BPS_PER_100_TURNOVER = 10.0  # 10 bps per 100% turnover

# VIX regime settings
VIX_TICKER = "^VIX"
VIX_SMOOTH_DAYS = 63      # ~3 months trading days
LOW_Q = 0.33              # bottom tercile
HIGH_Q = 0.67             # top tercile

# Outputs
SAVE_HOLDINGS_CHANGES_XLSX = "holdings_changes.xlsx"
SAVE_EQUITY_CURVES_CSV = "equity_curves.csv"
SAVE_REGIME_SUMMARY_CSV = "regime_summary.csv"

AUTO_OPEN_HOLDINGS_XLSX = True
AUTO_OPEN_EQUITY_CSV = True

# ==========================
# SYSTEM OPEN HELPER
# ==========================

def open_file(path: str):
    """Open a file with the system's default application."""
    try:
        if sys.platform.startswith("darwin"):      # macOS
            subprocess.call(("open", path))
        elif os.name == "nt":                      # Windows
            os.startfile(path)                     # type: ignore[attr-defined]
        elif os.name == "posix":                   # Linux
            subprocess.call(("xdg-open", path))
    except Exception as e:
        print(f"Could not open {path}: {e}")

# ==========================
# HELPERS
# ==========================

def safe_float(x) -> float:
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return s * 0
    return (s - s.mean(skipna=True)) / sd


def winsorize(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    ql, qh = s.quantile([lo, hi])
    return s.clip(ql, qh)


def download_prices(tickers, start, end) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    px = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df
    return px.dropna(how="all")


def download_close(ticker: str, start: str, end) -> pd.Series:
    df = yf.download([ticker], start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        s = df["Close"][ticker]
    else:
        s = df["Close"] if "Close" in df.columns else df.squeeze()
    return s.dropna()


# ==========================
# FACTORS
# ==========================

def compute_price_factors(prices: pd.DataFrame):
    rets = prices.pct_change()
    momentum = prices / prices.shift(MOMENTUM_LOOKBACK_DAYS) - 1
    vol = rets.rolling(VOL_LOOKBACK_DAYS).std() * np.sqrt(252)
    return momentum, vol


def fetch_fundamentals(tickers: List[str]) -> pd.DataFrame:
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


def make_scores(momentum_row: pd.Series, vol_row: pd.Series, fundamentals: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=fundamentals.index)
    df["value_pe"] = fundamentals["trailingPE"]
    df["profit_roe"] = fundamentals["ROE"]
    df["growth_rev"] = fundamentals["revenueGrowth"]
    df["risk_vol"] = vol_row
    df["risk_de"] = fundamentals["debtToEquity"]
    df["mom_12m"] = momentum_row

    for c in df.columns:
        df[c] = winsorize(df[c])

    z = pd.DataFrame(index=df.index)
    z["value_pe"] = -zscore(df["value_pe"])
    z["risk_vol"] = -zscore(df["risk_vol"])
    z["risk_de"] = -zscore(df["risk_de"])
    z["profit_roe"] = zscore(df["profit_roe"])
    z["growth_rev"] = zscore(df["growth_rev"])
    z["mom_12m"] = zscore(df["mom_12m"])

    # Weighted score (note: momentum is used as a tie-breaker, not in WEIGHTS by default)
    df["score"] = sum(z[k].fillna(0) * w for k, w in WEIGHTS.items())
    df["mom_z"] = z["mom_12m"]
    return df


# ==========================
# BACKTEST
# ==========================

def turnover(prev: Optional[List[str]], curr: List[str]) -> float:
    if prev is None:
        return np.nan
    return 1 - len(set(prev) & set(curr)) / len(curr)


def apply_costs(r: pd.Series, turn: float) -> pd.Series:
    if np.isnan(turn):
        return r
    cost = (TC_BPS_PER_100_TURNOVER / 10000) * turn
    r = r.copy()
    r.iloc[0] -= cost
    return r


def backtest(prices: pd.DataFrame, fundamentals: pd.DataFrame, bench_px: pd.Series):
    momentum, vol = compute_price_factors(prices)

    rebal_dates = prices.resample(REBALANCE).last().index
    rebal_dates = rebal_dates[rebal_dates.isin(prices.index)]

    gross, net, turns = [], [], []
    prev: Optional[List[str]] = None

    holdings_log_rows = []

    for d0, d1 in zip(rebal_dates[:-1], rebal_dates[1:]):
        scores = make_scores(momentum.loc[d0], vol.loc[d0], fundamentals)
        picks = scores.sort_values(["score", "mom_z"], ascending=False).head(TOP_N).index.tolist()

        if prev is None:
            added = sorted(picks)
            removed = []
            stayed = []
        else:
            added = sorted(set(picks) - set(prev))
            removed = sorted(set(prev) - set(picks))
            stayed = sorted(set(picks) & set(prev))

        turn = turnover(prev, picks)
        turns.append(turn)

        holdings_log_rows.append({
            "rebalance_date": pd.Timestamp(d0).date().isoformat(),
            "num_held": len(picks),
            "num_added": len(added),
            "num_removed": len(removed),
            "turnover_fraction": (None if np.isnan(turn) else float(turn)),
            "added": ", ".join(added),
            "removed": ", ".join(removed),
            "stayed": ", ".join(stayed),
            "held": ", ".join(sorted(picks)),
        })

        prev = picks

        period = prices.loc[d0:d1, picks].pct_change().dropna(how="all")
        if period.empty:
            continue

        g = period.mean(axis=1)
        n = apply_costs(g, turn)

        gross.append(g)
        net.append(n)

    gross = pd.concat(gross)
    net = pd.concat(net)

    bench = bench_px.pct_change().reindex(gross.index).dropna()
    gross = gross.reindex(bench.index)
    net = net.reindex(bench.index)

    holdings_changes = pd.DataFrame(holdings_log_rows)
    avg_turn = float(np.nanmean(turns)) if len(turns) else np.nan

    return gross, net, bench, avg_turn, holdings_changes


# ==========================
# METRICS
# ==========================

def ann_return(r: pd.Series) -> float:
    return (1 + r).prod() ** (252 / len(r)) - 1 if len(r) > 0 else np.nan

def ann_vol(r: pd.Series) -> float:
    return r.std() * np.sqrt(252) if len(r) > 1 else np.nan

def sharpe(r: pd.Series) -> float:
    sd = r.std()
    if len(r) < 2 or sd == 0 or np.isnan(sd):
        return np.nan
    return r.mean() / sd * np.sqrt(252)

def info_ratio(p: pd.Series, b: pd.Series) -> float:
    diff = (p - b).dropna()
    sd = diff.std()
    if len(diff) < 2 or sd == 0 or np.isnan(sd):
        return np.nan
    return diff.mean() / sd * np.sqrt(252)


# ==========================
# VIX REGIMES
# ==========================

def compute_vix_regimes(vix_close: pd.Series) -> pd.Series:
    """
    Returns a series of regime labels indexed by date: 'low_vol', 'mid', 'high_vol'
    based on terciles of a smoothed VIX series.
    """
    vix_smooth = vix_close.rolling(VIX_SMOOTH_DAYS).mean().dropna()
    lo_thr = float(vix_smooth.quantile(LOW_Q))
    hi_thr = float(vix_smooth.quantile(HIGH_Q))

    regime = pd.Series("mid", index=vix_smooth.index)
    regime[vix_smooth <= lo_thr] = "low_vol"
    regime[vix_smooth >= hi_thr] = "high_vol"
    return regime


def regime_stats(name: str, p: pd.Series, b: pd.Series) -> dict:
    p = p.dropna()
    b = b.reindex(p.index).dropna()
    p = p.reindex(b.index)

    return {
        "regime": name,
        "days": int(len(p)),
        "ann_return": float(ann_return(p)),
        "ann_vol": float(ann_vol(p)),
        "sharpe": float(sharpe(p)),
        "info_ratio_vs_spy": float(info_ratio(p, b)),
    }


# ==========================
# PLOT
# ==========================

def plot_equity_curves(gross: pd.Series, net: pd.Series, bench: pd.Series):
    plt.figure(figsize=(10, 6))

    plt.plot((1 + gross).cumprod(), label="Portfolio (Gross)", linewidth=2.2)
    plt.plot((1 + net).cumprod(), label="Portfolio (Net of Costs)", linestyle="--", linewidth=2.2)
    plt.plot((1 + bench).cumprod(), label="Benchmark (SPY)", linewidth=1.8)

    plt.title("Equity Curve: Portfolio vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==========================
# MAIN
# ==========================

def main():
    # Prices + fundamentals + benchmark
    prices = download_prices(TICKERS, START_DATE, END_DATE)
    fundamentals = fetch_fundamentals(TICKERS)
    bench_px = download_prices([BENCHMARK], START_DATE, END_DATE)[BENCHMARK]

    gross, net, bench, avg_turn, holdings_changes = backtest(prices, fundamentals, bench_px)

    # Save + open holdings changes
    holdings_changes.to_excel(SAVE_HOLDINGS_CHANGES_XLSX, index=False)
    print(f"Saved holdings changes to: {SAVE_HOLDINGS_CHANGES_XLSX}")
    if AUTO_OPEN_HOLDINGS_XLSX:
        open_file(SAVE_HOLDINGS_CHANGES_XLSX)

    # Save + open equity curves CSV
    equity = pd.DataFrame({
        "Portfolio_Gross": (1 + gross).cumprod(),
        "Portfolio_Net": (1 + net).cumprod(),
        "Benchmark_SPY": (1 + bench).cumprod()
    })
    equity.to_csv(SAVE_EQUITY_CURVES_CSV, index=True)
    print(f"Saved equity curves to: {SAVE_EQUITY_CURVES_CSV}")
    if AUTO_OPEN_EQUITY_CSV:
        open_file(SAVE_EQUITY_CURVES_CSV)

    # ==========================
    # VIX regime sensitivity
    # ==========================
    vix_close = download_close(VIX_TICKER, START_DATE, END_DATE)
    regimes = compute_vix_regimes(vix_close)

    # Align regimes to your return dates
    df = pd.DataFrame({
        "net": net,
        "bench": bench,
        "regime": regimes
    }).dropna()

    low = df[df["regime"] == "low_vol"]
    high = df[df["regime"] == "high_vol"]

    summary_rows = []
    summary_rows.append(regime_stats("ALL", df["net"], df["bench"]))
    summary_rows.append(regime_stats("LOW_VIX", low["net"], low["bench"]))
    summary_rows.append(regime_stats("HIGH_VIX", high["net"], high["bench"]))

    regime_summary = pd.DataFrame(summary_rows)

    print("\n=== VIX Regime Sensitivity (Net Portfolio vs SPY) ===")
    print(regime_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    regime_summary.to_csv(SAVE_REGIME_SUMMARY_CSV, index=False)
    print(f"\nSaved regime summary to: {SAVE_REGIME_SUMMARY_CSV}")

    # Plot (auto opens via plt.show)
    plot_equity_curves(gross, net, bench)

    print(f"\nAvg Turnover (fraction changed per rebalance): {avg_turn:.3f}" if np.isfinite(avg_turn) else "\nAvg Turnover: nan")


if __name__ == "__main__":
    main()
