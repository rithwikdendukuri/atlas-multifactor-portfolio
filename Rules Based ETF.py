from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

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
RISK_FREE_RATE_ANNUAL = 0.0
TC_BPS_PER_100_TURNOVER = 10.0  # 10 bps per 100% turnover

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


def download_prices(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    px = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df
    return px.dropna(how="all")


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

    df["score"] = sum(z[k].fillna(0) * w for k, w in WEIGHTS.items())
    df["mom_z"] = z["mom_12m"]
    return df


# ==========================
# BACKTEST
# ==========================

def turnover(prev, curr):
    if prev is None:
        return np.nan
    return 1 - len(set(prev) & set(curr)) / len(curr)


def apply_costs(r, turn):
    if np.isnan(turn):
        return r
    cost = (TC_BPS_PER_100_TURNOVER / 10000) * turn
    r = r.copy()
    r.iloc[0] -= cost
    return r


def backtest(prices, fundamentals, bench_px):
    momentum, vol = compute_price_factors(prices)

    rebal_dates = prices.resample(REBALANCE).last().index
    rebal_dates = rebal_dates[rebal_dates.isin(prices.index)]

    gross, net, turns = [], [], []
    prev = None

    for d0, d1 in zip(rebal_dates[:-1], rebal_dates[1:]):
        scores = make_scores(momentum.loc[d0], vol.loc[d0], fundamentals)
        picks = scores.sort_values(["score", "mom_z"], ascending=False).head(TOP_N).index.tolist()

        turn = turnover(prev, picks)
        turns.append(turn)
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

    return gross, net, bench, np.nanmean(turns)


# ==========================
# METRICS
# ==========================

def ann_return(r): return (1 + r).prod() ** (252 / len(r)) - 1
def ann_vol(r): return r.std() * np.sqrt(252)
def sharpe(r): return r.mean() / r.std() * np.sqrt(252)
def info_ratio(p, b): return (p - b).mean() / (p - b).std() * np.sqrt(252)

def alpha_beta(p, b):
    beta = p.cov(b) / b.var()
    alpha = (p - beta * b).mean() * 252
    return alpha, beta


# ==========================
# PLOT
# ==========================

def plot_equity_curves(gross, net, bench):
    plt.figure(figsize=(10, 6))

    plt.plot((1 + gross).cumprod(),
             label="Portfolio (Gross)",
             color="tab:blue",
             linewidth=2.2)

    plt.plot((1 + net).cumprod(),
             label="Portfolio (Net of Costs)",
             color="tab:orange",
             linestyle="--",
             linewidth=2.2)

    plt.plot((1 + bench).cumprod(),
             label="Benchmark",
             color="tab:gray",
             linewidth=1.8)

    plt.title("Equity Curve: Portfolio vs Benchmark")
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
    prices = download_prices(TICKERS, START_DATE, END_DATE)
    fundamentals = fetch_fundamentals(TICKERS)
    bench_px = download_prices([BENCHMARK], START_DATE, END_DATE)[BENCHMARK]

    gross, net, bench, avg_turn = backtest(prices, fundamentals, bench_px)

    alpha_g, beta_g = alpha_beta(gross, bench)
    alpha_n, beta_n = alpha_beta(net, bench)

    stats = {
        "Gross Ann Return": ann_return(gross),
        "Net Ann Return": ann_return(net),
        "Gross Sharpe": sharpe(gross),
        "Net Sharpe": sharpe(net),
        "Gross IR": info_ratio(gross, bench),
        "Net IR": info_ratio(net, bench),
        "Gross Alpha": alpha_g,
        "Net Alpha": alpha_n,
        "Avg Turnover": avg_turn,
    }

    for k, v in stats.items():
        print(f"{k}: {v:.3f}")

    plot_equity_curves(gross, net, bench)


if __name__ == "__main__":
    main()
