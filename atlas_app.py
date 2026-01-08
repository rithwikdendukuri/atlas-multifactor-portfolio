from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 130

DEFAULT_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","BRK-B","LLY","AVGO",
    "JPM","TSLA","V","XOM","UNH","MA","COST","HD","PG","JNJ",
    "ORCL","MRK","ABBV","CVX","NFLX","KO","CRM","BAC","WMT","PEP",
    "AMD","ADBE","TMO","MCD","QCOM","NKE","LIN","DIS","CSCO","ABT",
    "ACN","VZ","TXN","DHR","INTC","NEE","PM","UPS","MS","AMGN"
]

DEFAULT_WEIGHTS = {
    "value_pe": 0.20,
    "profit_roe": 0.20,
    "growth_rev": 0.20,
    "risk_vol": 0.20,
    "risk_de": 0.20,
}

def safe_float(x):
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

def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(w.values())
    if s <= 0:
        return {k: 1 / len(w) for k in w}
    return {k: v / s for k, v in w.items()}

@st.cache_data(show_spinner=False)
def download_prices(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    px = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df
    px = px.dropna(how="all")
    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])
    return px

@st.cache_data(show_spinner=False)
def download_close(ticker, start, end):
    df = yf.download([ticker], start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"][ticker].dropna()
    return df["Close"].dropna()

@st.cache_data(show_spinner=False)
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

def compute_price_factors(prices, mom_lb, vol_lb):
    rets = prices.pct_change()
    momentum = prices / prices.shift(mom_lb) - 1
    vol = rets.rolling(vol_lb).std() * np.sqrt(252)
    return momentum, vol

def make_scores_full(momentum_row, vol_row, fundamentals, weights):
    raw = pd.DataFrame(index=fundamentals.index)
    raw["value_pe"] = fundamentals["trailingPE"]
    raw["profit_roe"] = fundamentals["ROE"]
    raw["growth_rev"] = fundamentals["revenueGrowth"]
    raw["risk_vol"] = vol_row
    raw["risk_de"] = fundamentals["debtToEquity"]
    raw["mom_12m"] = momentum_row

    for c in raw.columns:
        raw[c] = winsorize(raw[c])

    z = pd.DataFrame(index=raw.index)
    z["z_value_pe"] = -zscore(raw["value_pe"])
    z["z_profit_roe"] = zscore(raw["profit_roe"])
    z["z_growth_rev"] = zscore(raw["growth_rev"])
    z["z_risk_vol"] = -zscore(raw["risk_vol"])
    z["z_risk_de"] = -zscore(raw["risk_de"])
    z["z_mom_12m"] = zscore(raw["mom_12m"])

    score = sum(z[f"z_{k}"] * weights[k] for k in weights)

    out = pd.concat([raw, z], axis=1)
    out["score"] = score
    out["mom_z"] = z["z_mom_12m"]
    return out

def turnover(prev, curr):
    if prev is None:
        return np.nan
    return 1 - len(set(prev) & set(curr)) / len(curr)

def apply_costs(r, turn, tc):
    if np.isnan(turn):
        return r
    r = r.copy()
    r.iloc[0] -= (tc / 10000) * turn
    return r

def backtest(prices, fundamentals, bench_px, top_n, rebalance, mom_lb, vol_lb, weights, tc):
    momentum, vol = compute_price_factors(prices, mom_lb, vol_lb)
    rebal_dates = prices.resample(rebalance).last().index
    rebal_dates = rebal_dates[rebal_dates.isin(prices.index)]

    gross, net, turns = [], [], []
    prev = None
    logs = []

    for d0, d1 in zip(rebal_dates[:-1], rebal_dates[1:]):
        scores = make_scores_full(momentum.loc[d0], vol.loc[d0], fundamentals, weights)
        picks = scores.sort_values(["score", "mom_z"], ascending=False).head(top_n).index.tolist()

        turn = turnover(prev, picks)
        turns.append(turn)

        logs.append({
            "rebalance_date": d0.date().isoformat(),
            "held": ", ".join(picks),
            "turnover": None if np.isnan(turn) else float(turn)
        })

        prev = picks
        period = prices.loc[d0:d1, picks].pct_change().dropna(how="all")
        if period.empty:
            continue

        g = period.mean(axis=1)
        n = apply_costs(g, turn, tc)

        gross.append(g)
        net.append(n)

    gross = pd.concat(gross)
    net = pd.concat(net)
    bench = bench_px.pct_change().reindex(gross.index).dropna()

    return gross.reindex(bench.index), net.reindex(bench.index), bench, np.nanmean(turns), pd.DataFrame(logs), rebal_dates

def equity_df(gross, net, bench):
    return pd.DataFrame({
        "Portfolio (Gross)": (1 + gross).cumprod(),
        "Portfolio (Net)": (1 + net).cumprod(),
        "SPY": (1 + bench).cumprod()
    })

st.set_page_config(layout="wide")
st.title("Atlas Multi-Factor Portfolio")

with st.sidebar:
    tickers = st.text_area("Tickers", ", ".join(DEFAULT_TICKERS)).split(",")
    tickers = [t.strip() for t in tickers if t.strip()]

    start = st.text_input("Start", "2016-01-01")
    end = st.text_input("End (optional)", "")
    end = None if end == "" else end

    top_n = st.slider("TOP N", 5, min(50, len(tickers)), 30)
    rebalance = st.selectbox("Rebalance", ["M", "Q", "W"])
    mom_lb = st.number_input("Momentum Lookback", 20, 756, 252)
    vol_lb = st.number_input("Volatility Lookback", 20, 756, 252)
    tc = st.number_input("TC (bps per 100% turnover)", 0.0, 200.0, 10.0)

    auto_norm = st.toggle("Normalize Weights", True)

    w = {
        "value_pe": st.number_input("value_pe", 0.0, 1.0, 0.2),
        "profit_roe": st.number_input("profit_roe", 0.0, 1.0, 0.2),
        "growth_rev": st.number_input("growth_rev", 0.0, 1.0, 0.2),
        "risk_vol": st.number_input("risk_vol", 0.0, 1.0, 0.2),
        "risk_de": st.number_input("risk_de", 0.0, 1.0, 0.2),
    }

    weights = normalize_weights(w) if auto_norm else w
    run = st.button("Run")

if run:
    prices = download_prices(tickers, start, end)
    bench_px = download_prices(["SPY"], start, end)["SPY"]
    fundamentals = fetch_fundamentals(tickers).reindex(prices.columns)

    gross, net, bench, avg_turn, logs, rebal_dates = backtest(
        prices, fundamentals, bench_px,
        top_n, rebalance, mom_lb, vol_lb, weights, tc
    )

    eq = equity_df(gross, net, bench)
    st.line_chart(eq)

    chosen = st.selectbox("Rebalance Date", rebal_dates[-60:])
    momentum, vol = compute_price_factors(prices, mom_lb, vol_lb)

    table = make_scores_full(
        momentum.loc[chosen],
        vol.loc[chosen],
        fundamentals,
        weights
    ).sort_values("score", ascending=False).head(top_n)

    st.dataframe(table)
