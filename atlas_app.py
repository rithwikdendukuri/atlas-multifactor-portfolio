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

DEFAULT_TICKERS: List[str] = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","BRK-B","LLY","AVGO",
    "JPM","TSLA","V","XOM","UNH","MA","COST","HD","PG","JNJ",
    "ORCL","MRK","ABBV","CVX","NFLX","KO","CRM","BAC","WMT","PEP",
    "AMD","ADBE","TMO","MCD","QCOM","NKE","LIN","DIS","CSCO","ABT",
    "ACN","VZ","TXN","DHR","INTC","NEE","PM","UPS","MS","AMGN"
]

DEFAULT_START = "2016-01-01"
DEFAULT_TOP_N = 30
DEFAULT_REBALANCE = "M"
DEFAULT_MOM_LB = 252
DEFAULT_VOL_LB = 252

DEFAULT_WEIGHTS: Dict[str, float] = {
    "value_pe": 0.20,
    "profit_roe": 0.20,
    "growth_rev": 0.20,
    "risk_vol": 0.20,
    "risk_de": 0.20,
}

DEFAULT_BENCHMARK = "SPY"
DEFAULT_TC_BPS_PER_100_TURNOVER = 10.0

DEFAULT_VIX_TICKER = "^VIX"
DEFAULT_VIX_SMOOTH_DAYS = 63
DEFAULT_LOW_Q = 0.33
DEFAULT_HIGH_Q = 0.67


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

def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(w.values()))
    if s <= 0:
        k = len(w)
        return {kk: 1.0 / k for kk in w}
    return {kk: float(v) / s for kk, v in w.items()}


@st.cache_data(show_spinner=False)
def download_prices(tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    px = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df
    px = px.dropna(how="all")
    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])
    return px

@st.cache_data(show_spinner=False)
def download_close(ticker: str, start: str, end: Optional[str]) -> pd.Series:
    df = yf.download([ticker], start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        s = df["Close"][ticker]
    else:
        s = df["Close"] if "Close" in df.columns else df.squeeze()
    return s.dropna()

@st.cache_data(show_spinner=False)
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


def compute_price_factors(prices: pd.DataFrame, mom_lb: int, vol_lb: int):
    rets = prices.pct_change()
    momentum = prices / prices.shift(mom_lb) - 1
    vol = rets.rolling(vol_lb).std() * np.sqrt(252)
    return momentum, vol


def make_scores_full(
    momentum_row: pd.Series,
    vol_row: pd.Series,
    fundamentals: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.DataFrame:
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
    z["z_risk_vol"] = -zscore(raw["risk_vol"])
    z["z_risk_de"] = -zscore(raw["risk_de"])
    z["z_profit_roe"] = zscore(raw["profit_roe"])
    z["z_growth_rev"] = zscore(raw["growth_rev"])
    z["z_mom_12m"] = zscore(raw["mom_12m"])

    out = pd.concat([raw, z], axis=1)
    out["score"] = (
        out["z_value_pe"].fillna(0) * float(weights["value_pe"])
        + out["z_profit_roe"].fillna(0) * float(weights["profit_roe"])
        + out["z_growth_rev"].fillna(0) * float(weights["growth_rev"])
        + out["z_risk_vol"].fillna(0) * float(weights["risk_vol"])
        + out["z_risk_de"].fillna(0) * float(weights["risk_de"])
    )
    out["mom_z"] = out["z_mom_12m"]
    return out


def turnover(prev: Optional[List[str]], curr: List[str]) -> float:
    if prev is None:
        return np.nan
    return 1 - len(set(prev) & set(curr)) / len(curr)

def apply_costs(r: pd.Series, turn: float, tc_bps_per_100_turnover: float) -> pd.Series:
    if np.isnan(turn):
        return r
    cost = (tc_bps_per_100_turnover / 10000) * turn
    r = r.copy()
    r.iloc[0] -= cost
    return r


def backtest(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    bench_px: pd.Series,
    top_n: int,
    rebalance: str,
    mom_lb: int,
    vol_lb: int,
    weights: Dict[str, float],
    tc_bps_per_100_turnover: float,
) -> Tuple[pd.Series, pd.Series, pd.Series, float, pd.DataFrame, pd.DatetimeIndex]:
    momentum, vol = compute_price_factors(prices, mom_lb=mom_lb, vol_lb=vol_lb)

    rebal_dates = prices.resample(rebalance).last().index
    rebal_dates = rebal_dates[rebal_dates.isin(prices.index)]

    gross, net, turns = [], [], []
    prev: Optional[List[str]] = None

    holdings_log_rows = []

    for d0, d1 in zip(rebal_dates[:-1], rebal_dates[1:]):
        scores = make_scores_full(momentum.loc[d0], vol.loc[d0], fundamentals, weights=weights)
        picks = scores.sort_values(["score", "mom_z"], ascending=False).head(top_n).index.tolist()

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
        n = apply_costs(g, turn, tc_bps_per_100_turnover)

        gross.append(g)
        net.append(n)

    gross = pd.concat(gross) if gross else pd.Series(dtype=float)
    net = pd.concat(net) if net else pd.Series(dtype=float)

    bench = bench_px.pct_change().reindex(gross.index).dropna()
    gross = gross.reindex(bench.index)
    net = net.reindex(bench.index)

    holdings_changes = pd.DataFrame(holdings_log_rows)
    avg_turn = float(np.nanmean(turns)) if len(turns) else np.nan

    return gross, net, bench, avg_turn, holdings_changes, rebal_dates


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


def compute_vix_regimes(vix_close: pd.Series, smooth_days: int, low_q: float, high_q: float) -> pd.Series:
    vix_smooth = vix_close.rolling(smooth_days).mean().dropna()
    lo_thr = float(vix_smooth.quantile(low_q))
    hi_thr = float(vix_smooth.quantile(high_q))

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


def equity_df(gross: pd.Series, net: pd.Series, bench: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "Portfolio (Gross)": (1 + gross).cumprod(),
        "Portfolio (Net of Costs)": (1 + net).cumprod(),
        "Benchmark (SPY)": (1 + bench).cumprod(),
    })


def equity_fig(df: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(10, 5))
    for c in df.columns:
        plt.plot(df.index, df[c], label=c, linewidth=2)
    plt.title("Equity Curve: Portfolio vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return fig


st.set_page_config(page_title="Atlas Multi-Factor Portfolio", layout="wide")
st.title("Atlas Multi-Factor Portfolio")

with st.sidebar:
    tickers_text = st.text_area("Tickers (comma-separated)", ", ".join(DEFAULT_TICKERS), height=120)
    tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]

    c1, c2 = st.columns(2)
    start = c1.text_input("Start (YYYY-MM-DD)", DEFAULT_START)
    end_in = c2.text_input("End (blank = today)", "")
    end = None if end_in.strip() == "" else end_in.strip()

    top_n = st.slider("TOP N", 5, min(50, len(tickers)), DEFAULT_TOP_N)
    rebalance = st.selectbox("Rebalance", ["M", "Q", "W"], index=0)

    st.subheader("Lookbacks")
    mom_lb = st.number_input("Momentum lookback (days)", 20, 756, DEFAULT_MOM_LB, step=5)
    vol_lb = st.number_input("Volatility lookback (days)", 20, 756, DEFAULT_VOL_LB, step=5)

    st.subheader("Transaction Costs")
    tc = st.number_input("TC bps per 100% turnover", 0.0, 200.0, float(DEFAULT_TC_BPS_PER_100_TURNOVER), step=1.0)

    st.subheader("Weights")
    auto_norm = st.toggle("Auto-normalize weights", True)

    w_raw = {
        "value_pe": st.number_input("value_pe", 0.0, 1.0, float(DEFAULT_WEIGHTS["value_pe"]), 0.05),
        "profit_roe": st.number_input("profit_roe", 0.0, 1.0, float(DEFAULT_WEIGHTS["profit_roe"]), 0.05),
        "growth_rev": st.number_input("growth_rev", 0.0, 1.0, float(DEFAULT_WEIGHTS["growth_rev"]), 0.05),
        "risk_vol": st.number_input("risk_vol", 0.0, 1.0, float(DEFAULT_WEIGHTS["risk_vol"]), 0.05),
        "risk_de": st.number_input("risk_de", 0.0, 1.0, float(DEFAULT_WEIGHTS["risk_de"]), 0.05),
    }
    weights = normalize_weights(w_raw) if auto_norm else w_raw

    st.subheader("Volatility Regimes (VIX)")
    vix_ticker = st.text_input("VIX ticker", DEFAULT_VIX_TICKER)
    vix_smooth = st.number_input("VIX smooth days", 5, 252, int(DEFAULT_VIX_SMOOTH_DAYS), 1)
    low_q = st.slider("LOW_Q", 0.05, 0.49, float(DEFAULT_LOW_Q), 0.01)
    high_q = st.slider("HIGH_Q", 0.51, 0.95, float(DEFAULT_HIGH_Q), 0.01)

    st.subheader("Benchmark")
    benchmark = st.text_input("Benchmark", DEFAULT_BENCHMARK)

    run = st.button("Run", type="primary")


if not run:
    st.write("Adjust settings in the sidebar and click **Run**.")
    st.stop()


prices = download_prices(tickers, start, end)
bench_px = download_prices([benchmark], start, end)[benchmark]
fundamentals = fetch_fundamentals(tickers).reindex(prices.columns)

gross, net, bench, avg_turn, holdings_changes, rebal_dates = backtest(
    prices=prices,
    fundamentals=fundamentals,
    bench_px=bench_px,
    top_n=int(top_n),
    rebalance=rebalance,
    mom_lb=int(mom_lb),
    vol_lb=int(vol_lb),
    weights=weights,
    tc_bps_per_100_turnover=float(tc),
)

if gross.empty or net.empty:
    st.error("Backtest produced no returns. Try a later start date or different frequency.")
    st.stop()

eq = equity_df(gross, net, bench)

left, right = st.columns([1.2, 0.8])

with left:
    st.subheader("Equity Curve")
    st.pyplot(equity_fig(eq), clear_figure=True)

    st.subheader("Top Holdings Explorer")
    momentum_all, vol_all = compute_price_factors(prices, mom_lb=int(mom_lb), vol_lb=int(vol_lb))
    valid_dates = [d for d in rebal_dates if d in prices.index]
    if len(valid_dates) > 120:
        valid_dates = valid_dates[-120:]
    default_ix = max(0, len(valid_dates) - 12)

    chosen = st.selectbox(
        "Rebalance date",
        options=valid_dates,
        index=default_ix if valid_dates else 0,
        format_func=lambda d: pd.Timestamp(d).date().isoformat()
    )

    scores_full = make_scores_full(
        momentum_all.loc[chosen],
        vol_all.loc[chosen],
        fundamentals,
        weights=weights,
    ).sort_values(["score", "mom_z"], ascending=False)

    top_table = scores_full.head(int(top_n))
    cols = [
        "score",
        "value_pe","profit_roe","growth_rev","risk_vol","risk_de","mom_12m",
        "z_value_pe","z_profit_roe","z_growth_rev","z_risk_vol","z_risk_de","z_mom_12m",
    ]
    st.dataframe(top_table[cols], use_container_width=True, height=430)

    with st.expander("Full universe factor table (ranked)"):
        st.dataframe(scores_full[cols], use_container_width=True, height=520)

with right:
    st.subheader("Headline Metrics (Net vs SPY)")
    m = regime_stats("ALL", net, bench)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ann. Return", f"{m['ann_return']*100:.2f}%")
    c2.metric("Ann. Vol", f"{m['ann_vol']*100:.2f}%")
    c3.metric("Sharpe", f"{m['sharpe']:.2f}")
    c4.metric("Info Ratio", f"{m['info_ratio_vs_spy']:.2f}")
    c5.metric("Avg Turnover", f"{avg_turn:.3f}" if np.isfinite(avg_turn) else "nan")

    st.subheader("VIX Regime Sensitivity (Net vs SPY)")
    vix_close = download_close(vix_ticker, start, end)
    regimes = compute_vix_regimes(vix_close, int(vix_smooth), float(low_q), float(high_q))

    df = pd.DataFrame({"net": net, "bench": bench, "regime": regimes}).dropna()
    low = df[df["regime"] == "low_vol"]
    high = df[df["regime"] == "high_vol"]

    regime_summary = pd.DataFrame([
        regime_stats("ALL", df["net"], df["bench"]),
        regime_stats("LOW_VIX", low["net"], low["bench"]),
        regime_stats("HIGH_VIX", high["net"], high["bench"]),
    ])

    st.dataframe(
        regime_summary.style.format({
            "ann_return": "{:.3%}",
            "ann_vol": "{:.3%}",
            "sharpe": "{:.3f}",
            "info_ratio_vs_spy": "{:.3f}",
        }),
        use_container_width=True
    )

    st.subheader("Holdings Changes Log")
    st.dataframe(holdings_changes, use_container_width=True, height=320)

    st.subheader("Downloads")
    eq_csv = eq.to_csv(index=True).encode("utf-8")
    reg_csv = regime_summary.to_csv(index=False).encode("utf-8")
    st.download_button("Download equity_curves.csv", data=eq_csv, file_name="equity_curves.csv", mime="text/csv")
    st.download_button("Download regime_summary.csv", data=reg_csv, file_name="regime_summary.csv", mime="text/csv")

    try:
        from io import BytesIO
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            holdings_changes.to_excel(writer, index=False, sheet_name="holdings_changes")
        st.download_button(
            "Download holdings_changes.xlsx",
            data=bio.getvalue(),
            file_name="holdings_changes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception:
        st.caption("Install openpyxl to enable Excel download.")

