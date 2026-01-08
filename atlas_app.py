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

# ---------------------------
# Defaults (match your original logic)
# ---------------------------

DEFAULT_TICKERS: List[str] = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","BRK-B","LLY","AVGO",
    "JPM","TSLA","V","XOM","UNH","MA","COST","HD","PG","JNJ",
    "ORCL","MRK","ABBV","CVX","NFLX","KO","CRM","BAC","WMT","PEP",
    "AMD","ADBE","TMO","MCD","QCOM","NKE","LIN","DIS","CSCO","ABT",
    "ACN","VZ","TXN","DHR","INTC","NEE","PM","UPS","MS","AMGN"
]

DEFAULT_BENCHMARK = "SPY"

# Your original: monthly by default, TOP_N=30
DEFAULT_TOP_N = 30
DEFAULT_REBALANCE = "M"

# Your original: 252 day lookbacks
DEFAULT_MOM_LB = 252
DEFAULT_VOL_LB = 252

DEFAULT_WEIGHTS: Dict[str, float] = {
    "value_pe": 0.20,
    "profit_roe": 0.20,
    "growth_rev": 0.20,
    "risk_vol": 0.20,
    "risk_de": 0.20,
}

# Your original: 10 bps per 100% turnover
DEFAULT_TC_BPS_PER_100_TURNOVER = 10.0

# Your original: VIX regime settings
DEFAULT_VIX_TICKER = "^VIX"
DEFAULT_VIX_SMOOTH_DAYS = 63
DEFAULT_LOW_Q = 0.33
DEFAULT_HIGH_Q = 0.67


# ---------------------------
# Helpers
# ---------------------------

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

def percent(x: float) -> str:
    return "—" if (x is None or np.isnan(x)) else f"{x*100:.2f}%"

def num(x: float, d: int = 2) -> str:
    return "—" if (x is None or np.isnan(x)) else f"{x:.{d}f}"


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


# ---------------------------
# Factors + Scores (match your original)
# ---------------------------

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


# ---------------------------
# Backtest (match your original)
# ---------------------------

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


# ---------------------------
# Metrics + VIX regimes (match your original)
# ---------------------------

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
    plt.title("Portfolio vs SPY (Cumulative Growth of $1)")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return fig


# ---------------------------
# UI (layman-friendly)
# ---------------------------

st.set_page_config(page_title="Atlas Portfolio Dashboard", layout="wide")
st.title("Atlas Portfolio Dashboard")
st.caption("An interactive prototype that builds a rules-based stock portfolio and compares it to SPY.")

with st.expander("How to use this (30 seconds)", expanded=True):
    st.markdown(
        """
1) Leave the defaults as-is for a clean first run  
2) Click **Run Analysis** in the sidebar  
3) Use the dropdown to explore **Top Holdings** by rebalance date  
4) Check **VIX Regime Sensitivity** to see how performance changes in calm vs volatile markets
        """
    )

with st.sidebar:
    st.header("1) Quick Start")

    simple_mode = st.toggle("Simple Mode (recommended)", value=True)

    if simple_mode:
        st.caption("Simple Mode uses safe defaults and hides advanced controls.")

    st.divider()
    st.header("2) Basics")

    benchmark = st.text_input("Benchmark (market comparison)", value=DEFAULT_BENCHMARK)
    vix_ticker = st.text_input("Volatility index (for regimes)", value=DEFAULT_VIX_TICKER)

    if simple_mode:
        start = st.text_input("Start date", value="2017-01-01")
        end_in = st.text_input("End date (optional)", value="")
        end = None if end_in.strip() == "" else end_in.strip()
    else:
        start = st.text_input("Start date (YYYY-MM-DD)", value="2016-01-01")
        end_in = st.text_input("End date (blank = today)", value="")
        end = None if end_in.strip() == "" else end_in.strip()

    st.divider()
    st.header("3) Portfolio Size & Rebalancing")

    top_n = st.slider(
        "How many stocks to hold (TOP N)",
        min_value=5,
        max_value=50,
        value=DEFAULT_TOP_N if not simple_mode else 30,
        help="The portfolio holds the top-ranked N stocks at each rebalance."
    )

    rebalance_map = {"Monthly": "M", "Quarterly": "Q", "Weekly": "W"}
    rebalance_label = st.selectbox(
        "How often to rebalance",
        options=list(rebalance_map.keys()),
        index=0,
        help="Monthly is the default and usually the most readable."
    )
    rebalance = rebalance_map[rebalance_label]

    st.divider()
    st.header("4) Costs (optional)")

    tc = st.number_input(
        "Transaction cost (bps per 100% turnover)",
        min_value=0.0,
        max_value=200.0,
        value=float(DEFAULT_TC_BPS_PER_100_TURNOVER) if not simple_mode else 10.0,
        step=1.0,
        help="Your original model: 10 bps cost if the portfolio fully turns over at a rebalance."
    )

    st.divider()
    st.header("5) Factor Settings")

    auto_norm = st.toggle("Auto-normalize weights (recommended)", value=True)
    st.caption("Weights control how much each factor matters. Auto-normalize keeps them totaling 100%.")

    preset = st.selectbox(
        "Preset (optional)",
        ["Balanced (default)", "Conservative (risk-sensitive)", "Aggressive (growth/profit focus)", "Custom"],
        index=0,
        help="Presets just adjust the factor weights."
    )

    w = dict(DEFAULT_WEIGHTS)
    if preset == "Conservative (risk-sensitive)":
        w = {"value_pe": 0.15, "profit_roe": 0.15, "growth_rev": 0.15, "risk_vol": 0.30, "risk_de": 0.25}
    elif preset == "Aggressive (growth/profit focus)":
        w = {"value_pe": 0.15, "profit_roe": 0.30, "growth_rev": 0.30, "risk_vol": 0.15, "risk_de": 0.10}

    if preset == "Custom" or not simple_mode:
        w["value_pe"] = st.slider("Valuation (P/E) weight", 0.0, 1.0, float(w["value_pe"]), 0.05)
        w["profit_roe"] = st.slider("Profitability (ROE) weight", 0.0, 1.0, float(w["profit_roe"]), 0.05)
        w["growth_rev"] = st.slider("Growth (revenue) weight", 0.0, 1.0, float(w["growth_rev"]), 0.05)
        w["risk_vol"] = st.slider("Risk (volatility) weight", 0.0, 1.0, float(w["risk_vol"]), 0.05)
        w["risk_de"] = st.slider("Risk (debt-to-equity) weight", 0.0, 1.0, float(w["risk_de"]), 0.05)

    weights = normalize_weights(w) if auto_norm else w

    if not simple_mode:
        st.divider()
        st.header("6) Advanced")
        mom_lb = st.number_input("Momentum lookback (days)", 20, 756, DEFAULT_MOM_LB, step=5)
        vol_lb = st.number_input("Volatility lookback (days)", 20, 756, DEFAULT_VOL_LB, step=5)
        vix_smooth = st.number_input("VIX smoothing window (days)", 5, 252, int(DEFAULT_VIX_SMOOTH_DAYS), 1)
        low_q = st.slider("LOW_Q (bottom quantile)", 0.05, 0.49, float(DEFAULT_LOW_Q), 0.01)
        high_q = st.slider("HIGH_Q (top quantile)", 0.51, 0.95, float(DEFAULT_HIGH_Q), 0.01)
    else:
        mom_lb = DEFAULT_MOM_LB
        vol_lb = DEFAULT_VOL_LB
        vix_smooth = DEFAULT_VIX_SMOOTH_DAYS
        low_q = DEFAULT_LOW_Q
        high_q = DEFAULT_HIGH_Q

    st.divider()
    st.header("Universe")

    if simple_mode:
        tickers = DEFAULT_TICKERS
        st.caption(f"Using default universe of {len(tickers)} large-cap tickers.")
        show_universe = st.toggle("Show / edit tickers", value=False)
        if show_universe:
            tickers_text = st.text_area("Tickers (comma-separated)", ", ".join(DEFAULT_TICKERS), height=120)
            tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]
    else:
        tickers_text = st.text_area("Tickers (comma-separated)", ", ".join(DEFAULT_TICKERS), height=120)
        tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]

    st.divider()
    run = st.button("Run Analysis", type="primary")


if not run:
    st.stop()

if len(tickers) < 5:
    st.error("Please provide at least 5 tickers.")
    st.stop()
if top_n > len(tickers):
    st.error("TOP N cannot exceed the number of tickers in the universe.")
    st.stop()

with st.spinner("Downloading price data (this can take a bit on first run)..."):
    prices = download_prices(tickers, start, end)
    bench_px = download_prices([benchmark], start, end)[benchmark]

with st.spinner("Fetching fundamentals snapshot..."):
    fundamentals = fetch_fundamentals(list(prices.columns)).reindex(prices.columns)

with st.spinner("Running backtest..."):
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
    st.error("No returns were produced. Try a later start date (e.g., 2019-01-01) or monthly rebalancing.")
    st.stop()

eq = equity_df(gross, net, bench)

left, right = st.columns([1.2, 0.8])

with left:
    st.subheader("Portfolio vs SPY")
    st.pyplot(equity_fig(eq), clear_figure=True)

    st.subheader("Top Holdings Explorer")

    momentum_all, vol_all = compute_price_factors(prices, mom_lb=int(mom_lb), vol_lb=int(vol_lb))
    valid_dates = [d for d in rebal_dates if d in prices.index]

    if len(valid_dates) > 120:
        valid_dates = valid_dates[-120:]

    if not valid_dates:
        st.warning("Not enough rebalance dates to explore holdings. Try a longer date range.")
    else:
        default_ix = max(0, len(valid_dates) - 12)
        chosen = st.selectbox(
            "Pick a rebalance date (the portfolio is re-selected on these dates)",
            options=valid_dates,
            index=default_ix,
            format_func=lambda d: pd.Timestamp(d).date().isoformat(),
        )

        scores_full = make_scores_full(
            momentum_all.loc[chosen],
            vol_all.loc[chosen],
            fundamentals,
            weights=weights,
        ).sort_values(["score", "mom_z"], ascending=False)

        cols = [
            "score",
            "value_pe","profit_roe","growth_rev","risk_vol","risk_de","mom_12m",
            "z_value_pe","z_profit_roe","z_growth_rev","z_risk_vol","z_risk_de","z_mom_12m",
        ]

        st.caption("Top-N table shows: factor inputs → z-scores → final score used for ranking.")
        st.dataframe(scores_full.head(int(top_n))[cols], use_container_width=True, height=430)

        with st.expander("What am I looking at?"):
            st.markdown(
                """
- **score**: the weighted total used to rank stocks (higher = better in this model)  
- **value_pe**: P/E (lower is better; converted to a positive z-score by multiplying by -1)  
- **risk_vol** and **risk_de**: lower is better (also multiplied by -1 before scoring)  
- **z_*** columns: standardized values so different metrics can be combined fairly  
- **mom_12m**: used as a tie-breaker (so two similar scores still rank consistently)
                """
            )

with right:
    st.subheader("Key Results (Net of Costs)")

    all_stats = regime_stats("ALL", net, bench)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Annual Return", percent(all_stats["ann_return"]))
    c2.metric("Annual Volatility", percent(all_stats["ann_vol"]))
    c3.metric("Sharpe", num(all_stats["sharpe"], 2))
    c4.metric("Info Ratio vs SPY", num(all_stats["info_ratio_vs_spy"], 2))
    c5.metric("Avg Turnover", num(avg_turn, 3))

    st.subheader("VIX Regime Sensitivity")

    with st.spinner("Downloading VIX and computing regimes..."):
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

    with st.expander("What does 'VIX regime sensitivity' mean?"):
        st.markdown(
            """
The **VIX** is often called a “fear gauge” for the stock market.

This app:
- Smooths VIX over ~3 months (63 trading days)
- Labels days as **low**, **mid**, or **high** volatility using quantiles
- Computes performance metrics separately in **LOW_VIX** and **HIGH_VIX** periods

This helps answer:  
**Does the strategy behave differently in calm markets vs turbulent markets?**
            """
        )

    st.subheader("Holdings Changes Log")
    st.caption("Shows what changed at each rebalance (adds/removes/turnover).")
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
