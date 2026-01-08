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

DEFAULT_BENCHMARK = "SPY"
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

def fmt_pct(x: float, decimals: int = 2) -> str:
    if x is None or np.isnan(x):
        return "—"
    return f"{x*100:.{decimals}f}%"

def fmt_num(x: float, decimals: int = 2) -> str:
    if x is None or np.isnan(x):
        return "—"
    return f"{x:.{decimals}f}"

def pct_change_series(px: pd.Series) -> pd.Series:
    return px.pct_change()

def safe_date_str(x) -> str:
    try:
        return pd.Timestamp(x).date().isoformat()
    except Exception:
        return str(x)


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

    bench = pct_change_series(bench_px).reindex(gross.index).dropna()
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
    plt.title("Growth of $1: Portfolio vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return fig


def make_downloads(eq: pd.DataFrame, regime_summary: pd.DataFrame, holdings_changes: pd.DataFrame):
    eq_csv = eq.to_csv(index=True).encode("utf-8")
    reg_csv = regime_summary.to_csv(index=False).encode("utf-8")
    st.download_button("Equity curve CSV", data=eq_csv, file_name="equity_curves.csv", mime="text/csv")
    st.download_button("Regime summary CSV", data=reg_csv, file_name="regime_summary.csv", mime="text/csv")

    try:
        from io import BytesIO
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            holdings_changes.to_excel(writer, index=False, sheet_name="holdings_changes")
        st.download_button(
            "Holdings log (Excel)",
            data=bio.getvalue(),
            file_name="holdings_changes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception:
        st.caption("Excel export needs openpyxl.")


st.set_page_config(page_title="Atlas Portfolio", layout="wide")
st.title("Atlas Portfolio")
st.caption("Build a simple rules-based portfolio and compare it to the market (SPY).")

with st.sidebar:
    st.subheader("Quick start")
    simple_mode = st.toggle("Simple mode", value=True)

    benchmark = st.text_input("Benchmark", value=DEFAULT_BENCHMARK)
    vix_ticker = st.text_input("VIX ticker", value=DEFAULT_VIX_TICKER)

    if simple_mode:
        start = st.text_input("Start", value="2017-01-01")
        end_in = st.text_input("End (optional)", value="")
    else:
        start = st.text_input("Start (YYYY-MM-DD)", value="2016-01-01")
        end_in = st.text_input("End (blank = today)", value="")

    end = None if end_in.strip() == "" else end_in.strip()

    st.divider()
    st.subheader("Portfolio")
    top_n = st.slider("Holdings (TOP N)", 5, 50, DEFAULT_TOP_N)
    rebalance_label = st.selectbox("Rebalance", ["Monthly", "Quarterly", "Weekly"], index=0)
    rebalance = {"Monthly": "M", "Quarterly": "Q", "Weekly": "W"}[rebalance_label]

    st.divider()
    st.subheader("Costs")
    tc = st.number_input("Cost (bps / 100% turnover)", 0.0, 200.0, float(DEFAULT_TC_BPS_PER_100_TURNOVER), 1.0)

    st.divider()
    st.subheader("Factors")
    auto_norm = st.toggle("Normalize weights", value=True)

    preset = st.selectbox("Preset", ["Balanced", "Conservative", "Aggressive", "Custom"], index=0)

    w = dict(DEFAULT_WEIGHTS)
    if preset == "Conservative":
        w = {"value_pe": 0.15, "profit_roe": 0.15, "growth_rev": 0.15, "risk_vol": 0.30, "risk_de": 0.25}
    elif preset == "Aggressive":
        w = {"value_pe": 0.15, "profit_roe": 0.30, "growth_rev": 0.30, "risk_vol": 0.15, "risk_de": 0.10}

    if preset == "Custom" or (not simple_mode):
        w["value_pe"] = st.slider("Valuation (P/E)", 0.0, 1.0, float(w["value_pe"]), 0.05)
        w["profit_roe"] = st.slider("Profitability (ROE)", 0.0, 1.0, float(w["profit_roe"]), 0.05)
        w["growth_rev"] = st.slider("Growth (revenue)", 0.0, 1.0, float(w["growth_rev"]), 0.05)
        w["risk_vol"] = st.slider("Low volatility", 0.0, 1.0, float(w["risk_vol"]), 0.05)
        w["risk_de"] = st.slider("Low debt", 0.0, 1.0, float(w["risk_de"]), 0.05)

    weights = normalize_weights(w) if auto_norm else w

    if not simple_mode:
        st.divider()
        st.subheader("Advanced")
        mom_lb = st.number_input("Momentum lookback (days)", 20, 756, DEFAULT_MOM_LB, 5)
        vol_lb = st.number_input("Volatility lookback (days)", 20, 756, DEFAULT_VOL_LB, 5)
        vix_smooth = st.number_input("VIX smooth days", 5, 252, int(DEFAULT_VIX_SMOOTH_DAYS), 1)
        low_q = st.slider("Low regime quantile", 0.05, 0.49, float(DEFAULT_LOW_Q), 0.01)
        high_q = st.slider("High regime quantile", 0.51, 0.95, float(DEFAULT_HIGH_Q), 0.01)
        show_full_factor_table = st.toggle("Show full factor table", value=False)
    else:
        mom_lb = DEFAULT_MOM_LB
        vol_lb = DEFAULT_VOL_LB
        vix_smooth = DEFAULT_VIX_SMOOTH_DAYS
        low_q = DEFAULT_LOW_Q
        high_q = DEFAULT_HIGH_Q
        show_full_factor_table = False

    st.divider()
    st.subheader("Universe")
    if simple_mode:
        tickers = DEFAULT_TICKERS
        edit_universe = st.toggle("Edit tickers", value=False)
        if edit_universe:
            tickers_text = st.text_area("Tickers (comma-separated)", ", ".join(DEFAULT_TICKERS), height=110)
            tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]
    else:
        tickers_text = st.text_area("Tickers (comma-separated)", ", ".join(DEFAULT_TICKERS), height=110)
        tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]

    st.divider()
    run = st.button("Run", type="primary")


if not run:
    st.stop()

if len(tickers) < 5:
    st.error("Add at least 5 tickers.")
    st.stop()

if top_n > len(tickers):
    st.error("TOP N cannot exceed the number of tickers.")
    st.stop()

with st.spinner("Loading data..."):
    prices = download_prices(tickers, start, end)
    bench_px = download_prices([benchmark], start, end)[benchmark]

with st.spinner("Loading fundamentals..."):
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
    st.error("No results. Try a later start date (e.g., 2019-01-01) or Monthly rebalance.")
    st.stop()

eq = equity_df(gross, net, bench)

tab1, tab2, tab3 = st.tabs(["Overview", "Holdings", "Volatility regimes"])

with tab1:
    st.subheader("Performance")
    st.pyplot(equity_fig(eq), clear_figure=True)

    all_stats = regime_stats("ALL", net, bench)
    a, b, c, d, e = st.columns(5)
    a.metric("Return", fmt_pct(all_stats["ann_return"], 2))
    b.metric("Vol", fmt_pct(all_stats["ann_vol"], 2))
    c.metric("Sharpe", fmt_num(all_stats["sharpe"], 2))
    d.metric("Info vs SPY", fmt_num(all_stats["info_ratio_vs_spy"], 2))
    e.metric("Turnover", fmt_num(avg_turn, 3))

    with st.expander("What do these mean?"):
        st.markdown(
            """
- **Return**: average yearly growth rate  
- **Vol**: how bumpy returns are (higher = more ups/downs)  
- **Sharpe**: return per unit of risk (higher is better)  
- **Info vs SPY**: how consistently you beat SPY (higher is better)  
- **Turnover**: how much the holdings change at each rebalance
            """
        )

with tab2:
    st.subheader("Top holdings")
    momentum_all, vol_all = compute_price_factors(prices, mom_lb=int(mom_lb), vol_lb=int(vol_lb))
    valid_dates = [d for d in rebal_dates if d in prices.index]
    if len(valid_dates) > 120:
        valid_dates = valid_dates[-120:]

    if not valid_dates:
        st.warning("Not enough rebalance dates to show holdings.")
    else:
        chosen = st.select_slider(
            "Rebalance date",
            options=valid_dates,
            value=valid_dates[-1],
            format_func=lambda d: safe_date_str(d),
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

        st.dataframe(scores_full.head(int(top_n))[cols], use_container_width=True, height=480)

        with st.expander("Holdings changes log"):
            st.dataframe(holdings_changes, use_container_width=True, height=340)

        if show_full_factor_table:
            with st.expander("Full factor table"):
                st.dataframe(scores_full[cols], use_container_width=True, height=520)

with tab3:
    st.subheader("VIX regime sensitivity (Net vs SPY)")
    with st.spinner("Computing regimes..."):
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

    regime_summary_display = regime_summary.copy()
    regime_summary_display["regime"] = regime_summary_display["regime"].replace({
        "ALL": "All days",
        "LOW_VIX": "Low VIX",
        "HIGH_VIX": "High VIX",
    })

    st.dataframe(
        regime_summary_display.style.format({
            "ann_return": "{:.2%}",
            "ann_vol": "{:.2%}",
            "sharpe": "{:.2f}",
            "info_ratio_vs_spy": "{:.2f}",
        }),
        use_container_width=True
    )

    with st.expander("What is this?"):
        st.markdown(
            """
The **VIX** is a popular measure of stock market volatility.

This splits time into:
- **Low VIX** (calm market)
- **High VIX** (turbulent market)

Then it recomputes the same metrics in each regime.
            """
        )

st.divider()
st.subheader("Downloads")
make_downloads(eq, regime_summary, holdings_changes)
