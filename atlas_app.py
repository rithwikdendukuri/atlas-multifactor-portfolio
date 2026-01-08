from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 140

# =========================
# Defaults (same methodology)
# =========================

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

APP_VERSION = "Atlas v1.0 (Streamlit)"


# =========================
# Theme switching (Beginner vs Advanced)
# =========================

def apply_mode_theme(advanced: bool) -> None:
    if advanced:
        css = """
        <style>
        :root {
          --bg: #0b1020;
          --panel: #111a33;
          --text: #e8ecff;
          --muted: #aab3d6;
          --accent: #7aa2ff;
          --accent2: #00ffb4;
          --border: rgba(255,255,255,0.10);
          --shadow: rgba(0,0,0,0.35);
        }
        .stApp {
          background: radial-gradient(1200px 600px at 15% 10%, rgba(122,162,255,0.25), transparent 60%),
                      radial-gradient(900px 500px at 90% 15%, rgba(0,255,180,0.10), transparent 55%),
                      var(--bg);
          color: var(--text);
        }
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, rgba(17,26,51,0.95), rgba(10,16,32,0.95));
          border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] * { color: var(--text) !important; }
        .stCaption, .stMarkdown p { color: var(--muted) !important; }
        .stButton>button {
          background: linear-gradient(180deg, rgba(122,162,255,0.95), rgba(88,129,255,0.95));
          color: #0b1020 !important;
          border: 1px solid rgba(255,255,255,0.10);
          border-radius: 14px;
          box-shadow: 0 10px 25px var(--shadow);
          font-weight: 800;
        }
        .stButton>button:hover { filter: brightness(1.05); transform: translateY(-1px); }
        [data-testid="stMetric"] {
          background: rgba(255,255,255,0.05);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 12px 12px;
          box-shadow: 0 10px 25px var(--shadow);
        }
        [data-testid="stDataFrame"] {
          background: rgba(255,255,255,0.03);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 6px;
          box-shadow: 0 10px 25px var(--shadow);
        }
        details {
          background: rgba(255,255,255,0.03);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 8px 12px;
        }
        button[role="tab"] { color: var(--muted) !important; }
        button[role="tab"][aria-selected="true"] {
          color: var(--text) !important;
          border-bottom: 2px solid var(--accent) !important;
        }
        </style>
        """
    else:
        css = """
        <style>
        :root{
          --bg:#ffffff; --panel:#f7f8fb; --text:#0f172a; --muted:#475569;
          --accent:#2563eb; --border:rgba(15,23,42,0.10); --shadow:rgba(15,23,42,0.06);
        }
        .stApp { background: var(--bg); color: var(--text); }
        [data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
        .stCaption, .stMarkdown p { color: var(--muted) !important; }
        .stButton>button {
          background: var(--accent);
          color: #ffffff !important;
          border: 1px solid var(--border);
          border-radius: 12px;
          font-weight: 800;
        }
        [data-testid="stMetric"]{
          background:#ffffff; border:1px solid var(--border); border-radius:14px;
          padding:12px 12px; box-shadow:0 10px 20px var(--shadow);
        }
        [data-testid="stDataFrame"]{
          background:#ffffff; border:1px solid var(--border); border-radius:14px;
          padding:6px; box-shadow:0 10px 20px var(--shadow);
        }
        details { background:#ffffff; border:1px solid var(--border); border-radius:14px; padding:8px 12px; }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


# =========================
# Helpers
# =========================

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

def safe_date_str(x) -> str:
    try:
        return pd.Timestamp(x).date().isoformat()
    except Exception:
        return str(x)

def clean_ticker_list(text: str) -> List[str]:
    raw = [t.strip().upper() for t in text.replace("\n", ",").split(",")]
    out = [t for t in raw if t]
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq

def annualized_return(r: pd.Series) -> float:
    return (1 + r).prod() ** (252 / len(r)) - 1 if len(r) > 0 else np.nan

def annualized_vol(r: pd.Series) -> float:
    return r.std() * np.sqrt(252) if len(r) > 1 else np.nan

def sharpe_ratio(r: pd.Series) -> float:
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

def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def cagr(equity: pd.Series) -> float:
    if equity.empty or len(equity) < 2:
        return np.nan
    days = (equity.index[-1] - equity.index[0]).days
    if days <= 0:
        return np.nan
    years = days / 365.25
    return float(equity.iloc[-1] ** (1 / years) - 1)

def rolling_vol(r: pd.Series, window: int = 63) -> pd.Series:
    return r.rolling(window).std() * np.sqrt(252)

def rolling_sharpe(r: pd.Series, window: int = 63) -> pd.Series:
    sd = r.rolling(window).std()
    mu = r.rolling(window).mean()
    out = (mu / sd) * np.sqrt(252)
    return out.replace([np.inf, -np.inf], np.nan)

def safe_bar_series(s: pd.Series) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================
# Cached data access
# =========================

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


# =========================
# Model logic (unchanged)
# =========================

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

@dataclass
class BacktestOutput:
    gross: pd.Series
    net: pd.Series
    bench: pd.Series
    avg_turn: float
    holdings_changes: pd.DataFrame
    rebal_dates: pd.DatetimeIndex

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
) -> BacktestOutput:
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

    return BacktestOutput(
        gross=gross,
        net=net,
        bench=bench,
        avg_turn=avg_turn,
        holdings_changes=holdings_changes,
        rebal_dates=rebal_dates,
    )

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
        "ann_return": float(annualized_return(p)),
        "ann_vol": float(annualized_vol(p)),
        "sharpe": float(sharpe_ratio(p)),
        "info_ratio_vs_spy": float(info_ratio(p, b)),
    }

def safe_regime_row(label: str, p: pd.Series, b: pd.Series, min_days: int = 30) -> dict:
    if len(p.dropna()) < min_days:
        return {
            "regime": label,
            "days": int(len(p.dropna())),
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "info_ratio_vs_spy": np.nan,
        }
    return regime_stats(label, p, b)

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
    plt.title("Growth of $1")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    return fig

def drawdowns_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0

def make_download_buttons(eq: pd.DataFrame, holdings_changes: pd.DataFrame, regime_export: pd.DataFrame):
    st.subheader("Downloads")
    st.caption("Export results for a write-up or deeper analysis.")

    st.download_button(
        "Equity curve CSV",
        data=eq.to_csv(index=True).encode("utf-8"),
        file_name="equity_curves.csv",
        mime="text/csv",
    )
    st.download_button(
        "Regime summary CSV",
        data=regime_export.to_csv(index=False).encode("utf-8"),
        file_name="regime_summary.csv",
        mime="text/csv",
    )

    try:
        from io import BytesIO
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            holdings_changes.to_excel(writer, index=False, sheet_name="holdings_changes")
        st.download_button(
            "Holdings log (Excel)",
            data=bio.getvalue(),
            file_name="holdings_changes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception:
        st.caption("Excel export needs openpyxl.")


# =========================
# App
# =========================

st.set_page_config(page_title="Atlas Portfolio", layout="wide")

# Sidebar first so theme applies immediately
with st.sidebar:
    st.caption(APP_VERSION)
    beginner = st.toggle("Beginner mode", value=True, help="Beginner = guided + simple. Advanced = full controls + pro theme.")
    advanced = not beginner
    apply_mode_theme(advanced)

    st.subheader("1) Style")
    preset = st.selectbox("Preset", ["Balanced", "Conservative", "Aggressive", "Custom"], index=0)

    st.subheader("2) Date range")
    if beginner:
        start = st.text_input("Start", value="2017-01-01")
        end_in = st.text_input("End (optional)", value="")
    else:
        start = st.text_input("Start (YYYY-MM-DD)", value="2016-01-01")
        end_in = st.text_input("End (blank=today)", value="")
    end = None if end_in.strip() == "" else end_in.strip()

    st.subheader("3) Portfolio")
    if beginner:
        rebalance_label = st.selectbox("Rebalance", ["Monthly", "Quarterly"], index=0)
        top_n = st.slider("Holdings (TOP N)", 10, 50, DEFAULT_TOP_N)
    else:
        rebalance_label = st.selectbox("Rebalance", ["Monthly", "Quarterly", "Weekly"], index=0)
        top_n = st.slider("Holdings (TOP N)", 5, 50, DEFAULT_TOP_N)

    rebalance = {"Monthly": "M", "Quarterly": "Q", "Weekly": "W"}[rebalance_label]

    tc = st.number_input("Costs (bps / 100% turnover)", 0.0, 200.0, float(DEFAULT_TC_BPS_PER_100_TURNOVER), 1.0)

    st.subheader("4) Benchmark + regimes")
    benchmark = st.text_input("Benchmark", value=DEFAULT_BENCHMARK)
    vix_ticker = st.text_input("VIX", value=DEFAULT_VIX_TICKER)

    if beginner:
        vix_smooth = DEFAULT_VIX_SMOOTH_DAYS
        low_q = DEFAULT_LOW_Q
        high_q = DEFAULT_HIGH_Q
        st.caption("Regimes use 3-month VIX smoothing.")
    else:
        vix_smooth = st.number_input("VIX smooth days", 5, 252, int(DEFAULT_VIX_SMOOTH_DAYS), 1)
        low_q = st.slider("Low quantile", 0.05, 0.49, float(DEFAULT_LOW_Q), 0.01)
        high_q = st.slider("High quantile", 0.51, 0.95, float(DEFAULT_HIGH_Q), 0.01)

    st.subheader("5) Factor weights")
    auto_norm = st.toggle("Normalize weights", value=True)

    w = dict(DEFAULT_WEIGHTS)
    if preset == "Conservative":
        w = {"value_pe": 0.15, "profit_roe": 0.15, "growth_rev": 0.15, "risk_vol": 0.30, "risk_de": 0.25}
    elif preset == "Aggressive":
        w = {"value_pe": 0.15, "profit_roe": 0.30, "growth_rev": 0.30, "risk_vol": 0.15, "risk_de": 0.10}

    if preset == "Custom" or advanced:
        w["value_pe"] = st.slider("Valuation (lower P/E)", 0.0, 1.0, float(w["value_pe"]), 0.05)
        w["profit_roe"] = st.slider("Profitability (higher ROE)", 0.0, 1.0, float(w["profit_roe"]), 0.05)
        w["growth_rev"] = st.slider("Growth (higher rev growth)", 0.0, 1.0, float(w["growth_rev"]), 0.05)
        w["risk_vol"] = st.slider("Lower volatility", 0.0, 1.0, float(w["risk_vol"]), 0.05)
        w["risk_de"] = st.slider("Lower debt", 0.0, 1.0, float(w["risk_de"]), 0.05)

    weights = normalize_weights(w) if auto_norm else w
    st.caption(f"Weight sum: {sum(weights.values()):.2f}")

    st.subheader("6) Universe")
    edit_universe = st.toggle("Edit tickers", value=False)
    if edit_universe:
        tickers_text = st.text_area("Tickers", ", ".join(DEFAULT_TICKERS), height=120)
        tickers = clean_ticker_list(tickers_text)
    else:
        tickers = DEFAULT_TICKERS

    st.divider()
    run = st.button("Run", type="primary")

# Header / status
st.title("Atlas Portfolio")
st.caption("Rules-based multi-factor portfolio prototype vs SPY, with VIX regime sensitivity.")

# Guided quick-start banner (beginner only, clean + admissions-friendly)
if not run:
    left, right = st.columns([1.4, 0.6])
    with left:
        st.markdown(
            """
### Quick start
1) Leave defaults  
2) Click **Run**  
3) Open **Holdings** to see how the model picks stocks  
4) Open **VIX regimes** to see calm vs turbulent market behavior
            """.strip()
        )
        with st.expander("What is this doing?"):
            st.markdown(
                """
- Each rebalance date, every stock gets a **score** from factor z-scores  
- The app buys the **top-ranked N** stocks and holds them equally weighted  
- It compares performance to **SPY** and measures turnover + costs  
- Then it splits performance by **VIX regimes** (low vs high volatility)
                """.strip()
            )
    with right:
        st.info("Tip: Use **Conservative** preset to see risk-focused behavior.")
    st.stop()

# Validation
if len(tickers) < 5:
    st.error("Please use at least 5 tickers.")
    st.stop()
if top_n > len(tickers):
    st.error("TOP N cannot exceed number of tickers.")
    st.stop()

# Pipeline with clear diagnostics
with st.spinner("Downloading prices..."):
    prices = download_prices(tickers, start, end)

if prices.empty or prices.shape[0] < 80:
    st.error("Not enough price data. Try a later start date or fewer tickers.")
    st.stop()

missing_px = [t for t in tickers if t not in prices.columns]
if missing_px:
    st.warning(
        "Some tickers had no price data and were dropped: "
        + ", ".join(missing_px[:10])
        + (" ..." if len(missing_px) > 10 else "")
    )

with st.spinner("Downloading benchmark..."):
    bench_df = download_prices([benchmark], start, end)
    bench_px = bench_df[benchmark] if benchmark in bench_df.columns else None

if bench_px is None or bench_px.dropna().empty:
    st.error("Benchmark data not found. Try SPY.")
    st.stop()

with st.spinner("Fetching fundamentals snapshot..."):
    fundamentals = fetch_fundamentals(list(prices.columns)).reindex(prices.columns)

fund_missing_rate = float(fundamentals.isna().mean().mean())
if fund_missing_rate > 0.6:
    st.warning("Many fundamentals are missing (data source limitation). Interpret results cautiously.")

with st.spinner("Running backtest..."):
    out = backtest(
        prices=prices,
        fundamentals=fundamentals,
        bench_px=bench_px,
        top_n=int(top_n),
        rebalance=rebalance,
        mom_lb=int(DEFAULT_MOM_LB),
        vol_lb=int(DEFAULT_VOL_LB),
        weights=weights,
        tc_bps_per_100_turnover=float(tc),
    )

if out.gross.empty or out.net.empty:
    st.error("No returns were produced. Try a later start date (e.g., 2019-01-01) or Monthly rebalance.")
    st.stop()

eq = equity_df(out.gross, out.net, out.bench)
net_equity = eq["Portfolio (Net of Costs)"]

# Status bar (admissions-polish)
status1, status2, status3, status4 = st.columns(4)
status1.metric("Universe", f"{len(prices.columns)} stocks")
status2.metric("Window", f"{safe_date_str(eq.index.min())} → {safe_date_str(eq.index.max())}")
status3.metric("Rebalance", rebalance_label)
status4.metric("Preset", preset)

# Tabs (tight, professional)
tab_overview, tab_risk, tab_holdings, tab_regimes, tab_downloads, tab_method = st.tabs(
    ["Overview", "Risk", "Holdings", "VIX regimes", "Downloads", "Method"]
)

with tab_overview:
    st.subheader("Performance")
    st.pyplot(equity_fig(eq), clear_figure=True)

    stats_all = regime_stats("ALL", out.net, out.bench)
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("CAGR", fmt_pct(cagr(net_equity), 2))
    c2.metric("Max DD", fmt_pct(max_drawdown(net_equity), 2))
    c3.metric("Return", fmt_pct(stats_all["ann_return"], 2))
    c4.metric("Vol", fmt_pct(stats_all["ann_vol"], 2))
    c5.metric("Sharpe", fmt_num(stats_all["sharpe"], 2))
    c6.metric("Info vs SPY", fmt_num(stats_all["info_ratio_vs_spy"], 2))

    st.caption("Net = after turnover-based transaction costs. Gross curve is still shown in downloads.")

with tab_risk:
    st.subheader("Risk lens")
    dd = drawdowns_series(net_equity)
    fig = plt.figure(figsize=(10, 3.8))
    plt.plot(dd.index, dd.values, linewidth=2)
    plt.title("Drawdown (net portfolio)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    rv = rolling_vol(out.net, window=63)
    rs = rolling_sharpe(out.net, window=63)
    colA, colB = st.columns(2)

    with colA:
        fig2 = plt.figure(figsize=(10, 3.8))
        plt.plot(rv.index, rv.values, linewidth=2)
        plt.title("Rolling volatility (63d)")
        plt.xlabel("Date")
        plt.ylabel("Vol")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig2, clear_figure=True)

    with colB:
        fig3 = plt.figure(figsize=(10, 3.8))
        plt.plot(rs.index, rs.values, linewidth=2)
        plt.title("Rolling Sharpe (63d)")
        plt.xlabel("Date")
        plt.ylabel("Sharpe")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig3, clear_figure=True)

    st.caption("Rolling metrics help non-quants see how behavior changes over time.")

with tab_holdings:
    st.subheader("Top holdings by rebalance date")

    momentum_all, vol_all = compute_price_factors(prices, mom_lb=DEFAULT_MOM_LB, vol_lb=DEFAULT_VOL_LB)
    valid_dates = [d for d in out.rebal_dates if d in prices.index]
    if len(valid_dates) > 160:
        valid_dates = valid_dates[-160:]

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

        st.dataframe(scores_full.head(int(top_n))[cols], use_container_width=True, height=560)

        # Simple “what changed” summary for that rebalance date
        hc = out.holdings_changes
        row = hc[hc["rebalance_date"] == pd.Timestamp(chosen).date().isoformat()]
        if not row.empty:
            r0 = row.iloc[0]
            st.caption(
                f"Added: {int(r0['num_added'])} | Removed: {int(r0['num_removed'])} | Turnover: "
                + (fmt_num(float(r0['turnover_fraction']), 3) if r0["turnover_fraction"] is not None else "—")
            )

        with st.expander("Holdings change log"):
            st.dataframe(out.holdings_changes, use_container_width=True, height=360)

        with st.expander("Plain-English guide to the table"):
            st.markdown(
                """
- **score** is the final rank used to pick the portfolio  
- Columns like **value_pe** are raw inputs (after winsorization)  
- Columns like **z_value_pe** are standardized (z-scores) so factors are comparable  
- For **P/E, volatility, and debt**, lower is better (so we multiply by -1 before scoring)  
- **Momentum** is used as a tie-breaker
                """.strip()
            )

with tab_regimes:
    st.subheader("VIX regime sensitivity (Net vs SPY)")

    with st.spinner("Downloading VIX and aligning regimes..."):
        vix_close = download_close(vix_ticker, start, end)
        regimes_raw = compute_vix_regimes(vix_close, int(vix_smooth), float(low_q), float(high_q))

        # Align regimes to return dates and forward fill (fixes missing labels)
        regimes = regimes_raw.reindex(out.net.index).ffill()

        df = pd.DataFrame({"net": out.net, "bench": out.bench, "regime": regimes}).dropna()
        low = df[df["regime"] == "low_vol"]
        high = df[df["regime"] == "high_vol"]

        regime_summary = pd.DataFrame([
            safe_regime_row("ALL", df["net"], df["bench"], min_days=30),
            safe_regime_row("LOW_VIX", low["net"], low["bench"], min_days=30),
            safe_regime_row("HIGH_VIX", high["net"], high["bench"], min_days=30),
        ])

    display = regime_summary.copy()
    display["regime"] = display["regime"].replace({
        "ALL": "All days",
        "LOW_VIX": "Low VIX (calm)",
        "HIGH_VIX": "High VIX (turbulent)",
    })

    st.dataframe(
        display.style.format({
            "ann_return": "{:.2%}",
            "ann_vol": "{:.2%}",
            "sharpe": "{:.2f}",
            "info_ratio_vs_spy": "{:.2f}",
        }, na_rep="—"),
        use_container_width=True
    )

    # Coverage bar chart (layman clarity)
    regime_counts = df["regime"].value_counts().reindex(["low_vol", "mid", "high_vol"]).fillna(0).astype(int)
    regime_counts.index = ["Low VIX", "Mid", "High VIX"]
    st.bar_chart(safe_bar_series(regime_counts))

    if (regime_summary["days"] < 30).any():
        st.caption("If you see —, widen the date range to get ~30+ days inside that regime.")

    with st.expander("What should a layman take away?"):
        st.markdown(
            """
This shows whether the strategy behaves differently when markets are:
- **Calm** (Low VIX)
- **Turbulent** (High VIX)

A good admissions-ready takeaway is **how risk/return changes across regimes**, not just overall performance.
            """.strip()
        )

with tab_downloads:
    # Export-friendly regimes (clean names)
    export_reg = display.copy()

    make_download_buttons(eq, out.holdings_changes, export_reg)

    with st.expander("One-paragraph summary (copy/paste for applications)"):
        st.markdown(
            f"""
I built **Atlas**, an interactive Streamlit research dashboard that constructs a monthly rebalanced,
equal-weighted equity portfolio from a fixed large-cap universe using a weighted factor score
(valuation, profitability, growth, and risk; momentum as a tie-breaker). The app compares results to SPY,
models turnover-based transaction costs, and analyzes performance across market volatility regimes defined
by a smoothed VIX series. Users can explore “top holdings at each rebalance” with the underlying factor table,
download results, and see how strategy behavior changes in calm vs turbulent markets.
            """.strip()
        )

with tab_method:
    st.subheader("Method (short and transparent)")
    st.markdown(
        """
**Portfolio**
- Fixed universe of tickers (editable)
- Rebalances on a schedule (monthly by default)
- Picks **TOP N** by factor score and equally weights them

**Factors**
- Value: trailing P/E (lower is better)
- Profitability: ROE (higher is better)
- Growth: revenue growth (higher is better)
- Risk: realized volatility + debt-to-equity (lower is better)
- Momentum: used as a tie-breaker

**Processing**
- Winsorize inputs (reduce outliers)
- Convert to z-scores cross-sectionally
- Weighted score = sum(weight × z-score)

**Transaction costs**
- Cost applied as: (bps per 100% turnover) × turnover at rebalance

**VIX regimes**
- Smooth VIX with a rolling mean
- Label Low/Mid/High using quantiles
- Compute metrics inside Low vs High regimes

**Limitation**
- Fundamentals come from a current snapshot (yfinance), not point-in-time historical fundamentals.
        """.strip()
    )
