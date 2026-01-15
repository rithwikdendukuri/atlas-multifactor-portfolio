from __future__ import annotations

import warnings
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 140

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

APP_VERSION = "Atlas"

NAVY_TEXT = "#0b1633"


def apply_mode_theme(advanced: bool) -> None:
    if advanced:
        css = f"""
        <style>
        :root{{
          --bg:#0b1020;
          --panel:#0f1730;
          --text:#f3f6ff;
          --muted:#cfd7ff;
          --muted2:#aeb9e8;
          --accent:#7aa2ff;
          --border:rgba(255,255,255,0.14);
          --shadow:rgba(0,0,0,0.35);
        }}

        .stApp{{
          background:
            radial-gradient(1200px 600px at 15% 10%, rgba(122,162,255,0.26), transparent 60%),
            radial-gradient(900px 500px at 90% 15%, rgba(0,255,180,0.11), transparent 55%),
            var(--bg);
          color:var(--text);
        }}

        html, body, [class*="css"]{{
          color:var(--text) !important;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }}

        .stMarkdown p, .stMarkdown li, .stMarkdown span {{ color: var(--text) !important; }}
        .stCaption, .stMarkdown small {{ color: rgba(207,215,255,0.82) !important; }}

        [data-testid="stSidebar"]{{
          background: linear-gradient(180deg, rgba(15,23,48,0.98), rgba(8,12,26,0.98));
          border-right:1px solid var(--border);
        }}
        [data-testid="stSidebar"] *{{ color:var(--text) !important; }}

        label, .stTextInput label, .stSelectbox label, .stNumberInput label,
        .stSlider label, .stRadio label, .stCheckbox label, .stToggle label{{
          color: var(--text) !important;
          font-weight: 650;
        }}

        .stButton>button{{
          background: linear-gradient(180deg, rgba(122,162,255,1.0), rgba(88,129,255,1.0));
          color:#0b1020 !important;
          border:1px solid rgba(255,255,255,0.16);
          border-radius:14px;
          box-shadow:0 10px 25px var(--shadow);
          font-weight:800;
        }}

        [data-testid="stMetric"]{{
          background: rgba(255,255,255,0.06);
          border:1px solid var(--border);
          border-radius:16px;
          padding:12px 12px;
          box-shadow:0 10px 25px var(--shadow);
        }}
        [data-testid="stMetric"] *{{ color: var(--text) !important; }}

        [data-testid="stDataFrame"]{{
          background: rgba(255,255,255,0.04);
          border:1px solid var(--border);
          border-radius:16px;
          padding:6px;
          box-shadow:0 10px 25px var(--shadow);
        }}

        details{{
          background: rgba(255,255,255,0.04);
          border:1px solid var(--border);
          border-radius:16px;
          padding:8px 12px;
        }}

        button[role="tab"]{{ color: rgba(207,215,255,0.75) !important; }}
        button[role="tab"][aria-selected="true"]{{
          color: var(--text) !important;
          border-bottom:2px solid var(--accent) !important;
        }}

        /* -----------------------------------------------------------------
           FIX: sidebar inputs show white box + white text in some versions.
           Force NAVY text inside all input boxes / selects / number inputs.
           ----------------------------------------------------------------- */

        [data-testid="stSidebar"] [data-testid="stTextInput"] input,
        [data-testid="stSidebar"] [data-testid="stTextArea"] textarea,
        [data-testid="stSidebar"] [data-testid="stNumberInput"] input{{
          background: #ffffff !important;
          color: {NAVY_TEXT} !important;
          -webkit-text-fill-color: {NAVY_TEXT} !important;
          border: 1px solid rgba(15,23,42,0.18) !important;
          border-radius: 12px !important;
          caret-color: {NAVY_TEXT} !important;
        }}

        [data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder,
        [data-testid="stSidebar"] [data-testid="stTextArea"] textarea::placeholder{{
          color: rgba(11,22,51,0.55) !important;
          -webkit-text-fill-color: rgba(11,22,51,0.55) !important;
        }}

        [data-testid="stSidebar"] [data-baseweb="input"] input,
        [data-testid="stSidebar"] [data-baseweb="textarea"] textarea{{
          background: #ffffff !important;
          color: {NAVY_TEXT} !important;
          -webkit-text-fill-color: {NAVY_TEXT} !important;
          caret-color: {NAVY_TEXT} !important;
        }}

        [data-testid="stSidebar"] [data-baseweb="select"] *{{
          color: {NAVY_TEXT} !important;
          -webkit-text-fill-color: {NAVY_TEXT} !important;
          fill: {NAVY_TEXT} !important;
        }}

        [data-testid="stSidebar"] [data-baseweb="select"] > div{{
          background: #ffffff !important;
          border: 1px solid rgba(15,23,42,0.18) !important;
          border-radius: 12px !important;
        }}

        [data-testid="stSidebar"] div[role="combobox"],
        [data-testid="stSidebar"] div[role="combobox"] *{{
          background: #ffffff !important;
          color: {NAVY_TEXT} !important;
          -webkit-text-fill-color: {NAVY_TEXT} !important;
          fill: {NAVY_TEXT} !important;
        }}

        div[role="listbox"]{{
          background: #ffffff !important;
          border: 1px solid rgba(15,23,42,0.18) !important;
          border-radius: 12px !important;
        }}
        div[role="option"]{{
          color: {NAVY_TEXT} !important;
          background: transparent !important;
        }}
        div[role="option"]:hover{{
          background: rgba(37,99,235,0.12) !important;
        }}

        [data-testid="stDownloadButton"] > button{{
          background: rgba(255,255,255,0.12) !important;
          color: var(--text) !important;
          border: 1px solid rgba(255,255,255,0.22) !important;
          border-radius: 14px !important;
          font-weight: 800 !important;
          box-shadow: 0 10px 22px var(--shadow) !important;
        }}
        [data-testid="stDownloadButton"] > button *{{
          color: var(--text) !important;
          fill: var(--text) !important;
        }}
        </style>
        """
    else:
        css = """
        <style>
        :root{
          --bg:#ffffff; --panel:#f7f8fb; --text:#0f172a; --muted:#475569;
          --accent:#2563eb; --border:rgba(15,23,42,0.10); --shadow:rgba(15,23,42,0.06);
        }
        .stApp{ background: var(--bg); color: var(--text); }
        [data-testid="stSidebar"]{ background: var(--panel); border-right:1px solid var(--border); }
        .stCaption, .stMarkdown p{ color: var(--muted) !important; }
        .stButton>button{
          background: var(--accent);
          color:#ffffff !important;
          border:1px solid var(--border);
          border-radius:12px;
          font-weight:800;
        }
        [data-testid="stMetric"]{
          background:#ffffff;
          border:1px solid var(--border);
          border-radius:14px;
          padding:12px 12px;
          box-shadow:0 10px 20px var(--shadow);
        }
        [data-testid="stDataFrame"]{
          background:#ffffff;
          border:1px solid var(--border);
          border-radius:14px;
          padding:6px;
          box-shadow:0 10px 20px var(--shadow);
        }
        details{ background:#ffffff; border:1px solid var(--border); border-radius:14px; padding:8px 12px; }
        [data-testid="stDownloadButton"] > button{
          background: var(--accent) !important;
          color:#ffffff !important;
          border:1px solid var(--border) !important;
          border-radius:12px !important;
          font-weight:800 !important;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


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


def safe_float(x) -> float:
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan


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


# -----------------------------
# Robust caching for downloads
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60)  # 1h
def download_prices(tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    px = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df
    px = px.dropna(how="all")
    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])
    return px


@st.cache_data(show_spinner=False, ttl=60 * 60)  # 1h
def download_close(ticker: str, start: str, end: Optional[str]) -> pd.Series:
    df = yf.download([ticker], start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        s = df["Close"][ticker]
    else:
        s = df["Close"] if "Close" in df.columns else df.squeeze()
    return s.dropna()


# -----------------------------
# Robust fundamentals fetching
# -----------------------------
try:
    from yfinance.exceptions import YFRateLimitError
except Exception:
    YFRateLimitError = Exception


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)  # 24h
def fetch_fundamentals(tickers: List[str]) -> pd.DataFrame:
   import threading

try:
    from yfinance.exceptions import YFRateLimitError
except Exception:
    YFRateLimitError = Exception


def _get_info_with_timeout(ticker: str, timeout_s: float = 2.0) -> Dict:
    """
    Run yf.Ticker(t).info in a thread so we can enforce a hard timeout.
    Returns {} on timeout or error.
    """
    out: Dict = {}
    err: List[Exception] = []

    def worker():
        nonlocal out
        try:
            out = yf.Ticker(ticker).info or {}
        except Exception as e:
            err.append(e)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    th.join(timeout=timeout_s)

    # If still alive, we timed out
    if th.is_alive():
        return {}

    # If it errored, treat as missing
    if err:
        return {}

    return out


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)  # cache 24h
def fetch_fundamentals(tickers: List[str]) -> pd.DataFrame:
    """
    Snapshot fundamentals from yfinance with:
    - hard per-ticker timeouts
    - a hard total time budget (won't block the app)
    - graceful degradation (returns NaNs instead of crashing)
    """
    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t and t.strip()]))

    rows = []
    failures = 0
    rate_limited = 0
    timeouts = 0

    # Hard budget: fundamentals step will never run longer than this.
    # Tune: 8–15 seconds is a good range on Streamlit Cloud.
    BUDGET_S = 12.0
    t0 = time.monotonic()

    for t in tickers:
        if (time.monotonic() - t0) > BUDGET_S:
            # Budget exceeded: fill remaining tickers with NaNs and exit fast
            failures += (len(tickers) - len(rows))
            break

        # tiny jitter helps avoid bursting in shared environments
        time.sleep(0.02 + random.random() * 0.03)

        info: Dict = {}
        try:
            info = _get_info_with_timeout(t, timeout_s=2.0)
        except YFRateLimitError:
            rate_limited += 1
            info = {}
        except Exception:
            info = {}

        if not info:
            # distinguish timeout vs other failure (best-effort)
            # (we can’t perfectly detect timeout here, but empty after timeout wrapper is a good proxy)
            timeouts += 1

        if not info:
            failures += 1

        rows.append({
            "ticker": t,
            "trailingPE": safe_float(info.get("trailingPE")),
            "ROE": safe_float(info.get("returnOnEquity")),
            "revenueGrowth": safe_float(info.get("revenueGrowth")),
            "debtToEquity": safe_float(info.get("debtToEquity")),
        })

    # If we broke early due to budget, pad remaining tickers so index lines up
    if len(rows) < len(tickers):
        remaining = tickers[len(rows):]
        for t in remaining:
            rows.append({
                "ticker": t,
                "trailingPE": np.nan,
                "ROE": np.nan,
                "revenueGrowth": np.nan,
                "debtToEquity": np.nan,
            })

    df = pd.DataFrame(rows).set_index("ticker")

    df.attrs["failures"] = int(failures)
    df.attrs["rate_limited"] = int(rate_limited)
    df.attrs["timeouts"] = int(timeouts)
    df.attrs["n"] = int(len(tickers))
    df.attrs["elapsed_s"] = float(time.monotonic() - t0)
    df.attrs["budget_s"] = float(BUDGET_S)

    return df
 df


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
        turns.append(turn)

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
        "info_ratio": float(info_ratio(p, b)),
    }


def safe_regime_row(label: str, p: pd.Series, b: pd.Series, min_days: int = 30) -> dict:
    if len(p.dropna()) < min_days:
        return {"regime": label, "days": int(len(p.dropna())), "ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "info_ratio": np.nan}
    return regime_stats(label, p, b)


def equity_df(gross: pd.Series, net: pd.Series, bench: pd.Series, bench_name: str) -> pd.DataFrame:
    return pd.DataFrame({
        "Portfolio (Gross)": (1 + gross).cumprod(),
        "Portfolio (Net)": (1 + net).cumprod(),
        f"Benchmark ({bench_name})": (1 + bench).cumprod(),
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


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


st.set_page_config(page_title="Atlas", layout="wide")

with st.sidebar:
    st.caption(APP_VERSION)

    beginner = st.toggle("Beginner mode", value=True)
    advanced = not beginner
    apply_mode_theme(advanced)

    st.subheader("Settings")

    profile = st.selectbox("Profile", ["Balanced", "Conservative", "Aggressive", "Custom"], index=0)

    st.markdown("**Dates**")
    start = st.text_input("Start", value="2017-01-01" if beginner else "2016-01-01")
    end_in = st.text_input("End (optional)", value="")
    end = None if end_in.strip() == "" else end_in.strip()

    st.markdown("**Portfolio**")
    rebalance_label = st.selectbox("Rebalance", ["Monthly", "Quarterly"] if beginner else ["Monthly", "Quarterly", "Weekly"], index=0)
    rebalance = {"Monthly": "M", "Quarterly": "Q", "Weekly": "W"}[rebalance_label]
    top_n = st.slider("Holdings", 10 if beginner else 5, 50, DEFAULT_TOP_N)
    tc = st.number_input("Trading costs (bps / 100% turnover)", 0.0, 200.0, float(DEFAULT_TC_BPS_PER_100_TURNOVER), 1.0)

    st.markdown("**Benchmark**")
    benchmark = st.text_input("Ticker", value=DEFAULT_BENCHMARK)

    st.markdown("**VIX**")
    vix_ticker = st.text_input("VIX ticker", value=DEFAULT_VIX_TICKER)

    if beginner:
        vix_smooth = DEFAULT_VIX_SMOOTH_DAYS
        low_q = DEFAULT_LOW_Q
        high_q = DEFAULT_HIGH_Q
    else:
        vix_smooth = st.number_input("Smoothing (days)", 5, 252, int(DEFAULT_VIX_SMOOTH_DAYS), 1)
        low_q = st.slider("Low quantile", 0.05, 0.49, float(DEFAULT_LOW_Q), 0.01)
        high_q = st.slider("High quantile", 0.51, 0.95, float(DEFAULT_HIGH_Q), 0.01)

    st.markdown("**Weights**")
    auto_norm = st.toggle("Normalize weights", value=True)

    w = dict(DEFAULT_WEIGHTS)
    if profile == "Conservative":
        w = {"value_pe": 0.15, "profit_roe": 0.15, "growth_rev": 0.15, "risk_vol": 0.30, "risk_de": 0.25}
    elif profile == "Aggressive":
        w = {"value_pe": 0.15, "profit_roe": 0.30, "growth_rev": 0.30, "risk_vol": 0.15, "risk_de": 0.10}

    if profile == "Custom" or advanced:
        w["value_pe"] = st.slider("Value (P/E)", 0.0, 1.0, float(w["value_pe"]), 0.05)
        w["profit_roe"] = st.slider("Profit (ROE)", 0.0, 1.0, float(w["profit_roe"]), 0.05)
        w["growth_rev"] = st.slider("Growth (rev)", 0.0, 1.0, float(w["growth_rev"]), 0.05)
        w["risk_vol"] = st.slider("Risk (vol)", 0.0, 1.0, float(w["risk_vol"]), 0.05)
        w["risk_de"] = st.slider("Risk (debt)", 0.0, 1.0, float(w["risk_de"]), 0.05)

    weights = normalize_weights(w) if auto_norm else w
    st.caption(f"Weight sum: {sum(weights.values()):.2f}")

    st.markdown("**Universe**")
    edit_universe = st.toggle("Edit tickers", value=False)
    if edit_universe:
        tickers_text = st.text_area("Tickers", ", ".join(DEFAULT_TICKERS), height=120)
        tickers = clean_ticker_list(tickers_text)
    else:
        tickers = DEFAULT_TICKERS

    st.divider()
    run = st.button("Run", type="primary")


st.title("Atlas")
st.caption("A rules-based stock ranking model with a backtest and a volatility view.")

st.subheader("What this is")
st.markdown(
    """
Atlas lets you test a simple idea: rank stocks with a few measurable signals, hold the top names, and rebalance on a schedule.

- Pick a universe of stocks (or use the default list)
- Choose how often to rebalance and how many holdings to keep
- Compare the result to any benchmark ticker you want
- See how results change when volatility is low vs high (using VIX)

This uses historical data and is for learning and research.
    """.strip()
)

if not run:
    st.info("Choose settings in the sidebar, then click **Run**.")
    st.stop()

if len(tickers) < 5:
    st.error("Add at least 5 tickers.")
    st.stop()

if top_n > len(tickers):
    st.error("Holdings can’t exceed the ticker count.")
    st.stop()

benchmark = benchmark.strip().upper()
vix_ticker = vix_ticker.strip().upper()

with st.spinner("Prices"):
    prices = download_prices(tickers, start, end)

if prices.empty or prices.shape[0] < 80:
    st.error("Not enough price data for this window. Try a later start date.")
    st.stop()

missing_px = [t for t in tickers if t not in prices.columns]
if missing_px:
    st.warning("Dropped (no price data): " + ", ".join(missing_px[:12]) + (" ..." if len(missing_px) > 12 else ""))

with st.spinner(f"Benchmark ({benchmark})"):
    bench_df = download_prices([benchmark], start, end)
    bench_px = bench_df[benchmark] if benchmark in bench_df.columns else None

if bench_px is None or bench_px.dropna().empty:
    st.error("Benchmark not found. Try a liquid ETF (SPY, QQQ, IWM) or a valid ticker.")
    st.stop()

with st.spinner("Fundamentals"):
    # sorted() stabilizes the cache key (reduces cache misses)
    fundamentals = fetch_fundamentals(sorted(list(prices.columns))).reindex(prices.columns)

failures = int(fundamentals.attrs.get("failures", 0))
rate_limited = int(fundamentals.attrs.get("rate_limited", 0))
n_f = int(fundamentals.attrs.get("n", max(1, len(fundamentals))))
fund_missing_rate = float(fundamentals.isna().mean().mean())

if rate_limited > 0:
    st.warning("Yahoo Finance is rate-limiting requests right now. Fundamentals may be incomplete; try again later.")
elif failures > 0 and failures / max(1, n_f) > 0.15:
    st.warning("Some fundamentals could not be fetched today. Results may be slightly noisier.")

if fund_missing_rate > 0.6:
    st.warning("Many fundamentals are missing from the data source. Treat results as directional.")

with st.spinner("Backtest"):
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
    st.error("No returns produced. Try Monthly rebalancing or a later start date.")
    st.stop()

eq = equity_df(out.gross, out.net, out.bench, bench_name=benchmark)
net_equity = eq["Portfolio (Net)"]

s1, s2, s3, s4 = st.columns(4)
s1.metric("Universe", f"{len(prices.columns)}")
s2.metric("Range", f"{safe_date_str(eq.index.min())} → {safe_date_str(eq.index.max())}")
s3.metric("Rebalance", rebalance_label)
s4.metric("Profile", profile)

tab_overview, tab_holdings, tab_regimes, tab_downloads, tab_method = st.tabs(
    ["Overview", "Holdings", "VIX", "Downloads", "Method"]
)

with tab_overview:
    st.subheader("Overview")
    st.pyplot(equity_fig(eq), clear_figure=True)

    stats_all = regime_stats("All", out.net, out.bench)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("CAGR", fmt_pct(cagr(net_equity), 2))
    c2.metric("Max drawdown", fmt_pct(max_drawdown(net_equity), 2))
    c3.metric("Annual return", fmt_pct(stats_all["ann_return"], 2))
    c4.metric("Annual vol", fmt_pct(stats_all["ann_vol"], 2))
    c5.metric("Sharpe", fmt_num(stats_all["sharpe"], 2))
    c6.metric(f"Info vs {benchmark}", fmt_num(stats_all["info_ratio"], 2))

    st.subheader("Drawdown (net)")
    dd = drawdown_series(net_equity)
    fig = plt.figure(figsize=(10, 3.8))
    plt.plot(dd.index, dd.values, linewidth=2)
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

with tab_holdings:
    st.subheader("Holdings")
    st.caption("Pick a rebalance date to see ranks and inputs.")

    momentum_all, vol_all = compute_price_factors(prices, mom_lb=DEFAULT_MOM_LB, vol_lb=DEFAULT_VOL_LB)
    valid_dates = [d for d in out.rebal_dates if d in prices.index]
    if len(valid_dates) > 160:
        valid_dates = valid_dates[-160:]

    if not valid_dates:
        st.warning("Not enough rebalance dates to display holdings.")
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

        with st.expander("Change log"):
            st.dataframe(out.holdings_changes, use_container_width=True, height=360)

with tab_regimes:
    st.subheader("VIX regimes")
    st.caption("Same metrics, split by volatility conditions from VIX.")

    with st.spinner("VIX"):
        vix_close = download_close(vix_ticker, start, end)
        regimes_raw = compute_vix_regimes(vix_close, int(vix_smooth), float(low_q), float(high_q))
        regimes = regimes_raw.reindex(out.net.index).ffill()

        df = pd.DataFrame({"net": out.net, "bench": out.bench, "regime": regimes}).dropna()
        low = df[df["regime"] == "low_vol"]
        high = df[df["regime"] == "high_vol"]

        regime_summary = pd.DataFrame([
            safe_regime_row("All days", df["net"], df["bench"], min_days=30),
            safe_regime_row("Low VIX", low["net"], low["bench"], min_days=30),
            safe_regime_row("High VIX", high["net"], high["bench"], min_days=30),
        ])

    st.dataframe(
        regime_summary.style.format(
            {"ann_return": "{:.2%}", "ann_vol": "{:.2%}", "sharpe": "{:.2f}", "info_ratio": "{:.2f}"},
            na_rep="—",
        ),
        use_container_width=True,
    )

    regime_counts = df["regime"].value_counts().reindex(["low_vol", "mid", "high_vol"]).fillna(0).astype(int)
    regime_counts.index = ["Low", "Mid", "High"]
    st.bar_chart(regime_counts)

with tab_downloads:
    st.subheader("Downloads")

    st.download_button(
        "Equity curves (CSV)",
        data=eq.to_csv(index=True).encode("utf-8"),
        file_name="equity_curves.csv",
        mime="text/csv",
    )

    reg_csv = regime_summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Regime summary (CSV)",
        data=reg_csv,
        file_name="regime_summary.csv",
        mime="text/csv",
    )

    try:
        from io import BytesIO
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            out.holdings_changes.to_excel(writer, index=False, sheet_name="holdings_changes")
        st.download_button(
            "Holdings log (Excel)",
            data=bio.getvalue(),
            file_name="holdings_changes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception:
        st.caption("Excel export needs openpyxl.")

with tab_method:
    st.subheader("Method")
    st.markdown(
        f"""
**Construction**
- Fixed universe (editable)
- Rebalance on a schedule
- Rank stocks, take top **N**, equal-weight

**Signals**
- Value: trailing P/E (lower is better)
- Profit: ROE (higher is better)
- Growth: revenue growth (higher is better)
- Risk: volatility + debt-to-equity (lower is better)
- Momentum: tie-breaker

**Scoring**
- Winsorize
- Cross-sectional z-scores
- Weighted sum

**Costs**
- Turnover-based cost applied on rebalance days

**Notes**
- Fundamentals are a current snapshot from yfinance (not point-in-time historical fundamentals)
- Benchmark is **{benchmark}**
        """.strip()
    )
