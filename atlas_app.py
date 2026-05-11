from __future__ import annotations

import time
import random
import threading
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import scipy
import scipy.stats as stats
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

try:
    from yfinance.exceptions import YFRateLimitError
except Exception:
    YFRateLimitError = Exception

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 140

TRADING_DAYS = 252

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — CORE RESEARCH ENGINE (ALL FUNCTIONS RETAINED)
# ═══════════════════════════════════════════════════════════════════════════════

def robust_zscore(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = np.median(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return pd.Series(0, index=x.index)
    return 0.6745 * (x - med) / mad


def winsorize_p1(x: pd.Series, l=0.01, u=0.99):
    lo, hi = x.quantile(l), x.quantile(u)
    return x.clip(lo, hi)


def sector_neutralize(signal: pd.Series, sector_map: Dict[str, str]) -> pd.Series:
    s = pd.Series(sector_map)
    out = signal.copy()
    for sec in s.unique():
        idx = s[s == sec].index
        out.loc[idx] = signal.loc[idx] - signal.loc[idx].mean()
    return out


def hierarchical_impute(df: pd.DataFrame, sector_map: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    s = pd.Series(sector_map)
    for col in out.columns:
        for sec in s.unique():
            idx = s[s == sec].index
            out.loc[idx, col] = out.loc[idx, col].fillna(out.loc[idx, col].median())
        out[col] = out[col].fillna(out[col].median())
    return out


def build_factor_matrix(
    raw: pd.DataFrame,
    sector_map: Dict[str, str],
    weights: Dict[str, float],
) -> pd.Series:
    df = hierarchical_impute(raw, sector_map)
    for c in df.columns:
        df[c] = winsorize_p1(df[c])
        df[c] = robust_zscore(df[c])
        df[c] = sector_neutralize(df[c], sector_map)
    score = pd.Series(0.0, index=df.index)
    for k, w in weights.items():
        score += df[k] * w
    return score


def turnover_p1(prev: List[str], curr: List[str]) -> float:
    if prev is None:
        return np.nan
    return 1 - len(set(prev) & set(curr)) / len(curr)


def transaction_cost_p1(turn: float, bps: float = 10) -> float:
    return turn * (bps / 10000)


def equal_weight(top: List[str]) -> pd.Series:
    w = 1 / len(top)
    return pd.Series({t: w for t in top})


def risk_parity(vol: pd.Series) -> pd.Series:
    inv = 1 / vol.replace(0, np.nan)
    inv = inv.fillna(0)
    return inv / inv.sum()


def ann_return(r):
    return (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1


def ann_vol(r):
    return r.std() * np.sqrt(TRADING_DAYS)


def sharpe_p1(r):
    return (r.mean() / r.std()) * np.sqrt(TRADING_DAYS)


def max_dd_p1(equity):
    peak = equity.cummax()
    return (equity / peak - 1).min()


def ic(factor: pd.Series, forward: pd.Series):
    df = pd.concat([factor, forward], axis=1).dropna()
    if len(df) < 10:
        return np.nan
    return stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])[0]


def ic_series(factor_df: pd.DataFrame, forward_df: pd.DataFrame):
    return pd.Series({
        d: ic(factor_df.loc[d], forward_df.loc[d])
        for d in factor_df.index
    })


def deciles(signal: pd.Series, returns: pd.Series):
    df = pd.concat([signal, returns], axis=1).dropna()
    df.columns = ["s", "r"]
    df["d"] = pd.qcut(df["s"], 10, labels=False)
    return df.groupby("d")["r"].mean()


def long_short(signal: pd.Series, returns: pd.Series):
    d = deciles(signal, returns)
    return d.loc[9] - d.loc[0]


def bootstrap_cagr(r: pd.Series, n=1000):
    vals = []
    x = r.dropna().values
    for _ in range(n):
        s = np.random.choice(x, len(x), replace=True)
        eq = np.cumprod(1 + s)
        yrs = len(s) / TRADING_DAYS
        vals.append(eq[-1] ** (1 / yrs) - 1)
    return np.percentile(vals, [2.5, 50, 97.5])


def rolling_beta(p, f, window=126):
    out = []
    for i in range(window, len(p)):
        y = p.iloc[i-window:i]
        x = f.iloc[i-window:i]
        df = pd.concat([y, x], axis=1).dropna()
        if len(df) < 30:
            out.append(np.nan)
            continue
        cov = df.iloc[:, 0].cov(df.iloc[:, 1])
        var = df.iloc[:, 1].var()
        out.append(cov / var if var != 0 else np.nan)
    return pd.Series(out, index=p.index[window:])


def vix_regime(vix: pd.Series, smooth=63):
    v = vix.rolling(smooth).mean()
    lo, hi = v.quantile(0.33), v.quantile(0.67)
    r = pd.Series("mid", index=v.index)
    r[v <= lo] = "low"
    r[v >= hi] = "high"
    return r


def summary(r: pd.Series):
    eq = (1 + r).cumprod()
    return {
        "CAGR": ann_return(r),
        "VOL": ann_vol(r),
        "SHARPE": sharpe_p1(r),
        "MAX_DD": max_dd_p1(eq),
        "BOOTSTRAP": bootstrap_cagr(r)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — CONSTANTS & CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_TICKERS: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "BRK-B", "LLY", "AVGO",
    "JPM", "TSLA", "V", "XOM", "UNH", "MA", "COST", "HD", "PG", "JNJ",
    "ORCL", "MRK", "ABBV", "CVX", "NFLX", "KO", "CRM", "BAC", "WMT", "PEP",
    "AMD", "ADBE", "TMO", "MCD", "QCOM", "NKE", "LIN", "DIS", "CSCO", "ABT",
    "ACN", "VZ", "TXN", "DHR", "INTC", "NEE", "PM", "UPS", "MS", "AMGN",
]

DEFAULT_BENCHMARK = "SPY"
DEFAULT_TOP_N = 30
DEFAULT_REBALANCE = "M"
DEFAULT_MOM_LB = 252
DEFAULT_VOL_LB = 252
DEFAULT_TC_BPS_PER_100_TURNOVER = 10.0
DEFAULT_VIX_TICKER = "^VIX"
DEFAULT_VIX_SMOOTH_DAYS = 63
DEFAULT_LOW_Q = 0.33
DEFAULT_HIGH_Q = 0.67

DEFAULT_WEIGHTS: Dict[str, float] = {
    "value_pe": 0.20,
    "profit_roe": 0.20,
    "growth_rev": 0.20,
    "risk_vol": 0.20,
    "risk_de": 0.20,
}

APP_VERSION = "Atlas"
NAVY_TEXT = "#0b1633"

EDGAR_HEADERS = {"User-Agent": "Atlas-Research rithwik.den@gmail.com"}
FILING_LAG_DAYS = 2
MAX_STALENESS = 460


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — UI THEME
# ═══════════════════════════════════════════════════════════════════════════════

def apply_mode_theme(advanced: bool) -> None:
    if advanced:
        css = f"""
        <style>
        :root{{
          --bg:#0b1020; --panel:#0f1730; --text:#f3f6ff; --muted:#cfd7ff;
          --muted2:#aeb9e8; --accent:#7aa2ff; --border:rgba(255,255,255,0.14);
          --shadow:rgba(0,0,0,0.35);
        }}
        .stApp{{
          background:
            radial-gradient(1200px 600px at 15% 10%, rgba(122,162,255,0.26), transparent 60%),
            radial-gradient(900px 500px at 90% 15%, rgba(0,255,180,0.11), transparent 55%),
            var(--bg);
          color:var(--text);
        }}
        html,body,[class*="css"]{{color:var(--text)!important;-webkit-font-smoothing:antialiased;}}
        .stMarkdown p,.stMarkdown li,.stMarkdown span{{color:var(--text)!important;}}
        .stCaption,.stMarkdown small{{color:rgba(207,215,255,0.82)!important;}}
        [data-testid="stSidebar"]{{background:linear-gradient(180deg,rgba(15,23,48,0.98),rgba(8,12,26,0.98));border-right:1px solid var(--border);}}
        [data-testid="stSidebar"] *{{color:var(--text)!important;}}
        label,.stTextInput label,.stSelectbox label,.stNumberInput label,
        .stSlider label,.stRadio label,.stCheckbox label,.stToggle label{{color:var(--text)!important;font-weight:650;}}
        .stButton>button{{background:linear-gradient(180deg,rgba(122,162,255,1.0),rgba(88,129,255,1.0));color:#0b1020!important;border:1px solid rgba(255,255,255,0.16);border-radius:14px;box-shadow:0 10px 25px var(--shadow);font-weight:800;}}
        [data-testid="stMetric"]{{background:rgba(255,255,255,0.06);border:1px solid var(--border);border-radius:16px;padding:12px 12px;box-shadow:0 10px 25px var(--shadow);}}
        [data-testid="stMetric"] *{{color:var(--text)!important;}}
        [data-testid="stDataFrame"]{{background:rgba(255,255,255,0.04);border:1px solid var(--border);border-radius:16px;padding:6px;box-shadow:0 10px 25px var(--shadow);}}
        details{{background:rgba(255,255,255,0.04);border:1px solid var(--border);border-radius:16px;padding:8px 12px;}}
        button[role="tab"]{{color:rgba(207,215,255,0.75)!important;}}
        button[role="tab"][aria-selected="true"]{{color:var(--text)!important;border-bottom:2px solid var(--accent)!important;}}
        [data-testid="stSidebar"] [data-testid="stTextInput"] input,
        [data-testid="stSidebar"] [data-testid="stTextArea"] textarea,
        [data-testid="stSidebar"] [data-testid="stNumberInput"] input{{background:#ffffff!important;color:{NAVY_TEXT}!important;-webkit-text-fill-color:{NAVY_TEXT}!important;border:1px solid rgba(15,23,42,0.18)!important;border-radius:12px!important;caret-color:{NAVY_TEXT}!important;}}
        [data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder,
        [data-testid="stSidebar"] [data-testid="stTextArea"] textarea::placeholder{{color:rgba(11,22,51,0.55)!important;-webkit-text-fill-color:rgba(11,22,51,0.55)!important;}}
        [data-testid="stSidebar"] [data-baseweb="input"] input,
        [data-testid="stSidebar"] [data-baseweb="textarea"] textarea{{background:#ffffff!important;color:{NAVY_TEXT}!important;-webkit-text-fill-color:{NAVY_TEXT}!important;caret-color:{NAVY_TEXT}!important;}}
        [data-testid="stSidebar"] [data-baseweb="select"] *{{color:{NAVY_TEXT}!important;-webkit-text-fill-color:{NAVY_TEXT}!important;fill:{NAVY_TEXT}!important;}}
        [data-testid="stSidebar"] [data-baseweb="select"] > div{{background:#ffffff!important;border:1px solid rgba(15,23,42,0.18)!important;border-radius:12px!important;}}
        [data-testid="stSidebar"] div[role="combobox"],
        [data-testid="stSidebar"] div[role="combobox"] *{{background:#ffffff!important;color:{NAVY_TEXT}!important;-webkit-text-fill-color:{NAVY_TEXT}!important;fill:{NAVY_TEXT}!important;}}
        div[role="listbox"]{{background:#ffffff!important;border:1px solid rgba(15,23,42,0.18)!important;border-radius:12px!important;}}
        div[role="option"]{{color:{NAVY_TEXT}!important;background:transparent!important;}}
        div[role="option"]:hover{{background:rgba(37,99,235,0.12)!important;}}
        [data-testid="stDownloadButton"] > button{{background:rgba(255,255,255,0.12)!important;color:var(--text)!important;border:1px solid rgba(255,255,255,0.22)!important;border-radius:14px!important;font-weight:800!important;box-shadow:0 10px 22px var(--shadow)!important;}}
        [data-testid="stDownloadButton"] > button *{{color:var(--text)!important;fill:var(--text)!important;}}
        </style>"""
    else:
        css = """
        <style>
        :root{--bg:#ffffff;--panel:#f7f8fb;--text:#0f172a;--muted:#475569;--accent:#2563eb;--border:rgba(15,23,42,0.10);--shadow:rgba(15,23,42,0.06);}
        .stApp{background:var(--bg);color:var(--text);}
        [data-testid="stSidebar"]{background:var(--panel);border-right:1px solid var(--border);}
        .stCaption,.stMarkdown p{color:var(--muted)!important;}
        .stButton>button{background:var(--accent);color:#ffffff!important;border:1px solid var(--border);border-radius:12px;font-weight:800;}
        [data-testid="stMetric"]{background:#ffffff;border:1px solid var(--border);border-radius:14px;padding:12px 12px;box-shadow:0 10px 20px var(--shadow);}
        [data-testid="stDataFrame"]{background:#ffffff;border:1px solid var(--border);border-radius:14px;padding:6px;box-shadow:0 10px 20px var(--shadow);}
        details{background:#ffffff;border:1px solid var(--border);border-radius:14px;padding:8px 12px;}
        [data-testid="stDownloadButton"] > button{background:var(--accent)!important;color:#ffffff!important;border:1px solid var(--border)!important;border-radius:12px!important;font-weight:800!important;}
        </style>"""
    st.markdown(css, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — MATH & FORMAT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return s * 0
    return (s - s.mean(skipna=True)) / sd


def winsorize(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
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
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:.{decimals}f}%"


def fmt_num(x: float, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
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


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def annualized_return(r: pd.Series) -> float:
    return (1 + r).prod() ** (252 / len(r)) - 1 if len(r) > 0 else np.nan


def annualized_vol(r: pd.Series) -> float:
    return r.std() * np.sqrt(252) if len(r) > 1 else np.nan


def sharpe_ratio(r: pd.Series) -> float:
    sd = r.std()
    if len(r) < 2 or sd == 0 or np.isnan(sd):
        return np.nan
    return r.mean() / sd * np.sqrt(252)


def sortino_ratio(r: pd.Series) -> float:
    downside = r[r < 0]
    dd = downside.std()
    if len(r) < 2 or dd == 0 or np.isnan(dd):
        return np.nan
    return r.mean() / dd * np.sqrt(252)


def calmar_ratio(equity: pd.Series) -> float:
    c = cagr(equity)
    mdd = max_drawdown(equity)
    if np.isnan(c) or np.isnan(mdd) or mdd == 0:
        return np.nan
    return c / abs(mdd)


def beta(p: pd.Series, b: pd.Series) -> float:
    df = pd.DataFrame({"p": p, "b": b}).dropna()
    if len(df) < 2:
        return np.nan
    cov = df["p"].cov(df["b"])
    var = df["b"].var()
    if var == 0 or np.isnan(var):
        return np.nan
    return float(cov / var)


def tracking_error(p: pd.Series, b: pd.Series) -> float:
    diff = (p - b).dropna()
    if len(diff) < 2:
        return np.nan
    return diff.std() * np.sqrt(252)


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


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — SEC EDGAR
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def load_cik_map() -> Dict[str, int]:
    url = "https://data.sec.gov/files/company_tickers.json"
    try:
        r = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        return {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}
    except Exception:
        return {}


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_company_facts(cik: int) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    try:
        r = requests.get(url, headers=EDGAR_HEADERS, timeout=20)
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _extract_series(facts: dict, concept: str, unit: str = "USD") -> pd.DataFrame:
    try:
        entries = facts["facts"]["us-gaap"][concept]["units"][unit]
    except KeyError:
        return pd.DataFrame()
    df = pd.DataFrame(entries)
    if df.empty or "filed" not in df.columns:
        return pd.DataFrame()
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df = df[df["form"].isin(["10-K", "10-K/A", "10-Q", "10-Q/A"])]
    df = df.dropna(subset=["end", "filed", "val"])
    return df.sort_values("filed").reset_index(drop=True)


def _pit_flow_ttm(df: pd.DataFrame, as_of: pd.Timestamp) -> float:
    if df.empty:
        return np.nan
    cutoff = as_of - pd.Timedelta(days=FILING_LAG_DAYS)
    avail = df[df["filed"] <= cutoff]
    if avail.empty:
        return np.nan
    annual = avail[avail["form"].isin(["10-K", "10-K/A"])].sort_values("end")
    if annual.empty:
        return np.nan
    last_annual = annual.iloc[-1]
    ann_end = last_annual["end"]
    if (cutoff - ann_end).days > MAX_STALENESS:
        return np.nan
    ann_val = float(last_annual["val"])
    all_qtrs = avail[avail["form"].isin(["10-Q", "10-Q/A"])].copy()
    if all_qtrs.empty:
        return ann_val
    all_qtrs = (
        all_qtrs.sort_values("filed")
        .drop_duplicates(subset=["end"], keep="last")
        .sort_values("end")
    )
    if len(all_qtrs) >= 4 and all_qtrs.iloc[-1]["end"] > ann_end:
        return float(all_qtrs.tail(4)["val"].sum())
    return ann_val


def _pit_balance_sheet(df: pd.DataFrame, as_of: pd.Timestamp) -> float:
    if df.empty:
        return np.nan
    cutoff = as_of - pd.Timedelta(days=FILING_LAG_DAYS)
    avail = df[df["filed"] <= cutoff].sort_values("filed")
    if avail.empty:
        return np.nan
    return float(avail.iloc[-1]["val"])


def _pit_revenue_growth(rev_df: pd.DataFrame, as_of: pd.Timestamp) -> float:
    if rev_df.empty:
        return np.nan
    cutoff = as_of - pd.Timedelta(days=FILING_LAG_DAYS)
    annual = (
        rev_df[rev_df["form"].isin(["10-K", "10-K/A"]) & (rev_df["filed"] <= cutoff)]
        .sort_values("end")
    )
    if len(annual) < 2:
        return np.nan
    curr = float(annual.iloc[-1]["val"])
    prev = float(annual.iloc[-2]["val"])
    if prev == 0 or np.isnan(prev):
        return np.nan
    return curr / prev - 1.0


_REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueGoodsNet",
]
_SHARES_CONCEPTS = [
    "CommonStockSharesOutstanding",
    "CommonStockSharesIssued",
]
_DEBT_CONCEPTS = [
    "LongTermDebt",
    "LongTermDebtAndCapitalLeaseObligations",
    "DebtAndCapitalLeaseObligations",
]


def _first_nonempty(facts: dict, concepts: List[str], unit: str = "USD") -> pd.DataFrame:
    for c in concepts:
        df = _extract_series(facts, c, unit)
        if not df.empty:
            return df
    return pd.DataFrame()


def _pit_snapshot_one(
    ticker: str,
    facts: dict,
    as_of: pd.Timestamp,
    price: float,
) -> Dict[str, float]:
    nan_row = {"trailingPE": np.nan, "ROE": np.nan,
               "revenueGrowth": np.nan, "debtToEquity": np.nan}
    if not facts or not np.isfinite(price) or price <= 0:
        return nan_row
    ni_df = _extract_series(facts, "NetIncomeLoss")
    ni_ttm = _pit_flow_ttm(ni_df, as_of)
    rev_df = _first_nonempty(facts, _REVENUE_CONCEPTS)
    rev_growth = _pit_revenue_growth(rev_df, as_of)
    eq_df = _extract_series(facts, "StockholdersEquity")
    equity = _pit_balance_sheet(eq_df, as_of)
    roe = np.nan
    if np.isfinite(ni_ttm) and np.isfinite(equity) and equity != 0:
        roe = ni_ttm / equity
    shares_df = _first_nonempty(facts, _SHARES_CONCEPTS, unit="shares")
    shares = _pit_balance_sheet(shares_df, as_of)
    pe = np.nan
    if np.isfinite(ni_ttm) and np.isfinite(shares) and shares > 0:
        eps = ni_ttm / shares
        if eps > 0:
            pe = price / eps
    debt_df = _first_nonempty(facts, _DEBT_CONCEPTS)
    debt = _pit_balance_sheet(debt_df, as_of)
    de = np.nan
    if np.isfinite(debt) and np.isfinite(equity) and equity != 0:
        de = debt / equity
    return {"trailingPE": pe, "ROE": roe,
            "revenueGrowth": rev_growth, "debtToEquity": de}


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def build_pit_fund_matrix(
    tickers: Tuple[str, ...],
    rebal_dates: Tuple[pd.Timestamp, ...],
    prices: pd.DataFrame,
    budget_s: float = 90.0,
) -> Dict[str, pd.DataFrame]:
    tickers_list = list(tickers)
    t0 = time.monotonic()
    cik_map = load_cik_map()
    facts_by_ticker: Dict[str, dict] = {}
    n = len(tickers_list)
    progress = st.progress(0, text="Downloading SEC EDGAR filings…")
    for i, tick in enumerate(tickers_list):
        if (time.monotonic() - t0) > budget_s:
            for t in tickers_list[i:]:
                facts_by_ticker[t] = {}
            break
        cik = cik_map.get(tick.upper())
        facts_by_ticker[tick] = fetch_company_facts(cik) if cik else {}
        time.sleep(0.12)
        progress.progress((i + 1) / n, text=f"EDGAR: {tick} ({i+1}/{n})")
    progress.empty()
    result: Dict[str, pd.DataFrame] = {}
    for date in rebal_dates:
        date_ts = pd.Timestamp(date)
        rows = []
        for tick in tickers_list:
            px_val = np.nan
            if date_ts in prices.index and tick in prices.columns:
                px_val = float(prices.loc[date_ts, tick])
            row = _pit_snapshot_one(
                ticker=tick,
                facts=facts_by_ticker.get(tick, {}),
                as_of=date_ts,
                price=px_val,
            )
            row["ticker"] = tick
            rows.append(row)
        df = pd.DataFrame(rows).set_index("ticker")
        result[date_ts.isoformat()] = df
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — PRICE DATA
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=60 * 60)
def download_prices(tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    px = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df
    px = px.dropna(how="all")
    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])
    return px


@st.cache_data(show_spinner=False, ttl=60 * 60)
def download_close(ticker: str, start: str, end: Optional[str]) -> pd.Series:
    df = yf.download([ticker], start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        s = df["Close"][ticker]
    else:
        s = df["Close"] if "Close" in df.columns else df.squeeze()
    return s.dropna()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — FACTOR SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def make_scores_pit(
    date: pd.Timestamp,
    momentum_row: pd.Series,
    vol_row: pd.Series,
    pit_fund_matrix: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
) -> pd.DataFrame:
    date_key = pd.Timestamp(date).isoformat()
    fund_df = pit_fund_matrix.get(date_key, pd.DataFrame())
    universe = momentum_row.index if not fund_df.empty else momentum_row.index
    raw = pd.DataFrame(index=universe)
    if not fund_df.empty:
        raw["value_pe"] = fund_df["trailingPE"].reindex(universe)
        raw["profit_roe"] = fund_df["ROE"].reindex(universe)
        raw["growth_rev"] = fund_df["revenueGrowth"].reindex(universe)
        raw["risk_de"] = fund_df["debtToEquity"].reindex(universe)
    else:
        raw["value_pe"] = np.nan
        raw["profit_roe"] = np.nan
        raw["growth_rev"] = np.nan
        raw["risk_de"] = np.nan
    raw["risk_vol"] = vol_row.reindex(universe)
    raw["mom_12m"] = momentum_row.reindex(universe)
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


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — PORTFOLIO & BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

def compute_price_factors(prices: pd.DataFrame, mom_lb: int, vol_lb: int):
    rets = prices.pct_change()
    momentum = prices / prices.shift(mom_lb) - 1
    vol = rets.rolling(vol_lb).std() * np.sqrt(252)
    return momentum, vol


def portfolio_turnover(prev: Optional[List[str]], curr: List[str]) -> float:
    if prev is None:
        return np.nan
    return 1 - len(set(prev) & set(curr)) / len(curr)


def apply_costs(r: pd.Series, turn: float, tc_bps: float) -> pd.Series:
    if np.isnan(turn):
        return r
    cost = (tc_bps / 10_000) * turn
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
    pit_fund_matrix: Dict[str, pd.DataFrame],
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
    holdings_log = []
    for d0, d1 in zip(rebal_dates[:-1], rebal_dates[1:]):
        scores = make_scores_pit(
            date=d0,
            momentum_row=momentum.loc[d0],
            vol_row=vol.loc[d0],
            pit_fund_matrix=pit_fund_matrix,
            weights=weights,
        )
        picks = scores.sort_values(["score", "mom_z"], ascending=False).head(top_n).index.tolist()
        added = sorted(set(picks) - set(prev)) if prev else sorted(picks)
        removed = sorted(set(prev) - set(picks)) if prev else []
        stayed = sorted(set(picks) & set(prev)) if prev else []
        turn = portfolio_turnover(prev, picks)
        prev = picks
        turns.append(turn)
        holdings_log.append({
            "rebalance_date": pd.Timestamp(d0).date().isoformat(),
            "num_held": len(picks),
            "num_added": len(added),
            "num_removed": len(removed),
            "turnover_fraction": None if np.isnan(turn) else float(turn),
            "added": ", ".join(added),
            "removed": ", ".join(removed),
            "stayed": ", ".join(stayed),
            "held": ", ".join(sorted(picks)),
        })
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
    return BacktestOutput(
        gross=gross,
        net=net,
        bench=bench,
        avg_turn=float(np.nanmean(turns)) if turns else np.nan,
        holdings_changes=pd.DataFrame(holdings_log),
        rebal_dates=rebal_dates,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — VIX REGIME
# ═══════════════════════════════════════════════════════════════════════════════

def compute_vix_regimes(
    vix_close: pd.Series,
    smooth_days: int,
    low_q: float,
    high_q: float,
    split_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.Series, float, float]:
    vix_smooth = vix_close.rolling(smooth_days).mean().dropna()
    if split_date is not None:
        in_sample = vix_smooth[vix_smooth.index < pd.Timestamp(split_date)]
        if len(in_sample) >= 30:
            lo_thr = float(in_sample.quantile(low_q))
            hi_thr = float(in_sample.quantile(high_q))
        else:
            lo_thr = float(vix_smooth.quantile(low_q))
            hi_thr = float(vix_smooth.quantile(high_q))
    else:
        lo_thr = float(vix_smooth.quantile(low_q))
        hi_thr = float(vix_smooth.quantile(high_q))
    regime = pd.Series("mid", index=vix_smooth.index)
    regime[vix_smooth <= lo_thr] = "low_vol"
    regime[vix_smooth >= hi_thr] = "high_vol"
    return regime, lo_thr, hi_thr


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
        "sortino": float(sortino_ratio(p)),
        "info_ratio": float(info_ratio(p, b)),
    }


def safe_regime_row(label: str, p: pd.Series, b: pd.Series, min_days: int = 30) -> dict:
    if len(p.dropna()) < min_days:
        return {"regime": label, "days": int(len(p.dropna())),
                "ann_return": np.nan, "ann_vol": np.nan,
                "sharpe": np.nan, "sortino": np.nan, "info_ratio": np.nan}
    return regime_stats(label, p, b)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — EQUITY CURVE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Atlas Quant Research", layout="wide")

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
    rebalance_label = st.selectbox(
        "Rebalance",
        ["Monthly", "Quarterly"] if beginner else ["Monthly", "Quarterly", "Weekly"],
        index=0,
    )
    rebalance = {"Monthly": "ME", "Quarterly": "QE", "Weekly": "W"}[rebalance_label]
    top_n = st.slider("Holdings", 10 if beginner else 5, 50, DEFAULT_TOP_N)
    tc = st.number_input("Trading costs (bps / 100% turnover)", 0.0, 200.0,
                         float(DEFAULT_TC_BPS_PER_100_TURNOVER), 1.0)
    st.markdown("**Benchmark**")
    benchmark = st.text_input("Ticker", value=DEFAULT_BENCHMARK)
    st.markdown("**VIX**")
    vix_ticker = st.text_input("VIX ticker", value=DEFAULT_VIX_TICKER)
    if beginner:
        vix_smooth = DEFAULT_VIX_SMOOTH_DAYS
        low_q = DEFAULT_LOW_Q
        high_q = DEFAULT_HIGH_Q
        use_wf = True
        split_date_in = None
    else:
        vix_smooth = st.number_input("Smoothing (days)", 5, 252, int(DEFAULT_VIX_SMOOTH_DAYS), 1)
        low_q = st.slider("Low quantile", 0.05, 0.49, float(DEFAULT_LOW_Q), 0.01)
        high_q = st.slider("High quantile", 0.51, 0.95, float(DEFAULT_HIGH_Q), 0.01)
        use_wf = st.toggle("Walk-forward VIX thresholds", value=True,
                           help="Compute VIX regime thresholds from in-sample data only, "
                                "then apply to the full period. Prevents look-ahead bias.")
        split_date_in = st.text_input("In-sample / OOS split date", value="2021-01-01",
                                      help="Thresholds are estimated on data before this date.") if use_wf else None
    st.markdown("**Lookback windows**")
    if advanced:
        mom_lb = st.number_input("Momentum lookback (days)", 63, 504, int(DEFAULT_MOM_LB), 21)
        vol_lb = st.number_input("Volatility lookback (days)", 63, 504, int(DEFAULT_VOL_LB), 21)
    else:
        mom_lb = DEFAULT_MOM_LB
        vol_lb = DEFAULT_VOL_LB
    st.markdown("**Weights**")
    auto_norm = st.toggle("Normalize weights", value=True)
    w = dict(DEFAULT_WEIGHTS)
    if profile == "Conservative":
        w = {"value_pe": 0.15, "profit_roe": 0.15, "growth_rev": 0.15,
             "risk_vol": 0.30, "risk_de": 0.25}
    elif profile == "Aggressive":
        w = {"value_pe": 0.15, "profit_roe": 0.30, "growth_rev": 0.30,
             "risk_vol": 0.15, "risk_de": 0.10}
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
st.info(
    "**Data sources:** Price data from Yahoo Finance (historical, point-in-time). "
    "Fundamental data (P/E, ROE, revenue growth, D/E) from **SEC EDGAR**, using only "
    "filings submitted on or before each rebalance date (+ 2-day buffer). "
    "VIX regime thresholds use walk-forward design: estimated on the in-sample period only.\n\n"
    "**Remaining limitations:** (1) The ticker universe is user-defined and static — companies "
    "that were delisted or went bankrupt during the backtest are excluded if not in the list "
    "(survivorship bias). (2) XBRL coverage is sparse before ~2012. "
    "(3) This is not a live trading system."
)

if not run:
    st.info("Choose settings in the sidebar, then click **Run**.")
    st.stop()

if len(tickers) < 5:
    st.error("Add at least 5 tickers.")
    st.stop()
if top_n > len(tickers):
    st.error("Holdings can't exceed the ticker count.")
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
    st.warning("Dropped (no price data): " + ", ".join(missing_px[:12]) +
               (" …" if len(missing_px) > 12 else ""))

with st.spinner(f"Benchmark ({benchmark})"):
    bench_df = download_prices([benchmark], start, end)
    bench_px = bench_df[benchmark] if benchmark in bench_df.columns else None
if bench_px is None or bench_px.dropna().empty:
    st.error("Benchmark not found. Try a liquid ETF (SPY, QQQ, IWM) or a valid ticker.")
    st.stop()

_all_rebal = prices.resample(rebalance).last().index
_all_rebal = _all_rebal[_all_rebal.isin(prices.index)]
rebal_dates_for_fund = tuple(_all_rebal.tolist())

with st.spinner("Fundamentals — SEC EDGAR (first run takes ~1 min; cached after that)"):
    pit_fund_matrix = build_pit_fund_matrix(
        tickers=tuple(sorted(prices.columns.tolist())),
        rebal_dates=rebal_dates_for_fund,
        prices=prices,
        budget_s=120.0,
    )

fund_vals = pd.concat(pit_fund_matrix.values()) if pit_fund_matrix else pd.DataFrame()
if not fund_vals.empty:
    missing_rate = float(fund_vals.isna().mean().mean())
    if missing_rate > 0.6:
        st.warning(
            f"EDGAR fundamental coverage is low ({100*(1-missing_rate):.0f}% of cells filled). "
            "This is normal for non-US tickers or tickers with little XBRL history before 2013. "
            "Missing values score as average (z=0) and don't distort rankings."
        )

with st.spinner("Backtest"):
    out = backtest(
        prices=prices,
        pit_fund_matrix=pit_fund_matrix,
        bench_px=bench_px,
        top_n=int(top_n),
        rebalance=rebalance,
        mom_lb=int(mom_lb),
        vol_lb=int(vol_lb),
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

tab_overview, tab_holdings, tab_regimes, tab_risk, tab_downloads, tab_method, tab_factor_engine = st.tabs(
    ["Overview", "Holdings", "VIX", "Risk", "Downloads", "Method", "Factor Engine"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab: Overview
# ═══════════════════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════════════════
# Tab: Holdings
# ═══════════════════════════════════════════════════════════════════════════════
with tab_holdings:
    st.subheader("Holdings")
    st.caption("Pick a rebalance date to see factor scores and raw inputs.")
    momentum_all, vol_all = compute_price_factors(prices, mom_lb=mom_lb, vol_lb=vol_lb)
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
        scores_full = make_scores_pit(
            date=chosen,
            momentum_row=momentum_all.loc[chosen],
            vol_row=vol_all.loc[chosen],
            pit_fund_matrix=pit_fund_matrix,
            weights=weights,
        ).sort_values(["score", "mom_z"], ascending=False)
        cols = [
            "score",
            "value_pe", "profit_roe", "growth_rev", "risk_vol", "risk_de", "mom_12m",
            "z_value_pe", "z_profit_roe", "z_growth_rev", "z_risk_vol", "z_risk_de", "z_mom_12m",
        ]
        st.dataframe(scores_full.head(int(top_n))[cols], use_container_width=True, height=560)
        with st.expander("Change log"):
            st.dataframe(out.holdings_changes, use_container_width=True, height=360)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab: VIX regimes
# ═══════════════════════════════════════════════════════════════════════════════
with tab_regimes:
    st.subheader("VIX regimes")
    st.caption("Same metrics, split by volatility conditions from VIX.")
    if use_wf:
        if split_date_in and split_date_in.strip():
            wf_split = pd.Timestamp(split_date_in.strip())
        else:
            idx = prices.index
            cutoff = idx[int(len(idx) * 0.70)]
            wf_split = pd.Timestamp(cutoff)
    else:
        wf_split = None
    with st.spinner("VIX"):
        vix_close = download_close(vix_ticker, start, end)
        regimes_raw, lo_thr, hi_thr = compute_vix_regimes(
            vix_close=vix_close,
            smooth_days=int(vix_smooth),
            low_q=float(low_q),
            high_q=float(high_q),
            split_date=wf_split,
        )
        regimes = regimes_raw.reindex(out.net.index).ffill()
        df_reg = pd.DataFrame({
            "net": out.net,
            "bench": out.bench,
            "regime": regimes,
        }).dropna()
        low_mask = df_reg["regime"] == "low_vol"
        high_mask = df_reg["regime"] == "high_vol"
        regime_summary = pd.DataFrame([
            safe_regime_row("All days", df_reg["net"], df_reg["bench"], min_days=30),
            safe_regime_row("Low VIX", df_reg.loc[low_mask, "net"], df_reg.loc[low_mask, "bench"], min_days=30),
            safe_regime_row("High VIX", df_reg.loc[high_mask, "net"], df_reg.loc[high_mask, "bench"], min_days=30),
        ])
    if wf_split:
        st.caption(
            f"Walk-forward: VIX thresholds estimated on data before "
            f"**{wf_split.date()}** (in-sample). "
            f"Low ≤ {lo_thr:.1f}, High ≥ {hi_thr:.1f}. "
            f"Thresholds applied to the full period without re-fitting."
        )
    else:
        st.caption(f"Full-sample thresholds: Low ≤ {lo_thr:.1f}, High ≥ {hi_thr:.1f}.")
    st.dataframe(
        regime_summary.style.format(
            {"ann_return": "{:.2%}", "ann_vol": "{:.2%}",
             "sharpe": "{:.2f}", "sortino": "{:.2f}", "info_ratio": "{:.2f}"},
            na_rep="—",
        ),
        use_container_width=True,
    )
    regime_counts = (
        df_reg["regime"]
        .value_counts()
        .reindex(["low_vol", "mid", "high_vol"])
        .fillna(0).astype(int)
    )
    regime_counts.index = ["Low", "Mid", "High"]
    st.bar_chart(regime_counts)
    if wf_split:
        st.subheader("In-sample vs out-of-sample")
        is_mask = df_reg.index < wf_split
        oos_mask = df_reg.index >= wf_split
        is_summary = pd.DataFrame([
            safe_regime_row("IS All", df_reg.loc[is_mask, "net"], df_reg.loc[is_mask, "bench"], 30),
            safe_regime_row("IS Low VIX", df_reg.loc[is_mask & low_mask, "net"], df_reg.loc[is_mask & low_mask, "bench"], 30),
            safe_regime_row("IS High VIX", df_reg.loc[is_mask & high_mask, "net"], df_reg.loc[is_mask & high_mask, "bench"], 30),
            safe_regime_row("OOS All", df_reg.loc[oos_mask, "net"], df_reg.loc[oos_mask, "bench"], 30),
            safe_regime_row("OOS Low VIX", df_reg.loc[oos_mask & low_mask, "net"], df_reg.loc[oos_mask & low_mask, "bench"], 30),
            safe_regime_row("OOS High VIX", df_reg.loc[oos_mask & high_mask, "net"], df_reg.loc[oos_mask & high_mask, "bench"], 30),
        ])
        st.dataframe(
            is_summary.style.format(
                {"ann_return": "{:.2%}", "ann_vol": "{:.2%}",
                 "sharpe": "{:.2f}", "sortino": "{:.2f}", "info_ratio": "{:.2f}"},
                na_rep="—",
            ),
            use_container_width=True,
        )
        st.caption(
            "IS = in-sample (thresholds estimated here). "
            "OOS = out-of-sample (thresholds applied, not re-estimated). "
            "OOS regime persistence is the key research finding."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab: Risk
# ═══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.subheader("Risk metrics")
    st.caption("Additional risk and performance statistics.")
    b = out.bench
    p = out.net
    risk_data = {
        "Metric": [
            "CAGR", "Annualized return", "Annualized volatility",
            "Max drawdown", "Sharpe ratio", "Sortino ratio",
            "Calmar ratio", "Beta to benchmark", "Tracking error",
            "Information ratio", "Average turnover",
        ],
        "Value": [
            cagr(net_equity),
            annualized_return(p),
            annualized_vol(p),
            max_drawdown(net_equity),
            sharpe_ratio(p),
            sortino_ratio(p),
            calmar_ratio(net_equity),
            beta(p, b),
            tracking_error(p, b),
            info_ratio(p, b),
            out.avg_turn,
        ],
    }
    risk_df = pd.DataFrame(risk_data)
    def fmt_risk_row(row):
        pct_metrics = {"CAGR", "Annualized return", "Annualized volatility", "Max drawdown"}
        if row["Metric"] in pct_metrics:
            return fmt_pct(row["Value"], 2)
        if row["Metric"] == "Average turnover":
            return fmt_num(row["Value"], 4)
        return fmt_num(row["Value"], 2)
    risk_df["Display"] = risk_df.apply(fmt_risk_row, axis=1)
    st.dataframe(
        risk_df[["Metric", "Display"]].rename(columns={"Display": "Value"}),
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Transaction cost sensitivity")
    st.caption("How net performance changes with different cost assumptions.")
    tc_sens = []
    for tc_test in [0, 5, 10, 25, 50, 100]:
        out_test = backtest(
            prices=prices,
            pit_fund_matrix=pit_fund_matrix,
            bench_px=bench_px,
            top_n=int(top_n),
            rebalance=rebalance,
            mom_lb=int(mom_lb),
            vol_lb=int(vol_lb),
            weights=weights,
            tc_bps_per_100_turnover=float(tc_test),
        )
        eq_t = equity_df(out_test.gross, out_test.net, out_test.bench, bench_name=benchmark)
        net_eq_t = eq_t["Portfolio (Net)"]
        tc_sens.append({
            "Cost (bps/100% turn)": tc_test,
            "CAGR": cagr(net_eq_t),
            "Sharpe": sharpe_ratio(out_test.net),
            "Max DD": max_drawdown(net_eq_t),
        })
    tc_df = pd.DataFrame(tc_sens)
    st.dataframe(
        tc_df.style.format(
            {"CAGR": "{:.2%}", "Sharpe": "{:.2f}", "Max DD": "{:.2%}"},
            na_rep="—",
        ),
        use_container_width=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab: Downloads
# ═══════════════════════════════════════════════════════════════════════════════
with tab_downloads:
    st.subheader("Downloads")
    st.download_button(
        "Equity curves (CSV)",
        data=eq.to_csv(index=True).encode("utf-8"),
        file_name="equity_curves.csv",
        mime="text/csv",
    )
    st.download_button(
        "Regime summary (CSV)",
        data=regime_summary.to_csv(index=False).encode("utf-8"),
        file_name="regime_summary.csv",
        mime="text/csv",
    )
    st.download_button(
        "Risk metrics (CSV)",
        data=risk_df.to_csv(index=False).encode("utf-8"),
        file_name="risk_metrics.csv",
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

# ═══════════════════════════════════════════════════════════════════════════════
# Tab: Method
# ═══════════════════════════════════════════════════════════════════════════════
with tab_method:
    st.subheader("Method")
    st.markdown(
        f"""
**Construction**
- User-defined ticker universe (editable in sidebar)
- Rebalance on a schedule (monthly / quarterly / weekly)
- Rank stocks by composite score; hold top N equal-weighted

**Signals**
- Value: trailing P/E computed from SEC EDGAR (lower is better)
- Profit: ROE from SEC EDGAR (higher is better)
- Growth: year-on-year revenue growth from SEC EDGAR (higher is better)
- Risk: annualized realized volatility from price history (lower is better)
- Risk: debt/equity from SEC EDGAR (lower is better)
- Momentum: 12-month trailing return — tie-breaker only, not in weighted score

**Scoring**
- Winsorize at 1st / 99th percentile
- Cross-sectional z-scores
- Weighted sum; missing z-scores filled with 0 (treated as cross-sectional average)

**Point-in-time fundamental data**
- Source: SEC EDGAR XBRL filings (free, no API key required)
- Only filings with `filed` date ≤ rebalance date − 2 days are used
- P/E is computed as price ÷ (trailing-12-month net income ÷ shares outstanding)
  using only earnings data that had been publicly filed at each rebalance date
- Concept fallback chains handle the fact that companies use different XBRL tags

**VIX regime classification — walk-forward design**
- VIX quantile thresholds are estimated on the **in-sample period only** (before split date)
- The same thresholds are applied to the full period without re-estimation
- This prevents the regime labels from incorporating future VIX distribution information

**Costs**
- Turnover-based cost applied on rebalance days
- Default: {DEFAULT_TC_BPS_PER_100_TURNOVER} bps per 100% turnover

**Risk metrics**
- Sharpe, Sortino, Calmar, beta, tracking error, information ratio

**Remaining limitations**
1. Static ticker universe — companies delisted during the backtest are excluded
   if not in the user's list (survivorship bias). Use a dynamic index constituent
   file to address this.
2. XBRL coverage is sparse before ~2012; fundamental signals will be mostly NaN
   for early periods.
3. This is not a live trading system. Equal-weighting ignores liquidity and capacity.
4. Benchmark: **{benchmark}**.
        """.strip()
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab: Factor Engine (Part 1 legacy view + unique analytics)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_factor_engine:
    st.subheader("Factor Engine — Part 1 Legacy View")
    st.caption("Original Part 1 diagnostics and additional analytics.")

    # ── Part 1 original main-page UI (retained exactly) ──
    eq_p1 = equity_df(out.gross, out.net, out.bench, benchmark)
    st.subheader("Equity Curve")
    st.line_chart(eq_p1)

    net_eq_p1 = eq_p1["Portfolio (Net)"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CAGR", fmt_pct(cagr(net_eq_p1)))
    col2.metric("Sharpe", fmt_num(sharpe_ratio(out.net)))
    col3.metric("Max DD", fmt_pct(max_drawdown(net_eq_p1)))
    col4.metric("Turnover", fmt_num(out.avg_turn))

    st.subheader("Holdings Evolution")
    st.dataframe(out.holdings_changes, use_container_width=True)

    st.subheader("Drawdown")
    dd_p1 = drawdown_series(net_eq_p1)
    st.line_chart(dd_p1)

    st.subheader("Benchmark Comparison")
    st.line_chart(pd.DataFrame({
        "Portfolio": net_eq_p1,
        "Benchmark": eq_p1[f"Benchmark ({benchmark})"]
    }))

    st.subheader("Notes")
    st.markdown(
        """
- All fundamentals are point-in-time via SEC EDGAR filings
- Momentum and volatility are rolling signals
- Portfolio is equal-weighted among top-ranked stocks
- Costs are applied proportional to turnover
        """
    )

    st.divider()
    st.subheader("Part 1 Unique Analytics")

    # Bootstrap CAGR
    st.markdown("**Bootstrap CAGR** (2.5 / 50 / 97.5 percentiles)")
    bs = bootstrap_cagr(out.net)
    st.write({"p2.5": f"{bs[0]:.2%}", "p50": f"{bs[1]:.2%}", "p97.5": f"{bs[2]:.2%}"})

    # Part 1 summary object
    st.markdown("**Part 1 Summary Object**")
    st.json(summary(out.net))

    # Rolling beta (portfolio vs benchmark)
    st.markdown("**Rolling Beta** (126-day window)")
    port_equity = (1 + out.net).cumprod()
    bench_equity = (1 + out.bench).cumprod()
    rb = rolling_beta(port_equity, bench_equity, window=126)
    if not rb.empty:
        st.line_chart(rb)
    else:
        st.caption("Not enough data for rolling beta.")

    # VIX regime using Part 1 function
    st.markdown("**VIX Regime (Part 1 function)**")
    try:
        vix_s = download_close(vix_ticker, start, end)
        if not vix_s.empty:
            regime_p1 = vix_regime(vix_s, smooth=63)
            regime_counts_p1 = regime_p1.value_counts()
            st.bar_chart(regime_counts_p1)
        else:
            st.caption("No VIX data available.")
    except Exception as e:
        st.caption(f"VIX regime error: {e}")
