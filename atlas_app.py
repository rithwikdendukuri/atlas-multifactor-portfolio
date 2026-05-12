from __future__ import annotations

import io
import math
import time
import warnings
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

try:
    import scipy.optimize as optimize
except Exception:
    optimize = None

try:
    import scipy.stats as stats
except Exception:
    stats = None

try:
    from yfinance.exceptions import YFRateLimitError
except Exception:
    YFRateLimitError = Exception

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 140

TRADING_DAYS = 252
HLZ_TSTAT_THRESHOLD = 3.0
FF3_DAILY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"


# =============================================================================
# PART 1 - CORE RESEARCH ENGINE
# =============================================================================

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
        idx = s[s == sec].index.intersection(signal.index)
        if len(idx) > 0:
            out.loc[idx] = signal.loc[idx] - signal.loc[idx].mean()
    return out


def hierarchical_impute(df: pd.DataFrame, sector_map: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    s = pd.Series(sector_map)
    for col in out.columns:
        for sec in s.unique():
            idx = s[s == sec].index.intersection(out.index)
            if len(idx) > 0:
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
