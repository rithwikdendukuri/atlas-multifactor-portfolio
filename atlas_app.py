# atlas_research_engine.py

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────
# ROBUST STATISTICS
# ─────────────────────────────────────────────────────────────

def robust_zscore(series: pd.Series) -> pd.Series:
    median = series.median()
    mad = np.median(np.abs(series - median))

    if mad == 0 or np.isnan(mad):
        return pd.Series(0, index=series.index)

    return 0.6745 * (series - median) / mad


def winsorize_series(series: pd.Series, lower=0.01, upper=0.99):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


# ─────────────────────────────────────────────────────────────
# HIERARCHICAL IMPUTATION
# ─────────────────────────────────────────────────────────────

def hierarchical_impute(
    df: pd.DataFrame,
    sector_map: Dict[str, str],
) -> pd.DataFrame:

    out = df.copy()
    sectors = pd.Series(sector_map)

    for col in out.columns:

        for sec in sectors.unique():

            members = sectors[sectors == sec].index
            sector_med = out.loc[members, col].median(skipna=True)

            out.loc[members, col] = (
                out.loc[members, col]
                .fillna(sector_med)
            )

        overall_med = out[col].median(skipna=True)
        out[col] = out[col].fillna(overall_med)

    return out


# ─────────────────────────────────────────────────────────────
# SECTOR NEUTRALIZATION
# ─────────────────────────────────────────────────────────────

def sector_neutralize(
    signal: pd.Series,
    sector_map: Dict[str, str],
) -> pd.Series:

    sectors = pd.Series(sector_map)
    adjusted = signal.copy()

    for sec in sectors.unique():

        members = sectors[sectors == sec].index

        adjusted.loc[members] = (
            signal.loc[members]
            - signal.loc[members].mean()
        )

    return adjusted


# ─────────────────────────────────────────────────────────────
# TRUE TURNOVER
# ─────────────────────────────────────────────────────────────

def compute_turnover(
    prev_weights: pd.Series,
    curr_weights: pd.Series,
) -> float:

    aligned = pd.concat(
        [prev_weights, curr_weights],
        axis=1
    ).fillna(0)

    aligned.columns = ["prev", "curr"]

    return 0.5 * np.abs(
        aligned["curr"] - aligned["prev"]
    ).sum()


# ─────────────────────────────────────────────────────────────
# TRANSACTION COST MODEL
# ─────────────────────────────────────────────────────────────

def estimate_transaction_costs(
    turnover: float,
    spread_bps: float = 5,
    impact_bps: float = 10,
) -> float:

    total_bps = spread_bps + impact_bps

    return turnover * total_bps / 10000


# ─────────────────────────────────────────────────────────────
# LIQUIDITY CONSTRAINTS
# ─────────────────────────────────────────────────────────────

def liquidity_filter(
    adv: pd.Series,
    threshold: float = 5_000_000,
) -> pd.Index:

    return adv[adv >= threshold].index


# ─────────────────────────────────────────────────────────────
# RISK PARITY
# ─────────────────────────────────────────────────────────────

def risk_parity_weights(
    vol: pd.Series,
) -> pd.Series:

    inv_vol = 1 / vol.replace(0, np.nan)

    inv_vol = (
        inv_vol
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return inv_vol / inv_vol.sum()


# ─────────────────────────────────────────────────────────────
# HIERARCHICAL RISK PARITY
# ─────────────────────────────────────────────────────────────

def hrp_weights(cov: pd.DataFrame):

    corr = cov.corr()

    dist = np.sqrt((1 - corr) / 2)

    link = linkage(
        squareform(dist.values),
        method="single"
    )

    ivp = 1 / np.diag(cov.values)
    ivp /= ivp.sum()

    return pd.Series(
        ivp,
        index=cov.index
    )


# ─────────────────────────────────────────────────────────────
# INFORMATION COEFFICIENT
# ─────────────────────────────────────────────────────────────

def information_coefficient(
    factor: pd.Series,
    future_returns: pd.Series,
):

    aligned = pd.concat(
        [factor, future_returns],
        axis=1
    ).dropna()

    if len(aligned) < 10:
        return np.nan

    return stats.spearmanr(
        aligned.iloc[:, 0],
        aligned.iloc[:, 1]
    )[0]


# ─────────────────────────────────────────────────────────────
# IC TIME SERIES
# ─────────────────────────────────────────────────────────────

def ic_series(
    factor_matrix: pd.DataFrame,
    future_return_matrix: pd.DataFrame,
):

    out = []

    for dt in factor_matrix.index:

        ic = information_coefficient(
            factor_matrix.loc[dt],
            future_return_matrix.loc[dt]
        )

        out.append(ic)

    return pd.Series(
        out,
        index=factor_matrix.index
    )


# ─────────────────────────────────────────────────────────────
# DECILE ANALYSIS
# ─────────────────────────────────────────────────────────────

def decile_returns(
    signal: pd.Series,
    returns: pd.Series,
):

    df = pd.concat(
        [signal, returns],
        axis=1
    ).dropna()

    df.columns = ["signal", "returns"]

    df["decile"] = pd.qcut(
        df["signal"],
        10,
        labels=False
    )

    return df.groupby(
        "decile"
    )["returns"].mean()


# ─────────────────────────────────────────────────────────────
# LONG SHORT SPREAD
# ─────────────────────────────────────────────────────────────

def long_short_spread(
    signal: pd.Series,
    returns: pd.Series,
):

    df = pd.concat(
        [signal, returns],
        axis=1
    ).dropna()

    df.columns = ["signal", "returns"]

    df["decile"] = pd.qcut(
        df["signal"],
        10,
        labels=False
    )

    top = df[df["decile"] == 9]["returns"].mean()
    bottom = df[df["decile"] == 0]["returns"].mean()

    return top - bottom


# ─────────────────────────────────────────────────────────────
# NEWEY WEST T STAT
# ─────────────────────────────────────────────────────────────

def newey_west_tstat(
    series: pd.Series,
    lag: int = 5,
):

    x = series.dropna().values

    if len(x) < lag + 5:
        return np.nan

    mean = np.mean(x)

    T = len(x)

    gamma0 = np.var(x, ddof=1)

    s = gamma0

    for l in range(1, lag + 1):

        weight = 1 - l / (lag + 1)

        gamma = np.cov(
            x[l:],
            x[:-l]
        )[0, 1]

        s += 2 * weight * gamma

    se = np.sqrt(s / T)

    return mean / se


# ─────────────────────────────────────────────────────────────
# BOOTSTRAP CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────

def bootstrap_cagr(
    returns: pd.Series,
    n_boot=2000,
):

    vals = []

    clean = returns.dropna().values

    for _ in range(n_boot):

        sample = np.random.choice(
            clean,
            size=len(clean),
            replace=True
        )

        equity = np.cumprod(1 + sample)

        years = len(sample) / TRADING_DAYS

        cagr = equity[-1] ** (1 / years) - 1

        vals.append(cagr)

    return np.percentile(
        vals,
        [2.5, 50, 97.5]
    )


# ─────────────────────────────────────────────────────────────
# FACTOR REGRESSION
# ─────────────────────────────────────────────────────────────

def rolling_beta(
    portfolio_returns: pd.Series,
    factor_returns: pd.Series,
    window: int = 126,
):

    out = []

    for i in range(window, len(portfolio_returns)):

        y = portfolio_returns.iloc[i-window:i]
        x = factor_returns.iloc[i-window:i]

        aligned = pd.concat(
            [y, x],
            axis=1
        ).dropna()

        if len(aligned) < 30:
            out.append(np.nan)
            continue

        cov = aligned.iloc[:, 0].cov(
            aligned.iloc[:, 1]
        )

        var = aligned.iloc[:, 1].var()

        out.append(
            cov / var if var != 0 else np.nan
        )

    idx = portfolio_returns.index[window:]

    return pd.Series(out, index=idx)


# ─────────────────────────────────────────────────────────────
# WALK FORWARD VALIDATION
# ─────────────────────────────────────────────────────────────

@dataclass
class WalkForwardSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_years=5,
    test_years=1,
):

    splits = []

    start_idx = 0

    while True:

        train_start = dates[start_idx]

        train_end = train_start + pd.DateOffset(
            years=train_years
        )

        test_end = train_end + pd.DateOffset(
            years=test_years
        )

        if test_end > dates[-1]:
            break

        splits.append(
            WalkForwardSplit(
                train_start=train_start,
                train_end=train_end,
                test_start=train_end,
                test_end=test_end
            )
        )

        start_idx += 252

    return splits


# ─────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────

def annualized_return(r):

    return (
        (1 + r).prod()
        ** (TRADING_DAYS / len(r))
        - 1
    )


def annualized_vol(r):

    return r.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(r):

    return (
        r.mean() / r.std()
    ) * np.sqrt(TRADING_DAYS)


def max_drawdown(equity):

    peak = equity.cummax()

    dd = equity / peak - 1

    return dd.min()


# ─────────────────────────────────────────────────────────────
# ALPHA DECAY
# ─────────────────────────────────────────────────────────────

def alpha_decay(
    factor: pd.Series,
    forward_returns: Dict[int, pd.Series],
):

    out = {}

    for horizon, rets in forward_returns.items():

        out[horizon] = information_coefficient(
            factor,
            rets
        )

    return pd.Series(out)


# ─────────────────────────────────────────────────────────────
# REGIME DETECTION
# ─────────────────────────────────────────────────────────────

def volatility_regime(
    vix: pd.Series,
    smooth=63,
):

    smoothed = vix.rolling(smooth).mean()

    low = smoothed.quantile(0.33)
    high = smoothed.quantile(0.67)

    out = pd.Series(
        "mid",
        index=vix.index
    )

    out[smoothed <= low] = "low_vol"
    out[smoothed >= high] = "high_vol"

    return out


# ─────────────────────────────────────────────────────────────
# FACTOR PIPELINE
# ─────────────────────────────────────────────────────────────

def build_factor_signal(
    raw_factors: pd.DataFrame,
    sector_map: Dict[str, str],
    weights: Dict[str, float],
):

    df = raw_factors.copy()

    df = hierarchical_impute(df, sector_map)

    for col in df.columns:

        df[col] = winsorize_series(df[col])

        df[col] = robust_zscore(df[col])

        df[col] = sector_neutralize(
            df[col],
            sector_map
        )

    composite = pd.Series(
        0,
        index=df.index,
        dtype=float
    )

    for factor, weight in weights.items():

        composite += df[factor] * weight

    return composite


# ─────────────────────────────────────────────────────────────
# POSITION SIZING
# ─────────────────────────────────────────────────────────────

def equal_weight_portfolio(
    tickers: List[str],
):

    w = pd.Series(
        1 / len(tickers),
        index=tickers
    )

    return w


# ─────────────────────────────────────────────────────────────
# FULL RESEARCH SUMMARY
# ─────────────────────────────────────────────────────────────

def research_summary(
    returns: pd.Series,
):

    equity = (1 + returns).cumprod()

    cagr = annualized_return(returns)

    vol = annualized_vol(returns)

    sharpe = sharpe_ratio(returns)

    mdd = max_drawdown(equity)

    ci = bootstrap_cagr(returns)

    return {
        "CAGR": cagr,
        "VOL": vol,
        "SHARPE": sharpe,
        "MAX_DRAWDOWN": mdd,
        "CAGR_CI_LOW": ci[0],
        "CAGR_CI_MEDIAN": ci[1],
        "CAGR_CI_HIGH": ci[2],
    }
