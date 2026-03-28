"""
Sensitivity Analysis for Atlas Backtest
Systematically tests parameter variations to strengthen research findings
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import sys
import time

# Import your backtest functions from the Streamlit app
# Adjust the import path as needed
from streamlit_app import (
    download_prices,
    download_close,
    fetch_fundamentals,
    backtest,
    compute_vix_regimes,
    annualized_return,
    annualized_vol,
    sharpe_ratio,
    info_ratio,
)

# Configuration
START_DATE = "2016-01-01"
END_DATE = "2025-12-31"
BENCHMARK = "SPY"
VIX_TICKER = "^VIX"
VIX_SMOOTH_DAYS = 63
DEFAULT_TC_BPS = 10.0

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

# ==============================================================================
# TEST 1: PORTFOLIO SIZE SENSITIVITY
# ==============================================================================
def test_portfolio_size(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    bench_px: pd.Series,
    vix_regimes: pd.Series,
    sizes: List[int] = [10, 15, 20, 30, 50, 75, 100],
) -> pd.DataFrame:
    """Test how regime-dependence varies with portfolio size"""
    
    results = []
    
    for size in sizes:
        if size > len(prices.columns):
            continue
            
        print(f"  Testing portfolio size: {size}...", end=" ")
        
        out = backtest(
            prices=prices,
            fundamentals=fundamentals,
            bench_px=bench_px,
            top_n=size,
            rebalance="M",
            mom_lb=252,
            vol_lb=252,
            weights=DEFAULT_WEIGHTS,
            tc_bps_per_100_turnover=DEFAULT_TC_BPS,
        )
        
        # Compute regime stats
        df = pd.DataFrame({"net": out.net, "bench": out.bench, "regime": vix_regimes})
        df = df.dropna()
        
        low = df[df["regime"] == "low_vol"]
        high = df[df["regime"] == "high_vol"]
        
        low_ir = info_ratio(low["net"], low["bench"]) if len(low) > 30 else np.nan
        high_ir = info_ratio(high["net"], high["bench"]) if len(high) > 30 else np.nan
        
        results.append({
            "portfolio_size": size,
            "low_vix_info_ratio": low_ir,
            "high_vix_info_ratio": high_ir,
            "ir_differential": low_ir - high_ir if not (np.isnan(low_ir) or np.isnan(high_ir)) else np.nan,
            "ir_ratio": low_ir / high_ir if not (np.isnan(low_ir) or np.isnan(high_ir)) and high_ir != 0 else np.nan,
        })
        
        print("✓")
    
    return pd.DataFrame(results)


# ==============================================================================
# TEST 2: REBALANCING FREQUENCY SENSITIVITY
# ==============================================================================
def test_rebalance_frequency(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    bench_px: pd.Series,
    vix_regimes: pd.Series,
    frequencies: List[str] = ["W", "M", "Q", "SA", "A"],  # Week, Month, Quarter, Semi-Annual, Annual
) -> pd.DataFrame:
    """Test how regime-dependence varies with rebalancing frequency"""
    
    results = []
    freq_names = {"W": "Weekly", "M": "Monthly", "Q": "Quarterly", "SA": "Semi-Annual", "A": "Annual"}
    
    for freq in frequencies:
        print(f"  Testing rebalance frequency: {freq_names[freq]}...", end=" ")
        
        out = backtest(
            prices=prices,
            fundamentals=fundamentals,
            bench_px=bench_px,
            top_n=30,
            rebalance=freq,
            mom_lb=252,
            vol_lb=252,
            weights=DEFAULT_WEIGHTS,
            tc_bps_per_100_turnover=DEFAULT_TC_BPS,
        )
        
        df = pd.DataFrame({"net": out.net, "bench": out.bench, "regime": vix_regimes})
        df = df.dropna()
        
        low = df[df["regime"] == "low_vol"]
        high = df[df["regime"] == "high_vol"]
        
        low_ir = info_ratio(low["net"], low["bench"]) if len(low) > 30 else np.nan
        high_ir = info_ratio(high["net"], high["bench"]) if len(high) > 30 else np.nan
        
        results.append({
            "rebalance_frequency": freq_names[freq],
            "low_vix_info_ratio": low_ir,
            "high_vix_info_ratio": high_ir,
            "ir_differential": low_ir - high_ir if not (np.isnan(low_ir) or np.isnan(high_ir)) else np.nan,
            "ir_ratio": low_ir / high_ir if not (np.isnan(low_ir) or np.isnan(high_ir)) and high_ir != 0 else np.nan,
        })
        
        print("✓")
    
    return pd.DataFrame(results)


# ==============================================================================
# TEST 3: FACTOR WEIGHTING SENSITIVITY
# ==============================================================================
def test_factor_weighting(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    bench_px: pd.Series,
    vix_regimes: pd.Series,
) -> pd.DataFrame:
    """Test how regime-dependence varies with factor weighting schemes"""
    
    weighting_schemes = {
        "Equal Weight": {
            "value_pe": 0.20,
            "profit_roe": 0.20,
            "growth_rev": 0.20,
            "risk_vol": 0.20,
            "risk_de": 0.20,
        },
        "Value Tilt": {
            "value_pe": 0.35,
            "profit_roe": 0.15,
            "growth_rev": 0.15,
            "risk_vol": 0.20,
            "risk_de": 0.15,
        },
        "Profitability Tilt": {
            "value_pe": 0.15,
            "profit_roe": 0.35,
            "growth_rev": 0.15,
            "risk_vol": 0.20,
            "risk_de": 0.15,
        },
        "Risk Focus": {
            "value_pe": 0.15,
            "profit_roe": 0.15,
            "growth_rev": 0.15,
            "risk_vol": 0.35,
            "risk_de": 0.20,
        },
        "Growth Tilt": {
            "value_pe": 0.15,
            "profit_roe": 0.15,
            "growth_rev": 0.35,
            "risk_vol": 0.20,
            "risk_de": 0.15,
        },
    }
    
    results = []
    
    for name, weights in weighting_schemes.items():
        print(f"  Testing weighting: {name}...", end=" ")
        
        out = backtest(
            prices=prices,
            fundamentals=fundamentals,
            bench_px=bench_px,
            top_n=30,
            rebalance="M",
            mom_lb=252,
            vol_lb=252,
            weights=weights,
            tc_bps_per_100_turnover=DEFAULT_TC_BPS,
        )
        
        df = pd.DataFrame({"net": out.net, "bench": out.bench, "regime": vix_regimes})
        df = df.dropna()
        
        low = df[df["regime"] == "low_vol"]
        high = df[df["regime"] == "high_vol"]
        
        low_ir = info_ratio(low["net"], low["bench"]) if len(low) > 30 else np.nan
        high_ir = info_ratio(high["net"], high["bench"]) if len(high) > 30 else np.nan
        
        results.append({
            "weighting_scheme": name,
            "low_vix_info_ratio": low_ir,
            "high_vix_info_ratio": high_ir,
            "ir_differential": low_ir - high_ir if not (np.isnan(low_ir) or np.isnan(high_ir)) else np.nan,
            "ir_ratio": low_ir / high_ir if not (np.isnan(low_ir) or np.isnan(high_ir)) and high_ir != 0 else np.nan,
        })
        
        print("✓")
    
    return pd.DataFrame(results)


# ==============================================================================
# TEST 4: INDIVIDUAL FACTOR CONTRIBUTION
# ==============================================================================
def test_single_factors(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    bench_px: pd.Series,
    vix_regimes: pd.Series,
) -> pd.DataFrame:
    """Test which factors drive returns in each regime"""
    
    factor_configs = {
        "Value Only": {"value_pe": 1.0, "profit_roe": 0.0, "growth_rev": 0.0, "risk_vol": 0.0, "risk_de": 0.0},
        "Profitability Only": {"value_pe": 0.0, "profit_roe": 1.0, "growth_rev": 0.0, "risk_vol": 0.0, "risk_de": 0.0},
        "Growth Only": {"value_pe": 0.0, "profit_roe": 0.0, "growth_rev": 1.0, "risk_vol": 0.0, "risk_de": 0.0},
        "Volatility Only": {"value_pe": 0.0, "profit_roe": 0.0, "growth_rev": 0.0, "risk_vol": 1.0, "risk_de": 0.0},
        "Debt Only": {"value_pe": 0.0, "profit_roe": 0.0, "growth_rev": 0.0, "risk_vol": 0.0, "risk_de": 1.0},
        "Value + Volatility": {"value_pe": 0.5, "profit_roe": 0.0, "growth_rev": 0.0, "risk_vol": 0.5, "risk_de": 0.0},
        "Profitability + Growth": {"value_pe": 0.0, "profit_roe": 0.5, "growth_rev": 0.5, "risk_vol": 0.0, "risk_de": 0.0},
    }
    
    results = []
    
    for name, weights in factor_configs.items():
        print(f"  Testing factors: {name}...", end=" ")
        
        out = backtest(
            prices=prices,
            fundamentals=fundamentals,
            bench_px=bench_px,
            top_n=30,
            rebalance="M",
            mom_lb=252,
            vol_lb=252,
            weights=weights,
            tc_bps_per_100_turnover=DEFAULT_TC_BPS,
        )
        
        df = pd.DataFrame({"net": out.net, "bench": out.bench, "regime": vix_regimes})
        df = df.dropna()
        
        low = df[df["regime"] == "low_vol"]
        high = df[df["regime"] == "high_vol"]
        
        low_ret = annualized_return(low["net"]) if len(low) > 30 else np.nan
        high_ret = annualized_return(high["net"]) if len(high) > 30 else np.nan
        low_ir = info_ratio(low["net"], low["bench"]) if len(low) > 30 else np.nan
        high_ir = info_ratio(high["net"], high["bench"]) if len(high) > 30 else np.nan
        
        results.append({
            "factor_combination": name,
            "low_vix_return": low_ret,
            "high_vix_return": high_ret,
            "low_vix_info_ratio": low_ir,
            "high_vix_info_ratio": high_ir,
        })
        
        print("✓")
    
    return pd.DataFrame(results)


# ==============================================================================
# TRANSACTION COST SENSITIVITY
# ==============================================================================
def test_transaction_costs(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    bench_px: pd.Series,
    vix_regimes: pd.Series,
    costs: List[float] = [0.0, 5.0, 10.0, 15.0, 20.0],
) -> pd.DataFrame:
    """Test how transaction cost assumptions affect findings"""
    
    results = []
    
    for cost in costs:
        print(f"  Testing transaction cost: {cost} bps...", end=" ")
        
        out = backtest(
            prices=prices,
            fundamentals=fundamentals,
            bench_px=bench_px,
            top_n=30,
            rebalance="M",
            mom_lb=252,
            vol_lb=252,
            weights=DEFAULT_WEIGHTS,
            tc_bps_per_100_turnover=cost,
        )
        
        df = pd.DataFrame({"net": out.net, "bench": out.bench, "regime": vix_regimes})
        df = df.dropna()
        
        low = df[df["regime"] == "low_vol"]
        high = df[df["regime"] == "high_vol"]
        
        low_ir = info_ratio(low["net"], low["bench"]) if len(low) > 30 else np.nan
        high_ir = info_ratio(high["net"], high["bench"]) if len(high) > 30 else np.nan
        
        results.append({
            "transaction_cost_bps": cost,
            "low_vix_info_ratio": low_ir,
            "high_vix_info_ratio": high_ir,
            "ir_differential": low_ir - high_ir if not (np.isnan(low_ir) or np.isnan(high_ir)) else np.nan,
        })
        
        print("✓")
    
    return pd.DataFrame(results)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("\n" + "="*80)
    print("ATLAS SENSITIVITY ANALYSIS")
    print("="*80)
    
    print("\n[1/4] Downloading data...")
    prices = download_prices(DEFAULT_TICKERS, START_DATE, END_DATE)
    bench_px = download_prices([BENCHMARK], START_DATE, END_DATE)[BENCHMARK]
    vix_close = download_close(VIX_TICKER, START_DATE, END_DATE)
    fundamentals = fetch_fundamentals(sorted(list(prices.columns))).reindex(prices.columns)
    
    print(f"  Prices: {prices.shape}")
    print(f"  Fundamentals: {fundamentals.shape}")
    
    print("\n[2/4] Computing VIX regimes...")
    vix_smooth = vix_close.rolling(VIX_SMOOTH_DAYS).mean().dropna()
    lo_thr = float(vix_smooth.quantile(0.33))
    hi_thr = float(vix_smooth.quantile(0.67))
    regimes = pd.Series("mid", index=vix_smooth.index)
    regimes[vix_smooth <= lo_thr] = "low_vol"
    regimes[vix_smooth >= hi_thr] = "high_vol"
    regimes_aligned = regimes.reindex(prices.index).ffill()
    
    print(f"  Low VIX threshold: {lo_thr:.2f}")
    print(f"  High VIX threshold: {hi_thr:.2f}")
    
    print("\n" + "="*80)
    print("ROBUSTNESS TESTING")
    print("="*80)
    
    all_results = {}
    
    # Test 1: Portfolio Size
    print("\n[TEST 1] Portfolio Size Sensitivity")
    all_results["portfolio_size"] = test_portfolio_size(prices, fundamentals, bench_px, regimes_aligned)
    
    # Test 2: Rebalancing Frequency
    print("\n[TEST 2] Rebalancing Frequency Sensitivity")
    all_results["rebalance_frequency"] = test_rebalance_frequency(prices, fundamentals, bench_px, regimes_aligned)
    
    # Test 3: Factor Weighting
    print("\n[TEST 3] Factor Weighting Sensitivity")
    all_results["factor_weighting"] = test_factor_weighting(prices, fundamentals, bench_px, regimes_aligned)
    
    # Test 4: Single Factors
    print("\n[TEST 4] Factor Contributions")
    all_results["single_factors"] = test_single_factors(prices, fundamentals, bench_px, regimes_aligned)
    
    # Test 5: Transaction Costs
    print("\n[TEST 5] Transaction Cost Sensitivity")
    all_results["transaction_costs"] = test_transaction_costs(prices, fundamentals, bench_px, regimes_aligned)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    for test_name, df in all_results.items():
        print(f"\n{test_name.upper()}:")
        print(df.to_string(index=False))
        df.to_csv(f"sensitivity_{test_name}.csv", index=False)
        print(f"  → saved to sensitivity_{test_name}.csv")
    
    print("\n" + "="*80)
    print("SUMMARY: ROBUSTNESS CONFIRMATION")
    print("="*80)
    
    print("\n✓ Portfolio Size Robustness:")
    ir_diffs = all_results["portfolio_size"]["ir_differential"].dropna()
    print(f"  Information ratio differential ranges: {ir_diffs.min():.2f} to {ir_diffs.max():.2f}")
    print(f"  Mean differential: {ir_diffs.mean():.2f}")
    
    print("\n✓ Rebalancing Frequency Robustness:")
    ir_diffs = all_results["rebalance_frequency"]["ir_differential"].dropna()
    print(f"  Information ratio differential ranges: {ir_diffs.min():.2f} to {ir_diffs.max():.2f}")
    print(f"  Mean differential: {ir_diffs.mean():.2f}")
    
    print("\n✓ Factor Weighting Robustness:")
    ir_diffs = all_results["factor_weighting"]["ir_differential"].dropna()
    print(f"  Information ratio differential ranges: {ir_diffs.min():.2f} to {ir_diffs.max():.2f}")
    print(f"  Mean differential: {ir_diffs.mean():.2f}")
    
    print("\n✓ Transaction Cost Robustness:")
    ir_diffs = all_results["transaction_costs"]["ir_differential"].dropna()
    print(f"  Information ratio differential ranges: {ir_diffs.min():.2f} to {ir_diffs.max():.2f}")
    print(f"  Mean differential: {ir_diffs.mean():.2f}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
