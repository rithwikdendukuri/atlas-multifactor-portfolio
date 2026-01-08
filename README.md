# Atlas Multi-Factor Equity Portfolio

Atlas is a rules-based, multi-factor equity portfolio implemented in Python and deployed as an interactive Streamlit web application.

The project is designed as a **methodological prototype** to study factor-based portfolio construction, turnover effects, and performance behavior across different market volatility regimes.

---

## Overview

- Fixed universe of large-cap U.S. equities
- Equal-weighted TOP-N portfolio selected via cross-sectional factor ranking
- Configurable rebalance frequency (monthly by default)
- Explicit transaction cost modeling
- Benchmark comparison against SPY
- Volatility regime analysis using the VIX
- Interactive web interface for exploration and analysis

---

## Factor Model

At each rebalance date, stocks are ranked using a weighted combination of the following factors:

- **Valuation**: Trailing P/E (lower is better)
- **Profitability**: Return on Equity (ROE)
- **Growth**: Revenue growth
- **Risk**:
  - Realized volatility
  - Debt-to-equity ratio
- **Momentum**: 12-month price momentum (used as a tie-breaker)

Raw factor values are winsorized and transformed into cross-sectional z-scores prior to aggregation.

---

## Portfolio Construction

- TOP-N securities selected by composite factor score
- Equal weighting across holdings
- Rebalanced on a fixed schedule (monthly by default)
- Turnover measured at each rebalance
- Transaction costs applied as a function of portfolio turnover

---

## Performance Evaluation

The application computes and visualizes:

- Cumulative equity curves (gross and net of transaction costs)
- Annualized return and volatility
- Sharpe ratio
- Information ratio versus SPY
- Average portfolio turnover

---

## Volatility Regime Analysis

Portfolio performance is additionally evaluated across market volatility regimes defined using the VIX.

- VIX levels are smoothed using a rolling moving average
- Days are classified into **low**, **mid**, and **high** volatility regimes based on distribution quantiles
- Risk-adjusted performance metrics are computed separately for:
  - All periods
  - Low-volatility regimes
  - High-volatility regimes

This analysis highlights how portfolio behavior and risk characteristics change across different market environments.

---

## Interactive Web Application

The Streamlit dashboard allows users to:

- Adjust factor weights (with optional auto-normalization)
- Modify rebalance frequency, lookback windows, and transaction costs
- View portfolio performance relative to SPY
- Explore **top holdings at each rebalance date**, including:
  - Raw factor inputs
  - Z-scores
  - Composite ranking scores
- Inspect turnover and holdings changes over time
- Analyze performance conditioned on volatility regimes

---

## Implementation Notes

- Core portfolio logic and analytics are implemented in Python using pandas and numpy
- Market data and fundamentals are sourced via `yfinance`
- The application interface is built using Streamlit
- Research logic and UI functionality are intentionally combined for clarity and reproducibility

---

## Limitations

- Fundamental data sourced via `yfinance` reflects current snapshots rather than point-in-time historical fundamentals
- As a result, this project should be interpreted as a **research and learning tool**, not a production-grade backtest
- The strategy is not intended for live trading or investment use

---

## Files

- `atlas_app.py` — interactive Streamlit web application
- `requirements.txt` — Python dependencies

---

## Live Demo

A live version of the application is deployed via Streamlit Community Cloud:

**[Insert Streamlit App URL here]**
