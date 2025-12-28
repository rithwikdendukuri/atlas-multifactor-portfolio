# Rules-Based Multi-Factor Equity Portfolio

This repository contains the code used to evaluate a transparent,
rules-based multi-factor equity portfolio. The objective of the
project is to test whether systematic factor exposure, implemented
using publicly available data and simple portfolio construction
rules, can outperform a passive market benchmark.

This repository accompanies a research paper and is intended to
support methodological transparency and reproducibility.

---

## Research Question

Can a transparent, rules-based multi-factor equity portfolio
constructed from publicly available data outperform a passive
market index on a risk-adjusted basis?

---

## Methodology Overview

- **Universe:** Fixed list of 50 large-cap U.S. equities  
- **Sample period:** January 2016 to present  
- **Rebalance frequency:** Monthly  
- **Portfolio size:** Top 30 ranked securities  
- **Portfolio weighting:** Implicitly equal-weighted (weights drift
  between rebalances)  
- **Benchmark:** SPDR S&P 500 ETF (SPY)

### Factors Used

**Price-based factors**
- Momentum: trailing 12-month return
- Volatility: trailing 12-month annualized standard deviation of returns

**Fundamental factors**
- Value: trailing price-to-earnings (P/E) ratio
- Profitability: return on equity (ROE)
- Growth: revenue growth
- Leverage risk: debt-to-equity ratio

All factors are winsorized at the 1st and 99th percentiles and
standardized using cross-sectional z-scores. Factors where lower
values are preferred (P/E, volatility, debt-to-equity) are inverted
so that higher scores consistently represent more favorable
characteristics.

Each factor receives equal weight in the composite score. Securities
are ranked by composite score, with momentum used as a secondary
tie-breaker.

---

## Data

Price and fundamental data are sourced from Yahoo Finance using the
`yfinance` Python package.

Fundamental variables are retrieved once and treated as static over
the sample period. As a result, fundamental information is not
point-in-time and may introduce look-ahead bias. Accordingly,
results involving fundamental factors should be interpreted as a
methodological prototype rather than a fully implementable historical
trading strategy.

---

## Files

- `rules_based_etf.py` — Main backtesting script  
- `prices.csv` — Historical adjusted price data  
- `fundamentals.csv` — Fundamental factor snapshot  
- `equity_curves.csv` — Cumulative return series for portfolio and benchmark  
- `holdings_log.csv` — Portfolio holdings at each rebalance date  

---

## Reproducibility

To reproduce the results:

```bash
pip install -r requirements.txt
python rules_based_etf.py
