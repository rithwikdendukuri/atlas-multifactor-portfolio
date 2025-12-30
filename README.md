# Atlas Multi-factor ETF Portfolio

This repository contains a rules-based equity portfolio prototype that applies a multi-factor ranking framework to a fixed universe of large-cap U.S. stocks. The strategy combines price-based and fundamental signals, rebalances monthly, and evaluates performance relative to the S&P 500 using both gross and net-of-cost returns.

The project is designed as a methodological prototype, emphasizing transparency, discipline, and awareness of empirical finance limitations rather than performance optimization.

---

## Overview

- Universe: 50 large-cap U.S. equities  
- Portfolio size: Top 30 ranked stocks  
- Rebalance frequency: Monthly  .
- Weighting: Equal-weighted with drift between rebalances  
- Benchmark: SPDR S&P 500 ETF (SPY)  
- Backtest period: January 2016 – present  

The model ranks securities using a combination of momentum, valuation, profitability, growth, and risk factors. Portfolio construction and evaluation follow explicit, rule-based procedures.

---

## Factor Framework

At each rebalance date, stocks are ranked cross-sectionally using the following factors:

Price-based factors:
- Momentum: Trailing 12-month total return (252 trading days). No skip-month adjustment is applied in this prototype.
- Volatility: Annualized standard deviation of daily returns over the prior 252 trading days  

Fundamental factors:
- Value: Trailing price-to-earnings (P/E) ratio (lower preferred)  
- Profitability: Return on equity (ROE) (higher preferred)  
- Growth: Revenue growth (higher preferred)  
- Leverage risk: Debt-to-equity ratio (lower preferred)  

Fundamental data are retrieved from Yahoo Finance and treated as static over the sample period. As a result, fundamental signals are not point-in-time and may introduce look-ahead bias. Accordingly, results involving fundamentals should be interpreted as illustrative rather than fully implementable.

---

## Data Processing

- All factor values are winsorized at the 1st and 99th percentiles at each rebalance date  
- Factors are standardized using cross-sectional z-scores  
- For factors where lower values are preferred (P/E, volatility, debt-to-equity), z-scores are sign-inverted  
- Missing factor values receive zero contribution after standardization to avoid mechanical advantage  

A composite score is computed as an equal-weighted sum of standardized factor scores.

---

## Portfolio Construction

- Securities are ranked by composite score, with momentum used as a secondary tie-breaker  
- The top 30 securities are selected at each rebalance  
- The portfolio is implicitly equal-weighted and allowed to drift between rebalances  

---

## Turnover and Transaction Costs

Turnover at each rebalance is defined as:

Turnover = 1 − |Holdings_t ∩ Holdings_(t−1)| / N

Transaction costs are modeled as a linear function of turnover:
- 10 basis points per 100 percent turnover
- Costs are deducted on the first trading day following each rebalance

Both gross (no costs) and net (after costs) portfolio returns are reported.

---

## Performance Metrics

The strategy is evaluated using:
- Annualized return  
- Annualized volatility  
- Sharpe ratio  
- Information ratio (relative to SPY)  
- Maximum drawdown  
- CAPM alpha and beta (estimated from daily returns)  
- Average monthly turnover  

Equity curves for gross returns, net-of-cost returns, and the benchmark are plotted automatically when the script is run.

---

## Outputs

Running the script generates:
- prices.csv — historical adjusted prices for all securities  
- fundamentals.csv — snapshot fundamentals pulled from Yahoo Finance  
- equity_curves.csv — cumulative returns for gross portfolio, net portfolio, and benchmark  
- holdings_log.csv — holdings at each rebalance date  
- turnover_log.csv — turnover at each rebalance date  

---

## Installation

pip install yfinance pandas numpy matplotlib

---

## Run

python atlas-multifactor-etf.py

When executed, the script will:
1. Download historical adjusted price data for all securities in the universe
2. Retrieve snapshot fundamental data from Yahoo Finance
3. Construct a monthly rebalanced, rules-based multi-factor portfolio
4. Compute both gross and net-of-transaction-cost returns
5. Calculate performance statistics including Sharpe ratio, information ratio, alpha/beta, drawdown, and turnover
6. Save all output files to disk
7. Automatically open a comparative equity curve plot showing portfolio and benchmark performance

---

## Notes and Limitations

- This project is based on historical backtesting and does not represent liv
