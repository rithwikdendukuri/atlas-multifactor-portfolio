# Results (Prototype Backtest)

This section summarizes the empirical results produced by `rules_based_etf.py`. The strategy is evaluated over the period January 2016 through the most recent available date using daily returns, with performance reported both gross of trading costs and net of a simple turnover-based transaction cost model.

## Headline Performance

The following statistics are printed directly by the script:

- Gross Annualized Return: 0.232  
- Net Annualized Return: 0.232  
- Gross Sharpe Ratio: 1.298  
- Net Sharpe Ratio: 1.297  
- Gross Information Ratio (vs SPY): 1.119  
- Net Information Ratio (vs SPY): 1.116  
- Gross Alpha (annualized): 0.074  
- Net Alpha (annualized): 0.074  
- Average Monthly Turnover: 0.024  

## Gross vs Net Performance

Gross and net annualized returns are nearly identical (both approximately 23.2 percent). This result is driven by the strategy’s very low average turnover of 2.4 percent per rebalance. Because transaction costs are modeled as 10 basis points per 100 percent turnover and are applied once per rebalance period, the implied cost drag is minimal.

The close alignment between the gross and net equity curves indicates that the strategy’s performance is not reliant on excessive trading and is relatively robust to modest trading frictions under the assumed cost model.

## Risk-Adjusted Returns

The gross Sharpe ratio of 1.298 and net Sharpe ratio of 1.297 indicate strong risk-adjusted performance. In this implementation, the Sharpe ratio is computed using daily returns with an implicit zero risk-free rate. As a result, the Sharpe ratio should be interpreted as a measure of return per unit of total volatility rather than excess return over cash.

The Information Ratio relative to SPY exceeds 1.1 on both a gross and net basis, suggesting consistent outperformance relative to the benchmark after accounting for the volatility of active returns.

## Market Exposure and Alpha

The strategy produces an annualized alpha of approximately 7.4 percent on both a gross and net basis. Alpha is computed as the annualized mean of (portfolio return minus beta times benchmark return), where beta is estimated using the covariance of portfolio and benchmark returns divided by the variance of benchmark returns.

Because the script does not estimate a full regression model or report statistical significance, alpha should be interpreted as a descriptive measure rather than a formally tested abnormal return.

## Turnover and Trading Frictions

Average monthly turnover is approximately 2.4 percent, indicating a high degree of holding persistence across rebalancing dates. This low turnover explains the minimal difference between gross and net results and suggests that the strategy’s signals are relatively stable over time.

Transaction costs are modeled in a simplified manner and do not incorporate bid-ask spreads, market impact, or taxes. Nevertheless, the low turnover implies that reasonable increases in assumed trading costs would be unlikely to materially alter the qualitative conclusions of the backtest.

## Summary Interpretation

Overall, the results suggest that the multi-factor ranking framework generates strong absolute and benchmark-relative performance over the sample period, with favorable risk-adjusted characteristics and minimal sensitivity to the assumed transaction cost model. However, these findings should be interpreted with caution given the use of static fundamental data and the inherent limitations of historical backtesting.

## Key Caveats

- Fundamental variables are treated as static and are not point-in-time, introducing potential look-ahead bias.  
- Alpha is computed using a simplified CAPM-style decomposition without formal statistical testing.  
- Transaction costs are approximated using a linear turnover-based model and do not reflect real-world execution dynamics.  
- Results reflect historical performance and do not guarantee future outcomes.
