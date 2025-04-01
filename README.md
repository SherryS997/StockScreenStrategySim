# StockScreenStrategySim

A powerful backtesting framework that simulates historical performance of stocks across various trading strategies for screening and recommendation.

## Overview

StockScreenStrategySim helps you identify promising stocks by testing them against a diverse collection of technical trading strategies. It evaluates which stocks perform well across multiple strategies, providing a robust selection mechanism that goes beyond single-strategy performance.

## Key Features

- Test a universe of stocks against 45+ technical trading strategies
- Parallel processing for efficient backtesting
- Smart caching system for both stock data and simulation results
- Comprehensive performance metrics (returns, drawdowns, Sharpe, Sortino, Calmar ratios)
- Price category-based recommendations
- Generates equity curve plots for visualization
- Detailed strategy performance analysis per stock

## Setup

### Dependencies

```bash
pip install numpy pandas matplotlib yfinance tqdm pywt scipy ripser
```

### Configuration

All simulation parameters are controlled via the `control.py` file:

- `RECOMMEND_STOCK_UNIVERSE`: List of stock symbols to analyze
- `RECOMMEND_START_DATE`/`RECOMMEND_END_DATE`: Backtest period
- `RECOMMEND_TOP_N`: Number of top stocks to recommend
- `RECOMMEND_RANKING_METRIC`: How to rank stocks ('risk_adjusted_pl', 'sortino', 'calmar', 'weighted_pl')
- `RECOMMEND_USE_SIMULATION_CACHE`: Enable/disable simulation result caching
- `RECOMMEND_EXCLUDE_STRATEGIES`: List of strategies to exclude from testing

## Usage

Run the main recommendation script:

```bash
python recommend_v2.py
```

The system will:
1. Download/load historical stock data
2. Run multiple trading strategies on each stock
3. Calculate performance metrics
4. Rank stocks based on their performance 
5. Generate recommendations by price category
6. Save results to CSV and produce equity curve plots

## Key Components

- `recommend_v2.py`: Main execution script
- `control.py`: Configuration parameters
- `trading_strategies_v2.py`: Collection of trading strategies
- `recommendation_results/`: Output directory for results
- `yf_cache/`: Cache directory for stock data
- `recommend_sim_cache/`: Cache directory for simulation results

## Interpreting Results

Results are categorized by price range:
- Above 1000
- 100 to 1000
- Below 100

For each stock, the system provides:
- Performance metrics averaged across all strategies
- Comparison to buy & hold performance
- List of top-performing strategies for that stock
- List of worst-performing strategies for that stock
- Consistency metrics (% of strategies that were profitable)

## Example Strategies

The system includes diverse trading approaches:
- Momentum-based strategies (RSI, MACD, etc.)
- Mean-reversion strategies (Bollinger Bands, etc.)
- Oscillator-based strategies (Stochastics, etc.)
- Volatility-based strategies (ATR, Keltner Channels, etc.)
- Pattern recognition strategies (Heikin-Ashi, etc.)
- Advanced mathematical models (Kalman filters, wavelets, etc.)

## Performance Considerations

- Backtesting large stock universes can be time-consuming
- Enable caching to speed up subsequent runs
- Consider testing strategy subsets for experimental runs

## Disclaimer

This tool is for research and educational purposes only. Past performance does not guarantee future results. Always conduct your own research before making investment decisions.
