import datetime
import logging
import sys
import time
from pathlib import Path
import os
import csv # Added for CSV output

# --- New Imports ---
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
# --- End New Imports ---


import numpy as np
import pandas as pd
import yfinance as yf
# Removed pyarrow import as it's usually implicit via pandas/parquet

# --- Project Imports ---
# Add parent directory to sys.path to find control.py etc.
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from control import (
        RECOMMEND_STOCK_UNIVERSE, RECOMMEND_START_DATE, RECOMMEND_END_DATE,
        RECOMMEND_TOP_N, RECOMMEND_NUM_CPUS, RECOMMEND_CACHE_EXPIRY_DAYS,
        RECOMMEND_COMMISSION_PER_TRADE_PCT, RECOMMEND_SLIPPAGE_PCT,
        RECOMMEND_RANKING_METRIC, RECOMMEND_INITIAL_CASH, RECOMMEND_MIN_HISTORY_DAYS,
        RECOMMEND_TOP_N_PER_CATEGORY # <-- Import the new parameter
    )
except ImportError as e:
    print(f"Error importing configuration from control.py: {e}")
    print("Please ensure control.py exists and contains the RECOMMEND_* parameters.")
    sys.exit(1)

# --- Configuration Constants (Loaded from control.py) ---
STOCK_UNIVERSE_SYMBOLS = RECOMMEND_STOCK_UNIVERSE
START_DATE = RECOMMEND_START_DATE
END_DATE = RECOMMEND_END_DATE
INITIAL_CASH = RECOMMEND_INITIAL_CASH
MIN_HISTORY_DAYS = RECOMMEND_MIN_HISTORY_DAYS
TOP_N_RECOMMEND = RECOMMEND_TOP_N
TOP_N_PER_CATEGORY = RECOMMEND_TOP_N_PER_CATEGORY # Use the new config
NUM_CPUS = RECOMMEND_NUM_CPUS
CACHE_EXPIRY_DAYS = RECOMMEND_CACHE_EXPIRY_DAYS
COMMISSION_PCT = RECOMMEND_COMMISSION_PER_TRADE_PCT
SLIPPAGE_PCT = RECOMMEND_SLIPPAGE_PCT
RANKING_METRIC = RECOMMEND_RANKING_METRIC
EPSILON = 1e-6 # Small number to avoid division by zero

# --- Caching Configuration ---
CACHE_DIR = Path("./yf_cache_recommend") # Use a separate cache directory
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s", # Added processName
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("yfinance").setLevel(logging.WARNING)
logger = logging.getLogger("StockRecommender")

# --- Dynamically Import Strategies ---
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !! WARNING: Using strategies from archived_strategies.                      !!
# !! These might differ from the strategies used in the main trading client.  !!
# !! Ensure this is intended or align strategy source for consistency.        !!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
STRATEGIES_DIR = Path("./strategies/archived_strategies") # <-- Using archived strategies
sys.path.insert(0, str(STRATEGIES_DIR.resolve()))

STRATEGIES_TO_TEST = []
FAILED_IMPORTS = []
try:
    logger.info(f"Attempting to import strategies from: {STRATEGIES_DIR}")
    # Import strategies from v2.1 as originally intended by this script's structure
    from trading_strategies_v2_1 import (
        adaptive_momentum_filter_strategy, complex_network_strategy, fractal_market_hypothesis_strategy,
        information_flow_strategy, kalman_filter_strategy, levy_distribution_strategy,
        pairs_trading_strategy, quantum_oscillator_strategy, regime_switching_strategy,
        simple_trend_reversal_strategy, topological_data_analysis_strategy, wavelet_momentum_strategy,
        zeta_potential_strategy
    )
    STRATEGIES_TO_TEST.extend([
        simple_trend_reversal_strategy, pairs_trading_strategy, regime_switching_strategy,
        kalman_filter_strategy, adaptive_momentum_filter_strategy, fractal_market_hypothesis_strategy,
        complex_network_strategy, information_flow_strategy, levy_distribution_strategy,
        quantum_oscillator_strategy, zeta_potential_strategy
    ])
    try:
        STRATEGIES_TO_TEST.append(wavelet_momentum_strategy); import pywt
    except ImportError: FAILED_IMPORTS.append("wavelet_momentum_strategy (requires PyWavelets)")
    try:
        STRATEGIES_TO_TEST.append(topological_data_analysis_strategy); import ripser
    except ImportError: FAILED_IMPORTS.append("topological_data_analysis_strategy (requires ripser)")

    # Remove duplicates if any were added conditionally
    STRATEGIES_TO_TEST = list(dict.fromkeys(STRATEGIES_TO_TEST))

    logger.info(f"Successfully imported {len(STRATEGIES_TO_TEST)} strategies for testing.")
    if FAILED_IMPORTS:
        logger.warning("Could not import the following optional strategies due to missing dependencies:")
        for failed in FAILED_IMPORTS: logger.warning(f"  - {failed}")

except ImportError as e:
    logger.error(f"Critical Error importing strategies from {STRATEGIES_DIR}: {e}.")
    logger.error("Check the path and ensure the strategy file (e.g., trading_strategies_v2_1.py) exists.")
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error during strategy import: {e}", exc_info=True)
    sys.exit(1)

# --- Data Fetching (Cached, adapted for NSE) ---
def fetch_stock_data(tickers, start_date, end_date):
    """ Fetches/Loads cached historical data for stock tickers. """
    # Assumes NSE tickers might need ".NS" suffix for yfinance
    symbols_map = {f"{ticker}.NS" if not ticker.endswith((".NS", ".BO")) else ticker: ticker for ticker in tickers}
    yf_symbols = list(symbols_map.keys())
    logger.info(f"Fetching/Loading data for {len(yf_symbols)} symbols...")

    fetch_start_date_dt = pd.to_datetime(start_date) - pd.Timedelta(days=MIN_HISTORY_DAYS * 2)
    fetch_start_date_str = fetch_start_date_dt.strftime("%Y-%m-%d")
    # Fetch a bit earlier to handle weekends/holidays at the start
    fetch_start_date_dt_dl = pd.to_datetime(start_date) - pd.Timedelta(days=MIN_HISTORY_DAYS * 2 + 7)
    fetch_start_date_str_dl = fetch_start_date_dt_dl.strftime("%Y-%m-%d")

    all_data = {}
    symbols_to_download = []
    cache_expiry_threshold = datetime.datetime.now() - datetime.timedelta(days=CACHE_EXPIRY_DAYS)

    # Check cache first
    for yf_ticker in yf_symbols:
        base_ticker = symbols_map[yf_ticker]
        # Standardize cache filename slightly
        cache_filename = f"{base_ticker.replace('.NS', '').replace('.BO', '')}_{fetch_start_date_str}_{end_date}.parquet"
        cache_file_path = CACHE_DIR / cache_filename
        load_from_cache = False
        if cache_file_path.exists():
            try:
                mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file_path))
                if mod_time > cache_expiry_threshold:
                    df = pd.read_parquet(cache_file_path)
                    # Basic validation on loaded data
                    if not df.empty and 'close' in [col.lower() for col in df.columns]:
                         all_data[base_ticker] = df
                         load_from_cache = True
                    else:
                        logger.warning(f"Cache file {cache_file_path} is empty or invalid. Re-downloading.")
                        cache_file_path.unlink() # Remove bad cache file
            except Exception as e:
                logger.warning(f"Error reading cache {cache_file_path}: {e}. Will re-download.")
                try: cache_file_path.unlink() # Attempt removal on error
                except OSError: pass
        if not load_from_cache:
            symbols_to_download.append(yf_ticker)

    # Download missing/expired data
    if symbols_to_download:
        logger.info(f"Need to download data for {len(symbols_to_download)} symbols.")
        batch_size = 100
        symbols_processed = set()
        for i in range(0, len(symbols_to_download), batch_size):
            batch = symbols_to_download[i:i+batch_size]
            logger.info(f"Downloading batch ({i//batch_size + 1}/{ (len(symbols_to_download)-1)//batch_size + 1 })...")
            try:
                # Use threads=True for potentially faster downloads
                data = yf.download(batch, start=fetch_start_date_str_dl, end=end_date, group_by='ticker', auto_adjust=True, prepost=False, threads=True)
                if data.empty:
                    logger.warning(f"No data returned for batch starting with {batch[0]}")
                    symbols_processed.update(batch)
                    continue

                for yf_ticker in batch:
                    base_ticker = symbols_map[yf_ticker]
                    symbols_processed.add(yf_ticker)
                    ticker_df = None
                    try:
                        # Handle yfinance download structure (MultiIndex or dict-like)
                        if isinstance(data.columns, pd.MultiIndex):
                            if yf_ticker in data.columns.levels[0]:
                                ticker_df = data[yf_ticker].copy()
                        elif yf_ticker in data: # Check if it's a key in a dict-like structure
                            ticker_df = data[yf_ticker].copy()
                        elif len(batch) == 1 and not isinstance(data.columns, pd.MultiIndex):
                             # Case where only one ticker was in the batch, and it returns a simple DataFrame
                            ticker_df = data.copy()

                        if ticker_df is None or ticker_df.empty:
                            logger.warning(f"No data found for {yf_ticker} within the downloaded batch.")
                            continue

                        # Basic cleaning and validation
                        ticker_df.dropna(subset=['Close'], inplace=True)
                        if ticker_df.empty: continue
                        ticker_df.columns = [col.lower() for col in ticker_df.columns] # Standardize cols
                        all_data[base_ticker] = ticker_df

                        # Save to cache
                        cache_filename = f"{base_ticker.replace('.NS', '').replace('.BO', '')}_{fetch_start_date_str}_{end_date}.parquet"
                        try:
                            ticker_df.to_parquet(CACHE_DIR / cache_filename)
                        except Exception as e:
                            logger.warning(f"Failed to write cache for {base_ticker}: {e}")

                    except KeyError:
                         logger.warning(f"Could not access data for {yf_ticker} in downloaded structure.")
                    except Exception as e:
                        logger.warning(f"Error processing downloaded data for {yf_ticker}: {e}")

            except Exception as e:
                logger.error(f"Error downloading batch starting with {batch[0]}: {e}", exc_info=False)
                symbols_processed.update(batch) # Mark batch as processed even on error to avoid retries
            time.sleep(0.5) # Small delay between batches

        not_processed = set(symbols_to_download) - symbols_processed
        if not_processed:
            logger.warning(f"Could not process download data for: {', '.join(not_processed)}")

    # Final validation of all data (from cache and download)
    validated_data = {}
    sim_start_dt = pd.to_datetime(start_date)
    sim_end_dt = pd.to_datetime(end_date)
    for base_ticker, df in all_data.items():
        if df is None or df.empty: continue
        df.columns = [col.lower() for col in df.columns] # Ensure lowercase cols
        # Check for required columns
        if not {'open', 'high', 'low', 'close', 'volume'}.issubset(df.columns):
            logger.warning(f"Data for {base_ticker} missing required columns. Skipping.")
            continue
        # Ensure data covers the necessary historical period for lookback
        if df.index.min() > fetch_start_date_dt:
            logger.warning(f"Insufficient history loaded/downloaded for {base_ticker} before {fetch_start_date_dt}. Skipping.")
            continue
        # Ensure there's data within the actual simulation period
        if df[(df.index >= sim_start_dt) & (df.index <= sim_end_dt)].empty:
            logger.warning(f"Data for {base_ticker} exists but has no entries within the simulation period {start_date} to {end_date}. Skipping.")
            continue

        validated_data[base_ticker] = df

    logger.info(f"Data ready for {len(validated_data)}/{len(tickers)} symbols.")
    if not validated_data:
        logger.error("No valid data available for any ticker. Exiting.")
        sys.exit(1)
    return validated_data


# --- Simulation Worker Function (Includes Costs) ---
def simulate_single_strategy_ticker(args):
    """Worker function to simulate one strategy on one ticker, including costs."""
    # Unpack arguments
    strategy, ticker, ticker_data, start_date_str, end_date_str, commission_pct, slippage_pct = args
    strategy_name = strategy.__name__

    try:
        ticker_data.columns = [col.lower() for col in ticker_data.columns]

        portfolio = {
            "cash": INITIAL_CASH, "quantity": 0.0, "total_portfolio_value": INITIAL_CASH,
            "trades": 0, "last_price": 0.0, "commission_paid": 0.0
        }

        sim_start_dt = pd.to_datetime(start_date_str)
        sim_end_dt = pd.to_datetime(end_date_str)

        # Filter data for the simulation period
        simulation_days = ticker_data[
            (ticker_data.index >= sim_start_dt) & (ticker_data.index <= sim_end_dt)
        ].copy() # Use copy to avoid SettingWithCopyWarning

        if simulation_days.empty:
            portfolio["final_value"] = INITIAL_CASH
            portfolio["profit_loss_pct"] = 0.0
            return strategy_name, ticker, portfolio # Return early

        # --- Simulation Loop ---
        for current_date in simulation_days.index:
            # Use .loc for robust indexing
            day_data = simulation_days.loc[current_date]
            current_price = day_data['close']

            if pd.isna(current_price) or current_price <= 0:
                portfolio["last_price"] = portfolio["last_price"] if portfolio["last_price"] > 0 else 0
                continue # Skip days with invalid prices
            portfolio["last_price"] = current_price

            # --- Prepare historical data for the strategy ---
            # Get the index location in the *original* dataframe for slicing
            loc_in_full_data = ticker_data.index.get_loc(current_date)
            if loc_in_full_data < MIN_HISTORY_DAYS:
                continue # Skip if not enough history

            # Slice from the *original* dataframe to include lookback period
            historical_slice = ticker_data.iloc[:loc_in_full_data + 1].copy() # Include current day for some indicators
            if historical_slice.empty or len(historical_slice) < MIN_HISTORY_DAYS:
                continue # Skip if slice is too short
            historical_slice.columns = [col.lower() for col in historical_slice.columns]

            try:
                # --- Get Strategy Signal ---
                # Assumes strategy returns: action (str), quantity_signal (float), ticker (str)
                action, quantity_signal, _ = strategy(
                    ticker=ticker, current_price=current_price, historical_data=historical_slice,
                    account_cash=portfolio["cash"], portfolio_qty=portfolio["quantity"],
                    total_portfolio_value=portfolio["total_portfolio_value"],
                )

                # --- Simulate Trade Execution with Costs ---
                trade_executed = False
                if action in ["buy", "strong buy"] and np.isfinite(quantity_signal) and quantity_signal > 0:
                    buy_quantity = quantity_signal
                    base_cost = buy_quantity * current_price
                    slippage_cost = base_cost * slippage_pct
                    commission_cost = base_cost * commission_pct
                    total_cost = base_cost + slippage_cost + commission_cost

                    if total_cost > 0 and portfolio["cash"] >= total_cost:
                        portfolio["cash"] -= total_cost
                        portfolio["quantity"] += buy_quantity
                        portfolio["trades"] += 1
                        portfolio["commission_paid"] += commission_cost
                        trade_executed = True

                elif action in ["sell", "strong sell"] and np.isfinite(quantity_signal) and quantity_signal > 0:
                    sell_quantity = min(portfolio["quantity"], quantity_signal)
                    if sell_quantity > 0:
                        base_revenue = sell_quantity * current_price
                        slippage_deduction = base_revenue * slippage_pct
                        commission_deduction = base_revenue * commission_pct
                        total_revenue = base_revenue - slippage_deduction - commission_deduction

                        if total_revenue > 0: # Ensure positive revenue after costs
                             portfolio["cash"] += total_revenue
                             portfolio["quantity"] -= sell_quantity
                             portfolio["trades"] += 1
                             portfolio["commission_paid"] += commission_deduction
                             trade_executed = True

                # --- Update Portfolio Value ---
                # Use the last known price if trade didn't execute or loop ends
                holding_value = portfolio["quantity"] * portfolio["last_price"]
                portfolio["total_portfolio_value"] = portfolio["cash"] + holding_value

            except Exception as strat_e:
                # Log less verbosely during parallel execution, maybe collect errors later
                # logger.warning(f"Error in {strategy_name} for {ticker} on {current_date}: {strat_e}")
                pass # Continue to next day

        # --- End of simulation for this ticker/strategy ---
        final_value = portfolio["cash"] + (portfolio["quantity"] * portfolio["last_price"] if portfolio["last_price"] > 0 else 0)
        if not np.isfinite(final_value):
            final_value = portfolio.get("cash", INITIAL_CASH) # Fallback

        portfolio["final_value"] = final_value
        if INITIAL_CASH > 0:
            portfolio["profit_loss_pct"] = ((final_value - INITIAL_CASH) / INITIAL_CASH) * 100
        else:
            portfolio["profit_loss_pct"] = 0.0

        # Clean up large data object before returning
        # portfolio.pop('historical_data_snapshot', None)

        return strategy_name, ticker, portfolio # Return results

    except Exception as worker_e:
        # Log error at the worker level
        # logger.error(f"Error in worker for {strategy_name} on {ticker}: {worker_e}", exc_info=True)
        # Return error indicator
        return strategy_name, ticker, {"error": str(worker_e), "final_value": INITIAL_CASH, "profit_loss_pct": 0.0, "trades": 0, "commission_paid": 0.0}


# --- Modified Simulation Runner ---
def run_simulation_parallel(all_ticker_data, strategies):
    """Runs the backtest simulation in parallel using multiprocessing."""
    logger.info(f"Preparing tasks for parallel simulation across {NUM_CPUS} cores...")

    tasks = []
    for strategy in strategies:
        for ticker, ticker_data in all_ticker_data.items():
            # Pass necessary data for one simulation run, including cost parameters
            tasks.append((strategy, ticker, ticker_data.copy(), START_DATE, END_DATE, COMMISSION_PCT, SLIPPAGE_PCT))

    logger.info(f"Total simulation tasks to run: {len(tasks)}")
    if not tasks:
        logger.warning("No simulation tasks generated. Check data and strategies.")
        return {}

    start_sim_time = time.time()
    results_list = [] # To store results from workers

    # Use context manager for the Pool
    with Pool(processes=NUM_CPUS) as pool:
        try:
            # Use imap_unordered for potentially better memory usage with tqdm
            pool_iterator = pool.imap_unordered(simulate_single_strategy_ticker, tasks)

            # Wrap the iterator with tqdm for progress bar
            for result_item in tqdm(pool_iterator, total=len(tasks), desc="Simulating", unit="task"):
                if result_item: # Check if worker returned successfully
                    results_list.append(result_item)
                # else: logger.warning("Worker returned None, possible error.") # Optional: Handle None returns

            logger.info(f"Parallel processing finished in {time.time() - start_sim_time:.2f} seconds.")

        except Exception as e:
            logger.error(f"Error during parallel execution with imap_unordered: {e}", exc_info=True)
            return {} # Return empty on pool error

    # Reconstruct the results dictionary
    final_results = {}
    error_count = 0
    successful_count = 0
    for result_item in results_list:
        if result_item: # Ensure the worker returned something
            strategy_name, ticker, portfolio = result_item
            if "error" in portfolio:
                error_count += 1
                # Optionally log the specific error reported by the worker
                # logger.warning(f"Simulation failed for {strategy_name} on {ticker}: {portfolio['error']}")
            else:
                successful_count += 1
                if strategy_name not in final_results:
                    final_results[strategy_name] = {}
                final_results[strategy_name][ticker] = portfolio
        else:
            error_count += 1 # Worker returned None, likely due to unhandled exception in worker

    logger.info(f"Processed results: {successful_count} successful, {error_count} failed/skipped.")
    return final_results


# --- Analysis and Recommendation (Enhanced Ranking) ---
def analyze_and_recommend(results):
    """
    Analyzes results, ranks stocks based on the configured metric,
    and categorizes the top recommendations by price.
    """
    logger.info("\n--- Analyzing Strategy Performance Across Tickers ---")
    ticker_scores = {}

    # Aggregate results per ticker
    for strategy_name, ticker_results in results.items():
        for ticker, portfolio in ticker_results.items():
            if "error" in portfolio: continue # Skip failed simulations

            if ticker not in ticker_scores:
                ticker_scores[ticker] = {
                    'total_pl_pct': 0.0, 'strategy_count': 0, 'positive_strategies': 0,
                    'pl_values': [], 'total_trades': 0, 'total_commission': 0.0,
                    'last_price': 0.0 # Initialize last_price
                }

            pl_pct = portfolio.get('profit_loss_pct', 0.0)
            last_price = portfolio.get('last_price', 0.0) # Get last price from simulation

            if np.isfinite(pl_pct):
                ticker_scores[ticker]['total_pl_pct'] += pl_pct
                ticker_scores[ticker]['strategy_count'] += 1
                ticker_scores[ticker]['pl_values'].append(pl_pct)
                ticker_scores[ticker]['total_trades'] += portfolio.get('trades', 0)
                ticker_scores[ticker]['total_commission'] += portfolio.get('commission_paid', 0.0)
                if pl_pct > 0:
                    ticker_scores[ticker]['positive_strategies'] += 1
                # Store the last price from any valid simulation run for this ticker
                if last_price > 0:
                     ticker_scores[ticker]['last_price'] = last_price

    # Calculate metrics and rank tickers
    ranked_tickers = []
    for ticker, scores in ticker_scores.items():
        if scores['strategy_count'] > 0 and scores['last_price'] > 0: # Ensure valid count and price
            avg_pl = scores['total_pl_pct'] / scores['strategy_count']
            pl_std_dev = np.std(scores['pl_values']) if len(scores['pl_values']) > 1 else 0.0
            positive_ratio = (scores['positive_strategies'] / scores['strategy_count'])
            avg_trades = scores['total_trades'] / scores['strategy_count']
            avg_commission = scores['total_commission'] / scores['strategy_count']

            # Calculate Ranking Score based on configuration
            if RANKING_METRIC == 'risk_adjusted_pl':
                ranking_score = avg_pl / (pl_std_dev + EPSILON)
            elif RANKING_METRIC == 'weighted_pl':
                ranking_score = avg_pl * positive_ratio
            else: # Default to 'avg_pl'
                ranking_score = avg_pl

            ranked_tickers.append({
                'ticker': ticker,
                'avg_pl_pct': avg_pl,
                'pl_std_dev': pl_std_dev,
                'positive_ratio': positive_ratio,
                'avg_trades': avg_trades,
                'avg_commission': avg_commission,
                'last_price': scores['last_price'], # Add last price here
                'score': ranking_score,
                'strategy_count': scores['strategy_count']
            })
        else:
            logger.warning(f"Ticker {ticker} had no valid results or price. Excluding from ranking.")

    # Sort all tickers by the calculated ranking score
    ranked_tickers.sort(key=lambda x: x['score'], reverse=True)

    # --- Categorize Recommendations ---
    recommendations_by_category = {
        "Above_1000": [],
        "100_to_1000": [],
        "Below_100": []
    }

    for item in ranked_tickers:
        price = item['last_price']
        if price > 1000:
            category = "Above_1000"
        elif 100 <= price <= 1000:
            category = "100_to_1000"
        else: # price < 100
            category = "Below_100"

        # Add to category list if not full
        if len(recommendations_by_category[category]) < TOP_N_PER_CATEGORY:
            recommendations_by_category[category].append(item)

    # --- Display Results By Category ---
    logger.info(f"\n--- Stock Recommendations by Price Category (Ranked by: {RANKING_METRIC}) ---")

    header_format = f"{'Rank':<5} {'Ticker':<15} {'Price':>10} {'Score':>15} {'Avg P/L (%)':>15} {'P/L Std Dev':>15} {'% Positive Strats':>20} {'Avg Trades/Strat':>15}"
    separator = "-" * 130 # Adjusted separator length

    for category, items in recommendations_by_category.items():
        print(f"\n\n--- Category: {category.replace('_', ' ')} (Top {TOP_N_PER_CATEGORY}) ---")
        print(separator)
        print(header_format)
        print(separator)
        if not items:
            print(f"No stocks found in this category or insufficient ranked stocks.")
        else:
            # Rank within the category is based on the overall score ranking
            for i, item in enumerate(items, 1):
                 print(f"{i:<5} {item['ticker']:<15} {item['last_price']:>10.2f} {item['score']:>15.4f} {item['avg_pl_pct']:>14.2f}% "
                      f"{item['pl_std_dev']:>15.2f} {item['positive_ratio']:>19.1%} "
                      f"{item['avg_trades']:>15.1f}")
        print(separator)


    # --- Disclaimer and Justification (remains the same) ---
    logger.info("\n--- Recommendation Justification & Expectations ---")
    logger.info(f"Ranking based on '{RANKING_METRIC}' across {len(STRATEGIES_TO_TEST)} strategies tested from {START_DATE} to {END_DATE}.")
    logger.info(f"Simulation included Commission: {COMMISSION_PCT*100:.3f}%, Slippage: {SLIPPAGE_PCT*100:.3f}% per trade.")
    logger.info("Recommendations are categorized based on the closing price on the simulation end date.")
    logger.info(" > Higher 'Score': Indicates better performance according to the chosen ranking metric.")
    logger.info(" > 'Avg P/L (%)': Average profit/loss across all tested strategies.")
    logger.info(" > 'P/L Std Dev': Volatility/consistency of P/L across strategies (lower is more consistent).")
    logger.info(" > '% Positive Strats': Percentage of strategies that resulted in a profit for this stock.")
    logger.info(" > 'Avg Trades/Strat': Average trading activity generated by the strategies for this stock.")
    # logger.info(" > 'Avg Comm./Strat': Average commission paid per strategy simulation.") # Removed as less relevant for final display
    logger.warning("\n**DISCLAIMER:** Historical backtest simulation. Past performance NOT indicative of future results. This is NOT financial advice.**")

    return ranked_tickers # Return the full ranked list for saving

# --- Function to Save Results ---
def save_recommendations_to_csv(ranked_data, filename="stock_recommendations.csv"):
    """Saves the ranked stock data to a CSV file."""
    if not ranked_data:
        logger.warning("No ranked data to save.")
        return

    # Create results directory if it doesn't exist
    results_dir = Path("./recommendation_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    filepath = results_dir / filename

    try:
        keys = ranked_data[0].keys()
        with open(filepath, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(ranked_data)
        logger.info(f"Recommendations saved successfully to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save recommendations to CSV: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    # Optional: Add freeze_support() for Windows compatibility if creating executables
    # from multiprocessing import freeze_support
    # freeze_support()

    if not STRATEGIES_TO_TEST:
        logger.error("No strategies were successfully imported or selected. Exiting.")
        sys.exit(1)

    logger.info(f"Starting Stock Recommendation Backtest")
    logger.info(f"Period: {START_DATE} to {END_DATE}")
    logger.info(f"Universe: {len(STOCK_UNIVERSE_SYMBOLS)} symbols")
    logger.info(f"Strategies: {len(STRATEGIES_TO_TEST)}")
    logger.info(f"Ranking Metric: {RANKING_METRIC}")
    logger.info(f"Using {NUM_CPUS} CPU cores for parallel simulation.")
    script_start_time = datetime.datetime.now()

    # Basic check on symbol list size
    if len(STOCK_UNIVERSE_SYMBOLS) < 10:
        logger.warning("Stock universe list seems small. Ensure configuration is correct.")
        # Optionally add confirmation prompt here if needed

    # Fetch Data
    all_data = fetch_stock_data(STOCK_UNIVERSE_SYMBOLS, START_DATE, END_DATE)

    if all_data:
        simulation_results = run_simulation_parallel(all_data, STRATEGIES_TO_TEST)

        if simulation_results:
            # Analyze, categorize and display recommendations
            full_ranked_list = analyze_and_recommend(simulation_results)

            # Save the *full* ranked list (not categorized) to CSV
            if full_ranked_list:
                 timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                 csv_filename = f"recommendations_full_ranked_{RANKING_METRIC}_{timestamp}.csv"
                 save_recommendations_to_csv(full_ranked_list, filename=csv_filename)
        else:
            logger.error("Simulation produced no results. Check logs for errors.")
    else:
        logger.error("Failed to fetch sufficient data. Aborting simulation.")

    script_end_time = datetime.datetime.now()
    logger.info(f"Full backtest, analysis, and recommendation finished in {script_end_time - script_start_time}")