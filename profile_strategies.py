import time
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

# --- Configuration ---
PROFILE_TICKER = "APLAPOLLO.NS"  # Choose a representative stock (e.g., liquid, good data)
PROFILE_START_DATE = "2023-01-01" # Use a reasonable period for historical data
PROFILE_END_DATE = "2024-01-01"
PROFILE_MIN_HISTORY_DAYS = 60 # Should match MIN_HISTORY_DAYS used in recommend.py
PROFILE_EXCLUDE_STRATEGIES = [] # Add function names (strings) to exclude if needed

# --- Setup ---
# Add parent directory to sys.path if needed (adjust if script location differs)
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Basic Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StrategyProfiler")
logging.getLogger("yfinance").setLevel(logging.WARNING) # Reduce yfinance noise

# --- Import Strategies ---
# Using the merged v2 file as requested
STRATEGIES_DIR = Path("./strategies/archived_strategies")
STRATEGY_MODULE_NAME = "trading_strategies_v2"
sys.path.insert(0, str(STRATEGIES_DIR.resolve()))

STRATEGIES_TO_PROFILE = []
logger.info(f"Attempting to import strategies from: {STRATEGIES_DIR / (STRATEGY_MODULE_NAME + '.py')}")
try:
    # **** IMPORTANT: Explicitly import ALL desired strategy functions from your merged file ****
    from trading_strategies_v2 import (
        # Original v2 Strategies (Examples)
        rsi_strategy, bollinger_bands_strategy, momentum_strategy, mean_reversion_strategy,
        triple_moving_average_strategy, volume_price_trend_strategy, keltner_channel_strategy,
        dual_thrust_strategy, adaptive_momentum_strategy, # Assuming one combined/renamed
        hull_moving_average_strategy, elder_ray_strategy, chande_momentum_strategy, dema_strategy,
        price_channel_strategy, mass_index_strategy, vortex_indicator_strategy, aroon_strategy,
        ultimate_oscillator_strategy, trix_strategy, kst_strategy, psar_strategy,
        stochastic_momentum_strategy, williams_vix_fix_strategy, conners_rsi_strategy,
        dpo_strategy, fisher_transform_strategy, ehlers_fisher_strategy, schaff_trend_cycle_strategy,
        rainbow_oscillator_strategy, heikin_ashi_strategy, volume_weighted_macd_strategy,
        fractal_adaptive_moving_average_strategy, relative_vigor_index_strategy,
        center_of_gravity_strategy, kauffman_efficiency_strategy, phase_change_strategy,
        volatility_breakout_strategy, momentum_divergence_strategy, adaptive_channel_strategy,
        # Added v2.1 Strategies (Examples - verify names in your merged file)
        pairs_trading_strategy, kalman_filter_strategy, regime_switching_strategy,
        adaptive_momentum_filter_strategy, # Potentially duplicate name - check your merge
        fractal_market_hypothesis_strategy, topological_data_analysis_strategy, levy_distribution_strategy,
        information_flow_strategy, wavelet_momentum_strategy, complex_network_strategy,
        zeta_potential_strategy, quantum_oscillator_strategy, simple_trend_reversal_strategy
        # Add *all* other strategy functions from your merged file here
    )

    all_imported_strategies = [
        # List all imported function objects again here
        rsi_strategy, bollinger_bands_strategy, momentum_strategy, mean_reversion_strategy,
        triple_moving_average_strategy, volume_price_trend_strategy, keltner_channel_strategy,
        dual_thrust_strategy, adaptive_momentum_strategy,
        hull_moving_average_strategy, elder_ray_strategy, chande_momentum_strategy, dema_strategy,
        price_channel_strategy, mass_index_strategy, vortex_indicator_strategy, aroon_strategy,
        ultimate_oscillator_strategy, trix_strategy, kst_strategy, psar_strategy,
        stochastic_momentum_strategy, williams_vix_fix_strategy, conners_rsi_strategy,
        dpo_strategy, fisher_transform_strategy, ehlers_fisher_strategy, schaff_trend_cycle_strategy,
        rainbow_oscillator_strategy, heikin_ashi_strategy, volume_weighted_macd_strategy,
        fractal_adaptive_moving_average_strategy, relative_vigor_index_strategy,
        center_of_gravity_strategy, kauffman_efficiency_strategy, phase_change_strategy,
        volatility_breakout_strategy, momentum_divergence_strategy, adaptive_channel_strategy,
        pairs_trading_strategy, kalman_filter_strategy, regime_switching_strategy,
        adaptive_momentum_filter_strategy,
        fractal_market_hypothesis_strategy, topological_data_analysis_strategy, levy_distribution_strategy,
        information_flow_strategy, wavelet_momentum_strategy, complex_network_strategy,
        zeta_potential_strategy, quantum_oscillator_strategy, simple_trend_reversal_strategy
    ]

     # Filter based on exclusion list and remove duplicates
    STRATEGIES_TO_PROFILE = [
        strat for strat in all_imported_strategies
        if strat.__name__ not in PROFILE_EXCLUDE_STRATEGIES
    ]
    STRATEGIES_TO_PROFILE = list(dict.fromkeys(STRATEGIES_TO_PROFILE))

    logger.info(f"Successfully prepared {len(STRATEGIES_TO_PROFILE)} strategies for profiling.")
    if PROFILE_EXCLUDE_STRATEGIES:
        logger.info(f"Excluded strategies: {', '.join(PROFILE_EXCLUDE_STRATEGIES)}")

except ImportError as e:
    logger.error(f"Critical Error importing strategies from {STRATEGY_MODULE_NAME}: {e}.")
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error during strategy import setup: {e}", exc_info=True)
    sys.exit(1)

if not STRATEGIES_TO_PROFILE:
    logger.error("No strategies selected for profiling. Exiting.")
    sys.exit(1)

# --- Data Fetching ---
# --- Data Fetching ---
def fetch_single_stock_data(ticker, start_date, end_date):
    """Fetches and preprocesses data for a single stock."""
    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        # --- ADD THIS CHECK ---
        if not isinstance(data, pd.DataFrame):
            logger.error(f"yf.download did not return a DataFrame for {ticker}. Received type: {type(data)}. Value: {data}")
            return None
        # --- END CHECK ---

        if data.empty:
            logger.error(f"No data downloaded for {ticker} (DataFrame is empty).")
            return None

        print(data.columns)        

        # Preprocess: lowercase columns, handle NaNs
        data.columns = [col[0].lower() for col in data.columns] # This line should now be safe
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(data.columns):
            logger.error(f"Data for {ticker} missing required columns ({required_cols - set(data.columns)}).")
            return None

        # Basic cleaning
        initial_len = len(data)
        # Ensure relevant columns are numeric before checking for NaNs/zeros
        for col in ['open', 'high', 'low', 'close']:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        data.dropna(subset=['close', 'open', 'high', 'low'], inplace=True)
        if len(data) < initial_len:
             logger.warning(f"Dropped {initial_len - len(data)} rows with NaNs in OHLC for {ticker}.")

        if data.empty:
             logger.error(f"Data for {ticker} became empty after cleaning.")
             return None

        logger.info(f"Successfully fetched and preprocessed data for {ticker} ({len(data)} rows).")
        return data
    except Exception as e:
        # Log the full traceback for unexpected errors during download/processing
        logger.error(f"Error during yfinance download or processing for {ticker}: {e}", exc_info=True)
        return None
    
# --- Main Profiling Logic ---
if __name__ == "__main__":
    logger.info("--- Starting Strategy Profiling ---")

    # 1. Fetch Data
    # Calculate needed history start date based on profile end date and min history
    hist_start_dt = pd.to_datetime(PROFILE_END_DATE) - pd.Timedelta(days=PROFILE_MIN_HISTORY_DAYS * 2 - 5) # Fetch more for buffer
    hist_start_date_str = hist_start_dt.strftime('%Y-%m-%d')
    historical_data = fetch_single_stock_data(PROFILE_TICKER, hist_start_date_str, PROFILE_END_DATE)

    if historical_data is None or len(historical_data) < PROFILE_MIN_HISTORY_DAYS:
        logger.error(f"Insufficient historical data ({len(historical_data) if historical_data is not None else 0} rows) for {PROFILE_TICKER} ending {PROFILE_END_DATE}. Need at least {PROFILE_MIN_HISTORY_DAYS}.")
        sys.exit(1)

    # 2. Prepare Dummy Inputs for Strategies
    # We use the *last* available data point and the full history
    # The actual trading logic isn't important here, just the execution time
    current_price = historical_data['close'].iloc[-1]
    account_cash = 100000.0  # Dummy value
    portfolio_qty = 10.0     # Dummy value
    total_portfolio_value = 110000.0 # Dummy value
    logger.info(f"Using data for {PROFILE_TICKER} ending {historical_data.index[-1].date()} for profiling.")
    logger.info(f"Current Price (dummy): {current_price:.2f}")


    # 3. Profile Each Strategy
    strategy_timings = []
    total_profiling_time = 0

    logger.info(f"\nProfiling {len(STRATEGIES_TO_PROFILE)} strategies...")
    for strategy in STRATEGIES_TO_PROFILE:
        strategy_name = strategy.__name__
        logger.debug(f"Profiling: {strategy_name}...") # Use debug level for less verbose output
        start_time = time.perf_counter()
        try:
            # Call the strategy function with the prepared data and dummy values
            # Pass the *entire* historical_data; strategies should handle slicing if needed
            _ = strategy( # We don't care about the output (action, qty, ticker)
                ticker=PROFILE_TICKER,
                current_price=current_price,
                historical_data=historical_data, # Pass the full DataFrame
                account_cash=account_cash,
                portfolio_qty=portfolio_qty,
                total_portfolio_value=total_portfolio_value
            )
            end_time = time.perf_counter()
            duration = end_time - start_time
            strategy_timings.append({"name": strategy_name, "duration_s": duration, "status": "Success"})
            total_profiling_time += duration
            logger.debug(f"Finished: {strategy_name} in {duration:.4f}s")

        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time # Time until error
            logger.warning(f"Error profiling {strategy_name}: {e}", exc_info=False) # Log warning, don't show full traceback unless needed
            strategy_timings.append({"name": strategy_name, "duration_s": duration, "status": f"Error: {type(e).__name__}"})
            total_profiling_time += duration # Include error time

    # 4. Report Results
    logger.info("\n--- Profiling Results ---")

    # Sort by duration, descending
    strategy_timings.sort(key=lambda x: x["duration_s"], reverse=True)

    print(f"\nExecution Time per Strategy (Single Call on {PROFILE_TICKER}, descending):")
    print("-" * 60)
    print(f"{'Strategy Name':<40} {'Time (ms)':>10}  {'Status'}")
    print("-" * 60)
    for timing in strategy_timings:
        duration_ms = timing['duration_s'] * 1000
        print(f"{timing['name']:<40} {duration_ms:>10.3f}  {timing['status']}")
    print("-" * 60)

    successful_timings = [t['duration_s'] for t in strategy_timings if t['status'] == 'Success']
    error_count = len(strategy_timings) - len(successful_timings)

    if successful_timings:
        avg_time_s = np.mean(successful_timings)
        median_time_s = np.median(successful_timings)
        max_time_s = np.max(successful_timings)
        min_time_s = np.min(successful_timings)

        print("\nSummary Statistics (Successful Strategies):")
        print(f"  Average Time : {avg_time_s * 1000:.3f} ms")
        print(f"  Median Time  : {median_time_s * 1000:.3f} ms")
        print(f"  Max Time     : {max_time_s * 1000:.3f} ms ({strategy_timings[0]['name']})") # Slowest is first after sort
        print(f"  Min Time     : {min_time_s * 1000:.3f} ms ({strategy_timings[-1-error_count]['name'] if error_count < len(strategy_timings) else 'N/A'})") # Fastest successful

    print(f"\nTotal Strategies Profiled: {len(strategy_timings)}")
    print(f"  Successful : {len(successful_timings)}")
    print(f"  Errors     : {error_count}")
    print(f"Total Profiling Time (all strategies): {total_profiling_time:.3f} seconds")
    logger.info("--- Profiling Complete ---")