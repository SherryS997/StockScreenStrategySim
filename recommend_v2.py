import datetime
import logging
import sys
import time
from pathlib import Path
import os
import csv
import warnings
import functools
import hashlib # For hashing parameters
import pickle # For saving/loading simulation results

# --- New Imports ---
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
# import quantstats as qs # Optional

# --- Filter specific warnings ---
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in log2")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log2")
warnings.filterwarnings("ignore", category=FutureWarning, message="Setting an item of incompatible dtype is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="Could not infer format") # From pandas/pickle


# --- Project Imports ---
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from control import (
        RECOMMEND_STOCK_UNIVERSE, RECOMMEND_START_DATE, RECOMMEND_END_DATE,
        RECOMMEND_TOP_N, RECOMMEND_NUM_CPUS, RECOMMEND_CACHE_EXPIRY_DAYS,
        RECOMMEND_COMMISSION_PER_TRADE_PCT, RECOMMEND_SLIPPAGE_PCT,
        RECOMMEND_RANKING_METRIC, RECOMMEND_INITIAL_CASH, RECOMMEND_MIN_HISTORY_DAYS,
        RECOMMEND_TOP_N_PER_CATEGORY,
        RECOMMEND_BENCHMARK_TICKER, RECOMMEND_EXCLUDE_STRATEGIES,
        RECOMMEND_GENERATE_PLOTS, RECOMMEND_PLOT_TOP_N,
        RECOMMEND_STRATEGY_DETAIL_COUNT, RECOMMEND_MIN_AVG_TRADES,
        # --- New Cache Config ---
        RECOMMEND_USE_SIMULATION_CACHE, RECOMMEND_SIM_CACHE_DIR
    )
except ImportError as e:
    print(f"Error importing configuration from control.py: {e}")
    print("Please ensure control.py exists and contains ALL required RECOMMEND_* parameters.")
    sys.exit(1)

# --- Configuration Constants ---
STOCK_UNIVERSE_SYMBOLS = RECOMMEND_STOCK_UNIVERSE
START_DATE = RECOMMEND_START_DATE
END_DATE = RECOMMEND_END_DATE
INITIAL_CASH = RECOMMEND_INITIAL_CASH
MIN_HISTORY_DAYS = RECOMMEND_MIN_HISTORY_DAYS
TOP_N_RECOMMEND = RECOMMEND_TOP_N
TOP_N_PER_CATEGORY = RECOMMEND_TOP_N_PER_CATEGORY
NUM_CPUS = RECOMMEND_NUM_CPUS
CACHE_EXPIRY_DAYS = RECOMMEND_CACHE_EXPIRY_DAYS
COMMISSION_PCT = RECOMMEND_COMMISSION_PER_TRADE_PCT
SLIPPAGE_PCT = RECOMMEND_SLIPPAGE_PCT
RANKING_METRIC = RECOMMEND_RANKING_METRIC
BENCHMARK_TICKER = RECOMMEND_BENCHMARK_TICKER
EXCLUDE_STRATEGIES = RECOMMEND_EXCLUDE_STRATEGIES
GENERATE_PLOTS = RECOMMEND_GENERATE_PLOTS
PLOT_TOP_N = RECOMMEND_PLOT_TOP_N
STRATEGY_DETAIL_COUNT = RECOMMEND_STRATEGY_DETAIL_COUNT
MIN_AVG_TRADES_FILTER = RECOMMEND_MIN_AVG_TRADES
EPSILON = 1e-9
# --- New Cache Config ---
USE_SIMULATION_CACHE = RECOMMEND_USE_SIMULATION_CACHE
SIM_CACHE_DIR = Path(RECOMMEND_SIM_CACHE_DIR)

# --- Caching & Output Configuration ---
YF_CACHE_DIR = Path("./yf_cache") # Renamed for clarity
YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("./recommendation_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = RESULTS_DIR / "plots"
if GENERATE_PLOTS:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
if USE_SIMULATION_CACHE:
    SIM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging Setup --- (Same as before)
log_file_path = RESULTS_DIR / f"recommend_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.WARNING)
logger = logging.getLogger("StockRecommender")


# --- Strategy Import Setup --- (Same as previous version - using merged v2)
STRATEGIES_DIR = Path(".")
STRATEGY_MODULE_NAME = "trading_strategies_v2"
sys.path.insert(0, str(STRATEGIES_DIR.resolve()))
STRATEGIES_TO_TEST = []
logger.warning("="*80)
logger.warning(f"INFO: Importing strategies from MODIFIED file: {STRATEGIES_DIR / (STRATEGY_MODULE_NAME + '.py')}")
logger.warning("This file should contain merged v2 and v2.1 strategies.")
logger.warning("="*80)
try:
    # Import ALL desired functions explicitly
    from trading_strategies_v2 import (
         rsi_strategy, bollinger_bands_strategy, momentum_strategy, mean_reversion_strategy,
        triple_moving_average_strategy, volume_price_trend_strategy, keltner_channel_strategy,
        dual_thrust_strategy, adaptive_momentum_strategy as adaptive_momentum_strategy_v2, # Renamed if needed
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
        adaptive_momentum_filter_strategy, # Potentially duplicate name - check your merge
        fractal_market_hypothesis_strategy, topological_data_analysis_strategy, levy_distribution_strategy,
        information_flow_strategy, wavelet_momentum_strategy, complex_network_strategy,
        zeta_potential_strategy, quantum_oscillator_strategy, simple_trend_reversal_strategy
    )
    # List all imported function objects again here
    all_imported_strategies = [
        rsi_strategy, bollinger_bands_strategy, momentum_strategy, mean_reversion_strategy,
        triple_moving_average_strategy, volume_price_trend_strategy, keltner_channel_strategy,
        dual_thrust_strategy, adaptive_momentum_strategy_v2, # Use potentially renamed one
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
        adaptive_momentum_filter_strategy, # The v2.1 version
        fractal_market_hypothesis_strategy, topological_data_analysis_strategy, levy_distribution_strategy,
        information_flow_strategy, wavelet_momentum_strategy, complex_network_strategy,
        zeta_potential_strategy, quantum_oscillator_strategy, simple_trend_reversal_strategy
    ]
    # Check for missing libs
    try: import pywt
    except ImportError: logger.warning("Missing dependency: 'pywt'")
    try: import ripser
    except ImportError: logger.warning("Missing dependency: 'ripser'")
    try: from scipy.special import zeta
    except ImportError: logger.warning("Missing dependency: 'scipy'")

    STRATEGIES_TO_TEST = [
        strat for strat in all_imported_strategies
        if strat.__name__ not in EXCLUDE_STRATEGIES
    ]
    STRATEGIES_TO_TEST = list(dict.fromkeys(STRATEGIES_TO_TEST))

    logger.info(f"Successfully prepared {len(STRATEGIES_TO_TEST)} strategies for testing from merged file.")
    if EXCLUDE_STRATEGIES: logger.info(f"Excluded strategies: {', '.join(EXCLUDE_STRATEGIES)}")

except ImportError as e:
    logger.error(f"Critical Error importing strategies from {STRATEGY_MODULE_NAME}: {e}.")
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error during strategy import setup: {e}", exc_info=True)
    sys.exit(1)

if not STRATEGIES_TO_TEST:
    logger.error("No strategies were successfully imported or selected. Exiting.")
    sys.exit(1)


# --- Performance Metrics Calculation --- (Same as previous version)
def calculate_performance_metrics(daily_values: pd.Series):
    """Calculates key performance metrics from a series of daily portfolio values."""
    if daily_values.empty or daily_values.isnull().all() or len(daily_values) < 2:
        return {'total_return_pct': 0.0, 'max_drawdown_pct': 0.0, 'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'calmar_ratio': 0.0}
    daily_values = daily_values.fillna(method='ffill').dropna()
    if len(daily_values) < 2: return {'total_return_pct': 0.0, 'max_drawdown_pct': 0.0, 'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'calmar_ratio': 0.0}
    total_return = (daily_values.iloc[-1] / daily_values.iloc[0]) - 1
    cumulative_max = daily_values.cummax()
    drawdown = (daily_values - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    daily_returns = daily_values.pct_change().dropna()
    if daily_returns.empty: return {'total_return_pct': total_return * 100, 'max_drawdown_pct': abs(max_drawdown * 100), 'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'calmar_ratio': 0.0}
    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    sharpe_ratio = (avg_daily_return / (std_daily_return + EPSILON)) * np.sqrt(252) if std_daily_return > EPSILON else 0.0
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = (avg_daily_return / (downside_std + EPSILON)) * np.sqrt(252) if downside_std > EPSILON else 0.0
    annualized_return = ((1 + total_return) ** (252 / len(daily_values))) - 1 if len(daily_values) > 0 else 0.0
    calmar_ratio = annualized_return / (abs(max_drawdown) + EPSILON) if max_drawdown < -EPSILON else 0.0
    return {'total_return_pct': total_return * 100, 'max_drawdown_pct': abs(max_drawdown * 100), 'sharpe_ratio': sharpe_ratio, 'sortino_ratio': sortino_ratio, 'calmar_ratio': calmar_ratio}


# --- Simulation Cache Helper Functions ---
def get_sim_cache_key_params():
    """Returns a dictionary of parameters affecting simulation outcome for hashing."""
    return {
        "initial_cash": INITIAL_CASH,
        "commission": COMMISSION_PCT,
        "slippage": SLIPPAGE_PCT,
        "min_hist": MIN_HISTORY_DAYS,
        # Add other crucial parameters from control.py if they affect strategy logic directly
    }

def get_sim_cache_filename(strategy_name, ticker, start_date, end_date):
    """Generates a consistent filename for simulation cache."""
    params_str = str(sorted(get_sim_cache_key_params().items()))
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8] # Short hash
    # Clean ticker name for filename
    safe_ticker = "".join(c for c in ticker if c.isalnum() or c in ('-', '_')).rstrip()
    filename = f"sim_{safe_ticker}_{strategy_name}_{start_date}_{end_date}_{params_hash}.pkl"
    return SIM_CACHE_DIR / filename

def load_from_sim_cache(cache_file_path):
    """Loads simulation result from a pickle file."""
    try:
        if cache_file_path.exists():
            with open(cache_file_path, 'rb') as f:
                return pickle.load(f)
        return None
    except (pickle.UnpicklingError, EOFError, FileNotFoundError, Exception) as e:
        logger.warning(f"Error loading simulation cache file {cache_file_path}: {e}. Will recompute.")
        try:
            if cache_file_path.exists(): cache_file_path.unlink() # Remove potentially corrupted file
        except OSError: pass
        return None

def save_to_sim_cache(result_data, cache_file_path):
    """Saves simulation result to a pickle file."""
    try:
        with open(cache_file_path, 'wb') as f:
            pickle.dump(result_data, f)
    except Exception as e:
        logger.error(f"Error saving simulation cache to {cache_file_path}: {e}")


# --- Data Fetching --- (Modified to return yf cache file path info)
def fetch_stock_data(tickers, benchmark_ticker, start_date, end_date):
    """ Fetches/Loads cached historical data. Returns dict of {ticker: df} and benchmark_df."""
    # ... (rest of the fetching logic is the same as previous version) ...

    all_symbols_to_fetch = tickers + ([benchmark_ticker] if benchmark_ticker else [])
    symbols_map = {f"{ticker}.NS" if not ticker.endswith((".NS", ".BO")) else ticker: ticker for ticker in all_symbols_to_fetch}
    yf_symbols = list(symbols_map.keys())
    logger.info(f"Fetching/Loading data for {len(tickers)} symbols + benchmark '{benchmark_ticker}'...")

    fetch_start_date_dt = pd.to_datetime(start_date) - pd.Timedelta(days=MIN_HISTORY_DAYS * 2)
    fetch_start_date_str = fetch_start_date_dt.strftime("%Y-%m-%d")
    fetch_start_date_dt_dl = pd.to_datetime(start_date) - pd.Timedelta(days=MIN_HISTORY_DAYS * 2 + 7)
    fetch_start_date_str_dl = fetch_start_date_dt_dl.strftime("%Y-%m-%d")

    all_data = {}
    yf_cache_info = {} # Store info about yfinance cache files
    symbols_to_download = []
    cache_expiry_threshold = datetime.datetime.now() - datetime.timedelta(days=CACHE_EXPIRY_DAYS)

    # Check cache first
    for yf_ticker in yf_symbols:
        base_ticker = symbols_map[yf_ticker]
        cache_filename = f"{base_ticker.replace('.NS', '').replace('.BO', '').replace('^','')}_{fetch_start_date_str}_{end_date}.parquet"
        cache_file_path = YF_CACHE_DIR / cache_filename # Use YF_CACHE_DIR
        load_from_cache = False
        if cache_file_path.exists():
            try:
                mod_time_ts = os.path.getmtime(cache_file_path)
                mod_time = datetime.datetime.fromtimestamp(mod_time_ts)
                if mod_time > cache_expiry_threshold:
                    df = pd.read_parquet(cache_file_path)
                    if not df.empty and 'close' in [col.lower() for col in df.columns]:
                        df.columns = [col.lower() for col in df.columns]
                        df.index = pd.to_datetime(df.index)
                        if df['close'].isnull().sum() / len(df) > 0.1 or (df['close'] <= 0).sum() > 5:
                             logger.warning(f"YF Cache file {cache_file_path} has excessive NaNs/zeros. Re-downloading {base_ticker}.")
                             if cache_file_path.exists(): cache_file_path.unlink()
                        else:
                             df.dropna(subset=['close'], inplace=True)
                             if not df.empty:
                                all_data[base_ticker] = df
                                yf_cache_info[base_ticker] = {'path': cache_file_path, 'mtime': mod_time_ts}
                                load_from_cache = True
                             else:
                                 logger.warning(f"YF Cache file {cache_file_path} empty after dropna. Re-downloading {base_ticker}.")
                                 if cache_file_path.exists(): cache_file_path.unlink()
                    else:
                        logger.warning(f"YF Cache file {cache_file_path} empty/invalid. Re-downloading {base_ticker}.")
                        if cache_file_path.exists(): cache_file_path.unlink()
            except Exception as e:
                logger.warning(f"Error reading YF cache {cache_file_path} for {base_ticker}: {e}. Will re-download.")
                try:
                    if cache_file_path.exists(): cache_file_path.unlink()
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
             logger.info(f"Downloading batch {i//batch_size + 1}/{ (len(symbols_to_download)-1)//batch_size + 1 } ({len(batch)} symbols)...")
             try:
                 data = yf.download(batch, start=fetch_start_date_str_dl, end=end_date, group_by='ticker', auto_adjust=True, prepost=False, threads=True, progress=False)
                 if data.empty: logger.warning(f"No data returned for batch {i//batch_size + 1}"); symbols_processed.update(batch); continue

                 for yf_ticker in batch:
                    base_ticker = symbols_map[yf_ticker]
                    symbols_processed.add(yf_ticker)
                    ticker_df = None
                    try:
                        if isinstance(data.columns, pd.MultiIndex):
                            if yf_ticker in data.columns.levels[0]: ticker_df = data[yf_ticker].copy()
                        elif yf_ticker in data: ticker_df = data[yf_ticker].copy()
                        elif len(batch) == 1 and not isinstance(data.columns, pd.MultiIndex): ticker_df = data.copy()

                        if ticker_df is None or ticker_df.empty: logger.warning(f"No data for {yf_ticker} ({base_ticker}) in batch."); continue

                        ticker_df.columns = [col.lower() for col in ticker_df.columns]
                        ticker_df.index = pd.to_datetime(ticker_df.index)
                        if ticker_df['close'].isnull().sum() / len(ticker_df) > 0.1 or (ticker_df['close'] <= 0).sum() > 5:
                             logger.warning(f"Downloaded data for {base_ticker} has excessive NaNs/zeros. Skipping."); continue
                        ticker_df.dropna(subset=['close'], inplace=True)
                        if ticker_df.empty: logger.warning(f"Downloaded data {base_ticker} empty after dropna. Skipping."); continue

                        all_data[base_ticker] = ticker_df

                        cache_filename = f"{base_ticker.replace('.NS', '').replace('.BO', '').replace('^','')}_{fetch_start_date_str}_{end_date}.parquet"
                        cache_file_path = YF_CACHE_DIR / cache_filename # Use YF_CACHE_DIR
                        try:
                            ticker_df.to_parquet(cache_file_path)
                            yf_cache_info[base_ticker] = {'path': cache_file_path, 'mtime': os.path.getmtime(cache_file_path)}
                        except Exception as e: logger.warning(f"Failed to write YF cache for {base_ticker}: {e}")

                    except Exception as e: logger.warning(f"Error processing download for {yf_ticker} ({base_ticker}): {e}", exc_info=False)

             except Exception as e: logger.error(f"Error downloading batch {i//batch_size + 1}: {e}", exc_info=False); symbols_processed.update(batch)
             time.sleep(0.5)
        not_processed = set(symbols_to_download) - symbols_processed
        if not_processed: logger.warning(f"Could not process download for: {', '.join(not_processed)}")

    # Final validation
    validated_data = {}
    benchmark_data = None
    sim_start_dt = pd.to_datetime(start_date)
    sim_end_dt = pd.to_datetime(end_date)
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    for base_ticker, df in all_data.items():
        if df is None or df.empty: continue
        df.columns = [col.lower() for col in df.columns]
        if not required_cols.issubset(df.columns): logger.warning(f"Data {base_ticker} missing columns. Skipping."); continue
        if df.index.min() > fetch_start_date_dt: logger.warning(f"Insufficient history {base_ticker}. Skipping."); continue
        if df[(df.index >= sim_start_dt) & (df.index <= sim_end_dt)].empty: logger.warning(f"Data {base_ticker} no entries in sim period. Skipping."); continue

        if base_ticker == benchmark_ticker: benchmark_data = df.copy()
        else: validated_data[base_ticker] = df.copy()

    logger.info(f"Data ready for {len(validated_data)}/{len(tickers)} symbols.")
    if not validated_data: logger.error("No valid stock data. Exiting."); sys.exit(1)
    if benchmark_ticker and benchmark_data is None: logger.warning(f"No valid benchmark data for {benchmark_ticker}. Skipping benchmark.")

    # Return data and cache info for simulation caching validation
    return validated_data, benchmark_data, yf_cache_info


# --- Simulation Worker Function --- (Same as previous version)
def simulate_single_strategy_ticker(args):
    """Worker function to simulate one strategy on one ticker, including costs and daily values."""
    strategy, ticker, ticker_data, start_date_str, end_date_str, commission_pct, slippage_pct = args
    strategy_name = strategy.__name__
    daily_portfolio_values = {}

    try:
        ticker_data.columns = [col.lower() for col in ticker_data.columns]
        portfolio = {"cash": INITIAL_CASH, "quantity": 0.0, "total_portfolio_value": INITIAL_CASH, "trades": 0, "last_price": 0.0, "commission_paid": 0.0, "final_pl_pct": 0.0}
        sim_start_dt = pd.to_datetime(start_date_str); sim_end_dt = pd.to_datetime(end_date_str)
        full_date_range = pd.date_range(start=sim_start_dt, end=sim_end_dt, freq='B')
        simulation_days_data = ticker_data[(ticker_data.index >= sim_start_dt) & (ticker_data.index <= sim_end_dt)].copy()

        if simulation_days_data.empty: return strategy_name, ticker, portfolio, pd.Series(dtype=float)
        first_valid_date = simulation_days_data.index[0]; daily_portfolio_values[first_valid_date] = INITIAL_CASH
        last_recorded_value = INITIAL_CASH

        for current_date in full_date_range:
            if current_date in simulation_days_data.index:
                day_data = simulation_days_data.loc[current_date]; current_price = day_data['close']
                if pd.isna(current_price) or current_price <= 0: daily_portfolio_values[current_date] = last_recorded_value; continue
                portfolio["last_price"] = current_price
                loc_in_full_data = ticker_data.index.get_loc(current_date)

                if loc_in_full_data >= MIN_HISTORY_DAYS:
                    historical_slice = ticker_data.iloc[:loc_in_full_data + 1].copy()
                    if not historical_slice.empty and len(historical_slice) >= MIN_HISTORY_DAYS:
                        historical_slice.columns = [col.lower() for col in historical_slice.columns]
                        try:
                            action, quantity_signal, _ = strategy(ticker=ticker, current_price=current_price, historical_data=historical_slice, account_cash=portfolio["cash"], portfolio_qty=portfolio["quantity"], total_portfolio_value=portfolio["total_portfolio_value"])
                            if action in ["buy", "strong buy"] and np.isfinite(quantity_signal) and quantity_signal > 0:
                                buy_quantity = quantity_signal; base_cost = buy_quantity * current_price; slippage_cost = base_cost * slippage_pct; commission_cost = base_cost * commission_pct; total_cost = base_cost + slippage_cost + commission_cost
                                if total_cost > 0 and portfolio["cash"] >= total_cost: portfolio["cash"] -= total_cost; portfolio["quantity"] += buy_quantity; portfolio["trades"] += 1; portfolio["commission_paid"] += commission_cost
                            elif action in ["sell", "strong sell"] and np.isfinite(quantity_signal) and quantity_signal > 0:
                                sell_quantity = min(portfolio["quantity"], quantity_signal)
                                if sell_quantity > 0:
                                    base_revenue = sell_quantity * current_price; slippage_deduction = base_revenue * slippage_pct; commission_deduction = base_revenue * commission_pct; total_revenue = base_revenue - slippage_deduction - commission_deduction
                                    if total_revenue > 0: portfolio["cash"] += total_revenue; portfolio["quantity"] -= sell_quantity; portfolio["trades"] += 1; portfolio["commission_paid"] += commission_deduction
                        except Exception: pass # Continue on strategy error

                holding_value = portfolio["quantity"] * portfolio["last_price"]
                portfolio["total_portfolio_value"] = portfolio["cash"] + holding_value
                daily_portfolio_values[current_date] = portfolio["total_portfolio_value"]
                last_recorded_value = portfolio["total_portfolio_value"]
            else:
                 if current_date >= first_valid_date: daily_portfolio_values[current_date] = last_recorded_value

        final_value = portfolio["total_portfolio_value"]; portfolio["final_value"] = final_value
        portfolio["final_pl_pct"] = ((final_value - INITIAL_CASH) / INITIAL_CASH) * 100 if INITIAL_CASH > 0 else 0.0
        daily_values_series = pd.Series(daily_portfolio_values).sort_index().reindex(full_date_range, method='ffill')
        return strategy_name, ticker, portfolio, daily_values_series
    except Exception:
        return strategy_name, ticker, {"error": "Worker Error", "final_value": INITIAL_CASH, "final_pl_pct": 0.0, "trades": 0, "commission_paid": 0.0}, pd.Series(dtype=float)


# --- Buy and Hold Simulation --- (Same as previous version)
def simulate_buy_and_hold(ticker_data, start_date_str, end_date_str, initial_cash):
    sim_start_dt=pd.to_datetime(start_date_str); sim_end_dt=pd.to_datetime(end_date_str)
    sim_data = ticker_data[(ticker_data.index >= sim_start_dt) & (ticker_data.index <= sim_end_dt)].copy()
    if sim_data.empty: return pd.Series(dtype=float)
    start_price = sim_data['close'].iloc[0]; quantity = initial_cash / start_price if start_price > 0 else 0
    daily_values = quantity * sim_data['close']; full_date_range = pd.date_range(start=sim_start_dt, end=sim_end_dt, freq='B')
    daily_values = daily_values.reindex(full_date_range, method='ffill').fillna(initial_cash)
    return daily_values

# --- Modified Simulation Runner (Parallel with Cache) ---
def run_simulation_parallel_cached(all_ticker_data, strategies, yf_cache_info):
    """Runs the backtest simulation in parallel, using simulation cache if enabled."""
    logger.info(f"Preparing tasks for parallel simulation across {NUM_CPUS} cores...")

    tasks_to_run = []
    cached_results = []
    cache_hits = 0
    total_tasks = 0

    for strategy in strategies:
        strategy_name = strategy.__name__
        for ticker, ticker_data in all_ticker_data.items():
            total_tasks += 1
            task_args = (strategy, ticker, ticker_data.copy(), START_DATE, END_DATE, COMMISSION_PCT, SLIPPAGE_PCT)

            if USE_SIMULATION_CACHE:
                cache_file = get_sim_cache_filename(strategy_name, ticker, START_DATE, END_DATE)
                cached_data = load_from_sim_cache(cache_file)
                if cached_data:
                    # Optional: Add validation against yf_cache modification time here if needed
                    # yf_info = yf_cache_info.get(ticker)
                    # sim_cache_mtime = os.path.getmtime(cache_file)
                    # if yf_info and sim_cache_mtime >= yf_info['mtime']: # Check if sim cache is newer than data cache
                    cached_results.append(cached_data)
                    cache_hits += 1
                    continue # Skip adding to tasks_to_run
                # else: cache miss or error loading

            tasks_to_run.append(task_args) # Add task if cache disabled or missed

    logger.info(f"Total simulation tasks: {total_tasks}")
    if USE_SIMULATION_CACHE:
        logger.info(f"Simulation cache hits: {cache_hits} ({cache_hits/total_tasks*100:.1f}%)")
    logger.info(f"Tasks requiring computation: {len(tasks_to_run)}")

    if not tasks_to_run and not cached_results:
        logger.warning("No simulation tasks to run and no cached results found.")
        return {}
    if not tasks_to_run:
        logger.info("All results loaded from simulation cache.")

    new_results = []
    if tasks_to_run:
        start_sim_time = time.time()
        with Pool(processes=NUM_CPUS) as pool:
            try:
                pool_iterator = pool.imap_unordered(simulate_single_strategy_ticker, tasks_to_run)
                for result_item in tqdm(pool_iterator, total=len(tasks_to_run), desc="Simulating Strategies", unit="task"):
                    if result_item:
                        new_results.append(result_item)
                        # Save newly computed result to cache if enabled
                        if USE_SIMULATION_CACHE:
                           strategy_name_res, ticker_res, _, _ = result_item # Get key info
                           cache_file_path = get_sim_cache_filename(strategy_name_res, ticker_res, START_DATE, END_DATE)
                           save_to_sim_cache(result_item, cache_file_path)

                logger.info(f"Parallel processing finished in {time.time() - start_sim_time:.2f} seconds.")
            except Exception as e:
                logger.error(f"Error during parallel execution: {e}", exc_info=True)
                # Combine whatever new results we got with cached ones
    else: # Only cached results
        pass


    # Combine cached and new results
    all_results_list = cached_results + new_results

    # Reconstruct the final results dictionary
    final_results = {} # {ticker: {strategy_name: {'portfolio': {...}, 'daily_values': pd.Series}}}
    error_count = 0
    successful_count = 0
    for result_item in all_results_list:
         if result_item: # Should always be true here, but check anyway
            strategy_name, ticker, portfolio, daily_values_series = result_item
            if "error" in portfolio:
                error_count += 1
            else:
                successful_count += 1
                if ticker not in final_results:
                    final_results[ticker] = {}
                final_results[ticker][strategy_name] = {
                    'portfolio': portfolio,
                    'daily_values': daily_values_series
                }
    logger.info(f"Total processed simulation results: {successful_count} successful, {error_count} failed/skipped.")
    return final_results


# --- Plotting Functions --- (Same as previous version)
def plot_equity_curves(ticker, avg_strategy_values, bh_values, benchmark_values, filename):
    plt.figure(figsize=(12, 7)); start_val = INITIAL_CASH
    if avg_strategy_values is not None and not avg_strategy_values.empty: plt.plot(avg_strategy_values.index, avg_strategy_values, label=f'Avg Strategy ({ticker})', color='blue'); start_val = avg_strategy_values.iloc[0]
    if bh_values is not None and not bh_values.empty: bh_normalized = (bh_values / bh_values.iloc[0]) * start_val if bh_values.iloc[0] > 0 else bh_values; plt.plot(bh_normalized.index, bh_normalized, label=f'Buy & Hold ({ticker})', color='green', linestyle='--'); start_val = bh_values.iloc[0] if start_val == INITIAL_CASH else start_val
    if benchmark_values is not None and not benchmark_values.empty: benchmark_normalized = (benchmark_values / benchmark_values.iloc[0]) * start_val if benchmark_values.iloc[0] > 0 else benchmark_values; plt.plot(benchmark_normalized.index, benchmark_normalized, label=f'Benchmark ({BENCHMARK_TICKER})', color='red', linestyle=':')
    plt.title(f'Equity Curve Comparison: {ticker} (Normalized Start)'); plt.xlabel('Date'); plt.ylabel('Portfolio Value'); plt.legend(); plt.grid(True); plt.tight_layout();
    try: plt.savefig(filename)
    except Exception as e: logger.error(f"Failed to save plot {filename}: {e}")
    plt.close()

def plot_score_distribution(scores, filename):
    plt.figure(figsize=(10, 6)); plt.hist(scores, bins=20, color='skyblue', edgecolor='black'); plt.title('Distribution of Ranking Scores'); plt.xlabel('Score'); plt.ylabel('Frequency'); plt.grid(axis='y', alpha=0.75);
    try: plt.savefig(filename)
    except Exception as e: logger.error(f"Failed to save score distribution plot {filename}: {e}")
    plt.close()

# --- Analysis and Recommendation --- (Same as previous version)
def analyze_and_recommend(simulation_results, buy_hold_results, benchmark_data):
    logger.info("\n--- Analyzing Strategy Performance Across Tickers ---"); ticker_analysis = {}; benchmark_metrics = {}; benchmark_daily_values = None
    if benchmark_data is not None:
        benchmark_sim_start = pd.to_datetime(START_DATE); benchmark_sim_end = pd.to_datetime(END_DATE)
        benchmark_relevant = benchmark_data[(benchmark_data.index >= benchmark_sim_start) & (benchmark_data.index <= benchmark_sim_end)]['close'].copy()
        if not benchmark_relevant.empty:
             benchmark_daily_values = (benchmark_relevant / benchmark_relevant.iloc[0]) * INITIAL_CASH; full_date_range = pd.date_range(start=benchmark_sim_start, end=benchmark_sim_end, freq='B'); benchmark_daily_values = benchmark_daily_values.reindex(full_date_range, method='ffill').fillna(INITIAL_CASH)
             benchmark_metrics = calculate_performance_metrics(benchmark_daily_values); logger.info(f"Benchmark ({BENCHMARK_TICKER}) Metrics: Return={benchmark_metrics.get('total_return_pct', 0.0):.2f}%, MaxDD={benchmark_metrics.get('max_drawdown_pct', 0.0):.2f}%, Sharpe={benchmark_metrics.get('sharpe_ratio', 0.0):.2f}")
        else: logger.warning(f"No benchmark data in period."); benchmark_metrics = {k: 0.0 for k in ['total_return_pct', 'max_drawdown_pct', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio']}

    for ticker, strategy_results in simulation_results.items():
        if not strategy_results: continue
        ticker_analysis[ticker] = {'strategy_pl_pct': {}, 'strategy_trades': {}, 'strategy_commissions': {}, 'all_daily_values': [], 'valid_strategy_count': 0, 'last_price': 0.0}
        for strategy_name, result_data in strategy_results.items():
            portfolio = result_data['portfolio']; daily_values = result_data['daily_values']
            if not daily_values.empty:
                ticker_analysis[ticker]['strategy_pl_pct'][strategy_name] = portfolio.get('final_pl_pct', 0.0); ticker_analysis[ticker]['strategy_trades'][strategy_name] = portfolio.get('trades', 0); ticker_analysis[ticker]['strategy_commissions'][strategy_name] = portfolio.get('commission_paid', 0.0)
                ticker_analysis[ticker]['all_daily_values'].append(daily_values); ticker_analysis[ticker]['valid_strategy_count'] += 1
                if portfolio.get('last_price', 0.0) > 0: ticker_analysis[ticker]['last_price'] = portfolio.get('last_price')

    ranked_tickers = []; all_scores = []
    for ticker, analysis in ticker_analysis.items():
        if analysis['valid_strategy_count'] == 0 or analysis['last_price'] <= 0: logger.warning(f"Ticker {ticker} invalid results/price. Excluding."); continue
        avg_daily_values = pd.concat(analysis['all_daily_values'], axis=1).mean(axis=1); avg_strategy_metrics = calculate_performance_metrics(avg_daily_values)
        pl_values = list(analysis['strategy_pl_pct'].values()); avg_pl_pct = np.mean(pl_values) if pl_values else 0.0; pl_std_dev = np.std(pl_values) if len(pl_values) > 1 else 0.0
        positive_strategies = sum(1 for pl in pl_values if pl > 0); positive_ratio = positive_strategies / analysis['valid_strategy_count'] if analysis['valid_strategy_count'] > 0 else 0.0
        avg_trades = np.mean(list(analysis['strategy_trades'].values())) if analysis['strategy_trades'] else 0.0; avg_commission = np.mean(list(analysis['strategy_commissions'].values())) if analysis['strategy_commissions'] else 0.0
        if avg_trades < MIN_AVG_TRADES_FILTER: logger.info(f"Filtering {ticker}, low avg trades ({avg_trades:.2f} < {MIN_AVG_TRADES_FILTER})."); continue

        if RANKING_METRIC == 'risk_adjusted_pl': ranking_score = avg_strategy_metrics['sharpe_ratio']
        elif RANKING_METRIC == 'sortino': ranking_score = avg_strategy_metrics['sortino_ratio']
        elif RANKING_METRIC == 'calmar': ranking_score = avg_strategy_metrics['calmar_ratio']
        elif RANKING_METRIC == 'weighted_pl': ranking_score = avg_pl_pct * positive_ratio
        else: ranking_score = avg_strategy_metrics['total_return_pct']
        all_scores.append(ranking_score)

        sorted_strats = sorted(analysis['strategy_pl_pct'].items(), key=lambda item: item[1], reverse=True); top_strategies = sorted_strats[:STRATEGY_DETAIL_COUNT]; worst_strategies = sorted_strats[-STRATEGY_DETAIL_COUNT:]
        bh_daily_values = buy_hold_results.get(ticker); bh_metrics = calculate_performance_metrics(bh_daily_values) if bh_daily_values is not None else {k: 0.0 for k in avg_strategy_metrics}

        ranked_tickers.append({
            'ticker': ticker, 'score': ranking_score, 'last_price': analysis['last_price'], 'strategy_count': analysis['valid_strategy_count'],
            'avg_strat_return_pct': avg_strategy_metrics['total_return_pct'], 'avg_strat_max_drawdown_pct': avg_strategy_metrics['max_drawdown_pct'],
            'avg_strat_sharpe': avg_strategy_metrics['sharpe_ratio'], 'avg_strat_sortino': avg_strategy_metrics['sortino_ratio'], 'avg_strat_calmar': avg_strategy_metrics['calmar_ratio'],
            'avg_pl_pct_all': avg_pl_pct, 'pl_std_dev': pl_std_dev, 'positive_strat_ratio': positive_ratio, 'avg_trades_per_strat': avg_trades, 'avg_commission_per_strat': avg_commission,
            'bh_return_pct': bh_metrics['total_return_pct'], 'bh_max_drawdown_pct': bh_metrics['max_drawdown_pct'], 'bh_sharpe': bh_metrics['sharpe_ratio'],
            'top_strategies': top_strategies, 'worst_strategies': worst_strategies, 'avg_daily_values': avg_daily_values, 'bh_daily_values': bh_daily_values
        })

    ranked_tickers.sort(key=lambda x: x['score'], reverse=True)

    if GENERATE_PLOTS:
        if all_scores: dist_filename = PLOTS_DIR / f"score_distribution_{RANKING_METRIC}_{datetime.datetime.now().strftime('%Y%m%d')}.png"; plot_score_distribution(all_scores, dist_filename)
        logger.info(f"Generating equity plots for top {PLOT_TOP_N} stocks...")
        for i, item in enumerate(ranked_tickers[:PLOT_TOP_N]):
            ticker = item['ticker']; plot_filename = PLOTS_DIR / f"equity_{ticker}_{datetime.datetime.now().strftime('%Y%m%d')}.png"
            plot_equity_curves(ticker, item.get('avg_daily_values'), item.get('bh_daily_values'), benchmark_daily_values, plot_filename)

    logger.info(f"\n--- Stock Recommendations by Price Category (Ranked by: {RANKING_METRIC}) ---")
    logger.info(f"--- Benchmark ({BENCHMARK_TICKER}): Return={benchmark_metrics.get('total_return_pct', 0.0):.2f}%, MaxDD={benchmark_metrics.get('max_drawdown_pct', 0.0):.2f}%, Sharpe={benchmark_metrics.get('sharpe_ratio', 0.0):.2f} ---")
    recommendations_by_category = {"Above_1000": [], "100_to_1000": [], "Below_100": []}
    for item in ranked_tickers:
        price = item['last_price']; category = "Above_1000" if price > 1000 else ("100_to_1000" if 100 <= price <= 1000 else "Below_100")
        if len(recommendations_by_category[category]) < TOP_N_PER_CATEGORY: recommendations_by_category[category].append(item)

    header1 = f"{'Rank':<5} {'Ticker':<12} {'Price':>9} {'Score':>10} | {'Avg Strat':^45} | {'Buy & Hold':^30} | {'Consistency':^36}"
    header2 = f"{'':<5} {'':<12} {'':>9} {'':>10} | {'Return%':>9} {'MaxDD%':>9} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} | {'Return%':>9} {'MaxDD%':>9} {'Sharpe':>8} | {'Strat +%':>9} {'StdDev%':>9} {'AvgTrades':>9}"
    separator = "-" * (len(header1) + 5)
    for category, items in recommendations_by_category.items():
        print(f"\n\n--- Category: {category.replace('_', ' ')} (Top {TOP_N_PER_CATEGORY}) ---"); print(separator); print(header1); print(header2); print(separator)
        if not items: print(f"No stocks found.")
        else:
            for i, item in enumerate(items, 1):
                 print(f"{i:<5} {item['ticker']:<12} {item['last_price']:>9.2f} {item['score']:>10.2f} | " f"{item['avg_strat_return_pct']:>9.2f} {item['avg_strat_max_drawdown_pct']:>9.2f} {item['avg_strat_sharpe']:>8.2f} {item['avg_strat_sortino']:>8.2f} {item['avg_strat_calmar']:>8.2f} | " f"{item['bh_return_pct']:>9.2f} {item['bh_max_drawdown_pct']:>9.2f} {item['bh_sharpe']:>8.2f} | " f"{item['positive_strat_ratio']:>8.1%} {item['pl_std_dev']:>9.2f} {item['avg_trades_per_strat']:>9.1f}")
                 top_strats_str = ", ".join([f"{s[0].replace('_strategy','')} ({s[1]:.1f}%)" for s in item['top_strategies']]); worst_strats_str = ", ".join([f"{s[0].replace('_strategy','')} ({s[1]:.1f}%)" for s in item['worst_strategies']])
                 print(f"{'':<5} {' ': <12} {' ':>9} {' ':>10} |   Top Strats : {top_strats_str}"); print(f"{'':<5} {' ': <12} {' ':>9} {' ':>10} |   Worst Strats: {worst_strats_str}")
        print(separator)

    logger.info("\n--- Recommendation Justification & Expectations ---"); logger.warning("!! Using strategies from MODIFIED file: {} !!".format(STRATEGIES_DIR / (STRATEGY_MODULE_NAME + '.py'))); logger.info(f"Ranking Metric: '{RANKING_METRIC}' across {len(STRATEGIES_TO_TEST)} strategies tested ({START_DATE} to {END_DATE})."); logger.info(f"Simulation Costs: Commission={COMMISSION_PCT*100:.3f}%, Slippage={SLIPPAGE_PCT*100:.3f}% per trade."); logger.info("Categorized by price on simulation end date."); logger.info("Metrics shown are averages across successful strategy simulations per stock."); logger.info(f"B&H = Buy and Hold performance for the stock itself."); logger.info(f"Benchmark = {BENCHMARK_TICKER} (Return: {benchmark_metrics.get('total_return_pct', 0.0):.2f}%)"); logger.warning("\n**DISCLAIMER:** Historical backtest simulation. Past performance NOT indicative of future results. This is NOT financial advice. Use results for research ONLY.**")
    return ranked_tickers


# --- Function to Save Results --- (Same as previous version)
def save_recommendations_to_csv(ranked_data, filename="stock_recommendations.csv"):
    if not ranked_data: logger.warning("No ranked data to save."); return
    filepath = RESULTS_DIR / filename
    try:
        flat_data = []
        for item in ranked_data:
            flat_item = item.copy(); flat_item['top_strategies'] = ", ".join([f"{s[0]}:{s[1]:.2f}" for s in item.get('top_strategies', [])]); flat_item['worst_strategies'] = ", ".join([f"{s[0]}:{s[1]:.2f}" for s in item.get('worst_strategies', [])])
            flat_item.pop('avg_daily_values', None); flat_item.pop('bh_daily_values', None); flat_data.append(flat_item)
        if not flat_data: logger.warning("No data left after flattening for CSV."); return
        keys = flat_data[0].keys()
        with open(filepath, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys); dict_writer.writeheader(); dict_writer.writerows(flat_data)
        logger.info(f"Recommendations saved successfully to {filepath}")
    except Exception as e: logger.error(f"Failed to save recommendations to CSV: {e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    logger.info(f"Starting Stock Recommendation Backtest")
    logger.info(f"Period: {START_DATE} to {END_DATE}")
    logger.info(f"Universe: {len(STOCK_UNIVERSE_SYMBOLS)} symbols")
    logger.info(f"Benchmark: {BENCHMARK_TICKER}")
    logger.info(f"Strategies Source: {STRATEGIES_DIR / (STRATEGY_MODULE_NAME + '.py')}")
    logger.info(f"Strategies Count: {len(STRATEGIES_TO_TEST)}")
    logger.info(f"Ranking Metric: {RANKING_METRIC}")
    logger.info(f"Simulation Cache Enabled: {USE_SIMULATION_CACHE}")
    logger.info(f"Using {NUM_CPUS} CPU cores for parallel simulation.")
    script_start_time = datetime.datetime.now()

    if len(STOCK_UNIVERSE_SYMBOLS) < 10: logger.warning("Stock universe list seems small.")

    all_stock_data, benchmark_data, yf_cache_info = fetch_stock_data(STOCK_UNIVERSE_SYMBOLS, BENCHMARK_TICKER, START_DATE, END_DATE)

    if all_stock_data:
        logger.info("Running Buy & Hold simulations...")
        buy_hold_results = {}
        for ticker, data in tqdm(all_stock_data.items(), desc="Simulating Buy&Hold", unit="ticker"):
             buy_hold_results[ticker] = simulate_buy_and_hold(data, START_DATE, END_DATE, INITIAL_CASH)

        # Use the cached simulation runner
        simulation_results = run_simulation_parallel_cached(all_stock_data, STRATEGIES_TO_TEST, yf_cache_info)

        if simulation_results:
            full_ranked_list = analyze_and_recommend(simulation_results, buy_hold_results, benchmark_data)
            if full_ranked_list:
                 timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                 csv_filename = f"recommendations_v2merged_{RANKING_METRIC}_{timestamp}.csv"
                 save_recommendations_to_csv(full_ranked_list, filename=csv_filename)
        else: logger.error("Strategy simulation produced no results. Check logs for errors.")
    else: logger.error("Failed to fetch sufficient stock data. Aborting simulation.")

    script_end_time = datetime.datetime.now()
    logger.info(f"Full backtest, analysis, and recommendation finished in {script_end_time - script_start_time}")
    logger.info(f"Log file saved to: {log_file_path}")
    if GENERATE_PLOTS: logger.info(f"Plots saved in: {PLOTS_DIR}")
    if USE_SIMULATION_CACHE: logger.info(f"Simulation cache stored in: {SIM_CACHE_DIR}")
    logger.info(f"CSV results saved in: {RESULTS_DIR}")