# generate_report.py
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import argparse
from pathlib import Path
import datetime
import logging
import sys
import re
import base64 # For embedding images
import io # For image handling

# --- Plotting (for generating charts within this script) ---
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

# --- Data Fetching (for Benchmark) & Performance Calc ---
import yfinance as yf
import numpy as np

# Set default template for plotly
pio.templates.default = "plotly_white"

# --- Configuration ---
try:
    parent_dir = Path(__file__).resolve().parent.parent
    if str(parent_dir) not in sys.path: sys.path.insert(0, str(parent_dir))
    from control import (
        RECOMMEND_TOP_N_PER_CATEGORY,
        RECOMMEND_RANKING_METRIC,
        RECOMMEND_MIN_AVG_TRADES,
        RECOMMEND_START_DATE,
        RECOMMEND_END_DATE,
        RECOMMEND_BENCHMARK_TICKER,
        RECOMMEND_TOP_N, # Use overall top N
        RECOMMEND_STRATEGY_DETAIL_COUNT,
        RECOMMEND_PLOT_TOP_N # How many plots were generated
    )
    TOP_N_PER_CATEGORY = RECOMMEND_TOP_N_PER_CATEGORY
    RANKING_METRIC = RECOMMEND_RANKING_METRIC
    MIN_AVG_TRADES_FILTER = RECOMMEND_MIN_AVG_TRADES
    DATE_RANGE = f"{RECOMMEND_START_DATE} to {RECOMMEND_END_DATE}"
    BENCHMARK_TICKER = RECOMMEND_BENCHMARK_TICKER
    TOP_N_OVERALL = RECOMMEND_TOP_N # Use the overall top N from control
    STRATEGY_DETAIL_COUNT = RECOMMEND_STRATEGY_DETAIL_COUNT
    PLOT_TOP_N = RECOMMEND_PLOT_TOP_N # Number of stocks for which plots were likely generated

except ImportError as e:
    print(f"Warning: Could not import from control.py: {e}. Using default values.")
    TOP_N_PER_CATEGORY = 5 # Reduced default
    RANKING_METRIC = 'risk_adjusted_pl'
    MIN_AVG_TRADES_FILTER = 0.5
    DATE_RANGE = "Unknown"
    BENCHMARK_TICKER = "^NSEI" # Default guess
    RECOMMEND_START_DATE = "Unknown"
    RECOMMEND_END_DATE = "Unknown"
    TOP_N_OVERALL = 10
    STRATEGY_DETAIL_COUNT = 3
    PLOT_TOP_N = 5


RESULTS_DIR = Path("./recommendation_results")
PLOTS_DIR = RESULTS_DIR / "plots" # Define plot directory relative to results
DOCS_DIR = Path("./docs") # Output directory for the dashboard
TEMPLATE_FILE = "dashboard_template.html" # Template filename
OUTPUT_HTML_FILE = DOCS_DIR / "index.html"

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DashboardGenerator")
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("plotly").setLevel(logging.WARNING)

# --- Helper Functions ---

def find_latest_recommendation_csv(directory: Path) -> Path | None:
    """Finds the most recently created recommendation CSV file in the directory."""
    latest_file = None
    latest_time = 0
    # Pattern matching recommend_v2.py output: recommendations_v2merged_<metric>_<timestamp>.csv
    pattern = r"recommendations_v2merged_[\w\-]+_\d{8}_\d{6}\.csv"

    try:
        if not directory.is_dir():
            logger.error(f"Results directory not found: {directory}")
            return None
        logger.info(f"Searching for pattern '{pattern}' in {directory}...")
        found_files = list(directory.glob('*.csv'))
        # logger.debug(f"Files found: {[f.name for f in found_files]}") # Debugging

        matched_files = []
        for file_path in found_files:
            if re.match(pattern, file_path.name):
                 matched_files.append(file_path)
            # else: # Debugging
            #     logger.debug(f"File '{file_path.name}' did not match pattern.")

        if not matched_files:
            logger.warning(f"No files matched the pattern '{pattern}' in {directory}.")
            return None

        for file_path in matched_files:
             try:
                mod_time = file_path.stat().st_mtime
                if mod_time > latest_time:
                    latest_time = mod_time
                    latest_file = file_path
             except FileNotFoundError:
                 logger.warning(f"File disappeared while checking: {file_path}")
                 continue # Skip this file

        return latest_file
    except Exception as e:
        logger.error(f"Error searching for latest CSV in {directory}: {e}", exc_info=True)
        return None

def format_dataframe_for_template(df):
    """Formats the dataframe columns for better display and ensures numeric types."""
    # Define expected columns and attempt numeric conversion
    numeric_cols = [
        'last_price', 'score', 'strategy_count',
        'avg_strat_return_pct', 'avg_strat_max_drawdown_pct', 'avg_strat_sharpe',
        'avg_strat_sortino', 'avg_strat_calmar', 'avg_pl_pct_all', 'pl_std_dev',
        'positive_strat_ratio', 'avg_trades_per_strat', 'avg_commission_per_strat',
        'bh_return_pct', 'bh_max_drawdown_pct', 'bh_sharpe'
    ]
    string_cols = ['ticker', 'top_strategies', 'worst_strategies']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            logger.warning(f"Expected numeric column '{col}' not found in CSV. Adding as NaN.")
            df[col] = np.nan # Add missing numeric columns as NaN

    for col in string_cols:
        if col not in df.columns:
             logger.warning(f"Expected string column '{col}' not found in CSV. Adding as empty string.")
             df[col] = '' # Add missing string columns as empty

    # Fill NaNs - Crucial after numeric conversion and adding missing cols
    df.fillna({'score': 0.0, 'last_price': 0.0}, inplace=True)
    fill_zero_cols = [col for col in numeric_cols if col not in ['score', 'last_price']]
    df[fill_zero_cols] = df[fill_zero_cols].fillna(0.0)
    df[string_cols] = df[string_cols].fillna('')

    # Ensure correct types after filling NaNs
    if 'strategy_count' in df.columns:
        df['strategy_count'] = df['strategy_count'].astype(int)

    # Format strategy strings for better display (split and maybe limit)
    def format_strategies(strat_string, limit=STRATEGY_DETAIL_COUNT):
        if not isinstance(strat_string, str) or not strat_string.strip():
            return []
        try:
            # Example format: "strat_name1:12.34, strat_name2:-5.67"
            parts = strat_string.split(',')
            formatted = []
            for part in parts[:limit]:
                name_val = part.split(':')
                if len(name_val) == 2:
                    name = name_val[0].strip().replace('_strategy', '') # Clean name
                    val = float(name_val[1])
                    formatted.append({'name': name, 'value': val})
            return formatted
        except Exception:
             # logger.warning(f"Could not parse strategy string: {strat_string}")
             return [{'name': strat_string[:30] + '...' if len(strat_string)>30 else strat_string, 'value': None}] # Fallback

    if 'top_strategies' in df.columns:
      df['top_strategies_list'] = df['top_strategies'].apply(format_strategies)
    if 'worst_strategies' in df.columns:
      df['worst_strategies_list'] = df['worst_strategies'].apply(format_strategies)


    return df

# --- Performance Calculation (for Benchmark) ---
def calculate_simple_performance(daily_values: pd.Series | None): # Allow None input
    """Calculates basic return, drawdown, and approximate Sharpe."""
    # --- More Robust Input Check ---
    if daily_values is None:
        logger.debug("calculate_simple_performance received None.")
        return {'total_return_pct': 0.0, 'max_drawdown_pct': 0.0, 'sharpe_ratio': 0.0}
    if not isinstance(daily_values, pd.Series):
        logger.warning(f"calculate_simple_performance received non-Series input (type: {type(daily_values)}).")
        return {'total_return_pct': 0.0, 'max_drawdown_pct': 0.0, 'sharpe_ratio': 0.0}
    if daily_values.empty:
        logger.debug("calculate_simple_performance received an empty Series.")
        return {'total_return_pct': 0.0, 'max_drawdown_pct': 0.0, 'sharpe_ratio': 0.0}
    # Check for all NaNs *after* ensuring it's not empty
    if daily_values.isnull().all():
        logger.debug("calculate_simple_performance received a Series with all NaNs.")
        return {'total_return_pct': 0.0, 'max_drawdown_pct': 0.0, 'sharpe_ratio': 0.0}
    # --- End Robust Check ---

    # Proceed with calculation only if checks pass
    daily_values = daily_values.dropna() # Drop NaNs again just in case
    if len(daily_values) < 2:
         logger.debug("calculate_simple_performance has < 2 valid data points after dropna.")
         return {'total_return_pct': 0.0, 'max_drawdown_pct': 0.0, 'sharpe_ratio': 0.0}

    # Ensure the first value used for return calculation is non-zero
    if daily_values.iloc[0] == 0:
        logger.warning("First value in daily_values is zero, cannot calculate total return percentage accurately.")
        total_return = 0.0 # Avoid division by zero
    else:
        total_return = (daily_values.iloc[-1] / daily_values.iloc[0]) - 1

    cumulative_max = daily_values.cummax()
    drawdown = (daily_values - cumulative_max) / cumulative_max.replace(0, np.nan) # Avoid division by zero in drawdown calc too
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0
    # Ensure max_drawdown is finite
    max_drawdown = max_drawdown if np.isfinite(max_drawdown) else 0.0

    # Approximate Annualized Sharpe (using daily returns, assuming 0 risk-free rate)
    daily_returns = daily_values.pct_change().dropna()
    sharpe_ratio = 0.0
    # Check std dev is positive and finite before calculating Sharpe
    std_dev = daily_returns.std()

    if not daily_returns.empty and np.isfinite(std_dev) and std_dev > 1e-9:
        mean_return = daily_returns.mean()
        if np.isfinite(mean_return): # Check mean is finite
             sharpe_ratio = (mean_return / std_dev) * np.sqrt(252) # Annualize
        else:
             logger.warning("Mean daily return is non-finite, Sharpe set to 0.")
    elif not daily_returns.empty:
         logger.warning(f"Std dev of daily returns is zero or non-finite ({std_dev}), Sharpe set to 0.")

    # Ensure final metrics are finite
    final_return = total_return * 100 if np.isfinite(total_return) else 0.0
    final_drawdown = abs(max_drawdown * 100) if np.isfinite(max_drawdown) else 0.0
    final_sharpe = sharpe_ratio if np.isfinite(sharpe_ratio) else 0.0

    return {'total_return_pct': final_return,
            'max_drawdown_pct': final_drawdown,
            'sharpe_ratio': final_sharpe
            }

# --- Benchmark Data Fetching ---
def get_benchmark_performance(ticker, start_date, end_date):
    """Fetches benchmark data and calculates performance."""
    if not ticker or start_date == "Unknown" or end_date == "Unknown":
        logger.warning("Benchmark ticker or date range is unknown. Skipping benchmark fetch.")
        return None
    try:
        logger.info(f"Fetching benchmark data for {ticker} ({start_date} to {end_date})...")
        fetch_start = (pd.to_datetime(start_date) - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        fetch_end = (pd.to_datetime(end_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        data = yf.download(ticker, start=fetch_start, end=fetch_end, progress=False, auto_adjust=True)

        if data.empty:
            logger.warning(f"No benchmark data found for {ticker} in range {fetch_start} to {fetch_end}.")
            return None

        data.index = pd.to_datetime(data.index) # Ensure index is datetime
        data = data[(data.index >= pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))]
        if data.empty:
            logger.warning(f"No benchmark data for {ticker} within exact simulation period {start_date} to {end_date}.")
            return None

        col_name = 'Close' if 'Close' in data.columns else 'close' if 'close' in data.columns else None
        if not col_name:
             logger.warning(f"'Close' column not found in benchmark data for {ticker}.")
             return None

        benchmark_series = data[col_name]

        # --- Add Check: Ensure it's a Series ---
        if not isinstance(benchmark_series, pd.Series):
            logger.error(f"Extracted benchmark data for {ticker} is not a pandas Series (type: {type(benchmark_series)}). Cannot calculate performance.")
            return None
        # --- End Check ---

        logger.info(f"Calculating benchmark performance for {ticker}...")
        return calculate_simple_performance(benchmark_series) # Pass the Series

    except Exception as e:
        logger.error(f"Error fetching or processing benchmark {ticker}: {e}", exc_info=True)
        return None

# --- Chart Generation Functions ---
def create_score_distribution_chart(df):
    """Generates Plotly histogram for score distribution."""
    if df.empty or 'score' not in df.columns: return None
    fig = px.histogram(df, x="score", nbins=30,
                       title=f"Distribution of Ranking Scores ({RANKING_METRIC.replace('_',' ').title()})",
                       labels={'score': 'Score'})
    fig.update_layout(bargap=0.1, title_x=0.5)
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

def create_top_n_comparison_chart(df, top_n=20, metric='score', title_metric_name=None):
    """Generates Plotly bar chart comparing a metric for top N stocks."""
    if df.empty or metric not in df.columns: return None
    if title_metric_name is None: title_metric_name = metric.replace('_pct',' (%)').replace('_',' ').title()

    top_df = df.head(top_n)
    fig = px.bar(top_df, x='ticker', y=metric,
                 title=f"Top {top_n} Stocks by {title_metric_name}",
                 labels={'ticker': 'Ticker', metric: title_metric_name},
                 hover_data=['last_price', 'score', 'avg_strat_return_pct'])
    fig.update_layout(title_x=0.5)
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

def create_risk_return_scatter(df, top_n=20):
    """Generates Plotly scatter plot for risk vs return."""
    if df.empty or 'avg_strat_return_pct' not in df.columns or 'avg_strat_max_drawdown_pct' not in df.columns: return None

    # Check for last_price needed for color coding
    use_color = True
    if 'last_price' not in df.columns:
        logger.warning("Cannot create risk-return scatter with color: 'last_price' column missing.")
        use_color = False
        # We'll still proceed but without the color dimension

    top_df = df.head(top_n).copy() # Use copy to avoid SettingWithCopyWarning

    hover_data_cols = ['ticker', 'score', 'avg_strat_sharpe']
    if 'last_price' in top_df.columns:
         hover_data_cols.append('last_price') # Add price to hover if available

    plot_args = {
        "data_frame": top_df, # Pass the actual DataFrame
        "x": 'avg_strat_max_drawdown_pct',
        "y": 'avg_strat_return_pct',
        "text": 'ticker',
        "hover_data": hover_data_cols,
        "title": f"Risk (Avg Strategy Max DD %) vs. Return (Avg Strategy %) for Top {top_n} Stocks",
        "labels": {
            'avg_strat_max_drawdown_pct': 'Avg Strategy Max Drawdown (%)',
            'avg_strat_return_pct': 'Avg Strategy Return (%)'
        }
    }

    # Add color only if the necessary column exists
    if use_color:
        def get_category(price):
            if pd.isna(price): return "Unknown" # Handle potential NaNs
            if price > 1000: return "Above 1000"
            if price >= 100: return "100 to 1000"
            return "Below 100"
        top_df['price_category'] = top_df['last_price'].apply(get_category)
        plot_args["color"] = 'price_category'
        plot_args["labels"]['price_category'] = 'Price Category'


    # **** THE FIX IS HERE: Replace the ellipsis with the actual function call ****
    # fig = px.scatter(...) # <--- This was the problem
    fig = px.scatter(**plot_args) # <--- Use the arguments defined above

    fig.update_traces(textposition='top center')
    fig.update_layout(title_x=0.5)
    if use_color: # Only add legend title if color was used
        fig.update_layout(legend_title_text='Price Category')
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

# --- Equity Plot Handling ---
def find_and_encode_equity_plot(ticker: str, plots_dir: Path) -> str | None:
    """Finds the latest equity plot PNG for a ticker and encodes it as base64."""
    try:
        if not plots_dir.is_dir(): return None

        # Sanitize ticker name for matching filenames (same as in recommend_v2.py)
        safe_ticker_name = "".join(c for c in ticker if c.isalnum() or c in ('-', '_', '.'))
        latest_plot = None
        latest_time = 0
        plot_pattern = f"equity_{safe_ticker_name}_*.png" # Use safe name in pattern

        found_plots = list(plots_dir.glob(plot_pattern))
        if not found_plots:
             logger.debug(f"No plot found for ticker {ticker} (safe: {safe_ticker_name}) with pattern {plot_pattern} in {plots_dir}")
             return None

        for plot_file in found_plots:
            try:
                mod_time = plot_file.stat().st_mtime
                if mod_time > latest_time:
                    latest_time = mod_time
                    latest_plot = plot_file
            except FileNotFoundError: continue

        if latest_plot:
            with open(latest_plot, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/png;base64,{encoded_string}"
        else:
            return None

    except Exception as e:
        logger.error(f"Error finding/encoding plot for {ticker}: {e}", exc_info=True)
        return None

# --- Main Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML dashboard from stock recommendation CSV.")
    parser.add_argument("csv_file", nargs='?', type=str, help="Path to the recommendation CSV file (optional, finds latest if omitted).")
    args = parser.parse_args()

    input_csv_path = None
    if args.csv_file:
        input_csv_path = Path(args.csv_file)
        if not input_csv_path.is_file():
            logger.error(f"Specified CSV file not found: {input_csv_path}")
            sys.exit(1)
        logger.info(f"Using specified file: {input_csv_path.name}")
    else:
        logger.info(f"No CSV file specified, searching for the latest in '{RESULTS_DIR}'...")
        input_csv_path = find_latest_recommendation_csv(RESULTS_DIR)
        if not input_csv_path:
            logger.error(f"Could not find a suitable recommendation CSV file in {RESULTS_DIR}.")
            logger.error("Please run recommend_v2.py first or specify the CSV file path.")
            sys.exit(1)
        logger.info(f"Using latest file: {input_csv_path.name}")

    # --- Load and Process Data ---
    try:
        df = pd.read_csv(input_csv_path)
        logger.info(f"Loaded data from {input_csv_path.name} ({len(df)} rows)")
    except Exception as e:
        logger.error(f"Error reading CSV file {input_csv_path}: {e}", exc_info=True)
        sys.exit(1)

    overall_top_stocks_list = []
    categories_data = { "Above 1000": [], "100 to 1000": [], "Below 100": [] }
    total_stocks_analyzed = 0
    strategies_count_from_data = 'N/A'

    if df.empty:
        logger.warning("CSV file is empty. Dashboard will be generated with no recommendation data.")
    else:
        total_stocks_analyzed = len(df)
        df = format_dataframe_for_template(df)

        if 'score' in df.columns:
            df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
        else:
            logger.warning("Column 'score' not found in CSV. Cannot sort. Using original order.")
            df['score'] = 0.0 # Add dummy score

        if 'strategy_count' in df.columns and not df['strategy_count'].empty:
            strategies_count_from_data = df['strategy_count'].iloc[0]

        # --- Determine which tickers need plots (Same logic as recommend_v2) ---
        tickers_needing_plots = set()
        # Add overall top N
        for i, row in df.head(TOP_N_OVERALL).iterrows():
             tickers_needing_plots.add(row['ticker'])
        # Add category top N
        temp_categories_report = {"Above 1000": [], "100 to 1000": [], "Below 100": []}
        for i, row in df.iterrows():
             price = row['last_price']
             category_name = "Above 1000" if price > 1000 else ("100 to 1000" if 100 <= price <= 1000 else "Below 100")
             if len(temp_categories_report[category_name]) < TOP_N_PER_CATEGORY:
                 temp_categories_report[category_name].append(row['ticker'])
        for category_tickers in temp_categories_report.values():
            tickers_needing_plots.update(category_tickers)
        logger.info(f"{len(tickers_needing_plots)} unique tickers identified for displaying plots in the report.")
        # --- End plot target identification ---

        logger.info(f"Processing {total_stocks_analyzed} stocks for dashboard...")
        processed_count = 0
        plot_found_count = 0
        for i, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict['rank'] = i + 1
            ticker = row_dict['ticker']

            # --- Find plot ONLY if ticker is in the required set ---
            if ticker in tickers_needing_plots:
                row_dict['equity_plot_base64'] = find_and_encode_equity_plot(ticker, PLOTS_DIR)
                if row_dict['equity_plot_base64']:
                    plot_found_count += 1
            else:
                 row_dict['equity_plot_base64'] = None # No plot needed/expected for this one
            # --- End plot finding ---

            # Add to overall top N list
            if i < TOP_N_OVERALL:
                overall_top_stocks_list.append(row_dict)

            # Add to category lists
            # ... (category assignment logic as before) ...
            price = row_dict['last_price']
            if price > 1000: category_name = "Above 1000"
            elif 100 <= price <= 1000: category_name = "100 to 1000"
            else: category_name = "Below 100"
            if len(categories_data[category_name]) < TOP_N_PER_CATEGORY:
                 categories_data[category_name].append(row_dict)


        logger.info(f"Finished processing stocks. Found plots for {plot_found_count}/{len(tickers_needing_plots)} targeted stocks.")

    # --- Generate Charts from Processed Data ---
    logger.info("Generating Plotly charts...")
    score_dist_chart_html = create_score_distribution_chart(df)
    top_n_score_chart_html = create_top_n_comparison_chart(df, top_n=min(TOP_N_OVERALL * 2, len(df)), metric='score', title_metric_name='Ranking Score')
    top_n_return_chart_html = create_top_n_comparison_chart(df, top_n=min(TOP_N_OVERALL * 2, len(df)), metric='avg_strat_return_pct', title_metric_name='Avg Strategy Return (%)')
    risk_return_chart_html = create_risk_return_scatter(df, top_n=min(TOP_N_OVERALL * 2, len(df)))


    # --- Get Benchmark Performance ---
    benchmark_perf = get_benchmark_performance(BENCHMARK_TICKER, RECOMMEND_START_DATE, RECOMMEND_END_DATE)


    # --- Prepare Template Data ---
    template_data = {
        "dashboard_title": "Stock Recommendation Dashboard",
        "generation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_csv": input_csv_path.name if input_csv_path else "N/A",
        "ranking_metric": RANKING_METRIC.replace('_', ' ').title(),
        "date_range": DATE_RANGE,
        "total_stocks_analyzed": total_stocks_analyzed,
        "strategies_count": strategies_count_from_data,
        "benchmark_ticker": BENCHMARK_TICKER or "N/A",
        "benchmark_perf": benchmark_perf, # Will be None if failed
        "min_avg_trades": MIN_AVG_TRADES_FILTER,
        "top_n_overall_limit": TOP_N_OVERALL,
        "top_overall_stocks": overall_top_stocks_list,
        "categories": [
            {"name": name, "stocks": stocks, "limit": TOP_N_PER_CATEGORY, "table_id": f"table-{name.lower().replace(' ','-')}"}
            for name, stocks in categories_data.items()
        ],
        "charts": {
             "score_distribution": score_dist_chart_html,
             "top_n_score": top_n_score_chart_html,
             "top_n_return": top_n_return_chart_html,
             "risk_return": risk_return_chart_html,
        },
        "PLOT_TOP_N": PLOT_TOP_N,
    }

    # --- Render HTML ---
    try:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        template_dir = Path(__file__).parent # Assumes template is in same dir as script
        if not (template_dir / TEMPLATE_FILE).exists():
             logger.error(f"Template file '{TEMPLATE_FILE}' not found in directory '{template_dir}'.")
             # Attempt to find it relative to the current working directory as a fallback
             template_dir = Path(".")
             if not (template_dir / TEMPLATE_FILE).exists():
                 logger.error(f"Also not found in current directory '{template_dir}'. Please ensure '{TEMPLATE_FILE}' exists.")
                 sys.exit(1)
             else:
                 logger.info(f"Found template in current directory: {template_dir.resolve() / TEMPLATE_FILE}")

        env = Environment(loader=FileSystemLoader(template_dir), autoescape=True) # Enable autoescaping
        template = env.get_template(TEMPLATE_FILE)
        html_output = template.render(template_data)

        with open(OUTPUT_HTML_FILE, "w", encoding="utf-8") as f:
            f.write(html_output)

        logger.info(f"Successfully generated dashboard: {OUTPUT_HTML_FILE.resolve()}")

    except FileNotFoundError: # Should be caught above now, but keep as safeguard
        logger.error(f"Template file '{TEMPLATE_FILE}' not found.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error generating HTML dashboard: {e}", exc_info=True)
        sys.exit(1)