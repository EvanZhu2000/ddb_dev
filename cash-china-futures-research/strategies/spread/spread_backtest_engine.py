import numpy as np
import pandas as pd

# ================================
# Utility Functions
# ================================

def compute_rolling_zscore(series, window):
    """
    Compute rolling z-score with a specified rolling window.
    The z-score is calculated as (x - mean) / std.
    Uses min_periods=window to ensure stable estimates.
    """
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    return (series - rolling_mean) / rolling_std

def compute_rolling_zscore_grouped(df: pd.DataFrame, value_col: str, window: int, group_col: str) -> pd.Series:
    """
    Compute rolling z-score within each group independently.

    Parameters:
        df (pd.DataFrame): The input DataFrame
        value_col (str): Name of the column on which to compute z-score
        window (int): Rolling window size
        group_col (str): Grouping column to isolate different regimes (e.g. contract boundaries)

    Returns:
        pd.Series: Rolling z-score, NaN for periods before window is filled
    """
    return df.groupby(group_col, observed=True)[value_col].transform(
        lambda x: (x - x.rolling(window, min_periods=window).mean()) /
                  x.rolling(window, min_periods=window).std()
    )

# Signal generation logic has been moved to signals/signal_generators.py

# ================================
# Execution Engine
# ================================

# Helper Function – Price Fetcher
def get_price_from_row(row, price_field: str, leg: str):
    """
    Generalized price fetcher to support arbitrary leg names (e.g., '_1', '_2').
    Extracts price for a given leg (1 or 2) and price type (open, close, high, low).
    """
    return row[f"{price_field}_{leg}"]

def run_backtest_loop(df: pd.DataFrame, signal_col: str, execution_config: dict):
    """
    Loop-based backtest engine with realistic execution and log/real-return computation.

    Parameters:
        df (pd.DataFrame): Merged dataframe with futures prices.
        signal_col (str): Name of the signal column (+1, -1, 0, or NaN).
        execution_config (dict): Execution rules including price type, lag, stop loss, etc.

    Returns:
        Tuple[
            pd.DataFrame,      # df_result with strategy returns and positions (minute-level)
            Dict[str, list]    # trade_data with per-trade summaries
        ]
    """

    # Extract config
    entry_price_type = execution_config["entry_execution_price"]
    exit_price_type = execution_config["exit_execution_price"]
    lag = execution_config.get("execution_lag", 1)

    # Initialize tracking
    current_position = 0
    entry_price_ratio = None
    entry_price_tuple = None  # (entry_1, entry_2)
    entry_time = None
    holding_minutes = 0
    force_exit = False # Flag to enforce exit due to stop-loss or max holding

    strategy_log_returns = []
    strategy_real_returns = []   # NEW
    trade_log_returns = []
    trade_real_returns = []      # NEW
    executed_position = []
    entry_times = []
    exit_times = []
    holding_durations = []

    current_trade_id = None
    trade_id_counter = 0
    trade_ids = [] # Track trade_id for each bar

    for i in range(len(df)):
        row = df.iloc[i]
        datetime_now = row["datetime"]

        # Apply lag to signal
        if i - lag >= 0:
            signal = df.iloc[i - lag][signal_col]
        else:
            signal = np.nan

        # Extract prices
        entry_price_1 = get_price_from_row(row, entry_price_type, "1")
        entry_price_2 = get_price_from_row(row, entry_price_type, "2")
        close_1 = get_price_from_row(row, "close", "1")
        close_2 = get_price_from_row(row, "close", "2")

        # --- Forced Exit Triggered from Previous Bar ---
        if force_exit and current_position != 0:
            exit_price_1 = get_price_from_row(row, exit_price_type, "1")
            exit_price_2 = get_price_from_row(row, exit_price_type, "2")

            exit_price_ratio = current_position * (np.log(exit_price_1) - np.log(exit_price_2))
            trade_log_returns.append(exit_price_ratio - entry_price_ratio)

            # Compute and track real trade return
            entry_1, entry_2 = entry_price_tuple
            trade_real_return = current_position * ((exit_price_1 / entry_1) - (exit_price_2 / entry_2))
            trade_real_returns.append(trade_real_return)

            # PnL from previous close to this exit
            prev_close_1 = get_price_from_row(df.iloc[i - 1], "close", "1") if i > 0 else close_1
            prev_close_2 = get_price_from_row(df.iloc[i - 1], "close", "2") if i > 0 else close_2

            log_ret = current_position * (
                (np.log(exit_price_1) - np.log(exit_price_2)) -
                (np.log(prev_close_1) - np.log(prev_close_2))
            )
            strategy_log_returns.append(log_ret)

            real_ret = current_position * (
                (exit_price_1 / prev_close_1) - (exit_price_2 / prev_close_2)
            )
            strategy_real_returns.append(real_ret)

            exit_times.append(datetime_now)
            holding_durations.append(holding_minutes)

            # Reset state
            current_position = 0
            entry_price_ratio = None
            entry_price_tuple = None  # ← Reset tuple here
            entry_time = None
            holding_minutes = 0
            force_exit = False # ← Clear the flag
            current_trade_id = None

            executed_position.append(current_position)
            trade_ids.append(current_trade_id)
            continue # Skip rest of this bar

        # --- Entry ---
        if current_position == 0 and signal in [1, -1] and not (row.get("is_roll_date_1", False) or row.get("is_roll_date_2", False)) and not (
            execution_config.get("max_gap_days") is not None and row.get("gap_days_to_next", 0) > execution_config["max_gap_days"]):

            current_position = signal
            entry_time = datetime_now
            trade_id_counter += 1
            current_trade_id = trade_id_counter
            entry_price_ratio = current_position * (np.log(entry_price_1) - np.log(entry_price_2))
            entry_price_tuple = (entry_price_1, entry_price_2)
            holding_minutes = 0

            # Log return from entry to close of this bar
            log_ret = current_position * (np.log(close_1) - np.log(close_2)) - entry_price_ratio
            strategy_log_returns.append(log_ret)

            # Real return from entry to close of this bar
            real_ret = current_position * ((close_1 / entry_price_1) - (close_2 / entry_price_2))
            strategy_real_returns.append(real_ret)

            entry_times.append(entry_time)

        # --- Exit ---
        elif current_position != 0 and signal == 0:
            exit_price_1 = get_price_from_row(row, exit_price_type, "1")
            exit_price_2 = get_price_from_row(row, exit_price_type, "2")

            exit_price_ratio = current_position * (np.log(exit_price_1) - np.log(exit_price_2))
            trade_log_returns.append(exit_price_ratio - entry_price_ratio)

            # --- Real Return from Entry to Exit ---
            entry_1, entry_2 = entry_price_tuple
            trade_real_ret = current_position * ((exit_price_1 / entry_1) - (exit_price_2 / entry_2))
            trade_real_returns.append(trade_real_ret)

            # --- Bar-to-bar Real Return from Previous Close ---
            prev_close_1 = get_price_from_row(df.iloc[i - 1], "close", "1") if i > 0 else close_1
            prev_close_2 = get_price_from_row(df.iloc[i - 1], "close", "2") if i > 0 else close_2

            log_ret = current_position * (
                (np.log(exit_price_1) - np.log(exit_price_2)) -
                (np.log(prev_close_1) - np.log(prev_close_2))
            )
            strategy_log_returns.append(log_ret)

            real_ret = current_position * (
                (exit_price_1 / prev_close_1) - (exit_price_2 / prev_close_2)
            )
            strategy_real_returns.append(real_ret)

            exit_times.append(datetime_now)
            holding_durations.append(holding_minutes)

            # --- Reset State ---
            current_position = 0
            entry_price_ratio = None
            entry_price_tuple = None
            entry_time = None
            holding_minutes = 0
            current_trade_id = None

        # --- Holding ---
        elif current_position != 0:
            prev_close_1 = get_price_from_row(df.iloc[i - 1], "close", "1") if i > 0 else close_1
            prev_close_2 = get_price_from_row(df.iloc[i - 1], "close", "2") if i > 0 else close_2

            # --- Per-bar PnL (log + real) ---
            log_ret = current_position * (
                (np.log(close_1) - np.log(close_2)) -
                (np.log(prev_close_1) - np.log(prev_close_2))
            )
            strategy_log_returns.append(log_ret)

            real_ret = current_position * (
                (close_1 / prev_close_1) - (close_2 / prev_close_2)
            )
            strategy_real_returns.append(real_ret)

            # --- Check Stop Loss (real cumulative return from entry) ---
            if execution_config.get("stop_loss_pct") is not None:
                if entry_price_tuple is not None:
                    entry_1, entry_2 = entry_price_tuple
                    cumulative_real_return = current_position * ((close_1 / entry_1) - (close_2 / entry_2))
                    if cumulative_real_return <= -execution_config["stop_loss_pct"]:
                        force_exit = True

            # --- Check Max Holding Duration ---
            if execution_config.get("max_holding_minutes") is not None:
                if holding_minutes + 1 >= execution_config["max_holding_minutes"]:
                    force_exit = True

            # --- Forced Exit on Roll Date at 14:58 ---
            if row.get("is_roll_date_1", False) or row.get("is_roll_date_2", False):
                current_time = datetime_now.time()
                if current_time.hour == 14 and current_time.minute == 58:
                    force_exit = True

            # --- Forced Exit on Big Gap Day at 14:58 ---
            if execution_config.get("max_gap_days") is not None:
                if row.get("gap_days_to_next", 0) > execution_config["max_gap_days"]:
                    current_time = datetime_now.time()
                    if current_time.hour == 14 and current_time.minute == 58:
                        force_exit = True

            holding_minutes += 1
        
        else:
            # Flat position = no return
            strategy_log_returns.append(0.0)
            strategy_real_returns.append(0.0)

        # Track current position (always)
        executed_position.append(current_position)
        trade_ids.append(current_trade_id)

    # Output
    df_result = df.copy()
    df_result["executed_position"] = executed_position
    df_result["strategy_log_return"] = strategy_log_returns
    df_result["strategy_real_return"] = strategy_real_returns
    df_result["trade_id"] = trade_ids

    trade_data = {
        "entry_times": entry_times,
        "exit_times": exit_times,
        "holding_durations": holding_durations,
        "trade_log_returns": trade_log_returns,
        "trade_real_returns": trade_real_returns,
    }

    return df_result, trade_data

# ================================
# Metrics Engine
# ================================

def compute_trading_metrics(df_result: pd.DataFrame, trade_data: dict, trade_directions: list):
    """
    Compute robust trading metrics from backtest results.
    
    Parameters:
        df_result (pd.DataFrame): Output from run_backtest_loop with 'strategy_real_return'
        trade_data (dict): Dictionary with 'trade_real_returns', 'holding_durations', etc.
        
    Returns:
        dict: Dictionary of trading metrics.
    """

    # Use real return directly
    if "strategy_real_return" not in df_result.columns:
        raise ValueError("Missing 'strategy_real_return' in df_result.")
    df_result = df_result.copy()
    returns = df_result["strategy_real_return"].dropna()
    
    # --- Intraday Sharpe / Sortino base ---
    minutes_per_day = 345
    annualization_factor = np.sqrt(minutes_per_day * 252)
    mean_return = returns.mean()
    std_return = returns.std()
    downside_std = returns[returns < 0].std()
    sharpe_ratio = (mean_return / std_return) * annualization_factor if std_return != 0 else np.nan
    sortino_ratio = (mean_return / downside_std) * annualization_factor if downside_std != 0 else np.nan

    # --- Equity Curve and Max Drawdown ---
    equity_curve = (1 + returns).cumprod()
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1
    max_drawdown = drawdown.min()

    # --- Trade-Level Metrics from real returns ---
    trade_returns = trade_data.get("trade_real_returns", [])
    pct_trade_returns = trade_returns  # already real
    
    holding_durations = trade_data.get("holding_durations", [])

    num_trades = len(pct_trade_returns)
    num_winning_trades = sum(1 for r in pct_trade_returns if r > 0)

    win_rate = num_winning_trades / num_trades if num_trades > 0 else np.nan
    avg_trade_return = np.mean(pct_trade_returns) if pct_trade_returns else np.nan
    avg_holding_minutes = np.mean(holding_durations) if holding_durations else np.nan

    # --- Split into long vs short trades ---
    long_returns = [r for r, d in zip(pct_trade_returns, trade_directions) if d == 1]
    short_returns = [r for r, d in zip(pct_trade_returns, trade_directions) if d == -1]

    long_win_rate = sum(1 for r in long_returns if r > 0) / len(long_returns) if long_returns else np.nan
    short_win_rate = sum(1 for r in short_returns if r > 0) / len(short_returns) if short_returns else np.nan

    avg_long_return = np.mean(long_returns) if long_returns else np.nan
    avg_short_return = np.mean(short_returns) if short_returns else np.nan

    # --- Additional Trade-Level Metrics ---

    # Median and standard deviation of trade returns
    median_trade_return = np.median(pct_trade_returns) if pct_trade_returns else np.nan
    std_trade_return = np.std(pct_trade_returns) if pct_trade_returns else np.nan

    # Profit Factor: sum of winning trades / abs(sum of losing trades)
    total_profit = sum(r for r in pct_trade_returns if r > 0)
    total_loss = -sum(r for r in pct_trade_returns if r < 0)
    profit_factor = (total_profit / total_loss) if total_loss != 0 else np.nan

    # Expectancy: avg_win * win_rate - avg_loss * loss_rate
    winning_returns = [r for r in pct_trade_returns if r > 0]
    losing_returns = [-r for r in pct_trade_returns if r < 0]
    avg_win = np.mean(winning_returns) if winning_returns else 0.0
    avg_loss = np.mean(losing_returns) if losing_returns else 0.0
    loss_rate = 1 - win_rate if win_rate is not np.nan else np.nan
    expectancy = avg_win * win_rate - avg_loss * loss_rate if win_rate is not np.nan else np.nan

    # Max consecutive losses
    max_consecutive_losses = 0
    current_losses = 0
    for r in pct_trade_returns:
        if r <= 0:
            current_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
        else:
            current_losses = 0

    # Date of maximum drawdown
    if not drawdown.empty and "datetime" in df_result.columns:
        max_dd_idx = drawdown.idxmin()
        date_of_max_drawdown = df_result.loc[max_dd_idx, "datetime"]
    else:
        date_of_max_drawdown = None

    # Cumulative return and annualization
    cumulative_real_return = (1 + returns).prod() - 1
    total_minutes = df_result.shape[0]
    minutes_per_year = 252 * 345
    annualized_return = (1 + cumulative_real_return) ** (minutes_per_year / total_minutes) - 1 if total_minutes > 0 else np.nan

    # --- Assemble Final Metrics Dictionary ---
    metrics = {
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
        "avg_holding_minutes": avg_holding_minutes,
        "total_return": equity_curve.iloc[-1] - 1 if not equity_curve.empty else np.nan,
        "cumulative_real_return": cumulative_real_return, # Equivalent to total_return, for clarity
        "annualized_return": annualized_return,
        "median_trade_return": median_trade_return,
        "std_trade_return": std_trade_return,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_consecutive_losses": max_consecutive_losses,
        "date_of_max_drawdown": date_of_max_drawdown,
        "avg_long_trade_return": avg_long_return,
        "avg_short_trade_return": avg_short_return,
        "long_win_rate": long_win_rate,
        "short_win_rate": short_win_rate,
        "num_long_trades": len(long_returns),
        "num_short_trades": len(short_returns),
    }

    return metrics