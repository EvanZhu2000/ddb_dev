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

# ================================
# Signal Generation
# ================================

def generate_trading_signals_directional(zscore_series,
                                         entry_threshold_long=-1.5,
                                         entry_threshold_short=1.5,
                                         exit_threshold_long=0.0,
                                         exit_threshold_short=0.0):
    """
    Generate trading signals based on direction-aware z-score thresholds.
    
    Parameters:
        zscore_series (pd.Series): z-score of log price ratio
        entry_threshold_long (float): Enter long when zscore < this
        entry_threshold_short (float): Enter short when zscore > this
        exit_threshold_long (float): Exit long when zscore > this
        exit_threshold_short (float): Exit short when zscore < this
    
    Returns:
        signal_series (pd.Series): raw signals at each bar (+1, -1, 0, or NaN)
    """
    signal_series = pd.Series(index=zscore_series.index, dtype="float")
    current_position = 0 # +1 (long), -1 (short), 0 (flat)
    
    for i in range(len(zscore_series)):
        z = zscore_series.iloc[i]

        # ENTRY logic
        if current_position == 0:
            if z < entry_threshold_long:
                current_position = 1 # Enter long
                signal_series.iloc[i] = 1
            elif z > entry_threshold_short:
                current_position = -1 # Enter short
                signal_series.iloc[i] = -1
            else:
                signal_series.iloc[i] = np.nan

        # EXIT logic
        elif current_position == 1: # Currently long
            if z > exit_threshold_long:
                current_position = 0
                signal_series.iloc[i] = 0
            else:
                signal_series.iloc[i] = np.nan

        elif current_position == -1: # Currently short
            if z < exit_threshold_short:
                current_position = 0
                signal_series.iloc[i] = 0
            else:
                signal_series.iloc[i] = np.nan

    return signal_series.ffill(), signal_series

# ================================
# Execution Engine
# ================================

# Helper Function – Price Fetcher
def get_price_from_row(row, price_field: str, leg: str):
    """
    Extracts price for a given leg (Y or M) and price type (open, close, high, low).

    Parameters:
        row (pd.Series): A row from the DataFrame
        price_field (str): Price type to use ("open", "close", "high", "low")
        leg (str): Either "Y" or "M"

    Returns:
        float: The price from the appropriate column
    """
    return row[f"{price_field}_{leg}"]

def run_backtest_loop(df: pd.DataFrame, signal_col: str, execution_config: dict):
    """
    Loop-based backtest engine with realistic execution and log-return computation.

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
    entry_time = None
    holding_minutes = 0
    force_exit = False # Flag to enforce exit due to stop-loss or max holding

    strategy_log_returns = []
    trade_log_returns = []
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
        entry_price_Y = get_price_from_row(row, entry_price_type, "Y")
        entry_price_M = get_price_from_row(row, entry_price_type, "M")
        close_Y = get_price_from_row(row, "close", "Y")
        close_M = get_price_from_row(row, "close", "M")

        # --- Forced Exit Triggered from Previous Bar ---
        if force_exit and current_position != 0:
            exit_price_Y = get_price_from_row(row, exit_price_type, "Y")
            exit_price_M = get_price_from_row(row, exit_price_type, "M")

            exit_price_ratio = current_position * (np.log(exit_price_Y) - np.log(exit_price_M))
            trade_log_returns.append(exit_price_ratio - entry_price_ratio)

            # PnL from previous close to this exit
            prev_close_Y = get_price_from_row(df.iloc[i - 1], "close", "Y") if i > 0 else close_Y
            prev_close_M = get_price_from_row(df.iloc[i - 1], "close", "M") if i > 0 else close_M

            log_ret = current_position * (
                (np.log(exit_price_Y) - np.log(exit_price_M)) -
                (np.log(prev_close_Y) - np.log(prev_close_M))
            )
            strategy_log_returns.append(log_ret)

            exit_times.append(datetime_now)
            holding_durations.append(holding_minutes)

            # Reset state
            current_position = 0
            entry_price_ratio = None
            entry_time = None
            holding_minutes = 0
            force_exit = False # ← Clear the flag
            current_trade_id = None

            executed_position.append(current_position)
            trade_ids.append(current_trade_id)
            continue # Skip rest of this bar

        # --- Entry ---
        if current_position == 0 and signal in [1, -1] and not row.get("is_roll_date", False) and not (
            execution_config.get("max_gap_days") is not None and row.get("gap_days_to_next", 0) > execution_config["max_gap_days"]):

            current_position = signal
            entry_time = datetime_now
            trade_id_counter += 1
            current_trade_id = trade_id_counter
            entry_price_ratio = current_position * (np.log(entry_price_Y) - np.log(entry_price_M))
            holding_minutes = 0

            # PnL from entry to close of this bar
            log_ret = current_position * (np.log(close_Y) - np.log(close_M)) - entry_price_ratio
            strategy_log_returns.append(log_ret)
            entry_times.append(entry_time)

        # --- Exit ---
        elif current_position != 0 and signal == 0:
            exit_price_Y = get_price_from_row(row, exit_price_type, "Y")
            exit_price_M = get_price_from_row(row, exit_price_type, "M")

            exit_price_ratio = current_position * (np.log(exit_price_Y) - np.log(exit_price_M))
            trade_log_returns.append(exit_price_ratio - entry_price_ratio)

            # PnL from previous close to this exit
            prev_close_Y = get_price_from_row(df.iloc[i - 1], "close", "Y") if i > 0 else close_Y
            prev_close_M = get_price_from_row(df.iloc[i - 1], "close", "M") if i > 0 else close_M

            log_ret = current_position * (
                (np.log(exit_price_Y) - np.log(exit_price_M)) -
                (np.log(prev_close_Y) - np.log(prev_close_M))
            )
            strategy_log_returns.append(log_ret)

            exit_times.append(datetime_now)
            holding_durations.append(holding_minutes)

            # Reset state
            current_position = 0
            entry_price_ratio = None
            entry_time = None
            holding_minutes = 0
            current_trade_id = None

        # --- Holding ---
        elif current_position != 0:
            prev_close_Y = get_price_from_row(df.iloc[i - 1], "close", "Y") if i > 0 else close_Y
            prev_close_M = get_price_from_row(df.iloc[i - 1], "close", "M") if i > 0 else close_M

            log_ret = current_position * (
                (np.log(close_Y) - np.log(close_M)) -
                (np.log(prev_close_Y) - np.log(prev_close_M))
            )
            strategy_log_returns.append(log_ret)

            # --- Check for Stop Loss ---
            if execution_config.get("stop_loss_pct") is not None:
                cumulative_log_return = (
                    current_position * (np.log(close_Y) - np.log(close_M)) - entry_price_ratio
                )
                cumulative_pct_return = np.exp(cumulative_log_return) - 1
                if cumulative_pct_return <= -execution_config["stop_loss_pct"]:
                    force_exit = True

            # --- Check for Max Holding Duration ---
            if execution_config.get("max_holding_minutes") is not None:
                if holding_minutes + 1 >= execution_config["max_holding_minutes"]:
                    force_exit = True

            # --- Check for Forced Exit on Roll Date at 14:58 ---
            if row.get("is_roll_date", False):
                current_time = datetime_now.time()
                if current_time.hour == 14 and current_time.minute == 58:
                    force_exit = True

            # --- Check for Forced Exit on Big Gap Day at 14:58 ---
            if execution_config.get("max_gap_days") is not None:
                if row.get("gap_days_to_next", 0) > execution_config["max_gap_days"]:
                    current_time = datetime_now.time()
                    if current_time.hour == 14 and current_time.minute == 58:
                        force_exit = True

            holding_minutes += 1
        
        else:
            # Flat position = no return
            strategy_log_returns.append(0.0)

        # Track current position (always)
        executed_position.append(current_position)
        trade_ids.append(current_trade_id)

    # Output
    df_result = df.copy()
    df_result["executed_position"] = executed_position
    df_result["strategy_log_return"] = strategy_log_returns
    df_result["trade_id"] = trade_ids

    trade_data = {
        "entry_times": entry_times,
        "exit_times": exit_times,
        "holding_durations": holding_durations,
        "trade_log_returns": trade_log_returns
    }

    return df_result, trade_data

# ================================
# Metrics Engine
# ================================

def compute_trading_metrics(df_result: pd.DataFrame, trade_data: dict, trade_directions: list):
    """
    Compute robust trading metrics from backtest results.
    
    Parameters:
        df_result (pd.DataFrame): Output from run_backtest_loop with 'strategy_log_return'
        trade_data (dict): Dictionary with 'trade_log_returns', 'holding_durations', etc.
        
    Returns:
        dict: Dictionary of trading metrics.
    """

    # --- Convert log returns to arithmetic returns for Sharpe/Sortino ---
    if "strategy_log_return" not in df_result.columns:
        raise ValueError("Missing 'strategy_log_return' in df_result.")

    df_result = df_result.copy()
    df_result["strategy_return"] = np.exp(df_result["strategy_log_return"]) - 1

    # --- Intraday Sharpe / Sortino base ---
    minutes_per_day = 345
    annualization_factor = np.sqrt(minutes_per_day * 252)

    returns = df_result["strategy_return"].dropna()
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

    # --- Trade-Level Metrics ---
    trade_returns = trade_data.get("trade_log_returns", [])
    holding_durations = trade_data.get("holding_durations", [])

    pct_trade_returns = [np.exp(r) - 1 for r in trade_returns]
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

    # --- Annualized Return ---
    cumulative_log_return = df_result["strategy_log_return"].sum()
    total_minutes = df_result.shape[0]
    minutes_per_year = 252 * 345
    annualized_return = np.exp(cumulative_log_return * minutes_per_year / total_minutes) - 1 if total_minutes > 0 else np.nan

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
        "cumulative_log_return": cumulative_log_return,
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