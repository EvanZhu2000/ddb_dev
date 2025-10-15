# config_params.py
"""
Parameter configuration for spread signal generation and execution.
Assumes generically named spread legs: close_1, close_2, log_price_ratio, etc.
"""

from itertools import product

# ================================
# Signal Parameter Grid
# ================================

ZSCORE_WINDOWS = [150, 180, 240]  # Rolling window in minutes
ENTRY_LONGS = [-1.5, -1.0]  # Long entry thresholds
ENTRY_SHORTS = [1.0, 1.5]  # Short entry thresholds
EXIT_LONGS = [-0.5, -0.25]  # Long exit thresholds
EXIT_SHORTS = [0.25, 0.5]  # Short exit thresholds

# ================================
# Execution Parameter Grid
# ================================

ENTRY_EXEC_PRICES = ["open"]       # Entry execution price: "open", "close"
EXIT_EXEC_PRICES = ["open"]        # Exit execution price
EXEC_LAG = [1]                     # Execution lag in minutes
STOP_LOSS_PCTS = [0.005]     # Stop loss percentage (None or 0.005, for example)
MAX_HOLD_MINUTES = [120]          # Max holding duration (in minutes)
MAX_DAY_GAP = [3]               # Max allowed gap between trades

# ================================
# Combine into Full Parameter Grid
# ================================

param_grid = []
for zwin, el, es, exl, exs, eep, exp, elg, sl, mh, mdg in product(
    ZSCORE_WINDOWS, ENTRY_LONGS, ENTRY_SHORTS, EXIT_LONGS, EXIT_SHORTS,
    ENTRY_EXEC_PRICES, EXIT_EXEC_PRICES, EXEC_LAG, STOP_LOSS_PCTS,
    MAX_HOLD_MINUTES, MAX_DAY_GAP
):
    param_grid.append({
        "signal": {
            "zscore_window": zwin,
            "entry_threshold_long": el,
            "entry_threshold_short": es,
            "exit_threshold_long": exl,
            "exit_threshold_short": exs
        },
        "execution": {
            "entry_execution_price": eep,
            "exit_execution_price": exp,
            "execution_lag": elg,
            "stop_loss_pct": sl,
            "max_holding_minutes": mh,
            "max_gap_days": mdg
        }
    })

# ================================
# Evaluation Metrics
# ================================

# Format: (metric_name, higher_is_better)
evaluation_metrics = [
    ("sharpe_ratio", True),
    ("sortino_ratio", True),
    ("max_drawdown", True),
    ("win_rate", True),
    ("avg_trade_return", True),
    ("annualized_return", True),
    ("num_trades", False),
]

# ================================
# Base Metrics for Visualization
# ================================
base_metrics = ["sharpe_ratio", "max_drawdown", "win_rate", "annualized_return"]