# signal_generators.py

import numpy as np
import pandas as pd

def generate_trading_signals_directional(zscore_series,
                                         entry_threshold_long=-1.5,
                                         entry_threshold_short=1.5,
                                         exit_threshold_long=0.0,
                                         exit_threshold_short=0.0):
    """
    Generate trading signals based on direction-aware z-score thresholds and
    confirmation from z-score reversal.

    Parameters:
        zscore_series (pd.Series): z-score of log price ratio
        entry_threshold_long (float): Enter long when zscore < this
        entry_threshold_short (float): Enter short when zscore > this
        exit_threshold_long (float): Exit long when zscore > this
        exit_threshold_short (float): Exit short when zscore < this

    Returns:
        signal_series (pd.Series): forward-filled signal (+1, -1, 0, or NaN)
        raw_signal_series (pd.Series): raw signal with NaNs at non-decision points
    """
    signal_series = pd.Series(index=zscore_series.index, dtype="float")
    current_position = 0  # +1 (long), -1 (short), 0 (flat)

    for i in range(len(zscore_series)):
        z = zscore_series.iloc[i]
        z_prev = zscore_series.iloc[i - 1] if i > 0 else z

        # ENTRY logic
        if current_position == 0:
            if z < entry_threshold_long:
                current_position = 1  # Enter long
                signal_series.iloc[i] = 1
            elif z > entry_threshold_short:
                current_position = -1  # Enter short
                signal_series.iloc[i] = -1
            else:
                signal_series.iloc[i] = np.nan

        # EXIT logic
        elif current_position == 1:
            if z > exit_threshold_long:
                current_position = 0
                signal_series.iloc[i] = 0
            else:
                signal_series.iloc[i] = np.nan

        elif current_position == -1:
            if z < exit_threshold_short:
                current_position = 0
                signal_series.iloc[i] = 0
            else:
                signal_series.iloc[i] = np.nan

    return signal_series.ffill(), signal_series