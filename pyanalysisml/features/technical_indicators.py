"""
Technical indicators module using native pandas and numpy implementations.
"""

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_indicators(df: pd.DataFrame, indicators: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe using native pandas implementations.

    Args:
        df: DataFrame with OHLCV data
        indicators: List of indicators to add (default: basic set of indicators)

    Returns:
        DataFrame with added indicators
    """
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()

    # Use default indicators if none provided
    if indicators is None:
        indicators = [
            "sma",
            "ema",
            "rsi",
            "macd",
            "bbands",
            "atr",
            "stoch",
            "adx",
            "cci",
            "obv",
            "roc",
        ]

    logger.info(f"Adding technical indicators: {indicators}")

    # Add indicators based on the list
    for indicator in indicators:
        if indicator.lower() == "sma":
            # Simple Moving Averages
            for period in [5, 20, 50, 200]:
                result_df[f"sma_{period}"] = result_df["close"].rolling(window=period).mean()

        elif indicator.lower() == "ema":
            # Exponential Moving Averages
            for period in [5, 20, 50, 200]:
                result_df[f"ema_{period}"] = (
                    result_df["close"].ewm(span=period, adjust=False).mean()
                )

        elif indicator.lower() == "rsi":
            # Relative Strength Index
            delta = result_df["close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()

            # First RSI calculation
            rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
            result_df["rsi_14"] = 100 - (100 / (1 + rs))

        elif indicator.lower() == "macd":
            # Moving Average Convergence Divergence
            ema_12 = result_df["close"].ewm(span=12, adjust=False).mean()
            ema_26 = result_df["close"].ewm(span=26, adjust=False).mean()
            result_df["macd"] = ema_12 - ema_26
            result_df["macd_signal"] = result_df["macd"].ewm(span=9, adjust=False).mean()
            result_df["macd_hist"] = result_df["macd"] - result_df["macd_signal"]

        elif indicator.lower() == "bbands":
            # Bollinger Bands
            result_df["bbands_middle"] = result_df["close"].rolling(window=20).mean()
            rolling_std = result_df["close"].rolling(window=20).std()
            result_df["bbands_upper"] = result_df["bbands_middle"] + (rolling_std * 2)
            result_df["bbands_lower"] = result_df["bbands_middle"] - (rolling_std * 2)

            # Add bandwidth and %B
            result_df["bbands_bandwidth"] = (
                result_df["bbands_upper"] - result_df["bbands_lower"]
            ) / result_df["bbands_middle"]
            result_df["bbands_pctb"] = (result_df["close"] - result_df["bbands_lower"]) / (
                result_df["bbands_upper"] - result_df["bbands_lower"]
            )

        elif indicator.lower() == "atr":
            # Average True Range
            high_low = result_df["high"] - result_df["low"]
            high_close = (result_df["high"] - result_df["close"].shift()).abs()
            low_close = (result_df["low"] - result_df["close"].shift()).abs()

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            result_df["atr_14"] = true_range.rolling(window=14).mean()

        elif indicator.lower() == "stoch":
            # Stochastic Oscillator
            low_min = result_df["low"].rolling(window=14).min()
            high_max = result_df["high"].rolling(window=14).max()

            # %K calculation
            result_df["stoch_k"] = 100 * (
                (result_df["close"] - low_min) / (high_max - low_min + np.finfo(float).eps)
            )
            # %D calculation (3-period SMA of %K)
            result_df["stoch_d"] = result_df["stoch_k"].rolling(window=3).mean()

        elif indicator.lower() == "obv":
            # On Balance Volume
            obv = pd.Series(0, index=result_df.index)
            for i in range(1, len(result_df)):
                if result_df["close"].iloc[i] > result_df["close"].iloc[i - 1]:
                    obv.iloc[i] = obv.iloc[i - 1] + result_df["volume"].iloc[i]
                elif result_df["close"].iloc[i] < result_df["close"].iloc[i - 1]:
                    obv.iloc[i] = obv.iloc[i - 1] - result_df["volume"].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i - 1]
            result_df["obv"] = obv

        elif indicator.lower() == "roc":
            # Rate of Change
            result_df["roc_10"] = result_df["close"].pct_change(periods=10) * 100

    # Calculate price momentum
    for period in [1, 3, 5, 10, 21]:
        result_df[f"return_{period}d"] = result_df["close"].pct_change(period) * 100

    logger.info(f"Added {len(result_df.columns) - len(df.columns)} indicator columns")
    return result_df


def add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add custom features beyond basic technical indicators.

    Args:
        df: DataFrame with OHLCV data and basic indicators

    Returns:
        DataFrame with added custom features
    """
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()

    # Calculate volatility
    for period in [5, 10, 21]:
        result_df[f"volatility_{period}d"] = (
            result_df["close"].pct_change().rolling(period).std() * 100
        )

    # Add day of week (for potential seasonality)
    if isinstance(result_df.index, pd.DatetimeIndex):
        result_df["day_of_week"] = result_df.index.dayofweek
        if hasattr(result_df.index, "hour"):  # Check if timestamp has hour component
            result_df["hour_of_day"] = result_df.index.hour

    # Price distance from moving averages (%)
    if "sma_20" in result_df.columns:
        result_df["dist_sma_20"] = (
            (result_df["close"] - result_df["sma_20"]) / result_df["sma_20"] * 100
        )

    if "sma_50" in result_df.columns:
        result_df["dist_sma_50"] = (
            (result_df["close"] - result_df["sma_50"]) / result_df["sma_50"] * 100
        )

    # Moving average crossovers (as binary signals)
    if "sma_5" in result_df.columns and "sma_20" in result_df.columns:
        result_df["sma_5_20_cross"] = np.where(
            result_df["sma_5"] > result_df["sma_20"],
            1,
            np.where(result_df["sma_5"] < result_df["sma_20"], -1, 0),
        )

    # High-Low range relative to price
    result_df["hl_range_pct"] = (result_df["high"] - result_df["low"]) / result_df["close"] * 100

    # Additional feature: Price momentum relative to volatility
    if "volatility_10d" in result_df.columns and "return_5d" in result_df.columns:
        # Avoid division by zero
        safe_volatility = result_df["volatility_10d"].replace(0, np.finfo(float).eps)
        result_df["momentum_per_volatility"] = result_df["return_5d"] / safe_volatility

    return result_df
