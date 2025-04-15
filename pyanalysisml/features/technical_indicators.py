"""
Technical indicators module for calculating various indicators using pandas-ta.
"""

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


def add_indicators(df: pd.DataFrame, indicators: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe using pandas-ta.

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
                result_df[f"sma_{period}"] = ta.sma(result_df["close"], length=period)

        elif indicator.lower() == "ema":
            # Exponential Moving Averages
            for period in [5, 20, 50, 200]:
                result_df[f"ema_{period}"] = ta.ema(result_df["close"], length=period)

        elif indicator.lower() == "rsi":
            # Relative Strength Index
            result_df["rsi_14"] = ta.rsi(result_df["close"], length=14)

        elif indicator.lower() == "macd":
            # Moving Average Convergence Divergence
            macd = ta.macd(result_df["close"], fast=12, slow=26, signal=9)
            result_df["macd"] = macd["MACD_12_26_9"]
            result_df["macd_signal"] = macd["MACDs_12_26_9"]
            result_df["macd_hist"] = macd["MACDh_12_26_9"]

        elif indicator.lower() == "bbands":
            # Bollinger Bands
            bbands = ta.bbands(result_df["close"], length=20, std=2)
            result_df["bbands_upper"] = bbands["BBU_20_2.0"]
            result_df["bbands_middle"] = bbands["BBM_20_2.0"]
            result_df["bbands_lower"] = bbands["BBL_20_2.0"]

            # Add bandwidth and %B
            result_df["bbands_bandwidth"] = (
                result_df["bbands_upper"] - result_df["bbands_lower"]
            ) / result_df["bbands_middle"]
            result_df["bbands_pctb"] = (result_df["close"] - result_df["bbands_lower"]) / (
                result_df["bbands_upper"] - result_df["bbands_lower"]
            )

        elif indicator.lower() == "atr":
            # Average True Range
            result_df["atr_14"] = ta.atr(
                result_df["high"], result_df["low"], result_df["close"], length=14
            )

        elif indicator.lower() == "stoch":
            # Stochastic Oscillator
            stoch = ta.stoch(
                result_df["high"], result_df["low"], result_df["close"], k=14, d=3, smooth_k=3
            )
            result_df["stoch_k"] = stoch["STOCHk_14_3_3"]
            result_df["stoch_d"] = stoch["STOCHd_14_3_3"]

        elif indicator.lower() == "adx":
            # Average Directional Index
            adx = ta.adx(result_df["high"], result_df["low"], result_df["close"], length=14)
            result_df["adx_14"] = adx["ADX_14"]

        elif indicator.lower() == "cci":
            # Commodity Channel Index
            result_df["cci_14"] = ta.cci(
                result_df["high"], result_df["low"], result_df["close"], length=14
            )

        elif indicator.lower() == "obv":
            # On Balance Volume
            result_df["obv"] = ta.obv(result_df["close"], result_df["volume"])

        elif indicator.lower() == "roc":
            # Rate of Change
            result_df["roc_10"] = ta.roc(result_df["close"], length=10)

    # Calculate price momentum
    for period in [1, 3, 5, 10, 21]:
        result_df[f"return_{period}d"] = result_df["close"].pct_change(period)

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
        result_df[f"volatility_{period}d"] = result_df["close"].pct_change().rolling(period).std()

    # Add day of week (for potential seasonality)
    if isinstance(result_df.index, pd.DatetimeIndex):
        result_df["day_of_week"] = result_df.index.dayofweek
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

    return result_df
