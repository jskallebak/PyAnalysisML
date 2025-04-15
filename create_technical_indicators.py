#!/usr/bin/env python3
"""
Create the technical indicators module for PyAnalysisML.
"""

content = """\"\"\"
Technical indicators module for calculating various TA-Lib indicators.
\"\"\"
import logging
from typing import List, Optional, Union

import pandas as pd
import numpy as np
import talib

logger = logging.getLogger(__name__)

def add_indicators(
    df: pd.DataFrame,
    indicators: Optional[List[str]] = None
) -> pd.DataFrame:
    \"\"\"
    Add technical indicators to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        indicators: List of indicators to add (default: basic set of indicators)
        
    Returns:
        DataFrame with added indicators
    \"\"\"
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Use default indicators if none provided
    if indicators is None:
        indicators = [
            'sma', 'ema', 'rsi', 'macd', 'bbands', 'atr',
            'stoch', 'adx', 'cci', 'obv', 'roc'
        ]
    
    # Extract price and volume data
    open_price = result_df['open'].values
    high_price = result_df['high'].values
    low_price = result_df['low'].values
    close_price = result_df['close'].values
    volume = result_df['volume'].values if 'volume' in result_df.columns else None
    
    logger.info(f"Adding technical indicators: {indicators}")
    
    # Add indicators based on the list
    for indicator in indicators:
        if indicator.lower() == 'sma':
            # Simple Moving Averages
            for period in [5, 20, 50, 200]:
                result_df[f'sma_{period}'] = talib.SMA(close_price, timeperiod=period)
        
        elif indicator.lower() == 'ema':
            # Exponential Moving Averages
            for period in [5, 20, 50, 200]:
                result_df[f'ema_{period}'] = talib.EMA(close_price, timeperiod=period)
        
        elif indicator.lower() == 'rsi':
            # Relative Strength Index
            result_df['rsi_14'] = talib.RSI(close_price, timeperiod=14)
        
        elif indicator.lower() == 'macd':
            # Moving Average Convergence Divergence
            macd, macd_signal, macd_hist = talib.MACD(
                close_price, fastperiod=12, slowperiod=26, signalperiod=9
            )
            result_df['macd'] = macd
            result_df['macd_signal'] = macd_signal
            result_df['macd_hist'] = macd_hist
        
        elif indicator.lower() == 'bbands':
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                close_price, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            result_df['bbands_upper'] = upper
            result_df['bbands_middle'] = middle
            result_df['bbands_lower'] = lower
            
            # Add bandwidth and %B
            result_df['bbands_bandwidth'] = (upper - lower) / middle
            result_df['bbands_pctb'] = (close_price - lower) / (upper - lower)
        
        elif indicator.lower() == 'atr':
            # Average True Range
            result_df['atr_14'] = talib.ATR(
                high_price, low_price, close_price, timeperiod=14
            )
        
        elif indicator.lower() == 'stoch':
            # Stochastic Oscillator
            slowk, slowd = talib.STOCH(
                high_price, low_price, close_price,
                fastk_period=14, slowk_period=3, slowk_matype=0,
                slowd_period=3, slowd_matype=0
            )
            result_df['stoch_k'] = slowk
            result_df['stoch_d'] = slowd
        
        elif indicator.lower() == 'adx':
            # Average Directional Index
            result_df['adx_14'] = talib.ADX(
                high_price, low_price, close_price, timeperiod=14
            )
        
        elif indicator.lower() == 'cci':
            # Commodity Channel Index
            result_df['cci_14'] = talib.CCI(
                high_price, low_price, close_price, timeperiod=14
            )
        
        elif indicator.lower() == 'obv':
            # On Balance Volume
            if volume is not None:
                result_df['obv'] = talib.OBV(close_price, volume)
        
        elif indicator.lower() == 'roc':
            # Rate of Change
            result_df['roc_10'] = talib.ROC(close_price, timeperiod=10)
    
    # Calculate price momentum
    for period in [1, 3, 5, 10, 21]:
        result_df[f'return_{period}d'] = result_df['close'].pct_change(period)
    
    logger.info(f"Added {len(result_df.columns) - len(df.columns)} indicator columns")
    return result_df

def add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"
    Add custom features beyond basic technical indicators.
    
    Args:
        df: DataFrame with OHLCV data and basic indicators
        
    Returns:
        DataFrame with added custom features
    \"\"\"
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Calculate volatility
    for period in [5, 10, 21]:
        result_df[f'volatility_{period}d'] = result_df['close'].pct_change().rolling(period).std()
    
    # Add day of week (for potential seasonality)
    if isinstance(result_df.index, pd.DatetimeIndex):
        result_df['day_of_week'] = result_df.index.dayofweek
        result_df['hour_of_day'] = result_df.index.hour
    
    # Price distance from moving averages (%)
    if 'sma_20' in result_df.columns:
        result_df['dist_sma_20'] = (result_df['close'] - result_df['sma_20']) / result_df['sma_20'] * 100
    
    if 'sma_50' in result_df.columns:
        result_df['dist_sma_50'] = (result_df['close'] - result_df['sma_50']) / result_df['sma_50'] * 100
    
    # Moving average crossovers (as binary signals)
    if 'sma_5' in result_df.columns and 'sma_20' in result_df.columns:
        result_df['sma_5_20_cross'] = np.where(
            result_df['sma_5'] > result_df['sma_20'], 1, 
            np.where(result_df['sma_5'] < result_df['sma_20'], -1, 0)
        )
    
    # High-Low range relative to price
    result_df['hl_range_pct'] = (result_df['high'] - result_df['low']) / result_df['close'] * 100
    
    return result_df
"""

# Write to file
with open("pyanalysisml/features/technical_indicators.py", "w") as f:
    f.write(content)

print("Created file: pyanalysisml/features/technical_indicators.py")
