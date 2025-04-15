"""
Feature engineering module for PyAnalysisML.

This module provides functions for creating advanced features
beyond basic technical indicators.
"""

import logging
from typing import List, Optional, Dict, Union, Callable

import numpy as np
import pandas as pd

from pyanalysisml.features.technical_indicators import add_indicators, add_custom_features
from pyanalysisml.config import DEFAULT_INDICATORS, PREDICTION_HORIZONS
from pyanalysisml.utils.logging_utils import get_logger

logger = get_logger(__name__)

def create_lag_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lag_periods: List[int] = [1, 2, 3, 5, 7, 14, 21],
) -> pd.DataFrame:
    """
    Create lagged features for time series forecasting.
    
    Args:
        df: DataFrame with time series data
        columns: Columns to create lags for (default: 'close')
        lag_periods: List of periods to lag
    
    Returns:
        DataFrame with added lag features
    """
    logger.info(f"Creating lag features with periods {lag_periods}")
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Default to close price if no columns specified
    if columns is None:
        if 'close' in result_df.columns:
            columns = ['close']
        else:
            logger.warning("No 'close' column found and no columns specified")
            return result_df
    
    # Create lag features
    for col in columns:
        if col in result_df.columns:
            for lag in lag_periods:
                result_df[f"{col}_lag_{lag}"] = result_df[col].shift(lag)
        else:
            logger.warning(f"Column {col} not found in DataFrame")
    
    logger.info(f"Added {len(columns) * len(lag_periods)} lag features")
    return result_df

def create_rolling_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    windows: List[int] = [5, 10, 20, 50],
    functions: Dict[str, Callable] = {'mean': np.mean, 'std': np.std, 'min': np.min, 'max': np.max},
) -> pd.DataFrame:
    """
    Create rolling window features like moving statistics.
    
    Args:
        df: DataFrame with time series data
        columns: Columns to create rolling features for (default: 'close', 'volume')
        windows: List of window sizes
        functions: Dictionary mapping function names to functions
    
    Returns:
        DataFrame with added rolling features
    """
    logger.info(f"Creating rolling features with windows {windows}")
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Default columns if not specified
    if columns is None:
        columns = []
        for col in ['close', 'volume', 'high', 'low']:
            if col in result_df.columns:
                columns.append(col)
        
        if not columns:
            logger.warning("No appropriate columns found for rolling features")
            return result_df
    
    # Create rolling features
    for col in columns:
        if col in result_df.columns:
            for window in windows:
                for func_name, func in functions.items():
                    result_df[f"{col}_roll_{func_name}_{window}"] = (
                        result_df[col].rolling(window=window).apply(func)
                    )
        else:
            logger.warning(f"Column {col} not found in DataFrame")
    
    total_features = len(columns) * len(windows) * len(functions)
    logger.info(f"Added {total_features} rolling features")
    return result_df

def create_return_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    periods: List[int] = PREDICTION_HORIZONS,
) -> pd.DataFrame:
    """
    Create price return/percent change features.
    
    Args:
        df: DataFrame with time series data
        columns: Columns to create return features for (default: 'close')
        periods: List of periods for calculating returns
    
    Returns:
        DataFrame with added return features
    """
    logger.info(f"Creating return features with periods {periods}")
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Default to close price if no columns specified
    if columns is None:
        if 'close' in result_df.columns:
            columns = ['close']
        else:
            logger.warning("No 'close' column found and no columns specified")
            return result_df
    
    # Create return features
    for col in columns:
        if col in result_df.columns:
            for period in periods:
                result_df[f"{col}_return_{period}"] = result_df[col].pct_change(period) * 100
        else:
            logger.warning(f"Column {col} not found in DataFrame")
    
    logger.info(f"Added {len(columns) * len(periods)} return features")
    return result_df

def create_cyclical_features(
    df: pd.DataFrame,
    drop_original: bool = False,
) -> pd.DataFrame:
    """
    Create cyclical time features (day of week, hour of day, etc.)
    using sine and cosine transforms.
    
    Args:
        df: DataFrame with datetime index
        drop_original: Whether to drop the original time features
    
    Returns:
        DataFrame with added cyclical features
    """
    logger.info("Creating cyclical time features")
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Check if index is datetime
    if not isinstance(result_df.index, pd.DatetimeIndex):
        logger.warning("DataFrame index is not datetime. Cannot create cyclical features.")
        return result_df
    
    # Extract time components
    result_df['day_of_week'] = result_df.index.dayofweek
    result_df['month'] = result_df.index.month
    result_df['day_of_month'] = result_df.index.day
    result_df['day_of_year'] = result_df.index.dayofyear
    
    # Add hour if available
    if result_df.index.inferred_type == 'datetime64' and hasattr(result_df.index, 'hour'):
        result_df['hour_of_day'] = result_df.index.hour
    
    # Create cyclical features
    time_features = {
        'day_of_week': 7,
        'month': 12,
        'day_of_month': 31,
        'day_of_year': 365,
        'hour_of_day': 24
    }
    
    # Only process features that exist in the DataFrame
    for feature, period in time_features.items():
        if feature in result_df.columns:
            result_df[f'{feature}_sin'] = np.sin(2 * np.pi * result_df[feature] / period)
            result_df[f'{feature}_cos'] = np.cos(2 * np.pi * result_df[feature] / period)
            
            # Drop original if requested
            if drop_original:
                result_df = result_df.drop(columns=[feature])
    
    cyclical_cols = [col for col in result_df.columns if '_sin' in col or '_cos' in col]
    logger.info(f"Added {len(cyclical_cols)} cyclical features")
    return result_df

def create_all_features(
    df: pd.DataFrame,
    add_technical: bool = True,
    add_lags: bool = True,
    add_rolling: bool = True,
    add_returns: bool = True,
    add_cyclical: bool = True,
    indicators: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create all features in one function call.
    
    Args:
        df: DataFrame with OHLCV data
        add_technical: Whether to add technical indicators
        add_lags: Whether to add lag features
        add_rolling: Whether to add rolling features
        add_returns: Whether to add return features
        add_cyclical: Whether to add cyclical features
        indicators: List of technical indicators to add (default from config)
    
    Returns:
        DataFrame with all requested features
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Add technical indicators
    if add_technical:
        result_df = add_indicators(result_df, indicators=indicators)
        result_df = add_custom_features(result_df)
    
    # Add lag features
    if add_lags:
        result_df = create_lag_features(result_df)
    
    # Add rolling features
    if add_rolling:
        result_df = create_rolling_features(result_df)
    
    # Add return features
    if add_returns:
        result_df = create_return_features(result_df)
    
    # Add cyclical features
    if add_cyclical:
        result_df = create_cyclical_features(result_df)
    
    # Drop NaN values resulting from the window calculations
    result_df = result_df.dropna()
    
    logger.info(f"Created feature DataFrame with {len(result_df.columns)} columns")
    return result_df 