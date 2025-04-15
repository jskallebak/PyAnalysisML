"""
Data loading functions for PyAnalysisML.

This module provides functions to load data from various sources:
- CSV files
- Pickle files
- Binance API (via binance_client.py)
"""

import os
import pickle
from typing import Dict, Optional, Union, List, Any
from datetime import datetime

import pandas as pd

from pyanalysisml.data.binance_client import BinanceClient
from pyanalysisml.config import DATA_DIR, DEFAULT_SYMBOL, DEFAULT_INTERVAL, DEFAULT_LOOKBACK
from pyanalysisml.utils.logging_utils import get_logger

logger = get_logger(__name__)

def load_from_csv(
    filepath: str,
    parse_dates: bool = True,
    date_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        parse_dates: Whether to parse date columns
        date_column: Column to use as index (if it contains dates)
    
    Returns:
        DataFrame with loaded data
    """
    logger.info(f"Loading data from CSV: {filepath}")
    
    if date_column:
        df = pd.read_csv(filepath, parse_dates=[date_column])
        df.set_index(date_column, inplace=True)
    elif parse_dates:
        df = pd.read_csv(filepath, parse_dates=True, infer_datetime_format=True)
        # Try to set the index to a datetime column if it exists
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            df.set_index(date_cols[0], inplace=True)
    else:
        df = pd.read_csv(filepath)
        
    logger.info(f"Loaded {len(df)} rows from CSV")
    return df

def save_to_csv(
    df: pd.DataFrame,
    filename: str,
    directory: Optional[str] = None,
) -> str:
    """
    Save a DataFrame to a CSV file.
    
    Args:
        df: DataFrame to save
        filename: Name of the file
        directory: Directory to save to (default: DATA_DIR from config)
    
    Returns:
        Path to the saved file
    """
    directory = directory or DATA_DIR
    os.makedirs(directory, exist_ok=True)
    
    # Add .csv extension if not present
    if not filename.endswith('.csv'):
        filename += '.csv'
        
    filepath = os.path.join(directory, filename)
    df.to_csv(filepath)
    logger.info(f"Saved {len(df)} rows to {filepath}")
    return filepath

def load_from_pickle(filepath: str) -> Any:
    """
    Load data from a pickle file.
    
    Args:
        filepath: Path to the pickle file
    
    Returns:
        Object loaded from pickle
    """
    logger.info(f"Loading data from pickle: {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def save_to_pickle(
    obj: Any,
    filename: str,
    directory: Optional[str] = None,
) -> str:
    """
    Save an object to a pickle file.
    
    Args:
        obj: Object to save
        filename: Name of the file
        directory: Directory to save to (default: DATA_DIR from config)
    
    Returns:
        Path to the saved file
    """
    directory = directory or DATA_DIR
    os.makedirs(directory, exist_ok=True)
    
    # Add .pkl extension if not present
    if not filename.endswith('.pkl'):
        filename += '.pkl'
        
    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Saved object to {filepath}")
    return filepath

def load_from_binance(
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    start_str: Union[str, int, datetime] = DEFAULT_LOOKBACK,
    end_str: Optional[Union[str, int, datetime]] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    save_csv: bool = False,
    csv_filename: Optional[str] = None,
    chunk_size: Optional[int] = 30,
    max_workers: int = 8
) -> pd.DataFrame:
    """
    Load data from Binance API.
    
    This function fetches historical OHLCV (Open, High, Low, Close, Volume) data 
    from Binance for cryptocurrency analysis. It supports parallel processing to
    improve performance when fetching large datasets.
    
    The fetching process:
    1. Initializes a Binance client
    2. Fetches data (potentially in parallel chunks)
    3. Optionally saves the result to a CSV file
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Kline interval (e.g., '1m', '5m', '1h', '1d')
        start_str: Start time in various formats ('1 day ago', '1 Jan 2021', timestamp)
        end_str: End time (optional, default is now)
        api_key: Binance API key (optional, default from config)
        api_secret: Binance API secret (optional, default from config)
        save_csv: Whether to save the data to a CSV file
        csv_filename: Name of the CSV file (optional, default is {symbol}_{interval}.csv)
        chunk_size: Number of days per chunk for parallel processing (optional)
                   If provided, the date range will be split into chunks of this size
                   for parallel fetching. Recommended for large date ranges.
        max_workers: Maximum number of parallel workers for fetching data chunks
    
    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Loading {interval} data for {symbol} from Binance")
    
    # Initialize Binance client
    client = BinanceClient(api_key=api_key, api_secret=api_secret)
    
    # Fetch data with support for parallel processing
    df = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_str,
        end_str=end_str,
        chunk_size=chunk_size,
        max_workers=max_workers
    )
    
    # Save to CSV if requested
    if save_csv and not df.empty:
        csv_filename = csv_filename or f"{symbol}_{interval}.csv"
        save_to_csv(df, csv_filename)
    
    return df 