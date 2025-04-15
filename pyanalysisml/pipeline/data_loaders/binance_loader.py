"""
Binance data loader implementation for the PyAnalysisML pipeline.

This module provides a DataLoader implementation for loading historical
cryptocurrency data from Binance.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from pyanalysisml.data.binance_client import BinanceClient
from pyanalysisml.pipeline.base import DataLoader
from pyanalysisml.pipeline.registry import register_data_loader

logger = logging.getLogger(__name__)


@register_data_loader("BinanceDataLoader")
class BinanceDataLoader(DataLoader):
    """
    DataLoader implementation for loading historical OHLCV data from Binance.
    
    This loader uses the BinanceClient to fetch historical klines data
    and transform it into a pandas DataFrame suitable for analysis.
    """
    
    def __init__(self, symbol: str = "BTCUSDT", interval: str = "1d",
                 start_str: str = "1 year ago", end_str: Optional[str] = None,
                 limit: int = 1000, chunk_size: int = 30,
                 max_workers: int = 4, **kwargs: Any) -> None:
        """
        Initialize the Binance data loader.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Time interval for the data points (e.g., "1d", "4h", "1m")
            start_str: Start date for the data (can be a string like "1 day ago" or a date)
            end_str: End date for the data (optional)
            limit: Maximum number of data points to fetch per API call
            chunk_size: Number of days per chunk for parallel processing
            max_workers: Maximum number of parallel workers for fetching data
            **kwargs: Additional arguments to pass to the BinanceClient
        """
        self.symbol = symbol
        self.interval = interval
        self.start_str = start_str
        self.end_str = end_str
        self.limit = limit
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.kwargs = kwargs
        
        self.client = BinanceClient()
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load historical OHLCV data from Binance.
        
        Returns:
            DataFrame containing the historical price data
        """
        logger.info(f"Loading Binance data for {self.symbol} at {self.interval} interval")
        
        # Fetch historical klines data
        data = self.client.get_historical_klines(
            symbol=self.symbol,
            interval=self.interval,
            start_str=self.start_str,
            end_str=self.end_str,
            limit=self.limit,
            chunk_size=self.chunk_size,
            max_workers=self.max_workers
        )
        
        if data is None or data.empty:
            logger.error(f"Failed to load data for {self.symbol}")
            return pd.DataFrame()
        
        logger.info(f"Successfully loaded {len(data)} records")
        self.data = data
        
        return data
    
    def preprocess_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Preprocess the loaded data by:
        1. Setting the index to datetime
        2. Sorting by time
        3. Removing any duplicate entries
        4. Converting numeric columns to appropriate types
        
        Args:
            data: DataFrame to preprocess (uses self.data if None)
            
        Returns:
            Preprocessed DataFrame
        """
        if data is None:
            if self.data is None:
                logger.warning("No data available for preprocessing. Loading data first.")
                self.load_data()
            data = self.data
        
        if data is None or data.empty:
            logger.error("No data available for preprocessing")
            return pd.DataFrame()
        
        logger.info("Preprocessing data")
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure datetime index
        if 'open_time' in df.columns and not pd.api.types.is_datetime64_dtype(df.index):
            df.set_index('open_time', inplace=True)
            
        # Sort by time
        df.sort_index(inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
        
        logger.info(f"Preprocessing complete, data shape: {df.shape}")
        self.data = df
        
        return df
    
    def split_data(self, data: Optional[pd.DataFrame] = None, 
                  test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and testing sets based on time.
        
        Args:
            data: DataFrame to split (uses self.data if None)
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (training_data, testing_data)
        """
        if data is None:
            if self.data is None:
                logger.warning("No data available for splitting. Loading data first.")
                self.load_data()
                self.preprocess_data()
            data = self.data
        
        if data is None or data.empty:
            logger.error("No data available for splitting")
            return pd.DataFrame(), pd.DataFrame()
        
        # Calculate the split point
        split_idx = int(len(data) * (1 - test_size))
        
        # Split the data
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        logger.info(f"Data split into train ({len(train_data)} rows) and test ({len(test_data)} rows)")
        
        return train_data, test_data 