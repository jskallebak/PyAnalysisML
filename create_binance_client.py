#!/usr/bin/env python3
"""
Create the Binance client module for PyAnalysisML.
"""

content = """\"\"\"
Binance client module for fetching OHLC cryptocurrency data.
\"\"\"
import os
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime

import pandas as pd
from binance.client import Client

logger = logging.getLogger(__name__)

class BinanceClient:
    \"\"\"Client for accessing Binance API to fetch historical OHLC data.\"\"\"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False
    ):
        \"\"\"
        Initialize Binance client.
        
        Args:
            api_key: Binance API key. If None, will try to get from environment variables.
            api_secret: Binance API secret. If None, will try to get from environment variables.
            testnet: Whether to use the testnet.
        \"\"\"
        # Use provided keys or get from environment
        self.api_key = api_key or os.environ.get("BINANCE_API_KEY")
        self.api_secret = api_secret or os.environ.get("BINANCE_API_SECRET")
        
        # Initialize client
        self.client = Client(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=testnet
        )
        
        logger.info("Binance client initialized")
        
    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_str: Union[str, int, datetime],
        end_str: Optional[Union[str, int, datetime]] = None
    ) -> pd.DataFrame:
        \"\"\"
        Get historical klines (OHLCV) data from Binance.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1m', '5m', '1h', '1d')
            start_str: Start time in various formats ('1 day ago', '1 Jan 2021', timestamp)
            end_str: End time (optional, default is now)
            
        Returns:
            DataFrame with OHLCV data
        \"\"\"
        logger.info(f"Fetching {interval} klines for {symbol} from {start_str}")
        
        # Fetch klines from Binance
        klines = self.client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_str,
            end_str=end_str
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(
            klines,
            columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
        )
        
        # Clean and convert data types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Convert price and volume columns to numeric
        numeric_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'quote_asset_volume', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume'
        ]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Fetched {len(df)} rows of data")
        return df
    
    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict:
        \"\"\"
        Get exchange information for a specific symbol or all symbols.
        
        Args:
            symbol: Trading pair symbol (optional)
            
        Returns:
            Dictionary with exchange information
        \"\"\"
        if symbol:
            return self.client.get_symbol_info(symbol)
        else:
            return self.client.get_exchange_info()
    
    def get_available_intervals(self) -> List[str]:
        \"\"\"
        Get available kline intervals.
        
        Returns:
            List of available interval strings
        \"\"\"
        return [
            '1m', '3m', '5m', '15m', '30m',
            '1h', '2h', '4h', '6h', '8h', '12h',
            '1d', '3d', '1w', '1M'
        ]
"""

# Write to file
with open("pyanalysisml/data/binance_client.py", "w") as f:
    f.write(content)

print("Created file: pyanalysisml/data/binance_client.py")
