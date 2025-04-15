"""
Binance client module for fetching OHLC cryptocurrency data.
"""
import os
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import concurrent.futures
import pandas as pd
from binance.client import Client

logger = logging.getLogger(__name__)

class BinanceClient:
    """Client for accessing Binance API to fetch historical OHLC data."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False
    ):
        """
        Initialize Binance client.
        
        Args:
            api_key: Binance API key. If None, will try to get from environment variables.
            api_secret: Binance API secret. If None, will try to get from environment variables.
            testnet: Whether to use the testnet.
        """
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
        end_str: Optional[Union[str, int, datetime]] = None,
        chunk_size: Optional[int] = None,
        max_workers: int = 8
    ) -> pd.DataFrame:
        """
        Get historical klines (OHLCV) data from Binance.
        
        This function fetches historical candlestick (kline) data from Binance for a specified trading pair
        and timeframe. It supports parallel processing by splitting the date range into chunks that can be
        fetched concurrently.
        
        The fetching process involves:
        1. Converting time inputs to datetime objects
        2. Splitting the time range into chunks if specified
        3. Fetching data for each chunk in parallel
        4. Combining chunks into a single dataframe
        5. Processing the raw data into a clean pandas DataFrame
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1m', '5m', '1h', '1d')
            start_str: Start time in various formats ('1 day ago', '1 Jan 2021', timestamp)
            end_str: End time (optional, default is now)
            chunk_size: Number of days per chunk for parallel processing (optional)
                        If None, fetches all data in a single request
            max_workers: Maximum number of parallel workers for fetching data chunks
            
        Returns:
            DataFrame with OHLCV data with columns:
            - timestamp (index): Open time of the candle
            - open: Opening price
            - high: Highest price during the interval
            - low: Lowest price during the interval
            - close: Closing price
            - volume: Trading volume
            - close_time: Close time of the candle
            - quote_asset_volume: Volume in quote asset
            - number_of_trades: Number of trades
            - taker_buy_base_asset_volume: Taker buy base asset volume
            - taker_buy_quote_asset_volume: Taker buy quote asset volume
        """
        logger.info(f"Fetching {interval} klines for {symbol} from {start_str}")
        
        # If no chunk_size specified, fetch all data in a single request
        if chunk_size is None:
            return self._fetch_and_process_klines(symbol, interval, start_str, end_str)
        
        # Convert start and end to datetime objects for chunking
        if isinstance(start_str, str) and not start_str.isdigit():
            # Handle relative times like "1 day ago"
            if "ago" in start_str:
                start_date = self.client._get_earliest_valid_timestamp(symbol, interval, start_str)
                start_date = datetime.fromtimestamp(start_date / 1000)
            else:
                # Handle date string
                start_date = datetime.strptime(start_str, "%Y-%m-%d")
        elif isinstance(start_str, (int, float)):
            # Handle timestamp
            start_date = datetime.fromtimestamp(start_str / 1000 if start_str > 10**10 else start_str)
        else:
            # Already a datetime
            start_date = start_str
            
        # Handle end time
        if end_str is None:
            end_date = datetime.now()
        elif isinstance(end_str, str) and not end_str.isdigit():
            if "ago" in end_str:
                end_date = self.client._get_earliest_valid_timestamp(symbol, interval, end_str)
                end_date = datetime.fromtimestamp(end_date / 1000)
            else:
                end_date = datetime.strptime(end_str, "%Y-%m-%d")
        elif isinstance(end_str, (int, float)):
            end_date = datetime.fromtimestamp(end_str / 1000 if end_str > 10**10 else end_str)
        else:
            end_date = end_str
            
        logger.info(f"Splitting fetch from {start_date} to {end_date} into chunks of {chunk_size} days")
        
        # Create time chunks
        time_chunks = self._create_time_chunks(start_date, end_date, chunk_size)
        
        # Fetch data in parallel
        all_dfs = []
        chunk_futures = {}  # Map futures to their time chunks for logging
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for chunk_start, chunk_end in time_chunks:
                future = executor.submit(
                    self._fetch_and_process_klines,
                    symbol, 
                    interval,
                    chunk_start,
                    chunk_end
                )
                futures.append(future)
                chunk_futures[future] = (chunk_start, chunk_end)
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    chunk_df = future.result()
                    chunk_start, chunk_end = chunk_futures[future]
                    if not chunk_df.empty:
                        all_dfs.append(chunk_df)
                        logger.info(f"Fetched chunk {chunk_start.date()} to {chunk_end.date()} with {len(chunk_df)} rows")
                    else:
                        logger.warning(f"No data for chunk {chunk_start.date()} to {chunk_end.date()}")
                except Exception as e:
                    chunk_start, chunk_end = chunk_futures[future]
                    logger.error(f"Error fetching chunk {chunk_start.date()} to {chunk_end.date()}: {str(e)}")
        
        # Combine all chunks
        if not all_dfs:
            logger.warning("No data fetched from any chunk")
            return pd.DataFrame()
            
        df = pd.concat(all_dfs)
        
        # Remove duplicates that might occur at chunk boundaries
        df = df.loc[~df.index.duplicated(keep='first')]
        
        # Sort by timestamp
        df = df.sort_index()
        
        logger.info(f"Fetched a total of {len(df)} rows of data from {start_date.date()} to {end_date.date()} using {len(time_chunks)} chunks")
        return df
    
    def _create_time_chunks(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        chunk_size: int
    ) -> List[Tuple[datetime, datetime]]:
        """
        Split a date range into chunks of specified size.
        
        Args:
            start_date: Start date
            end_date: End date
            chunk_size: Size of each chunk in days
            
        Returns:
            List of (chunk_start, chunk_end) tuples
        """
        chunks = []
        current_start = start_date
        
        while current_start < end_date:
            current_end = current_start + timedelta(days=chunk_size)
            if current_end > end_date:
                current_end = end_date
                
            chunks.append((current_start, current_end))
            current_start = current_end
            
        return chunks
    
    def _fetch_and_process_klines(
        self,
        symbol: str,
        interval: str,
        start_str: Union[str, int, datetime],
        end_str: Optional[Union[str, int, datetime]] = None
    ) -> pd.DataFrame:
        """
        Fetch and process klines for a specific time range.
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            start_str: Start time
            end_str: End time
            
        Returns:
            Processed DataFrame with kline data
        """
        # Convert datetime objects to timestamp strings for Binance API
        if isinstance(start_str, datetime):
            start_str = int(start_str.timestamp() * 1000)  # Convert to millisecond timestamp
        
        if end_str is not None and isinstance(end_str, datetime):
            end_str = int(end_str.timestamp() * 1000)  # Convert to millisecond timestamp
        
        # Fetch klines from Binance
        try:
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str
            )
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
        
        if not klines:
            logger.warning(f"No data returned for {symbol} {interval} from {start_str} to {end_str}")
            return pd.DataFrame()
        
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
        
        return df
    
    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict:
        """
        Get exchange information for a specific symbol or all symbols.
        
        Args:
            symbol: Trading pair symbol (optional)
            
        Returns:
            Dictionary with exchange information
        """
        if symbol:
            return self.client.get_symbol_info(symbol)
        else:
            return self.client.get_exchange_info()
    
    def get_available_intervals(self) -> List[str]:
        """
        Get available kline intervals.
        
        Returns:
            List of available interval strings
        """
        return [
            '1m', '3m', '5m', '15m', '30m',
            '1h', '2h', '4h', '6h', '8h', '12h',
            '1d', '3d', '1w', '1M'
        ]
