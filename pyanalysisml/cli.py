"""
Command Line Interface for PyAnalysisML.
"""
import argparse
import logging
import sys

from pyanalysisml.data.data_loader import load_from_binance
from pyanalysisml.features.technical_indicators import add_indicators, add_custom_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="PyAnalysisML - Cryptocurrency data analysis and ML")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Fetch data command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch historical OHLC data")
    fetch_parser.add_argument("symbol", help="Trading pair symbol (e.g., BTCUSDT)")
    fetch_parser.add_argument("--interval", default="1d", help="Kline interval (e.g., 1m, 5m, 1h, 1d)")
    fetch_parser.add_argument("--start", default="1 month ago", help="Start time (e.g., '1 month ago', '2021-01-01')")
    fetch_parser.add_argument("--end", default=None, help="End time (optional, default is now)")
    fetch_parser.add_argument("--output", default="data.csv", help="Output CSV file")
    fetch_parser.add_argument("--chunk-size", type=int, default=30, 
                             help="Number of days per chunk for parallel processing (default: 30)")
    fetch_parser.add_argument("--max-workers", type=int, default=8,
                             help="Maximum number of parallel workers (default: 8)")
    
    # Indicators command
    indicators_parser = subparsers.add_parser("indicators", help="Calculate technical indicators")
    indicators_parser.add_argument("input", help="Input CSV file with OHLC data")
    indicators_parser.add_argument("--output", default="data_with_indicators.csv", help="Output CSV file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # No command specified
    if args.command is None:
        parser.print_help()
        return 1
    
    try:
        if args.command == "fetch":
            logger.info(f"Fetching {args.interval} data for {args.symbol} from {args.start}")
            
            # Use the load_from_binance function which supports parallel processing
            df = load_from_binance(
                symbol=args.symbol,
                interval=args.interval,
                start_str=args.start,
                end_str=args.end,
                save_csv=False,
                chunk_size=args.chunk_size,
                max_workers=args.max_workers
            )
            
            df.to_csv(args.output)
            logger.info(f"Saved {len(df)} rows of data to {args.output}")
            
        elif args.command == "indicators":
            logger.info(f"Calculating indicators for {args.input}")
            import pandas as pd
            df = pd.read_csv(args.input, index_col=0, parse_dates=True)
            df = add_indicators(df)
            df = add_custom_features(df)
            df.to_csv(args.output)
            logger.info(f"Saved data with indicators to {args.output}")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 