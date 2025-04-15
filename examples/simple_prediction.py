"""
Example script for fetching OHLC data, calculating indicators, and making predictions.
"""

import logging
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Add package to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyanalysisml.data.binance_client import BinanceClient
from pyanalysisml.features.technical_indicators import add_custom_features, add_indicators

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_features(df, target_column="close", prediction_horizon=1):
    """
    Prepare features for machine learning.

    Args:
        df: DataFrame with OHLC data and indicators
        target_column: Column to predict
        prediction_horizon: Number of periods ahead to predict

    Returns:
        X, y: Features and target variables
    """
    # Create target variable (future price change)
    df["target"] = df[target_column].pct_change(prediction_horizon).shift(-prediction_horizon)

    # Drop NaN values
    df = df.dropna()

    # Define features to use (exclude non-feature columns)
    exclude_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
        "target",
    ]

    # Get feature columns (all columns except excluded ones)
    feature_columns = [col for col in df.columns if col not in exclude_columns]

    # Create features and target arrays
    X = df[feature_columns].values
    y = df["target"].values

    return X, y, df, feature_columns


def main():
    """Main function to run the example."""
    # Initialize Binance client
    client = BinanceClient()

    # Fetch historical data
    symbol = "BTCUSDT"
    interval = "1d"
    start_date = "1 year ago"

    logger.info(f"Fetching {interval} data for {symbol} from {start_date}")
    df = client.get_historical_klines(symbol, interval, start_date)

    # Calculate technical indicators
    df = add_indicators(df)

    # Add custom features
    df = add_custom_features(df)

    # Prepare features for machine learning
    prediction_horizon = 5  # predict 5 days ahead
    X, y, processed_df, feature_columns = prepare_features(
        df, target_column="close", prediction_horizon=prediction_horizon
    )

    # Split data into training and testing sets
    test_size = 30  # Use last 30 days for testing
    X_train, y_train = X[:-test_size], y[:-test_size]
    X_test, y_test = X[-test_size:], y[-test_size:]

    # Train a Random Forest model
    logger.info("Training Random Forest model")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Model Evaluation:")
    logger.info(f"Mean Absolute Error: {mae:.4f}")
    logger.info(f"Root Mean Squared Error: {rmse:.4f}")
    logger.info(f"R^2 Score: {r2:.4f}")

    # Get feature importances
    feature_importance = pd.DataFrame(
        {"Feature": feature_columns, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    logger.info("\nTop 10 Important Features:")
    logger.info(feature_importance.head(10))

    # Visualize actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(processed_df.index[-test_size:], y_test, label="Actual")
    plt.plot(processed_df.index[-test_size:], y_pred, label="Predicted")
    plt.title(f"{symbol} Price Change Prediction ({prediction_horizon} days ahead)")
    plt.xlabel("Date")
    plt.ylabel(f"{prediction_horizon}-day Price Change (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{symbol}_prediction_{prediction_horizon}d.png")
    logger.info(f"Saved prediction plot to {symbol}_prediction_{prediction_horizon}d.png")


if __name__ == "__main__":
    main()
