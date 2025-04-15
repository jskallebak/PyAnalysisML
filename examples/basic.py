# Basic usage example
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyanalysisml.data.data_loader import load_from_binance
from pyanalysisml.data.data_processor import clean_dataframe, normalize_dataframe
from pyanalysisml.features.feature_engineering import create_all_features
from pyanalysisml.models.model_factory import create_model
from pyanalysisml.models.train import prepare_data, train_model
from pyanalysisml.models.evaluate import evaluate_regression_model, feature_importance
from pyanalysisml.utils.visualization import plot_predictions
from pyanalysisml.utils.logging_utils import setup_logging
import time

# Set up logging
setup_logging()

# 1. Load data from Binance
print("Loading data using parallel processing...")
start_time = time.time()
df = load_from_binance(
    symbol="BTCUSDT",
    interval="1d",
    start_str="2 years ago",  # Use more data to demonstrate parallel processing
    save_csv=True,  # Save to CSV for future use
    chunk_size=30,  # Split into 30-day chunks
    max_workers=4   # Use 4 workers in parallel
)
print(f"Loaded data shape: {df.shape}")
print(f"Parallel loading took {time.time() - start_time:.2f} seconds")

# For comparison, load a smaller amount of data without parallel processing
print("\nLoading data without parallel processing (for comparison)...")
start_time = time.time()
df_small = load_from_binance(
    symbol="BTCUSDT",
    interval="1d",
    start_str="3 months ago",
    save_csv=False
)
print(f"Non-parallel loading took {time.time() - start_time:.2f} seconds")

# 2. Clean and preprocess data
df = clean_dataframe(df, fill_method='ffill', drop_na=False)  # Don't drop NAs yet
print(f"Cleaned data shape: {df.shape}")

# 3. Engineer features - only use essential features to minimize NaN values
df_features = df.copy()
# Add technical indicators with minimal lookback periods
from pyanalysisml.features.technical_indicators import add_indicators
df_features = add_indicators(df_features, indicators=['sma', 'ema', 'rsi', 'macd'])
print(f"After adding indicators shape: {df_features.shape}")

# Add minimal lag features and returns
df_features['close_lag_1'] = df_features['close'].shift(1)
df_features['close_lag_5'] = df_features['close'].shift(5)
df_features['return_1d'] = df_features['close'].pct_change() * 100
df_features['return_5d'] = df_features['close'].pct_change(5) * 100
print(f"After adding lags shape: {df_features.shape}")

# Now drop NaNs
df_features = df_features.dropna()
print(f"After dropping NaNs shape: {df_features.shape}")

# 4. Prepare data for modeling (predict 1-day price change)
X_train, y_train, X_test, y_test = prepare_data(
    df_features,
    target_column="close",
    prediction_horizon=1,
    test_size=30  # Use last 30 days for testing
)

print(f"Training data shape: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Test data shape: X_test {X_test.shape}, y_test {y_test.shape}")

# Get feature columns (needed for feature importance later)
exclude_columns = ["open", "high", "low", "close", "volume", 
                   "close_time", "quote_asset_volume", "number_of_trades",
                   "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore", "target"]
feature_columns = [col for col in df_features.columns if col not in exclude_columns]
print(f"Number of feature columns: {len(feature_columns)}")

# 5. Create and train model
model = create_model("random_forest")
model = train_model(model, X_train, y_train)

# 6. Evaluate model
metrics = evaluate_regression_model(model, X_test, y_test, X_train, y_train)
print(f"Test RMSE: {metrics['test_rmse']:.4f}")
print(f"Test R²: {metrics['test_r2']:.4f}")

# 7. Get feature importance
importances = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
})
importances = importances.sort_values('Importance', ascending=False).head(10)
print("Top 10 important features:")
print(importances)

# 8. Plot predictions
predictions = model.predict(X_test)
fig = plot_predictions(
    df_features.iloc[-len(y_test):],
    y_test,
    predictions,
    title="BTC Price Prediction"
)

plt.show()