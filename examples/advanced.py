"""
Advanced example of using PyAnalysisML.

This example demonstrates:
1. Loading data from Binance
2. Advanced feature engineering
3. Model comparison (RandomForest, XGBoost, LightGBM)
4. Cross-validation
5. Feature importance visualization
6. Prediction visualization
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from pyanalysisml.data.data_loader import load_from_binance
from pyanalysisml.data.data_processor import clean_dataframe, normalize_dataframe, handle_outliers
from pyanalysisml.features.feature_engineering import create_all_features
from pyanalysisml.models.model_factory import create_model, get_default_params
from pyanalysisml.models.train import prepare_data, train_model
from pyanalysisml.models.evaluate import (
    evaluate_regression_model, 
    cross_validate_model,
    feature_importance,
    prediction_vs_actual,
    calculate_directional_accuracy
)
from pyanalysisml.utils.visualization import plot_predictions, plot_feature_importance
from pyanalysisml.utils.logging_utils import setup_logging

# Set up logging
setup_logging()

# Parameters
SYMBOL = "BTCUSDT"
INTERVAL = "1d"
START_DATE = "2 years ago"
TEST_SIZE = 60  # 60 days for testing
PREDICTION_HORIZON = 5  # Predict 5-day price change

print(f"-- Loading data for {SYMBOL} --")
# 1. Load data
df = load_from_binance(
    symbol=SYMBOL,
    interval=INTERVAL,
    start_str=START_DATE,
    save_csv=True
)

# 2. Clean and preprocess data
print("\n-- Preprocessing data --")
df = clean_dataframe(df, fill_method='ffill', drop_na=False)
df = handle_outliers(df, columns=['close', 'high', 'low', 'volume'], method='clip')

# 3. Feature engineering - limiting to avoid too many NaNs
df_features = df.copy()

# Add technical indicators with minimal lookback
from pyanalysisml.features.technical_indicators import add_indicators
df_features = add_indicators(df_features, indicators=['sma', 'ema', 'rsi', 'macd', 'bbands'])

# Add some lag features
for col in ['close', 'volume']:
    for lag in [1, 5, 10]:
        df_features[f'{col}_lag_{lag}'] = df_features[col].shift(lag)

# Add price returns
for period in [1, 5, 10, 20]:
    df_features[f'return_{period}d'] = df_features['close'].pct_change(period) * 100

# Add some volatility features
for window in [5, 10, 20]:
    df_features[f'volatility_{window}d'] = df_features['close'].pct_change().rolling(window).std() * 100

# Add cyclical time features
if isinstance(df_features.index, pd.DatetimeIndex):
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['month'] = df_features.index.month
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)

# Now drop NaNs
df_features = df_features.dropna()
print(f"Final features shape: {df_features.shape}")

# 4. Prepare data for modeling
print("\n-- Preparing data --")
X_train, y_train, X_test, y_test = prepare_data(
    df_features,
    target_column="close",
    prediction_horizon=PREDICTION_HORIZON,
    test_size=TEST_SIZE
)

print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")

# Get feature columns for later use
feature_columns = [col for col in df_features.columns if col not in [
    "open", "high", "low", "close", "volume", "close_time", 
    "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", 
    "taker_buy_quote_asset_volume", "ignore", "target"
]]

# 5. Model training and comparison
print("\n-- Training and comparing models --")
models = {}
predictions = {}
metrics = {}

# Train multiple models
for model_name in ["random_forest", "xgboost", "lightgbm"]:
    print(f"\nTraining {model_name}...")
    
    # Get default parameters and create model
    params = get_default_params(model_name)
    models[model_name] = create_model(model_name, params)
    
    # Train model
    models[model_name] = train_model(models[model_name], X_train, y_train)
    
    # Evaluate model
    metrics[model_name] = evaluate_regression_model(
        models[model_name], X_test, y_test, X_train, y_train
    )
    
    # Make predictions
    predictions[model_name] = models[model_name].predict(X_test)
    
    # Calculate directional accuracy
    directional_acc = calculate_directional_accuracy(y_test, predictions[model_name])
    metrics[model_name]['directional_accuracy'] = directional_acc
    
    print(f"{model_name.upper()} - RMSE: {metrics[model_name]['test_rmse']:.4f}, "
          f"R²: {metrics[model_name]['test_r2']:.4f}, "
          f"Directional Accuracy: {directional_acc:.4f}")

# 6. Cross-validation of the best model
best_model_name = min(metrics, key=lambda m: metrics[m]['test_rmse'])
print(f"\nBest model: {best_model_name}")

print("\n-- Performing cross-validation on best model --")
cv_results = cross_validate_model(
    models[best_model_name],
    X_train, y_train,
    cv=5,
    scoring='neg_root_mean_squared_error',
    time_series=True
)

print(f"Cross-validation RMSE: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")

# 7. Feature importance for the best model
print("\n-- Feature importance analysis --")
importances = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': models[best_model_name].feature_importances_
})
importances = importances.sort_values('Importance', ascending=False)

print("Top 10 important features:")
print(importances.head(10))

# 8. Create comparison DataFrame of actual vs predicted values
print("\n-- Creating prediction comparison --")
result_df = pd.DataFrame({
    'Actual': y_test,
    'RandomForest': predictions['random_forest'],
    'XGBoost': predictions['xgboost'],
    'LightGBM': predictions['lightgbm'],
})

# 9. Visualization
print("\n-- Generating visualizations --")

# Set the style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 7]

# Plot 1: Model Comparison
plt.figure(figsize=(12, 6))
plt.plot(result_df.index, result_df['Actual'], 'b-', label='Actual', linewidth=2)
plt.plot(result_df.index, result_df['RandomForest'], 'r--', label='RandomForest')
plt.plot(result_df.index, result_df['XGBoost'], 'g-.', label='XGBoost')
plt.plot(result_df.index, result_df['LightGBM'], 'm:', label='LightGBM')
plt.title(f'{SYMBOL} {PREDICTION_HORIZON}-Day Price Prediction Comparison')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.savefig(f'{SYMBOL}_model_comparison.png')

# Plot 2: Feature Importance
plt.figure(figsize=(10, 8))
top_n = 15
plot_feature_importance(
    feature_names=importances['Feature'].tolist()[:top_n],
    importance_values=importances['Importance'].tolist()[:top_n],
    save_path=f'{SYMBOL}_feature_importance.png'
)

# Plot 3: Prediction Error Analysis
plt.figure(figsize=(12, 6))
errors = result_df.copy()
for model in ['RandomForest', 'XGBoost', 'LightGBM']:
    errors[f'{model}_Error'] = errors['Actual'] - errors[model]

plt.plot(errors.index, errors['RandomForest_Error'], 'r-', label='RandomForest Error')
plt.plot(errors.index, errors['XGBoost_Error'], 'g-', label='XGBoost Error')
plt.plot(errors.index, errors['LightGBM_Error'], 'm-', label='LightGBM Error')
plt.axhline(y=0, color='b', linestyle='-')
plt.title(f'{SYMBOL} Prediction Error Analysis')
plt.xlabel('Date')
plt.ylabel('Error (Actual - Predicted)')
plt.legend()
plt.tight_layout()
plt.savefig(f'{SYMBOL}_error_analysis.png')

print("\nAnalysis complete! Check the generated PNG files for visualizations.")
plt.show() 