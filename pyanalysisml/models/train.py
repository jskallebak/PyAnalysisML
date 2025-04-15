"""
Model training and evaluation utilities.
"""
import logging
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

def prepare_data(
    df: pd.DataFrame,
    target_column: str = "close",
    prediction_horizon: int = 1,
    test_size: Union[int, float] = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for machine learning by creating features and target variables.
    
    Args:
        df: DataFrame with OHLC data and indicators
        target_column: Column to predict
        prediction_horizon: Number of periods ahead to predict
        test_size: Size of test set (fraction or number of samples)
        random_state: Random seed for train/test split
    
    Returns:
        X_train, y_train, X_test, y_test: Training and testing data
    """
    # Create target variable (future price change)
    df = df.copy()
    df["target"] = df[target_column].pct_change(prediction_horizon).shift(-prediction_horizon)
    
    # Drop NaN values
    df = df.dropna()
    
    # Define columns to exclude from features
    exclude_columns = [
        "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
        "ignore", "target"
    ]
    
    # Get feature columns
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    # Create features and target arrays
    X = df[feature_columns].values
    y = df["target"].values
    
    # Split data
    if isinstance(test_size, int):
        # Use last n samples for testing if test_size is an integer
        X_train, y_train = X[:-test_size], y[:-test_size]
        X_test, y_test = X[-test_size:], y[-test_size:]
    else:
        # Otherwise use train_test_split with fraction
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    logger.info(f"Prepared data with {X_train.shape[0]} training samples and {X_test.shape[0]} test samples")
    return X_train, y_train, X_test, y_test

def train_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    **fit_params
) -> BaseEstimator:
    """
    Train a machine learning model.
    
    Args:
        model: Model instance to train
        X_train: Training features
        y_train: Training targets
        **fit_params: Additional parameters to pass to model.fit()
    
    Returns:
        Trained model
    """
    logger.info(f"Training {model.__class__.__name__} model on {X_train.shape[0]} samples")
    model.fit(X_train, y_train, **fit_params)
    return model

def evaluate_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate a trained model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
    
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }
    
    logger.info(f"Model evaluation metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    return metrics 