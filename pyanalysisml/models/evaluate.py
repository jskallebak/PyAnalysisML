"""
Model evaluation utilities for PyAnalysisML.

This module provides functions for evaluating machine learning models.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error,
)
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from pyanalysisml.utils.logging_utils import get_logger

logger = get_logger(__name__)

def evaluate_regression_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Evaluate a regression model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True test values
        X_train: Training features (for calculating training metrics, optional)
        y_train: True training values (optional)
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating {model.__class__.__name__} model")
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Calculate test metrics
    metrics = {
        'test_mae': mean_absolute_error(y_test, y_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'test_r2': r2_score(y_test, y_pred),
        'test_explained_variance': explained_variance_score(y_test, y_pred),
    }
    
    # Calculate MAPE if no zero values in y_test
    if not np.any(y_test == 0):
        metrics['test_mape'] = mean_absolute_percentage_error(y_test, y_pred)
    
    # Calculate train metrics if provided
    if X_train is not None and y_train is not None:
        y_train_pred = model.predict(X_train)
        
        metrics['train_mae'] = mean_absolute_error(y_train, y_train_pred)
        metrics['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_train_pred))
        metrics['train_r2'] = r2_score(y_train, y_train_pred)
        metrics['train_explained_variance'] = explained_variance_score(y_train, y_train_pred)
        
        if not np.any(y_train == 0):
            metrics['train_mape'] = mean_absolute_percentage_error(y_train, y_train_pred)
    
    logger.info(f"Test metrics: MAE={metrics['test_mae']:.4f}, RMSE={metrics['test_rmse']:.4f}, R²={metrics['test_r2']:.4f}")
    return metrics

def cross_validate_model(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = 'neg_mean_squared_error',
    time_series: bool = True,
) -> Dict[str, Union[float, List[float]]]:
    """
    Perform cross-validation on a model.
    
    Args:
        model: Model to cross-validate
        X: Features
        y: Target values
        cv: Number of folds
        scoring: Scoring metric
        time_series: Whether to use time series cross-validation
    
    Returns:
        Dictionary with cross-validation results
    """
    logger.info(f"Performing {cv}-fold cross-validation with {scoring} metric")
    
    # Select cross-validation strategy
    if time_series:
        cv_strategy = TimeSeriesSplit(n_splits=cv)
    else:
        cv_strategy = cv
    
    # Perform cross-validation
    cv_scores = cross_val_score(
        model, X, y, cv=cv_strategy, scoring=scoring
    )
    
    # Calculate metrics
    if scoring.startswith('neg_'):
        cv_scores = -cv_scores  # Convert back to positive values for easier interpretation
    
    results = {
        'cv_scores': cv_scores.tolist(),
        'mean_score': cv_scores.mean(),
        'std_score': cv_scores.std(),
        'min_score': cv_scores.min(),
        'max_score': cv_scores.max(),
    }
    
    logger.info(f"Cross-validation results: mean={results['mean_score']:.4f}, std={results['std_score']:.4f}")
    return results

def feature_importance(
    model: BaseEstimator,
    feature_names: List[str],
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Get feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: Names of features
        top_n: Number of top features to return (optional)
    
    Returns:
        DataFrame with feature importance
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        logger.warning(f"Model {model.__class__.__name__} does not have feature_importances_ attribute")
        return pd.DataFrame()
    
    # Create DataFrame with feature importance
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    
    # Sort by importance
    importance = importance.sort_values('Importance', ascending=False)
    
    # Get top N features if specified
    if top_n is not None:
        importance = importance.head(top_n)
    
    logger.info(f"Extracted feature importance for {len(importance)} features")
    return importance

def prediction_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[Union[List[str], pd.DatetimeIndex]] = None,
) -> pd.DataFrame:
    """
    Create a DataFrame with actual vs predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        dates: Dates corresponding to the values (optional)
    
    Returns:
        DataFrame with actual and predicted values
    """
    # Create DataFrame
    if dates is not None:
        df = pd.DataFrame({
            'Date': dates,
            'Actual': y_true,
            'Predicted': y_pred,
            'Error': y_true - y_pred,
            'AbsError': np.abs(y_true - y_pred),
            'PercentError': np.abs((y_true - y_pred) / y_true) * 100 if not np.any(y_true == 0) else np.nan
        })
        df.set_index('Date', inplace=True)
    else:
        df = pd.DataFrame({
            'Actual': y_true,
            'Predicted': y_pred,
            'Error': y_true - y_pred,
            'AbsError': np.abs(y_true - y_pred),
            'PercentError': np.abs((y_true - y_pred) / y_true) * 100 if not np.any(y_true == 0) else np.nan
        })
    
    logger.info(f"Created prediction vs actual DataFrame with {len(df)} rows")
    return df

def calculate_directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate directional accuracy (percentage of correct direction predictions).
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Directional accuracy (0-1)
    """
    # Calculate direction (signs)
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    
    # Calculate directional accuracy
    correct_directions = np.sum(true_direction == pred_direction)
    directional_accuracy = correct_directions / len(y_true)
    
    logger.info(f"Directional accuracy: {directional_accuracy:.4f}")
    return directional_accuracy 