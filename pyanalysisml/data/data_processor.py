"""
Data preprocessing functions for PyAnalysisML.

This module provides functions for data preprocessing:
- Data cleaning
- Normalization
- Handling missing values
- Train-test splitting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from pyanalysisml.utils.logging_utils import get_logger
from pyanalysisml.config import DEFAULT_TEST_SIZE

logger = get_logger(__name__)

def clean_dataframe(
    df: pd.DataFrame,
    drop_na: bool = True,
    fill_method: Optional[str] = None,
    drop_columns: Optional[List[str]] = None,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Clean a DataFrame by handling missing values and duplicate rows.
    
    Args:
        df: DataFrame to clean
        drop_na: Whether to drop rows with missing values
        fill_method: Method to fill missing values ('ffill', 'bfill', 'mean', 'median', 'zero')
        drop_columns: Columns to drop
        drop_duplicates: Whether to drop duplicate rows
    
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Cleaning DataFrame with shape {df.shape}")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Drop specified columns
    if drop_columns:
        df = df.drop(columns=drop_columns, errors='ignore')
        
    # Handle duplicates
    if drop_duplicates:
        original_len = len(df)
        df = df.drop_duplicates()
        if len(df) < original_len:
            logger.info(f"Dropped {original_len - len(df)} duplicate rows")
    
    # Handle missing values
    if fill_method:
        if fill_method == 'ffill':
            df = df.ffill()
        elif fill_method == 'bfill':
            df = df.bfill()
        elif fill_method == 'mean':
            df = df.fillna(df.mean(numeric_only=True))
        elif fill_method == 'median':
            df = df.fillna(df.median(numeric_only=True))
        elif fill_method == 'zero':
            df = df.fillna(0)
        else:
            logger.warning(f"Unknown fill method: {fill_method}. Skipping.")
    
    # Drop rows with missing values if requested
    if drop_na:
        original_len = len(df)
        df = df.dropna()
        if len(df) < original_len:
            logger.info(f"Dropped {original_len - len(df)} rows with missing values")
    
    logger.info(f"Cleaned DataFrame has shape {df.shape}")
    return df

def normalize_dataframe(
    df: pd.DataFrame,
    method: str = 'standard',
    columns: Optional[List[str]] = None,
    return_scaler: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Union[StandardScaler, MinMaxScaler]]]:
    """
    Normalize numerical columns in a DataFrame.
    
    Args:
        df: DataFrame to normalize
        method: Normalization method ('standard', 'minmax')
        columns: Columns to normalize (default: all numeric columns)
        return_scaler: Whether to return the scaler object
    
    Returns:
        Normalized DataFrame and optionally the scaler object
    """
    logger.info(f"Normalizing DataFrame using {method} scaling")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Determine which columns to normalize
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    else:
        # Only include columns that exist and are numeric
        numeric_cols = df.select_dtypes(include=['number']).columns
        columns = [col for col in columns if col in numeric_cols]
    
    # Select the scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Normalize the data
    if columns:
        df[columns] = scaler.fit_transform(df[columns])
        logger.info(f"Normalized {len(columns)} columns")
    else:
        logger.warning("No numeric columns to normalize")
    
    if return_scaler:
        return df, scaler
    else:
        return df

def split_data(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    test_size: float = DEFAULT_TEST_SIZE,
    validation_size: Optional[float] = None,
    random_state: int = 42,
    shuffle: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Split a DataFrame into train, test, and optionally validation sets.
    
    Args:
        df: DataFrame to split
        target_column: Column containing the target variable
        feature_columns: Columns to use as features (default: all except target)
        test_size: Proportion of data to use for testing
        validation_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle the data before splitting
    
    Returns:
        Dictionary with keys 'X_train', 'y_train', 'X_test', 'y_test',
        and optionally 'X_val', 'y_val'
    """
    logger.info(f"Splitting data with test_size={test_size}")
    
    # Determine feature columns if not specified
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    # Extract features and target
    X = df[feature_columns].values
    y = df[target_column].values
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
    )
    
    result = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_columns': feature_columns
    }
    
    # Split train set into train and validation if requested
    if validation_size:
        # Adjust validation size to be relative to the training set
        val_size = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state, shuffle=shuffle
        )
        result['X_train'] = X_train
        result['y_train'] = y_train
        result['X_val'] = X_val
        result['y_val'] = y_val
        
        logger.info(f"Split data into train ({len(X_train)}), validation ({len(X_val)}), and test ({len(X_test)}) sets")
    else:
        logger.info(f"Split data into train ({len(X_train)}) and test ({len(X_test)}) sets")
    
    return result

def handle_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'clip',
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Handle outliers in numerical columns.
    
    Args:
        df: DataFrame to process
        columns: Columns to check for outliers (default: all numeric columns)
        method: Method to handle outliers ('clip', 'remove', 'iqr')
        threshold: Threshold for outlier detection (for z-score method)
    
    Returns:
        DataFrame with outliers handled
    """
    logger.info(f"Handling outliers using {method} method")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Determine which columns to check
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    if method == 'clip':
        # Clip values outside of z-score threshold
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = df[col].clip(lower=mean - threshold * std, upper=mean + threshold * std)
            
    elif method == 'remove':
        # Remove rows with values outside of z-score threshold
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            df = df[((df[col] - mean) / std).abs() < threshold]
            
    elif method == 'iqr':
        # Use IQR method (remove values outside 1.5*IQR)
        for col in columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    else:
        logger.warning(f"Unknown outlier handling method: {method}. Skipping.")
    
    logger.info(f"Handled outliers in {len(columns)} columns")
    return df 