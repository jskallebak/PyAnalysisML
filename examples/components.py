#!/usr/bin/env python3
"""
Example components for use with the PyAnalysisML Pipeline.

This module registers various test components for use in example scripts.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.datasets import load_iris

from pyanalysisml.pipeline.registry import component_registry


@component_registry.register
class IrisDataLoader:
    """Simple data loader for the Iris dataset."""
    
    def __init__(self, **kwargs):
        """Initialize the Iris data loader."""
        self.params = kwargs
    
    def load(self, **kwargs):
        """Load the Iris dataset."""
        iris = load_iris()
        data = pd.DataFrame(
            data=np.c_[iris['data'], iris['target']],
            columns=iris['feature_names'] + ['target']
        )
        return data


@component_registry.register
class StandardScaler:
    """Feature engineering component that standardizes features."""
    
    def __init__(self, with_mean=True, with_std=True, **kwargs):
        """Initialize the standard scaler."""
        self.with_mean = with_mean
        self.with_std = with_std
        self.scaler = SklearnStandardScaler(with_mean=with_mean, with_std=with_std)
        self.target_col = None
    
    def fit(self, data, target_column='target', **kwargs):
        """
        Fit the scaler on the data.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
        """
        self.target_col = target_column
        X = data.drop(columns=[target_column])
        self.scaler.fit(X)
        return self
    
    def transform(self, data, target_column='target', **kwargs):
        """
        Transform the data using the fitted scaler.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            **kwargs: Additional parameters
            
        Returns:
            Transformed DataFrame
        """
        self.target_col = target_column
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # If not fitted, fit the scaler
        if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
            self.fit(data, target_column)
        
        X_scaled = self.scaler.transform(X)
        
        # Convert to DataFrame maintaining column names
        result = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        result[target_column] = y
        
        return result


@component_registry.register
class StandardSplitter:
    """Splits data into train, validation, and test sets."""
    
    def __init__(self, test_size=0.2, val_size=0.2, random_state=None, **kwargs):
        """Initialize the standard splitter."""
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
    
    def split(self, data, target_column='target', **kwargs):
        """
        Split the data into train, validation, and test sets.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with train, validation, and test datasets
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # First split to get test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Second split to get validation set
        if self.val_size > 0:
            # Calculate the validation size relative to the train_val set
            val_size_adjusted = self.val_size / (1 - self.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_size_adjusted, random_state=self.random_state
            )
            
            return {
                'train': (X_train, y_train),
                'val': (X_val, y_val),
                'test': (X_test, y_test)
            }
        else:
            return {
                'train': (X_train_val, y_train_val),
                'test': (X_test, y_test)
            }


@component_registry.register
class RandomForestFactory:
    """Factory for creating Random Forest models."""
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=None, **kwargs):
        """Initialize the random forest factory."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.params = kwargs
    
    def create(self, **kwargs):
        """
        Create a new Random Forest model instance.
        
        Args:
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Random Forest model instance
        """
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.random_state,
            **self.params,
            **kwargs
        }
        
        return RandomForestClassifier(**params)


@component_registry.register
class StandardTrainer:
    """Standard trainer for scikit-learn models."""
    
    def __init__(self, cv=None, scoring=None, **kwargs):
        """Initialize the standard trainer."""
        self.cv = cv
        self.scoring = scoring
        self.params = kwargs
    
    def train(self, model, train_data, val_data=None, **kwargs):
        """
        Train the model on the provided data.
        
        Args:
            model: The model to train
            train_data: Tuple of (X_train, y_train)
            val_data: Optional tuple of (X_val, y_val)
            **kwargs: Additional training parameters
            
        Returns:
            Tuple of (trained_model, training_metrics)
        """
        X_train, y_train = train_data
        
        # Cross-validation if requested
        metrics = {}
        if self.cv is not None:
            cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv, scoring=self.scoring)
            metrics['cv_scores'] = cv_scores
            metrics['cv_mean_score'] = np.mean(cv_scores)
            metrics['cv_std_score'] = np.std(cv_scores)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Get training accuracy
        y_train_pred = model.predict(X_train)
        metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
        
        # Evaluate on validation set if provided
        if val_data is not None:
            X_val, y_val = val_data
            y_val_pred = model.predict(X_val)
            metrics['val_accuracy'] = accuracy_score(y_val, y_val_pred)
        
        return model, metrics


@component_registry.register
class ClassificationEvaluator:
    """Evaluator for classification models."""
    
    def __init__(self, metrics=None, average='weighted', **kwargs):
        """Initialize the classification evaluator."""
        self.metrics = metrics or ['accuracy', 'precision', 'recall', 'f1']
        self.average = average
        self.params = kwargs
    
    def evaluate(self, model, test_data, **kwargs):
        """
        Evaluate the model on test data.
        
        Args:
            model: Trained model
            test_data: Tuple of (X_test, y_test)
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary of evaluation metrics
        """
        X_test, y_test = test_data
        y_pred = model.predict(X_test)
        
        metrics = {}
        
        if 'accuracy' in self.metrics:
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
        
        if 'precision' in self.metrics:
            metrics['precision'] = precision_score(y_test, y_pred, average=self.average, zero_division=0)
        
        if 'recall' in self.metrics:
            metrics['recall'] = recall_score(y_test, y_pred, average=self.average, zero_division=0)
        
        if 'f1' in self.metrics:
            metrics['f1'] = f1_score(y_test, y_pred, average=self.average, zero_division=0)
        
        return metrics


@component_registry.register
class RegressionEvaluator:
    """Evaluator for regression models."""
    
    def __init__(self, metrics=None, **kwargs):
        """Initialize the regression evaluator."""
        self.metrics = metrics or ['rmse', 'mae', 'r2']
        self.params = kwargs
    
    def evaluate(self, model, test_data, **kwargs):
        """
        Evaluate the model on test data.
        
        Args:
            model: Trained model
            test_data: Tuple of (X_test, y_test)
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary of evaluation metrics
        """
        X_test, y_test = test_data
        y_pred = model.predict(X_test)
        
        metrics = {}
        
        if 'rmse' in self.metrics:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        
        if 'mae' in self.metrics:
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
        
        if 'r2' in self.metrics:
            metrics['r2'] = r2_score(y_test, y_pred)
        
        return metrics


@component_registry.register
class StandardPredictor:
    """Standard predictor for scikit-learn models."""
    
    def __init__(self, **kwargs):
        """Initialize the standard predictor."""
        self.params = kwargs
    
    def predict(self, model, data, **kwargs):
        """
        Generate predictions using the trained model.
        
        Args:
            model: Trained model
            data: Input data, either as DataFrame or tuple (X, y)
            **kwargs: Additional prediction parameters
            
        Returns:
            Predictions
        """
        if isinstance(data, tuple):
            X, _ = data
        else:
            X = data
        
        return model.predict(X)


# Register all components
def register_all():
    """Register all components with the component registry."""
    # Components are registered by decorators, so this function just ensures 
    # that this module is imported and the decorators are executed
    pass


if __name__ == "__main__":
    # Print all registered components
    components = component_registry.list_components()
    for name, component in components.items():
        print(f"Registered component: {name}") 