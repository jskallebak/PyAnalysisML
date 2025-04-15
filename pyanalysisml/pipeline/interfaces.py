"""
Base interfaces for pipeline components.

This module defines the core interfaces that all pipeline components
must implement to be compatible with the PyAnalysisML pipeline system.
"""

import abc
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class DataLoader(abc.ABC):
    """Interface for components that load data for the pipeline."""
    
    @abc.abstractmethod
    def load(self, **kwargs) -> pd.DataFrame:
        """
        Load the data into a pandas DataFrame.
        
        Args:
            **kwargs: Additional arguments for data loading
            
        Returns:
            A pandas DataFrame containing the loaded data
        """
        pass


class FeatureEngineer(abc.ABC):
    """Interface for components that perform feature engineering."""
    
    @abc.abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> "FeatureEngineer":
        """
        Fit the feature engineering process on the provided data.
        
        Args:
            data: Input data to fit on
            **kwargs: Additional arguments for fitting
            
        Returns:
            Self, for method chaining
        """
        pass
    
    @abc.abstractmethod
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply feature engineering to the provided data.
        
        Args:
            data: Input data to transform
            **kwargs: Additional arguments for transformation
            
        Returns:
            Transformed DataFrame with engineered features
        """
        pass
    
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Fit and then transform the data in one step.
        
        Args:
            data: Input data to fit and transform
            **kwargs: Additional arguments
            
        Returns:
            Transformed DataFrame with engineered features
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)


class DataSplitter(abc.ABC):
    """Interface for components that split data for training/validation/testing."""
    
    @abc.abstractmethod
    def split(
        self, 
        data: pd.DataFrame, 
        target_column: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]]:
        """
        Split the data into training, validation, and test sets.
        
        Args:
            data: The input data to split
            target_column: Optional name of the target column (for supervised learning)
            **kwargs: Additional arguments for splitting
            
        Returns:
            Dictionary containing the split datasets, with keys such as:
            - 'train': Training data
            - 'val': Validation data
            - 'test': Test data
            
            If target_column is provided, each value will be a tuple of (X, y)
        """
        pass


class ModelFactory(abc.ABC):
    """Interface for components that create and configure models."""
    
    @abc.abstractmethod
    def create_model(self, **kwargs) -> Any:
        """
        Create and return a new model instance.
        
        Args:
            **kwargs: Configuration parameters for the model
            
        Returns:
            A new model instance ready for training
        """
        pass


class Trainer(abc.ABC):
    """Interface for components that train models."""
    
    @abc.abstractmethod
    def train(
        self,
        model: Any,
        train_data: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]],
        val_data: Optional[Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]] = None,
        **kwargs
    ) -> Any:
        """
        Train the model on the provided data.
        
        Args:
            model: The model to train
            train_data: Training data, either as a DataFrame or (X, y) tuple
            val_data: Optional validation data
            **kwargs: Additional arguments for training
            
        Returns:
            The trained model
        """
        pass


class Evaluator(abc.ABC):
    """Interface for components that evaluate model performance."""
    
    @abc.abstractmethod
    def evaluate(
        self,
        model: Any,
        test_data: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]],
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            model: The trained model to evaluate
            test_data: Test data, either as a DataFrame or (X, y) tuple
            **kwargs: Additional arguments for evaluation
            
        Returns:
            Dictionary of metric names and their values
        """
        pass


class Predictor(abc.ABC):
    """Interface for components that make predictions using trained models."""
    
    @abc.abstractmethod
    def predict(
        self,
        model: Any,
        data: pd.DataFrame,
        **kwargs
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Generate predictions using the trained model.
        
        Args:
            model: The trained model to use for predictions
            data: Input data for prediction
            **kwargs: Additional arguments for prediction
            
        Returns:
            Model predictions in the form of numpy array, pandas Series, or DataFrame
        """
        pass


class PipelineComponent(abc.ABC):
    """Generic interface for any pipeline component."""
    
    def __init__(self, **kwargs):
        """
        Initialize the component with the provided parameters.
        
        Args:
            **kwargs: Component-specific parameters
        """
        self.params = kwargs
    
    @abc.abstractmethod
    def run(self, **kwargs) -> Any:
        """
        Execute the component's main functionality.
        
        Args:
            **kwargs: Runtime parameters
            
        Returns:
            Component-specific output
        """
        pass 