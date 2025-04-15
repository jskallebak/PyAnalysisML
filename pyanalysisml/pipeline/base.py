"""
Base interfaces for pipeline components.

This module defines the interfaces for all pipeline components including
data loaders, feature engineers, model factories, trainers, evaluators, and predictors.
These interfaces define the contract that concrete implementations must adhere to.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


class DataLoader(ABC):
    """
    Interface for data loading components.
    
    Data loaders are responsible for loading data from various sources like
    files, databases, APIs, etc. and returning them in a standardized format.
    """
    
    @abstractmethod
    def load(self, **kwargs) -> Dict[str, Any]:
        """
        Load data from a source.
        
        Args:
            **kwargs: Additional arguments specific to the concrete implementation
            
        Returns:
            Dictionary containing the loaded data with standardized keys
        """
        pass
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate that the loaded data meets the requirements.
        
        Args:
            data: The data to validate
            
        Returns:
            True if the data is valid, False otherwise
        """
        return True


class FeatureEngineer(ABC):
    """
    Interface for feature engineering components.
    
    Feature engineers transform raw data into features that better represent
    the underlying problem, improving model performance.
    """
    
    @abstractmethod
    def transform(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Transform the input data by engineering features.
        
        Args:
            data: The input data to transform
            **kwargs: Additional arguments specific to the concrete implementation
            
        Returns:
            Dictionary containing the transformed data with features
        """
        pass
    
    def fit(self, data: Dict[str, Any], **kwargs) -> "FeatureEngineer":
        """
        Fit the feature engineer to the data.
        
        This method is optional and only needed for stateful feature engineers
        that need to learn parameters from the data.
        
        Args:
            data: The input data to fit on
            **kwargs: Additional arguments specific to the concrete implementation
            
        Returns:
            Self, to allow method chaining
        """
        return self


class ModelFactory(ABC):
    """
    Interface for model factory components.
    
    Model factories are responsible for creating machine learning models
    based on the specified configuration.
    """
    
    @abstractmethod
    def create(self, **kwargs) -> Any:
        """
        Create a new model instance.
        
        Args:
            **kwargs: Additional arguments specific to the concrete implementation
            
        Returns:
            A new model instance
        """
        pass


class Trainer(ABC):
    """
    Interface for model training components.
    
    Trainers are responsible for training machine learning models
    on the provided data.
    """
    
    @abstractmethod
    def train(self, model: Any, data: Dict[str, Any], **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a model on the provided data.
        
        Args:
            model: The model to train
            data: The data to train on
            **kwargs: Additional arguments specific to the concrete implementation
            
        Returns:
            Tuple containing the trained model and training metrics
        """
        pass


class Evaluator(ABC):
    """
    Interface for model evaluation components.
    
    Evaluators assess the performance of trained models on
    validation or test data.
    """
    
    @abstractmethod
    def evaluate(self, model: Any, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Evaluate a model on the provided data.
        
        Args:
            model: The model to evaluate
            data: The data to evaluate on
            **kwargs: Additional arguments specific to the concrete implementation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass


class Predictor(ABC):
    """
    Interface for prediction components.
    
    Predictors use trained models to make predictions on new data.
    """
    
    @abstractmethod
    def predict(self, model: Any, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Make predictions using a model on the provided data.
        
        Args:
            model: The model to use for predictions
            data: The data to make predictions on
            **kwargs: Additional arguments specific to the concrete implementation
            
        Returns:
            Dictionary containing the predictions
        """
        pass


class DataSplitter(ABC):
    """
    Interface for data splitting components.
    
    Data splitters divide data into training, validation, and test sets.
    """
    
    @abstractmethod
    def split(self, data: Dict[str, Any], **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Split the data into training, validation, and test sets.
        
        Args:
            data: The data to split
            **kwargs: Additional arguments specific to the concrete implementation
            
        Returns:
            Dictionary containing the split data with keys 'train', 'val', and 'test'
        """
        pass 