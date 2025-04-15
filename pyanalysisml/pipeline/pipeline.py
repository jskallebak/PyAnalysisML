"""
Core Pipeline implementation.

This module contains the Pipeline class that orchestrates the execution of a machine
learning pipeline by connecting various components together.
"""

import logging
import os
import pickle
import json
from typing import Any, Dict, List, Optional, Tuple, Union, Type
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .base import (
    DataLoader, 
    FeatureEngineer,
    ModelFactory,
    Trainer,
    Evaluator,
    Predictor,
    DataSplitter
)
from .registry import component_registry
from .config import PipelineConfig, ConfigLoader, ComponentConfig, ConfigManager


class Pipeline:
    """
    Main pipeline class that orchestrates the execution of ML workflow.
    
    The Pipeline connects various components like data loaders, feature engineers,
    model factories, trainers, evaluators, and predictors to form a complete
    machine learning workflow.
    
    Attributes:
        config (PipelineConfig): The pipeline configuration
        data_loader (Optional[DataLoader]): Data loader component
        feature_engineer (Optional[FeatureEngineer]): Feature engineering component
        data_splitter (Optional[DataSplitter]): Data splitting component
        model_factory (Optional[ModelFactory]): Model factory component
        trainer (Optional[Trainer]): Trainer component
        evaluator (Optional[Evaluator]): Evaluator component
        predictor (Optional[Predictor]): Predictor component
        logger (logging.Logger): Logger instance
        results (Dict[str, Any]): Results from the last pipeline run
    """
    
    def __init__(self, config: Union[Dict[str, Any], PipelineConfig, str, Path]):
        """
        Initialize the pipeline with a configuration.
        
        Args:
            config: Configuration for the pipeline. Can be a:
                - Dictionary containing the pipeline configuration
                - PipelineConfig instance
                - Path to a configuration file (YAML or JSON)
        """
        # Initialize components to None
        self.data_loader = None
        self.feature_engineer = None
        self.data_splitter = None
        self.model_factory = None
        self.trainer = None
        self.evaluator = None
        self.predictor = None
        self.results = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self._load_config(config)
        
        # Set up components based on config
        self._setup_components()
    
    def _load_config(self, config: Union[Dict[str, Any], PipelineConfig, str, Path]) -> None:
        """
        Load the configuration from the provided source.
        
        Args:
            config: Configuration source (dict, PipelineConfig, or file path)
            
        Raises:
            TypeError: If the config type is not supported
        """
        if isinstance(config, dict):
            self.config = ConfigLoader.load_from_dict(config)
        elif isinstance(config, PipelineConfig):
            self.config = config
        elif isinstance(config, (str, Path)):
            self.config = ConfigLoader.load_from_file(config)
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")
        
        # Configure logging based on config
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(level=log_level)
        self.logger.setLevel(log_level)
        
        self.logger.info(f"Initialized pipeline: {self.config.name}")
        if self.config.description:
            self.logger.info(f"Description: {self.config.description}")
    
    def _setup_components(self) -> None:
        """Set up pipeline components based on the configuration."""
        # Set up data loader
        self._setup_component("data_loader", self.config.data_loader)
        
        # Set up feature engineer
        self._setup_component("feature_engineer", self.config.feature_engineer)
        
        # Set up data splitter
        self._setup_component("data_splitter", self.config.data_splitter)
        
        # Set up model factory
        self._setup_component("model_factory", self.config.model_factory)
        
        # Set up trainer
        self._setup_component("trainer", self.config.trainer)
        
        # Set up evaluator
        self._setup_component("evaluator", self.config.evaluator)
        
        # Set up predictor
        self._setup_component("predictor", self.config.predictor)
        
        # Set up custom components
        for name, comp_config in self.config.custom_components.items():
            self._setup_component(name, comp_config)
    
    def _setup_component(self, name: str, component_config: ComponentConfig) -> None:
        """
        Set up an individual pipeline component.
        
        Args:
            name: Name of the component
            component_config: Configuration for the component
        """
        component_type = component_config.type
        component_params = component_config.params
        
        if component_type:
            component_cls = component_registry.get(component_type)
            if component_cls:
                try:
                    component_instance = component_cls(**component_params)
                    setattr(self, name, component_instance)
                    self.logger.info(f"Set {name}: {component_type}")
                except Exception as e:
                    self.logger.error(f"Error initializing {name} ({component_type}): {str(e)}")
            else:
                self.logger.error(f"{name.replace('_', ' ').title()} '{component_type}' not found in registry")
        else:
            self.logger.warning(f"No {name.replace('_', ' ')} specified in config")
    
    def set_data_loader(self, data_loader_config: Dict[str, Any]) -> "Pipeline":
        """
        Set the data loader component.
        
        Args:
            data_loader_config: Configuration for the data loader
            
        Returns:
            Self, for method chaining
        """
        component_name = data_loader_config.get("name")
        if component_name:
            data_loader_cls = component_registry.get(component_name)
            if data_loader_cls:
                self.data_loader = data_loader_cls(**data_loader_config.get("params", {}))
                self.logger.info(f"Set data loader: {component_name}")
            else:
                self.logger.error(f"Data loader '{component_name}' not found in registry")
        else:
            self.logger.warning("No data loader specified in config")
        return self
    
    def set_feature_engineer(self, feature_engineer_config: Dict[str, Any]) -> "Pipeline":
        """
        Set the feature engineer component.
        
        Args:
            feature_engineer_config: Configuration for the feature engineer
            
        Returns:
            Self, for method chaining
        """
        component_name = feature_engineer_config.get("name")
        if component_name:
            feature_engineer_cls = component_registry.get(component_name)
            if feature_engineer_cls:
                self.feature_engineer = feature_engineer_cls(**feature_engineer_config.get("params", {}))
                self.logger.info(f"Set feature engineer: {component_name}")
            else:
                self.logger.error(f"Feature engineer '{component_name}' not found in registry")
        else:
            self.logger.warning("No feature engineer specified in config")
        return self
    
    def set_data_splitter(self, data_splitter_config: Dict[str, Any]) -> "Pipeline":
        """
        Set the data splitter component.
        
        Args:
            data_splitter_config: Configuration for the data splitter
            
        Returns:
            Self, for method chaining
        """
        component_name = data_splitter_config.get("name")
        if component_name:
            data_splitter_cls = component_registry.get(component_name)
            if data_splitter_cls:
                self.data_splitter = data_splitter_cls(**data_splitter_config.get("params", {}))
                self.logger.info(f"Set data splitter: {component_name}")
            else:
                self.logger.error(f"Data splitter '{component_name}' not found in registry")
        else:
            self.logger.warning("No data splitter specified in config")
        return self
    
    def set_model_factory(self, model_factory_config: Dict[str, Any]) -> "Pipeline":
        """
        Set the model factory component.
        
        Args:
            model_factory_config: Configuration for the model factory
            
        Returns:
            Self, for method chaining
        """
        component_name = model_factory_config.get("name")
        if component_name:
            model_factory_cls = component_registry.get(component_name)
            if model_factory_cls:
                self.model_factory = model_factory_cls(**model_factory_config.get("params", {}))
                self.logger.info(f"Set model factory: {component_name}")
            else:
                self.logger.error(f"Model factory '{component_name}' not found in registry")
        else:
            self.logger.warning("No model factory specified in config")
        return self
    
    def set_trainer(self, trainer_config: Dict[str, Any]) -> "Pipeline":
        """
        Set the trainer component.
        
        Args:
            trainer_config: Configuration for the trainer
            
        Returns:
            Self, for method chaining
        """
        component_name = trainer_config.get("name")
        if component_name:
            trainer_cls = component_registry.get(component_name)
            if trainer_cls:
                self.trainer = trainer_cls(**trainer_config.get("params", {}))
                self.logger.info(f"Set trainer: {component_name}")
            else:
                self.logger.error(f"Trainer '{component_name}' not found in registry")
        else:
            self.logger.warning("No trainer specified in config")
        return self
    
    def set_evaluator(self, evaluator_config: Dict[str, Any]) -> "Pipeline":
        """
        Set the evaluator component.
        
        Args:
            evaluator_config: Configuration for the evaluator
            
        Returns:
            Self, for method chaining
        """
        component_name = evaluator_config.get("name")
        if component_name:
            evaluator_cls = component_registry.get(component_name)
            if evaluator_cls:
                self.evaluator = evaluator_cls(**evaluator_config.get("params", {}))
                self.logger.info(f"Set evaluator: {component_name}")
            else:
                self.logger.error(f"Evaluator '{component_name}' not found in registry")
        else:
            self.logger.warning("No evaluator specified in config")
        return self
    
    def set_predictor(self, predictor_config: Dict[str, Any]) -> "Pipeline":
        """
        Set the predictor component.
        
        Args:
            predictor_config: Configuration for the predictor
            
        Returns:
            Self, for method chaining
        """
        component_name = predictor_config.get("name")
        if component_name:
            predictor_cls = component_registry.get(component_name)
            if predictor_cls:
                self.predictor = predictor_cls(**predictor_config.get("params", {}))
                self.logger.info(f"Set predictor: {component_name}")
            else:
                self.logger.error(f"Predictor '{component_name}' not found in registry")
        else:
            self.logger.warning("No predictor specified in config")
        return self
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        This method executes the entire machine learning pipeline from data loading
        to prediction using the configured components.
        
        Args:
            **kwargs: Additional arguments to pass to the pipeline components
            
        Returns:
            Dictionary containing the pipeline results including model, predictions,
            and evaluation metrics
        """
        self.logger.info("Starting pipeline execution")
        self.results = {}
        
        # Create output directory if it doesn't exist
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        if not self.data_loader:
            raise ValueError("Data loader not set")
        
        self.logger.info("Loading data")
        data = self.data_loader.load(**kwargs.get("data_loader_args", {}))
        self.results["data"] = data
        
        # Engineer features
        if self.feature_engineer:
            self.logger.info("Engineering features")
            data = self.feature_engineer.transform(data, **kwargs.get("feature_engineer_args", {}))
            self.results["processed_data"] = data
        
        # Split data
        if not self.data_splitter:
            raise ValueError("Data splitter not set")
        
        self.logger.info("Splitting data")
        split_data = self.data_splitter.split(data, **kwargs.get("data_splitter_args", {}))
        self.results["split_data"] = split_data
        
        # Create model
        if not self.model_factory:
            raise ValueError("Model factory not set")
        
        self.logger.info("Creating model")
        model = self.model_factory.create(**kwargs.get("model_factory_args", {}))
        self.results["model"] = model
        
        # Train model
        if not self.trainer:
            raise ValueError("Trainer not set")
        
        self.logger.info("Training model")
        trained_model, training_metrics = self.trainer.train(
            model, 
            split_data["train"], 
            **kwargs.get("trainer_args", {})
        )
        self.results["trained_model"] = trained_model
        self.results["training_metrics"] = training_metrics
        
        # Evaluate model
        if self.evaluator:
            self.logger.info("Evaluating model")
            
            if "val" in split_data:
                val_metrics = self.evaluator.evaluate(
                    trained_model, 
                    split_data["val"], 
                    **kwargs.get("evaluator_args", {})
                )
                self.results["validation_metrics"] = val_metrics
            
            if "test" in split_data:
                test_metrics = self.evaluator.evaluate(
                    trained_model, 
                    split_data["test"], 
                    **kwargs.get("evaluator_args", {})
                )
                self.results["test_metrics"] = test_metrics
        
        # Make predictions
        if self.predictor and kwargs.get("predict", True):
            self.logger.info("Making predictions")
            predictions_data = kwargs.get("prediction_data", split_data.get("test", None))
            
            if predictions_data:
                predictions = self.predictor.predict(
                    trained_model, 
                    predictions_data, 
                    **kwargs.get("predictor_args", {})
                )
                self.results["predictions"] = predictions
        
        self.logger.info("Pipeline execution completed")
        
        # Save results if specified
        if kwargs.get("save_results", True):
            self.save_results(Path(self.config.output_dir) / f"{self.config.name}_results")
        
        return self.results
    
    def predict(self, data: Optional[pd.DataFrame] = None, **kwargs) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Generate predictions using the trained model.
        
        Args:
            data: Input data for prediction. If None, uses the data from the last run.
            **kwargs: Additional arguments for prediction
            
        Returns:
            Model predictions
            
        Raises:
            ValueError: If no trained model exists or no data is provided
        """
        if "trained_model" not in self.results:
            raise ValueError("No trained model available. Run the pipeline first.")
        
        if not self.predictor:
            raise ValueError("No predictor set")
        
        if data is None:
            if "test" in self.results.get("split_data", {}):
                data = self.results["split_data"]["test"]
            else:
                raise ValueError("No data provided for prediction and no test data available from previous run.")
        
        self.logger.info("Making predictions with trained model")
        predictions = self.predictor.predict(
            self.results["trained_model"],
            data,
            **kwargs
        )
        
        return predictions
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the pipeline (config and state) to disk.
        
        Args:
            path: Path where to save the pipeline (without extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_dict = {
            "name": self.config.name,
            "description": self.config.description,
            "output_dir": self.config.output_dir,
            "log_level": self.config.log_level,
        }
        
        # Add component configurations
        for component_name in ["data_loader", "feature_engineer", "data_splitter", 
                              "model_factory", "trainer", "evaluator", "predictor"]:
            component_config = getattr(self.config, component_name)
            config_dict[component_name] = {
                "type": component_config.type,
                "params": component_config.params
            }
        
        # Add custom components
        if self.config.custom_components:
            config_dict["custom_components"] = {}
            for name, component_config in self.config.custom_components.items():
                config_dict["custom_components"][name] = {
                    "type": component_config.type,
                    "params": component_config.params
                }
        
        # Save config
        ConfigManager.save_json(config_dict, f"{path}_config.json")
        
        # Save state if we have a trained model
        if "trained_model" in self.results:
            with open(f"{path}_state.pkl", "wb") as f:
                pickle.dump(self.results, f)
            
        self.logger.info(f"Pipeline saved to {path}")
    
    def save_results(self, path: Union[str, Path]) -> None:
        """
        Save pipeline results to disk.
        
        Args:
            path: Path where to save the results (without extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics = {}
        for metric_key in ["training_metrics", "validation_metrics", "test_metrics"]:
            if metric_key in self.results:
                metrics[metric_key] = self.results[metric_key]
        
        if metrics:
            with open(f"{path}_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
        
        # Save predictions if available
        if "predictions" in self.results:
            predictions = self.results["predictions"]
            if isinstance(predictions, (pd.DataFrame, pd.Series)):
                predictions.to_csv(f"{path}_predictions.csv")
            elif isinstance(predictions, np.ndarray):
                np.save(f"{path}_predictions.npy", predictions)
        
        # Save trained model if available
        if "trained_model" in self.results:
            try:
                joblib.dump(self.results["trained_model"], f"{path}_model.joblib")
            except Exception as e:
                self.logger.error(f"Error saving model: {str(e)}")
        
        self.logger.info(f"Pipeline results saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Pipeline":
        """
        Load a pipeline from disk.
        
        Args:
            path: Path to the saved pipeline (without extension)
            
        Returns:
            Loaded Pipeline instance
            
        Raises:
            FileNotFoundError: If the pipeline files don't exist
        """
        path = Path(path)
        config_path = f"{path}_config.json"
        state_path = f"{path}_state.pkl"
        
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Pipeline config not found at {config_path}")
        
        # Create pipeline from config
        pipeline = cls(config_path)
        
        # Load state if available
        if Path(state_path).exists():
            with open(state_path, "rb") as f:
                pipeline.results = pickle.load(f)
        
        return pipeline
    
    def get_component_config(self, component_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the configuration for a specific component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Component configuration or None if not found
        """
        if hasattr(self.config, component_name):
            component_config = getattr(self.config, component_name)
            return {
                "type": component_config.type,
                "params": component_config.params
            }
        elif component_name in self.config.custom_components:
            component_config = self.config.custom_components[component_name]
            return {
                "type": component_config.type,
                "params": component_config.params
            }
        else:
            return None
    
    def update_config(self, component_name: str, component_config: Dict[str, Any]) -> "Pipeline":
        """
        Update the configuration of a specific component.
        
        Args:
            component_name: Name of the component to update
            component_config: New configuration
            
        Returns:
            Self, for method chaining
            
        Raises:
            ValueError: If the component doesn't exist
        """
        component_type = component_config.get("type")
        component_params = component_config.get("params", {})
        
        if hasattr(self.config, component_name):
            # Update built-in component
            setattr(self.config, component_name, ComponentConfig(
                type=component_type,
                params=component_params
            ))
            
            # Re-initialize the component
            self._setup_component(component_name, getattr(self.config, component_name))
        elif component_name in self.config.custom_components:
            # Update custom component
            self.config.custom_components[component_name] = ComponentConfig(
                type=component_type,
                params=component_params
            )
            
            # Re-initialize the component
            self._setup_component(component_name, self.config.custom_components[component_name])
        else:
            raise ValueError(f"Component '{component_name}' not found in pipeline configuration")
        
        return self 