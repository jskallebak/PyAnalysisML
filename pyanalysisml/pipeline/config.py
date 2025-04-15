"""
Configuration system for ML pipelines.

This module provides functionality for loading, validating, and accessing
pipeline configurations.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


logger = logging.getLogger(__name__)


@dataclass
class ComponentConfig:
    """
    Configuration for a pipeline component.
    
    Attributes:
        type: The type/name of the component (must be registered)
        params: Parameters to pass to the component during initialization
    """
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """
    Configuration for a complete ML pipeline.
    
    Attributes:
        name: Name of the pipeline
        description: Optional description of the pipeline
        data_loader: Configuration for the data loader component
        feature_engineer: Configuration for the feature engineering component
        data_splitter: Configuration for the data splitting component
        model_factory: Configuration for the model factory component
        trainer: Configuration for the trainer component
        evaluator: Configuration for the evaluator component
        predictor: Configuration for the predictor component
        output_dir: Directory where outputs should be saved
        log_level: Logging level for the pipeline
        custom_components: Optional dictionary of additional component configurations
    """
    name: str
    description: Optional[str] = None
    data_loader: ComponentConfig = field(default_factory=lambda: ComponentConfig(type="DefaultDataLoader"))
    feature_engineer: ComponentConfig = field(default_factory=lambda: ComponentConfig(type="DefaultFeatureEngineer"))
    data_splitter: ComponentConfig = field(default_factory=lambda: ComponentConfig(type="DefaultDataSplitter"))
    model_factory: ComponentConfig = field(default_factory=lambda: ComponentConfig(type="DefaultModelFactory"))
    trainer: ComponentConfig = field(default_factory=lambda: ComponentConfig(type="DefaultTrainer"))
    evaluator: ComponentConfig = field(default_factory=lambda: ComponentConfig(type="DefaultEvaluator"))
    predictor: ComponentConfig = field(default_factory=lambda: ComponentConfig(type="DefaultPredictor"))
    output_dir: str = "outputs"
    log_level: str = "INFO"
    custom_components: Dict[str, ComponentConfig] = field(default_factory=dict)


class ConfigLoader:
    """Utility class for loading pipeline configurations from various sources."""
    
    @staticmethod
    def load_from_file(config_path: Union[str, Path]) -> PipelineConfig:
        """
        Load a pipeline configuration from a file.
        
        Args:
            config_path: Path to the configuration file (JSON or YAML)
            
        Returns:
            A PipelineConfig object
            
        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the file format is not supported
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        suffix = config_path.suffix.lower()
        config_dict = {}
        
        if suffix == ".json":
            with open(config_path, "r") as f:
                config_dict = json.load(f)
        elif suffix in (".yaml", ".yml"):
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {suffix}")
        
        return ConfigLoader._dict_to_config(config_dict)
    
    @staticmethod
    def load_from_dict(config_dict: Dict[str, Any]) -> PipelineConfig:
        """
        Load a pipeline configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            A PipelineConfig object
        """
        return ConfigLoader._dict_to_config(config_dict)
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> PipelineConfig:
        """
        Convert a configuration dictionary to a PipelineConfig object.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            A PipelineConfig object
        """
        # Process component configs
        component_keys = [
            "data_loader", "feature_engineer", "data_splitter", 
            "model_factory", "trainer", "evaluator", "predictor"
        ]
        
        for key in component_keys:
            if key in config_dict and isinstance(config_dict[key], dict):
                component_dict = config_dict[key]
                config_dict[key] = ComponentConfig(
                    type=component_dict.get("type", f"Default{key.title().replace('_', '')}"),
                    params=component_dict.get("params", {})
                )
        
        # Process custom components
        if "custom_components" in config_dict and isinstance(config_dict["custom_components"], dict):
            custom_components = {}
            for name, component_dict in config_dict["custom_components"].items():
                if isinstance(component_dict, dict):
                    custom_components[name] = ComponentConfig(
                        type=component_dict.get("type", name),
                        params=component_dict.get("params", {})
                    )
            config_dict["custom_components"] = custom_components
        
        return PipelineConfig(**config_dict)


class ConfigManager:
    """
    Configuration manager for pipeline configurations.
    
    Handles loading, validating, and saving pipeline configurations.
    """
    
    @staticmethod
    def load_yaml(path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            path: Path to the YAML configuration file
            
        Returns:
            Configuration dictionary
        """
        logger.info(f"Loading configuration from: {path}")
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def load_json(path: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Args:
            path: Path to the JSON configuration file
            
        Returns:
            Configuration dictionary
        """
        logger.info(f"Loading configuration from: {path}")
        with open(path, 'r') as f:
            config = json.load(f)
        
        return config
    
    @staticmethod
    def save_yaml(config: Dict[str, Any], path: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            config: Configuration dictionary
            path: Path to save the YAML configuration
        """
        logger.info(f"Saving configuration to: {path}")
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    @staticmethod
    def save_json(config: Dict[str, Any], path: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            config: Configuration dictionary
            path: Path to save the JSON configuration
        """
        logger.info(f"Saving configuration to: {path}")
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
    
    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        """
        Load configuration from a file, auto-detecting the format.
        
        Args:
            path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        _, ext = os.path.splitext(path)
        
        if ext.lower() in ['.yaml', '.yml']:
            return ConfigManager.load_yaml(path)
        elif ext.lower() == '.json':
            return ConfigManager.load_json(path)
        else:
            raise ValueError(f"Unsupported configuration format: {ext}")
    
    @staticmethod
    def save(config: Dict[str, Any], path: str) -> None:
        """
        Save configuration to a file, using the format based on the file extension.
        
        Args:
            config: Configuration dictionary
            path: Path to save the configuration
        """
        _, ext = os.path.splitext(path)
        
        if ext.lower() in ['.yaml', '.yml']:
            ConfigManager.save_yaml(config, path)
        elif ext.lower() == '.json':
            ConfigManager.save_json(config, path)
        else:
            raise ValueError(f"Unsupported configuration format: {ext}")
    
    @staticmethod
    def validate(config: Dict[str, Any]) -> List[str]:
        """
        Validate the configuration structure and required fields.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check if components section exists
        if "components" not in config:
            errors.append("Missing 'components' section in configuration")
            return errors
        
        components = config.get("components", {})
        
        # Check for required components
        required_components = ["data_loader", "feature_engineer", "model_factory", "trainer"]
        for component in required_components:
            if component not in components:
                errors.append(f"Missing required component: {component}")
        
        # Check if target column is specified
        if "target_column" not in config:
            errors.append("Missing 'target_column' in configuration")
        
        # Check individual component configurations
        for component_name, component_config in components.items():
            if not isinstance(component_config, dict):
                errors.append(f"Component '{component_name}' configuration must be a dictionary")
                continue
            
            # Check if component type is specified
            if "type" not in component_config:
                errors.append(f"Missing 'type' for component: {component_name}")
            
            # Check for model_type in model_factory
            if component_name == "model_factory" and "model_type" not in component_config:
                errors.append("Missing 'model_type' in model_factory configuration")
        
        return errors
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Generate a default pipeline configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "pipeline_name": "default_pipeline",
            "target_column": "close",
            "components": {
                "data_loader": {
                    "type": "BinanceDataLoader",
                    "params": {
                        "symbol": "BTCUSDT",
                        "interval": "1d",
                        "start_str": "1 year ago",
                        "chunk_size": 30,
                        "max_workers": 8
                    }
                },
                "feature_engineer": {
                    "type": "TechnicalFeatureEngineer",
                    "params": {
                        "indicators": ["sma", "ema", "rsi", "macd", "bbands"],
                        "selection_method": "correlation",
                        "min_correlation": 0.05
                    }
                },
                "model_factory": {
                    "type": "StandardModelFactory",
                    "model_type": "random_forest",
                    "params": {
                        "n_estimators": 100,
                        "max_depth": 10,
                        "random_state": 42
                    }
                },
                "trainer": {
                    "type": "StandardTrainer",
                    "params": {
                        "test_size": 0.2,
                        "prediction_horizon": 1,
                        "shuffle": False
                    }
                },
                "evaluator": {
                    "type": "RegressionEvaluator",
                    "params": {
                        "metrics": ["rmse", "mae", "r2"],
                        "evaluate_train": True
                    }
                },
                "predictor": {
                    "type": "StandardPredictor",
                    "params": {}
                }
            }
        } 