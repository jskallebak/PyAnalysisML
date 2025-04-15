#!/usr/bin/env python3
"""
Example script demonstrating how to use the PyAnalysisML Pipeline with different configuration methods.

This example shows:
1. Creating a pipeline from a dictionary configuration
2. Creating a pipeline from a YAML/JSON file
3. Creating a pipeline programmatically with a PipelineConfig object
4. Saving and loading a pipeline
5. Updating component configurations
"""

import logging
import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# Import the components to register them
from examples.components import register_all
register_all()  # Ensure components are registered

from pyanalysisml.pipeline.pipeline import Pipeline
from pyanalysisml.pipeline.config import PipelineConfig, ComponentConfig, ConfigManager


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_example_config_file():
    """Create an example configuration file for demonstration."""
    config = {
        "name": "example_pipeline",
        "description": "Example pipeline for Iris dataset classification",
        "output_dir": "outputs/iris",
        "log_level": "INFO",
        "data_loader": {
            "type": "IrisDataLoader",
            "params": {}
        },
        "feature_engineer": {
            "type": "StandardScaler",
            "params": {
                "with_mean": True,
                "with_std": True
            }
        },
        "data_splitter": {
            "type": "StandardSplitter",
            "params": {
                "test_size": 0.2,
                "random_state": 42
            }
        },
        "model_factory": {
            "type": "RandomForestFactory",
            "params": {
                "n_estimators": 100,
                "max_depth": 5,
                "random_state": 42
            }
        },
        "trainer": {
            "type": "StandardTrainer",
            "params": {
                "cv": 5,
                "scoring": "accuracy"
            }
        },
        "evaluator": {
            "type": "ClassificationEvaluator",
            "params": {
                "metrics": ["accuracy", "precision", "recall", "f1"]
            }
        },
        "predictor": {
            "type": "StandardPredictor",
            "params": {}
        }
    }
    
    # Create output directory if it doesn't exist
    output_dir = Path("examples/configs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON and YAML
    config_path_json = output_dir / "example_pipeline.json"
    config_path_yaml = output_dir / "example_pipeline.yaml"
    
    ConfigManager.save_json(config, str(config_path_json))
    ConfigManager.save_yaml(config, str(config_path_yaml))
    
    logger.info(f"Created example config files at {config_path_json} and {config_path_yaml}")
    
    return config, config_path_json, config_path_yaml


def example_1_dict_config():
    """Example: Create a pipeline from a dictionary configuration."""
    logger.info("EXAMPLE 1: Creating pipeline from dictionary configuration")
    
    # Create configuration dictionary
    config_dict = {
        "name": "dict_config_pipeline",
        "description": "Pipeline created from dictionary configuration",
        "output_dir": "outputs/dict_example",
        "log_level": "INFO",
        "data_loader": {
            "type": "IrisDataLoader",
            "params": {}
        },
        "feature_engineer": {
            "type": "StandardScaler",
            "params": {
                "with_mean": True,
                "with_std": True
            }
        },
        "data_splitter": {
            "type": "StandardSplitter",
            "params": {
                "test_size": 0.2,
                "random_state": 42
            }
        },
        "model_factory": {
            "type": "RandomForestFactory",
            "params": {
                "n_estimators": 100,
                "max_depth": 5,
                "random_state": 42
            }
        },
        "trainer": {
            "type": "StandardTrainer",
            "params": {}
        },
        "evaluator": {
            "type": "ClassificationEvaluator",
            "params": {
                "metrics": ["accuracy", "precision", "recall", "f1"]
            }
        },
        "predictor": {
            "type": "StandardPredictor",
            "params": {}
        }
    }
    
    # Create pipeline from dictionary
    pipeline = Pipeline(config_dict)
    
    # Print pipeline configuration
    logger.info(f"Pipeline name: {pipeline.config.name}")
    logger.info(f"Pipeline description: {pipeline.config.description}")
    
    return pipeline


def example_2_file_config(config_path):
    """Example: Create a pipeline from a configuration file."""
    logger.info(f"EXAMPLE 2: Creating pipeline from file configuration: {config_path}")
    
    # Create pipeline from file
    pipeline = Pipeline(config_path)
    
    # Print pipeline configuration
    logger.info(f"Pipeline name: {pipeline.config.name}")
    logger.info(f"Pipeline description: {pipeline.config.description}")
    
    return pipeline


def example_3_pipeline_config_object():
    """Example: Create a pipeline from a PipelineConfig object."""
    logger.info("EXAMPLE 3: Creating pipeline from PipelineConfig object")
    
    # Create PipelineConfig object programmatically
    config = PipelineConfig(
        name="programmatic_pipeline",
        description="Pipeline created programmatically",
        output_dir="outputs/programmatic_example",
        log_level="INFO",
        data_loader=ComponentConfig(
            type="IrisDataLoader",
            params={}
        ),
        feature_engineer=ComponentConfig(
            type="StandardScaler",
            params={
                "with_mean": True,
                "with_std": True
            }
        ),
        data_splitter=ComponentConfig(
            type="StandardSplitter",
            params={
                "test_size": 0.2,
                "random_state": 42
            }
        ),
        model_factory=ComponentConfig(
            type="RandomForestFactory",
            params={
                "n_estimators": 100,
                "max_depth": 5,
                "random_state": 42
            }
        ),
        trainer=ComponentConfig(
            type="StandardTrainer",
            params={}
        ),
        evaluator=ComponentConfig(
            type="ClassificationEvaluator",
            params={
                "metrics": ["accuracy", "precision", "recall", "f1"]
            }
        ),
        predictor=ComponentConfig(
            type="StandardPredictor",
            params={}
        )
    )
    
    # Create pipeline from PipelineConfig
    pipeline = Pipeline(config)
    
    # Print pipeline configuration
    logger.info(f"Pipeline name: {pipeline.config.name}")
    logger.info(f"Pipeline description: {pipeline.config.description}")
    
    return pipeline


def example_4_save_load_pipeline(pipeline):
    """Example: Save and load a pipeline."""
    logger.info("EXAMPLE 4: Saving and loading a pipeline")
    
    # Save pipeline
    save_path = Path("outputs/saved_pipeline/my_pipeline")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save(save_path)
    
    # Load pipeline
    loaded_pipeline = Pipeline.load(save_path)
    
    # Print loaded pipeline configuration
    logger.info(f"Loaded pipeline name: {loaded_pipeline.config.name}")
    logger.info(f"Loaded pipeline description: {loaded_pipeline.config.description}")
    
    return loaded_pipeline


def example_5_update_component_config(pipeline):
    """Example: Update a component configuration."""
    logger.info("EXAMPLE 5: Updating component configuration")
    
    # Print current model factory configuration
    current_config = pipeline.get_component_config("model_factory")
    logger.info(f"Current model factory configuration: {current_config}")
    
    # Update model factory configuration
    new_config = {
        "type": "RandomForestFactory",
        "params": {
            "n_estimators": 200,  # Changed from 100 to 200
            "max_depth": 10,      # Changed from 5 to 10
            "random_state": 42
        }
    }
    
    pipeline.update_config("model_factory", new_config)
    
    # Print updated model factory configuration
    updated_config = pipeline.get_component_config("model_factory")
    logger.info(f"Updated model factory configuration: {updated_config}")
    
    return pipeline


def example_6_run_pipeline(pipeline):
    """Example: Run a pipeline."""
    logger.info("EXAMPLE 6: Running a pipeline")
    
    # Run the pipeline
    results = pipeline.run()
    
    # Print results
    logger.info("Pipeline run completed")
    
    if "validation_metrics" in results:
        logger.info(f"Validation metrics: {results['validation_metrics']}")
    
    if "test_metrics" in results:
        logger.info(f"Test metrics: {results['test_metrics']}")
    
    return results


if __name__ == "__main__":
    # Create example configuration files
    _, config_path_json, config_path_yaml = create_example_config_file()
    
    # Run examples
    pipeline1 = example_1_dict_config()
    pipeline2 = example_2_file_config(config_path_json)
    pipeline3 = example_3_pipeline_config_object()
    pipeline4 = example_4_save_load_pipeline(pipeline1)
    pipeline5 = example_5_update_component_config(pipeline1)
    
    # Run the pipeline
    results = example_6_run_pipeline(pipeline1)
    
    logger.info("All examples completed successfully!") 