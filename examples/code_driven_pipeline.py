#!/usr/bin/env python3
"""
Example demonstrating how to create a PyAnalysisML pipeline directly with code
instead of using the configuration system.

This approach gives you direct control over component instances and allows for tighter
integration with your application code.
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

# Import the component classes directly
from examples.components import (
    IrisDataLoader,
    StandardScaler,
    StandardSplitter,
    RandomForestFactory,
    StandardTrainer,
    ClassificationEvaluator,
    StandardPredictor
)

from pyanalysisml.pipeline.pipeline import Pipeline


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_code_driven_pipeline():
    """Create a pipeline by directly instantiating and configuring components."""
    logger.info("Creating a pipeline using the code-driven approach")
    
    # Create components directly with specific parameters
    data_loader = IrisDataLoader()
    
    feature_engineer = StandardScaler(
        with_mean=True,
        with_std=True
    )
    
    data_splitter = StandardSplitter(
        test_size=0.2,
        val_size=0.15,
        random_state=42
    )
    
    model_factory = RandomForestFactory(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    
    trainer = StandardTrainer(
        cv=5,
        scoring="accuracy"
    )
    
    evaluator = ClassificationEvaluator(
        metrics=["accuracy", "precision", "recall", "f1"],
        average="weighted"
    )
    
    predictor = StandardPredictor()
    
    # Create a minimal configuration dictionary
    # This is still needed for basic pipeline metadata
    config = {
        "name": "code_driven_pipeline",
        "description": "Pipeline created directly with code",
        "output_dir": "outputs/code_pipeline",
        "log_level": "INFO"
    }
    
    # Create the pipeline
    pipeline = Pipeline(config)
    
    # Set components manually
    pipeline.data_loader = data_loader
    pipeline.feature_engineer = feature_engineer
    pipeline.data_splitter = data_splitter
    pipeline.model_factory = model_factory
    pipeline.trainer = trainer
    pipeline.evaluator = evaluator
    pipeline.predictor = predictor
    
    logger.info("Pipeline components configured manually")
    
    return pipeline


def run_pipeline(pipeline):
    """Run the pipeline and display the results."""
    logger.info("Running the code-driven pipeline")
    
    # Create output directory if it doesn't exist
    output_dir = Path("outputs/code_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the pipeline
    results = pipeline.run()
    
    # Print results
    logger.info("Pipeline execution completed")
    
    if "validation_metrics" in results:
        logger.info(f"Validation metrics: {results['validation_metrics']}")
    
    if "test_metrics" in results:
        logger.info(f"Test metrics: {results['test_metrics']}")
    
    # You have direct access to pipeline components for inspection or modification
    # For example, you can inspect the feature engineer's scaler attributes
    if hasattr(pipeline.feature_engineer, "scaler"):
        logger.info(f"Feature means: {pipeline.feature_engineer.scaler.mean_}")
    
    # Or access the model's feature importances
    if "trained_model" in results and hasattr(results["trained_model"], "feature_importances_"):
        importances = results["trained_model"].feature_importances_
        feature_names = pipeline.data_loader.load().columns[:-1]  # Exclude target column
        
        importance_data = list(zip(feature_names, importances))
        importance_data.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("Feature importances:")
        for name, importance in importance_data:
            logger.info(f"  {name}: {importance:.4f}")
    
    return results


def dynamic_parameter_example():
    """
    Example showing how to dynamically configure pipeline components
    based on runtime conditions or other factors.
    """
    logger.info("Creating a pipeline with dynamically determined parameters")
    
    # Determine parameters dynamically
    data = IrisDataLoader().load()
    
    # Calculate number of estimators based on data size
    n_samples = len(data)
    n_estimators = min(10 * int(np.sqrt(n_samples)), 200)
    
    # Determine class distribution for class weighting
    class_counts = data['target'].value_counts()
    if class_counts.nunique() > 1:
        class_weight = "balanced"
    else:
        class_weight = None
    
    # Create model factory with dynamic parameters
    model_factory = RandomForestFactory(
        n_estimators=n_estimators,
        max_depth=5,
        random_state=42,
        class_weight=class_weight
    )
    
    logger.info(f"Dynamically determined n_estimators={n_estimators}, class_weight={class_weight}")
    
    # Create other components
    pipeline = create_code_driven_pipeline()
    
    # Override the model factory with our dynamically configured one
    pipeline.model_factory = model_factory
    
    return pipeline


if __name__ == "__main__":
    # Run the basic code-driven pipeline example
    logger.info("EXAMPLE 1: Basic code-driven pipeline")
    pipeline = create_code_driven_pipeline()
    results = run_pipeline(pipeline)
    
    # Run the dynamic parameter example
    logger.info("\nEXAMPLE 2: Pipeline with dynamically determined parameters")
    dynamic_pipeline = dynamic_parameter_example()
    dynamic_results = run_pipeline(dynamic_pipeline)
    
    logger.info("All examples completed successfully!") 