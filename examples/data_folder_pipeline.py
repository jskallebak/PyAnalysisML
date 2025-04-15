#!/usr/bin/env python3
"""
Example demonstrating how to create a PyAnalysisML pipeline that works with data
in the data folder, which might have been fetched using the CLI fetch command.

This example shows how to:
1. Create a data loader that reads from the data folder
2. Build a complete pipeline using code (not configuration)
3. Process the data and generate results
"""

import logging
import os
from pathlib import Path
import sys

import pandas as pd
import numpy as np

# Import the components to register them
from examples.components import register_all
register_all()

# Import the component classes directly
from examples.components import (
    StandardScaler,
    StandardSplitter,
    RandomForestFactory,
    StandardTrainer,
    ClassificationEvaluator,
    StandardPredictor
)

from pyanalysisml.pipeline.pipeline import Pipeline
from pyanalysisml.pipeline.registry import component_registry


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@component_registry.register
class DataFolderLoader:
    """
    Data loader that loads files from the data folder.
    
    This loader is designed to work with data that may have been fetched
    using the CLI fetch command, which stores files in the data folder.
    """
    
    def __init__(self, filename, format="csv", data_dir="data", **kwargs):
        """
        Initialize the data folder loader.
        
        Args:
            filename: Name of the file to load
            format: File format (csv, parquet, etc.)
            data_dir: Data directory (relative to project root)
            **kwargs: Additional parameters for reading the file
        """
        self.filename = filename
        self.format = format.lower()
        self.data_dir = data_dir
        self.params = kwargs
    
    def load(self, **kwargs):
        """
        Load the data file from the data folder.
        
        Args:
            **kwargs: Additional parameters for reading the file
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
        """
        # Combine params from initialization and runtime
        params = {**self.params, **kwargs}
        
        # Build the full path to the data file
        file_path = Path(self.data_dir) / self.filename
        
        logger.info(f"Loading data from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load data based on format
        if self.format == "csv":
            return pd.read_csv(file_path, **params)
        elif self.format == "parquet":
            return pd.read_parquet(file_path, **params)
        elif self.format == "json":
            return pd.read_json(file_path, **params)
        elif self.format == "excel" or self.format == "xlsx":
            return pd.read_excel(file_path, **params)
        else:
            raise ValueError(f"Unsupported file format: {self.format}")


def check_data_file(filename, data_dir="data"):
    """
    Check if a data file exists and print information about it.
    
    Args:
        filename: Name of the file to check
        data_dir: Data directory (relative to project root)
        
    Returns:
        True if the file exists, False otherwise
    """
    file_path = Path(data_dir) / filename
    
    if file_path.exists():
        file_size = file_path.stat().st_size / 1024  # Size in KB
        logger.info(f"Data file found: {file_path} ({file_size:.2f} KB)")
        return True
    else:
        logger.warning(f"Data file not found: {file_path}")
        logger.info("You may need to fetch the data using the CLI command:")
        logger.info(f"  python -m pyanalysisml.cli fetch --dataset {filename}")
        return False


def create_data_folder_pipeline(filename, target_column="target"):
    """
    Create a pipeline that works with data from the data folder.
    
    Args:
        filename: Name of the data file
        target_column: Name of the target column
        
    Returns:
        Configured Pipeline instance
    """
    logger.info(f"Creating pipeline for data file: {filename}")
    
    # Check if the data file exists
    if not check_data_file(filename):
        logger.error("Cannot create pipeline: data file not found")
        return None
    
    # Create components directly with specific parameters
    data_loader = DataFolderLoader(
        filename=filename,
        format=Path(filename).suffix[1:],  # Remove the leading dot
        data_dir="data"
    )
    
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
    
    # Create a configuration dictionary
    config = {
        "name": f"data_folder_pipeline_{Path(filename).stem}",
        "description": f"Pipeline for data from {filename}",
        "output_dir": f"outputs/data_folder/{Path(filename).stem}",
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
    
    logger.info("Pipeline configured successfully")
    
    return pipeline


def run_pipeline(pipeline):
    """
    Run the pipeline and display the results.
    
    Args:
        pipeline: The pipeline to run
        
    Returns:
        Results from the pipeline run
    """
    if pipeline is None:
        logger.error("Cannot run pipeline: pipeline is None")
        return None
    
    logger.info(f"Running pipeline: {pipeline.config.name}")
    
    # Create output directory
    output_dir = Path(pipeline.config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run the pipeline
        results = pipeline.run()
        
        # Print results
        logger.info("Pipeline execution completed")
        
        if "validation_metrics" in results:
            logger.info(f"Validation metrics: {results['validation_metrics']}")
        
        if "test_metrics" in results:
            logger.info(f"Test metrics: {results['test_metrics']}")
        
        # Save results
        pipeline.save_results(output_dir / "results")
        logger.info(f"Results saved to {output_dir / 'results'}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    """Main function to demonstrate the data folder pipeline."""
    # Check command line arguments
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # Default to iris.csv if no filename provided
        filename = "iris.csv"
        
        # Create dummy iris.csv if it doesn't exist
        if not Path("data/iris.csv").exists():
            logger.info("Creating dummy iris.csv file for demonstration")
            
            # Create data directory if it doesn't exist
            Path("data").mkdir(exist_ok=True)
            
            # Create dummy iris.csv
            from sklearn.datasets import load_iris
            iris = load_iris()
            df = pd.DataFrame(
                data=np.c_[iris['data'], iris['target']],
                columns=iris['feature_names'] + ['target']
            )
            df.to_csv("data/iris.csv", index=False)
    
    # Create and run the pipeline
    pipeline = create_data_folder_pipeline(filename)
    results = run_pipeline(pipeline)
    
    if results is not None:
        logger.info("Pipeline executed successfully")
    

if __name__ == "__main__":
    main() 