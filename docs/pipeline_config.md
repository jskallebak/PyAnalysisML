# Pipeline Configuration System

PyAnalysisML provides a flexible configuration system for setting up and managing machine learning pipelines. This document explains how to use the configuration system to create, configure, save, and load pipelines.

## Overview

The configuration system allows you to:

1. Define pipeline components and their parameters
2. Create pipelines from various configuration sources (dictionaries, files, objects)
3. Save and load pipeline configurations
4. Update component configurations at runtime
5. Execute pipelines with the configured components

## Configuration Structure

A pipeline configuration consists of:

- **Pipeline metadata**: Name, description, output directory, logging level
- **Component configurations**: Configuration for each component in the pipeline
- **Custom components**: Optional additional components

Example configuration structure:

```json
{
  "name": "my_pipeline",
  "description": "Example pipeline for classification",
  "output_dir": "outputs/my_pipeline",
  "log_level": "INFO",
  "data_loader": {
    "type": "CSVDataLoader",
    "params": {
      "file_path": "data/train.csv"
    }
  },
  "feature_engineer": {
    "type": "StandardScaler",
    "params": {
      "with_mean": true,
      "with_std": true
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
      "max_depth": 5
    }
  },
  "trainer": {
    "type": "StandardTrainer",
    "params": {
      "cv": 5
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
  },
  "custom_components": {
    "feature_selector": {
      "type": "CorrelationSelector",
      "params": {
        "threshold": 0.7
      }
    }
  }
}
```

## Creating a Pipeline

### Method 1: From a Dictionary

You can create a pipeline directly from a Python dictionary:

```python
from pyanalysisml.pipeline.pipeline import Pipeline

config_dict = {
    "name": "dict_pipeline",
    "description": "Pipeline from dictionary",
    "data_loader": {
        "type": "CSVDataLoader",
        "params": {"file_path": "data/train.csv"}
    },
    # ... other components
}

pipeline = Pipeline(config_dict)
```

### Method 2: From a Configuration File

You can load pipeline configuration from JSON or YAML files:

```python
from pyanalysisml.pipeline.pipeline import Pipeline

# From JSON
pipeline = Pipeline("configs/my_pipeline.json")

# From YAML
pipeline = Pipeline("configs/my_pipeline.yaml")
```

### Method 3: From a PipelineConfig Object

For programmatic creation, you can use the PipelineConfig class:

```python
from pyanalysisml.pipeline.pipeline import Pipeline
from pyanalysisml.pipeline.config import PipelineConfig, ComponentConfig

config = PipelineConfig(
    name="programmatic_pipeline",
    description="Pipeline created programmatically",
    data_loader=ComponentConfig(
        type="CSVDataLoader",
        params={"file_path": "data/train.csv"}
    ),
    # ... other components
)

pipeline = Pipeline(config)
```

## Running the Pipeline

Once configured, you can run the pipeline:

```python
# Run the complete pipeline
results = pipeline.run()

# Access the results
trained_model = results["trained_model"]
test_metrics = results["test_metrics"]
predictions = results["predictions"]
```

You can also pass additional arguments to the pipeline components:

```python
results = pipeline.run(
    data_loader_args={"encoding": "utf-8"},
    trainer_args={"early_stopping": True},
    evaluator_args={"detailed_report": True}
)
```

## Saving and Loading Pipelines

### Saving a Pipeline

```python
# Save pipeline configuration and state
pipeline.save("outputs/saved_pipelines/my_pipeline")
```

This creates:
- `my_pipeline_config.json`: The pipeline configuration
- `my_pipeline_state.pkl`: The pipeline state (if it has been run)

### Loading a Pipeline

```python
# Load a saved pipeline
loaded_pipeline = Pipeline.load("outputs/saved_pipelines/my_pipeline")
```

## Saving Results

Results can be saved separately:

```python
# Save just the results from a pipeline run
pipeline.save_results("outputs/results/my_pipeline_results")
```

This creates:
- `my_pipeline_results_metrics.json`: Evaluation metrics
- `my_pipeline_results_predictions.csv`: Predictions (if available)
- `my_pipeline_results_model.joblib`: Trained model (if available)

## Updating Component Configurations

You can update component configurations at runtime:

```python
# Get current configuration
current_config = pipeline.get_component_config("model_factory")

# Update configuration
new_config = {
    "type": "RandomForestFactory",
    "params": {
        "n_estimators": 200,
        "max_depth": 10
    }
}
pipeline.update_config("model_factory", new_config)
```

## Component Registry

All components used in the pipeline must be registered with the component registry:

```python
from pyanalysisml.pipeline.registry import component_registry

@component_registry.register
class MyCustomDataLoader:
    # Component implementation
    pass

# Or register manually
component_registry.register(MyCustomDataLoader, name="MyDataLoader")
```

## Complete Example

```python
from pyanalysisml.pipeline.pipeline import Pipeline
from pyanalysisml.pipeline.config import ConfigManager

# Create and save a configuration
config = {
    "name": "example_pipeline",
    "description": "Example pipeline for classification",
    "data_loader": {
        "type": "CSVDataLoader",
        "params": {"file_path": "data/train.csv"}
    },
    # ... other components
}

ConfigManager.save_json(config, "configs/example_pipeline.json")

# Create pipeline from saved config
pipeline = Pipeline("configs/example_pipeline.json")

# Run the pipeline
results = pipeline.run()

# Save the pipeline and results
pipeline.save("outputs/pipelines/example_pipeline")
```

## Configuration File Types

The configuration system supports both JSON and YAML formats. Choose the format that best suits your needs:

- **JSON**: More strict, better for programmatic generation
- **YAML**: More human-readable, better for manual editing

## Best Practices

1. **Component Registration**: Always register components before creating a pipeline
2. **Configuration Version Control**: Keep your configurations under version control
3. **Parameterization**: Parameterize configurations for different environments/datasets
4. **Validation**: Validate configurations before using them in production
5. **Documentation**: Document the purpose and parameters of each component 