# PyAnalysisML Pipeline Configuration Tutorial

This tutorial explains what the pipeline configuration system is, why it's important for machine learning workflows, and how to use it effectively in your projects.

## What is the Pipeline Configuration System?

The pipeline configuration system in PyAnalysisML is a framework that allows you to:

1. **Define ML pipelines** through configuration rather than code
2. **Modularize** your machine learning workflows into reusable components
3. **Configure** components with different parameters without changing code
4. **Save and load** pipeline configurations for reproducibility
5. **Share** configurations between team members or projects

At its core, the system separates the *what* (pipeline structure) from the *how* (component implementations), allowing for more flexible and maintainable machine learning workflows.

## Why Use a Configuration System?

### 1. Separation of Concerns

By separating configuration from implementation:
- **Data scientists** can focus on creating effective ML components (models, feature engineering)
- **ML engineers** can focus on infrastructure and system architecture
- **End users** can run pre-configured pipelines without understanding the code

### 2. Reproducibility

Configuration files ensure that:
- Experiments can be **precisely reproduced**
- The exact pipeline structure and parameters are **documented**
- Different runs can be **compared** by examining configuration differences

### 3. Flexibility

The configuration system enables:
- **Swapping components** without changing code (e.g., trying different models)
- **Parameter tuning** by simply changing configuration values
- **Progressive refinement** of your ML workflow by iterating on configurations

### 4. Productivity

Using configurations increases productivity by:
- **Reducing boilerplate code** for pipeline setup
- **Minimizing errors** in pipeline construction
- **Simplifying experimentation** with different components and parameters

## How the Configuration System Works

### Core Components

1. **Pipeline**: The main orchestrator that executes the ML workflow
2. **ComponentConfig**: Configuration for individual pipeline components
3. **PipelineConfig**: Overall pipeline configuration
4. **ConfigLoader**: Loads configurations from various sources
5. **ComponentRegistry**: Registers components for use in pipelines

### Configuration Structure

Configurations typically define:
- **Metadata**: Pipeline name, description, outputs location
- **Component specifications**: What components to use and their parameters
- **Execution settings**: Logging level, custom settings

## Tutorial: Building Your First Configured Pipeline

Let's walk through creating, configuring, and running a complete ML pipeline using the configuration system.

### Step 1: Define Component Implementations

First, we need components that implement the required interfaces. Here's a simple data loader:

```python
from pyanalysisml.pipeline.registry import component_registry
from pyanalysisml.pipeline.base import DataLoader

@component_registry.register
class CSVDataLoader(DataLoader):
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        
    def load(self, **kwargs):
        import pandas as pd
        return pd.read_csv(self.file_path)
```

### Step 2: Create a Configuration

Now we can create a pipeline configuration in JSON:

```json
{
  "name": "simple_classifier",
  "description": "A simple classification pipeline",
  "output_dir": "outputs/simple_classifier",
  "log_level": "INFO",
  "data_loader": {
    "type": "CSVDataLoader",
    "params": {
      "file_path": "data/iris.csv"
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
    "params": {}
  },
  "evaluator": {
    "type": "ClassificationEvaluator",
    "params": {
      "metrics": ["accuracy", "f1"]
    }
  },
  "predictor": {
    "type": "StandardPredictor",
    "params": {}
  }
}
```

Save this as `configs/simple_classifier.json`.

### Step 3: Create and Run the Pipeline

Now we can create and run the pipeline from our configuration:

```python
from pyanalysisml.pipeline.pipeline import Pipeline

# Create pipeline from configuration file
pipeline = Pipeline("configs/simple_classifier.json")

# Run the pipeline
results = pipeline.run()

# Access results
print(f"Test accuracy: {results['test_metrics']['accuracy']:.4f}")
```

### Step 4: Experiment with Different Configurations

One of the key benefits is the ability to quickly experiment with different configurations:

```python
# Create a new configuration by modifying an existing one
from pyanalysisml.pipeline.config import ConfigManager

# Load the existing config
config = ConfigManager.load("configs/simple_classifier.json")

# Modify the model configuration
config["model_factory"]["params"]["n_estimators"] = 200
config["model_factory"]["params"]["max_depth"] = 10

# Save as a new configuration
ConfigManager.save(config, "configs/simple_classifier_large.json")

# Create and run the new pipeline
pipeline_large = Pipeline("configs/simple_classifier_large.json")
results_large = pipeline_large.run()

# Compare results
print(f"Original model accuracy: {results['test_metrics']['accuracy']:.4f}")
print(f"Large model accuracy: {results_large['test_metrics']['accuracy']:.4f}")
```

## Code-Driven Pipeline Approach

While the configuration system offers many advantages, you can also create and use pipelines directly with code if you prefer more programmatic control or are prototyping rapidly.

### Creating a Pipeline with Pure Code

```python
from pyanalysisml.pipeline.pipeline import Pipeline
from examples.components import IrisDataLoader, StandardScaler, StandardSplitter
from examples.components import RandomForestFactory, StandardTrainer, ClassificationEvaluator, StandardPredictor

# Create components directly
data_loader = IrisDataLoader()
feature_engineer = StandardScaler(with_mean=True, with_std=True)
data_splitter = StandardSplitter(test_size=0.2, random_state=42)
model_factory = RandomForestFactory(n_estimators=100, max_depth=5, random_state=42)
trainer = StandardTrainer(cv=5, scoring="accuracy")
evaluator = ClassificationEvaluator(metrics=["accuracy", "precision", "recall", "f1"])
predictor = StandardPredictor()

# Create a minimal configuration dictionary
config = {
    "name": "code_driven_pipeline",
    "description": "Pipeline created with code",
    "output_dir": "outputs/code_pipeline"
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

# Run the pipeline
results = pipeline.run()
```

This approach gives you direct control over the component instances and allows you to integrate more tightly with the rest of your code. It's particularly useful when:

- You're rapidly prototyping and don't want to maintain separate configuration files
- You need to dynamically generate component parameters based on runtime conditions
- You want to directly access component properties or methods before/after the pipeline runs
- You're integrating the pipeline into a larger application with its own configuration system

You can still take advantage of the pipeline's orchestration capabilities while having the flexibility of direct code manipulation.

## Advanced Usage

### Programmatic Configuration

You can also create configurations programmatically:

```python
from pyanalysisml.pipeline.config import PipelineConfig, ComponentConfig
from pyanalysisml.pipeline.pipeline import Pipeline

# Create configuration objects
config = PipelineConfig(
    name="programmatic_pipeline",
    description="Pipeline created programmatically",
    data_loader=ComponentConfig(
        type="CSVDataLoader",
        params={"file_path": "data/iris.csv"}
    ),
    # Add other components...
)

# Create pipeline
pipeline = Pipeline(config)
```

### Saving and Loading Pipelines

Save your pipeline for later use:

```python
# Save the pipeline configuration and state
pipeline.save("outputs/saved_pipelines/my_pipeline")

# Later, load it back
loaded_pipeline = Pipeline.load("outputs/saved_pipelines/my_pipeline")

# Make predictions with the loaded pipeline
new_predictions = loaded_pipeline.predict(new_data)
```

### Custom Components

You can also create and register custom components:

```python
from pyanalysisml.pipeline.registry import component_registry
from pyanalysisml.pipeline.base import FeatureEngineer

@component_registry.register(name="MyCustomFeatureEngineer")
class CustomFeatureEngineer(FeatureEngineer):
    def __init__(self, custom_param=1.0, **kwargs):
        self.custom_param = custom_param
        
    def transform(self, data, **kwargs):
        # Custom feature engineering logic
        return transformed_data
```

Then use it in your configuration:

```json
"feature_engineer": {
  "type": "MyCustomFeatureEngineer",
  "params": {
    "custom_param": 2.5
  }
}
```

## Best Practices

1. **Organize configurations** in a dedicated directory
2. **Version control** your configurations
3. **Document parameters** for each component
4. **Use descriptive names** for pipelines and components
5. **Validate configurations** before running pipelines
6. **Create templates** for common pipeline patterns
7. **Choose the right approach** - configuration-driven for reproducibility and sharing, code-driven for rapid prototyping and tight integration

## Conclusion

The pipeline configuration system in PyAnalysisML enables you to build modular, reproducible, and maintainable machine learning workflows. By separating the configuration from the implementation, you gain flexibility, improve team collaboration, and make your ML processes more robust.

Start by creating simple configurations and gradually move to more complex pipelines as you become familiar with the system. The ability to experiment rapidly with different components and parameters will significantly accelerate your ML development process. 

Remember that you can always use the code-driven approach when it better suits your needs, giving you the best of both worlds - structured orchestration with the flexibility of direct code control. 