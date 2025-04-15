#!/usr/bin/env python3
"""
Create the model factory module for PyAnalysisML.
"""

content = """\"\"\"
Model factory module for creating various ML models.
\"\"\"
import logging
from typing import Dict, Any, Optional, Union

# ML models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)

def create_model(
    model_name: str,
    model_params: Optional[Dict[str, Any]] = None
) -> Any:
    \"\"\"
    Create a machine learning model.
    
    Args:
        model_name: Name of the model to create
        model_params: Parameters for the model (optional)
        
    Returns:
        Initialized model instance
    \"\"\"
    if model_params is None:
        model_params = {}
    
    # Dictionary of available models
    models = {
        # Linear models
        "linear_regression": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "elastic_net": ElasticNet,
        
        # Tree-based models
        "decision_tree": DecisionTreeRegressor,
        "random_forest": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "xgboost": xgb.XGBRegressor,
        "lightgbm": lgb.LGBMRegressor,
        
        # Other models
        "svr": SVR,
        "knn": KNeighborsRegressor,
        "mlp": MLPRegressor,
    }
    
    # Check if the model is supported
    if model_name not in models:
        supported_models = list(models.keys())
        raise ValueError(
            f"Model '{model_name}' not supported. "
            f"Supported models: {supported_models}"
        )
    
    # Create and return the model
    model_class = models[model_name]
    model = model_class(**model_params)
    
    logger.info(f"Created {model_name} model with parameters: {model_params}")
    return model

def get_default_params(model_name: str) -> Dict[str, Any]:
    \"\"\"
    Get default parameters for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of default parameters
    \"\"\"
    default_params = {
        # Linear models
        "linear_regression": {},
        "ridge": {"alpha": 1.0},
        "lasso": {"alpha": 0.1},
        "elastic_net": {"alpha": 0.1, "l1_ratio": 0.5},
        
        # Tree-based models
        "decision_tree": {"max_depth": 5, "random_state": 42},
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 42
        },
        "xgboost": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "random_state": 42
        },
        "lightgbm": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "num_leaves": 31,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "random_state": 42
        },
        
        # Other models
        "svr": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1},
        "knn": {"n_neighbors": 5, "weights": "uniform"},
        "mlp": {
            "hidden_layer_sizes": (100,),
            "activation": "relu",
            "alpha": 0.0001,
            "learning_rate": "adaptive",
            "max_iter": 200,
            "random_state": 42
        }
    }
    
    if model_name not in default_params:
        logger.warning(f"No default parameters defined for model '{model_name}'")
        return {}
    
    return default_params[model_name]
"""

# Write to file
with open("pyanalysisml/models/model_factory.py", "w") as f:
    f.write(content)

print("Created file: pyanalysisml/models/model_factory.py")
