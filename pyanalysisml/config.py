"""
Configuration settings for PyAnalysisML.

This module provides configuration settings for the entire package, including:
- API credentials
- Data paths
- Default parameters for data fetching, technical indicators, and models
- Logging configuration

Most settings can be overridden through environment variables.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any

# Project directories
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = os.environ.get("PYANALYSISML_DATA_DIR", os.path.join(ROOT_DIR, "data"))
MODELS_DIR = os.environ.get("PYANALYSISML_MODELS_DIR", os.path.join(ROOT_DIR, "models"))
LOGS_DIR = os.environ.get("PYANALYSISML_LOGS_DIR", os.path.join(ROOT_DIR, "logs"))

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# API credentials for Binance
# Default to environment variables for security
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
BINANCE_TESTNET = os.environ.get("BINANCE_TESTNET", "False").lower() == "true"

# Data fetching default parameters
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1d"
DEFAULT_LOOKBACK = "1 year ago"

# Technical indicators to calculate by default
DEFAULT_INDICATORS = [
    "sma",    # Simple Moving Average
    "ema",    # Exponential Moving Average
    "rsi",    # Relative Strength Index
    "macd",   # Moving Average Convergence Divergence
    "bbands", # Bollinger Bands
    "atr",    # Average True Range
    "stoch",  # Stochastic Oscillator
    "obv",    # On Balance Volume
    "roc"     # Rate of Change
]

# Default model parameters
DEFAULT_MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "n_jobs": -1
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
}

# Default model to use
DEFAULT_MODEL = "random_forest"

# Feature engineering parameters
PREDICTION_HORIZONS = [1, 3, 5, 7, 14, 21, 30]  # Days ahead to predict
DEFAULT_PREDICTION_HORIZON = 5
DEFAULT_TEST_SIZE = 0.2

# Logging configuration
LOG_LEVEL = os.environ.get("PYANALYSISML_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(LOGS_DIR, "pyanalysisml.log")

# Create a logger configuration dictionary
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": LOG_FORMAT
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": LOG_LEVEL,
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": LOG_LEVEL,
            "formatter": "standard",
            "filename": LOG_FILE,
            "mode": "a"
        }
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file"],
            "level": LOG_LEVEL,
            "propagate": True
        },
        "pyanalysisml": {
            "handlers": ["console", "file"],
            "level": LOG_LEVEL,
            "propagate": False
        }
    }
} 