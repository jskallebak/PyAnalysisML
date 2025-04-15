"""
Logging utilities for PyAnalysisML.

This module provides functions to set up logging for the package.
"""

import logging
import logging.config
from typing import Optional

from pyanalysisml.config import LOGGING_CONFIG, LOG_LEVEL, LOG_FORMAT

def setup_logging(config: Optional[dict] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        config: Logging configuration dictionary. If None, use the default from config.py.
    """
    if config is None:
        config = LOGGING_CONFIG
    
    logging.config.dictConfig(config)
    logging.info("Logging configured successfully")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Name of the logger
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level if not already set
    if not logger.level:
        logger.setLevel(LOG_LEVEL)
    
    # Add a console handler if no handlers exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger 