# PyAnalysisML - Instructions for Cursor AI

## Project Overview
PyAnalysisML is a Python library for cryptocurrency OHLC data analysis, feature engineering with pandas-ta, and price prediction using machine learning. The project aims to provide a comprehensive toolkit for cryptocurrency traders and data scientists to analyze market data and build predictive models.

## Project Goals
- Create robust and efficient data loading and preprocessing capabilities
- Implement comprehensive technical indicators using pandas-ta
- Develop flexible feature engineering pipelines
- Support multiple machine learning models for price prediction
- Provide clear visualization utilities for data and model performance
- Maintain high code quality with proper testing and documentation

## Code Organization
- `pyanalysisml/` - Main package directory
  - `data/` - Data loading, cleaning, and processing
  - `features/` - Feature engineering and technical indicators
  - `models/` - Model creation, training, and evaluation
  - `utils/` - Visualization, logging, and helper functions
  - `cli/` - Command-line interface functionality
- `examples/` - Example scripts demonstrating usage
- `tests/` - Unit and integration tests
- `notebooks/` - Jupyter notebooks for exploration and analysis

## Coding Standards
- Follow PEP 8 style guidelines with a line length of 100
- Use type hints for function parameters and return values
- Write comprehensive docstrings for all functions, classes, and modules
- Keep functions focused on a single responsibility
- Implement proper error handling and logging
- Write tests for all new functionality

## Development Priorities
1. Fix existing bugs and inconsistencies in the codebase
2. Complete missing modules (CLI, training, visualization)
3. Improve test coverage
4. Enhance feature engineering capabilities
5. Add support for more ML algorithms
6. Optimize performance of data processing pipelines

## Guidelines for AI Assistance
- When suggesting code changes, prioritize correctness and clarity over cleverness
- Match the existing code style and patterns used in the project
- Provide explanations for non-trivial solutions
- Consider performance implications, especially for data processing functions
- Suggest improvements to existing code where appropriate
- Help identify potential bugs or issues in the codebase

## Dependencies
The project relies on these key libraries:
- pandas and pandas-ta for data manipulation and technical indicators
- scikit-learn, XGBoost, and LightGBM for machine learning
- matplotlib and seaborn for visualization
- python-binance for data acquisition

When suggesting new functionality, prefer using these existing dependencies rather than introducing new ones. 