#!/usr/bin/env python3
"""
Create the base files for PyAnalysisML.
"""

# README.md
readme_content = """# PyAnalysisML

A Python library for cryptocurrency OHLC data analysis, feature engineering with TA-Lib, and price prediction using machine learning.

## Features

- Fetch historical OHLC data from Binance
- Calculate technical indicators using TA-Lib
- Engineer custom features for improved prediction
- Train and evaluate machine learning models
- Visualize results and predictions

## Installation

### Prerequisites

- Python 3.8 or higher
- TA-Lib (see below for installation instructions)

TA-Lib is a technical analysis library with C/C++ dependencies. Before installing this package, you need to install TA-Lib:

#### On Ubuntu/Debian:
```bash
sudo apt install ta-lib
```

#### On macOS:
```bash
brew install ta-lib
```

#### On Windows:
Download and install the pre-built binary from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib).

### Package Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pyanalysisml.git
cd pyanalysisml

# Install the package in development mode
pip install -e .
```

## Usage

```python
from pyanalysisml.data.binance_client import BinanceClient
from pyanalysisml.features.technical_indicators import add_indicators
from pyanalysisml.models.model_factory import create_model
from pyanalysisml.models.train import train_model
from pyanalysisml.utils.visualization import plot_predictions

# Initialize Binance client and fetch data
client = BinanceClient()
df = client.get_historical_klines("BTCUSDT", "1d", "1 year ago")

# Add technical indicators
df = add_indicators(df)

# Train model
X, y, X_test, y_test = prepare_data(df)
model = create_model("random_forest")
model = train_model(model, X, y)

# Evaluate and visualize
predictions = model.predict(X_test)
plot_predictions(df.iloc[-len(y_test):], y_test, predictions)
```

Check the `examples/` directory for more detailed usage examples.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
"""

# pyproject.toml
pyproject_content = """[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pyanalysisml"
version = "0.1.0"
description = "OHLC data analysis with TA-Lib and ML for cryptocurrency prediction"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
dependencies = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "python-binance>=1.0.16",
    "ta-lib>=0.4.0",
    "scikit-learn>=1.0.2",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.2",
    "jupyter>=1.0.0",
    "pytest>=7.0.0",
]

[project.optional-dependencies]
dev = [
    "black",
    "ruff",
    "mypy",
    "pytest",
    "pytest-cov",
]

[project.scripts]
pyanalysisml = "pyanalysisml.cli:main"

[tool.ruff]
line-length = 100
target-version = "py38"

[tool.black]
line-length = 100
target-version = ["py38"]
"""

# setup.py
setup_py_content = """from setuptools import setup, find_packages

setup(
    name="pyanalysisml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "python-binance>=1.0.16",
        "ta-lib>=0.4.0",
        "scikit-learn>=1.0.2",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.2",
    ],
    extras_require={
        "dev": [
            "black",
            "ruff",
            "mypy",
            "pytest",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            "pyanalysisml=pyanalysisml.cli:main",
        ],
    },
    python_requires=">=3.8",
)
"""

# Write files
with open("README.md", "w") as f:
    f.write(readme_content)
print("Created file: README.md")

with open("pyproject.toml", "w") as f:
    f.write(pyproject_content)
print("Created file: pyproject.toml")

with open("setup.py", "w") as f:
    f.write(setup_py_content)
print("Created file: setup.py")

# Create a custom .gitignore
gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
dist/
build/
*.egg-info/

# Virtual environments
venv/
env/
.env/
.venv/

# IDE files
.idea/
.vscode/
*.sublime-*

# Jupyter notebooks
.ipynb_checkpoints/

# Test coverage
htmlcov/
.coverage
.coverage.*

# Logs
*.log

# Local configuration
.env
"""

with open(".gitignore", "w") as f:
    f.write(gitignore_content)
print("Created file: .gitignore")

print("Base files created successfully!")
