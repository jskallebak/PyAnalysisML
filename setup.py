from setuptools import find_packages, setup

setup(
    name="pyanalysisml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "python-binance>=1.0.16",
        "pandas-ta>=0.3.14b0",  # Using pandas-ta instead of TA-Lib
        "scikit-learn>=1.0.2",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.2",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "jupyter>=1.0.0",
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
