#!/usr/bin/env python3
"""
Create the project structure for PyAnalysisML.
"""
import os
import shutil

# Directories to create
directories = [
    "pyanalysisml",
    "pyanalysisml/data",
    "pyanalysisml/features",
    "pyanalysisml/models",
    "pyanalysisml/utils",
    "tests",
    "examples",
    "notebooks",
]

# Create directories
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

# Create __init__.py files
init_files = [
    "pyanalysisml/__init__.py",
    "pyanalysisml/data/__init__.py",
    "pyanalysisml/features/__init__.py",
    "pyanalysisml/models/__init__.py",
    "pyanalysisml/utils/__init__.py",
    "tests/__init__.py",
]

for init_file in init_files:
    with open(init_file, "w") as f:
        module_name = os.path.dirname(init_file).split("/")[-1]
        if module_name == "pyanalysisml":
            f.write(
                '"""\nPyAnalysisML - OHLC data analysis with TA-Lib and ML for cryptocurrency prediction.\n"""\n__version__ = "0.1.0"\n'
            )
        else:
            f.write(f'"""\n{module_name.capitalize()} module for PyAnalysisML.\n"""\n')
    print(f"Created file: {init_file}")

print("Project structure created successfully!")
