#!/usr/bin/env python3
"""
Master script to create the PyAnalysisML project.

This script will run all the component scripts to create the full project structure.
"""
import os
import subprocess
import sys

# List of scripts to run
scripts = [
    "create_project_structure.py",
    "create_base_files.py",
    "create_binance_client.py",
    "create_technical_indicators.py",
    "create_model_factory.py",
]

print("Creating PyAnalysisML project...")

# Make scripts executable
for script in scripts:
    os.chmod(script, 0o755)

# Run each script
for script in scripts:
    print(f"\nRunning {script}...")
    result = subprocess.run([f"./{script}"], shell=True)
    if result.returncode != 0:
        print(f"Error running {script}. Exiting.")
        sys.exit(1)

print("\nPyAnalysisML project created successfully!")
print("You can now install the package with:")
print("\npip install -e .")
print("\nThe project is set up with the core components. You can find:")
print("- Binance client for data acquisition in pyanalysisml/data/binance_client.py")
print("- Technical indicators in pyanalysisml/features/technical_indicators.py")
print("- Model factory in pyanalysisml/models/model_factory.py")
print("\nNext steps:")
print("1. Install TA-Lib following the instructions in README.md")
print("2. Set up your Binance API credentials if needed")
print("3. Start building your ML models for OHLC data analysis!")
