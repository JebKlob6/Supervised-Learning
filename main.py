"""
CS7641 Supervised Learning Project - Main Execution Script
Runs comprehensive analysis on Global Cancer Patients and Company Bankruptcy datasets
Using KNN, SVM, and Neural Networks

Author: [Your Name]
Date: Summer 2025
"""

import os
import importlib.util
from pathlib import Path

def run_experiment_file(file_path):
    """Run a single experiment file."""
    print(f"\n{'='*80}")
    print(f"Running experiment: {file_path}")
    print(f"{'='*80}\n")
    
    # Import the module
    spec = importlib.util.spec_from_file_location("experiment", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Run the main function if it exists
    if hasattr(module, 'main'):
        module.main()
    else:
        print(f"Warning: No main() function found in {file_path}")

def main():
    """Run all experiment files."""
    # Create necessary directories
    Path('figures').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    Path('data_analysis').mkdir(exist_ok=True)
    
    # Get all experiment files
    experiment_files = sorted([f for f in os.listdir('.') if f.endswith('_CP1.py') or f.endswith('_CP2.py')])
    
    # Run each experiment
    for file in experiment_files:
        run_experiment_file(file)
    
    print("\nAll experiments completed!")

if __name__ == "__main__":
    main()
