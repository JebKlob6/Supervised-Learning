from __future__ import annotations

import os
import random
import time
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score)
from sklearn.model_selection import (train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (StandardScaler)

# Set random seeds for reproducibility
RANDOM_STATE = 42343289
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


def create_output_directories():
    """Create necessary output directories."""
    directories = ["figures", "figures/nncp2", "results", "data_analysis"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def load_and_prepare():
    """Load and prepare the company bankruptcy dataset with specific features."""
    print("\n▶ Loading and Preparing Data")
    print("-" * 45)

    # Load the dataset
    data = pd.read_csv('data/company_bankruptcy_data.csv')

    # Clean column names: strip whitespace and convert to lowercase
    data.columns = [col.strip().lower() for col in data.columns]

    # Define the selected features based on the summary
    selected_features = [
        # Profitability
        'roa(a) before interest and % after tax',
        'realized sales gross margin',

        # Liquidity
        'current ratio',
        'cash/current liability',

        # Leverage
        'debt ratio %',
        'interest coverage ratio (interest expense to ebit)',

        # Efficiency
        'total asset turnover',
        'accounts receivable turnover',

        # Growth
        'after-tax net profit growth rate',
        'total asset growth rate',

        # Cash Flow
        'cash flow to sales'
    ]

    # Target variable
    target = 'bankrupt?'

    # Verify all features exist in the dataset
    missing_features = [f for f in selected_features if f not in data.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataset: {missing_features}")

    # Prepare X and y
    X = data[selected_features]
    y = data[target]

    # Handle missing values
    print("\nHandling missing values...")
    missing_before = X.isnull().sum().sum()
    if missing_before > 0:
        print(f"Found {missing_before} missing values")
        # Fill missing values with median for each feature
        X = X.fillna(X.median())
        print("Missing values filled with median values")
    else:
        print("No missing values found")

    # Scale the features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=selected_features)

    # Split the data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Print dataset information
    print("\nDataset Information:")
    print(f"Total samples: {len(data)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of features: {len(selected_features)}")
    print("\nClass distribution:")
    print(y.value_counts(normalize=True).round(3))

    # Save feature names for later use
    feature_names = selected_features

    return X_train, X_test, y_train, y_test, None, feature_names


def run_experiment(X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   activation: str,
                   train_sizes: List[float] = [0.2, 0.6, 1.0],
                   max_iter: int = 50) -> Dict:
    """Run experiment with different training set sizes."""
    results = {
        'train_errors': [],
        'test_errors': [],
        'train_times': [],
        'train_losses': []
    }

    for train_size in train_sizes:
        # Calculate actual number of samples
        n_samples = int(len(X_train) * train_size)
        X_train_subset = X_train[:n_samples]
        y_train_subset = y_train[:n_samples]

        # Initialize model
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation=activation,
            max_iter=max_iter,
            random_state=RANDOM_STATE,
            verbose=True
        )

        # Training
        start_time = time.time()
        model.fit(X_train_subset, y_train_subset)
        train_time = time.time() - start_time

        # Evaluate
        train_pred = model.predict(X_train_subset)
        test_pred = model.predict(X_test)

        train_error = 1 - accuracy_score(y_train_subset, train_pred)
        test_error = 1 - accuracy_score(y_test, test_pred)

        # Store results
        results['train_errors'].append(train_error)
        results['test_errors'].append(test_error)
        results['train_times'].append(train_time)
        results['train_losses'].append(model.loss_curve_)

        print(f"\nResults for {train_size * 100}% training data:")
        print(f"Training error: {train_error:.4f}")
        print(f"Test error: {test_error:.4f}")
        print(f"Training time: {train_time:.2f} seconds")

    return results


def plot_learning_curves(results_relu: Dict, results_sigmoid: Dict, train_sizes: List[float]):
    """Plot learning curves for both activation functions."""
    plt.figure(figsize=(12, 8))

    # Plot training and test errors
    plt.subplot(2, 2, 1)
    plt.plot(train_sizes, results_relu['train_errors'], 'o-', label='ReLU (Train)')
    plt.plot(train_sizes, results_relu['test_errors'], 'o--', label='ReLU (Test)')
    plt.plot(train_sizes, results_sigmoid['train_errors'], 's-', label='Sigmoid (Train)')
    plt.plot(train_sizes, results_sigmoid['test_errors'], 's--', label='Sigmoid (Test)')
    plt.xlabel('Training Set Size')
    plt.ylabel('Error Rate')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)

    # Plot training times
    plt.subplot(2, 2, 2)
    plt.plot(train_sizes, results_relu['train_times'], 'o-', label='ReLU')
    plt.plot(train_sizes, results_sigmoid['train_times'], 's-', label='Sigmoid')
    plt.xlabel('Training Set Size')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time vs Dataset Size')
    plt.legend()
    plt.grid(True)

    # Plot loss curves for ReLU
    plt.subplot(2, 2, 3)
    for i, loss_curve in enumerate(results_relu['train_losses']):
        plt.plot(loss_curve, label=f'{train_sizes[i] * 100}% data')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curves (ReLU)')
    plt.legend()
    plt.grid(True)

    # Plot loss curves for Sigmoid
    plt.subplot(2, 2, 4)
    for i, loss_curve in enumerate(results_sigmoid['train_losses']):
        plt.plot(loss_curve, label=f'{train_sizes[i] * 100}% data')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curves (Sigmoid)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('figures/nncp2/nn_learning_curves.png')
    plt.close()


def analyze_learning_rates(X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_test: np.ndarray,
                           y_test: np.ndarray) -> None:
    """Analyze the impact of different learning rates on model performance."""
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    activations = ['relu', 'logistic']

    results = {
        'relu': {'train_errors': [], 'test_errors': []},
        'logistic': {'train_errors': [], 'test_errors': []}
    }

    for lr in learning_rates:
        print(f"\nAnalyzing learning rate: {lr}")
        for activation in activations:
            # Initialize model
            model = MLPClassifier(
                hidden_layer_sizes=(64, 64),
                activation=activation,
                max_iter=50,
                batch_size=64,
                solver='adam',
                learning_rate_init=lr,
                random_state=RANDOM_STATE,
                verbose=True
            )

            # Train model
            model.fit(X_train, y_train)

            # Evaluate
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_error = 1 - accuracy_score(y_train, train_pred)
            test_error = 1 - accuracy_score(y_test, test_pred)

            # Store results
            results[activation]['train_errors'].append(train_error)
            results[activation]['test_errors'].append(test_error)

            print(f"{activation.capitalize()} - Train error: {train_error:.4f}, Test error: {test_error:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogx(learning_rates, results['relu']['train_errors'], 'o-', label='ReLU (Train)')
    plt.semilogx(learning_rates, results['relu']['test_errors'], 'o--', label='ReLU (Test)')
    plt.semilogx(learning_rates, results['logistic']['train_errors'], 's-', label='Sigmoid (Train)')
    plt.semilogx(learning_rates, results['logistic']['test_errors'], 's--', label='Sigmoid (Test)')

    plt.xlabel('Learning Rate')
    plt.ylabel('Error Rate')
    plt.title('NN Error vs Learning Rate for ReLU & Sigmoid')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('figures/nncp2/nn_learning_rate_analysis.png')
    plt.close()


def main():
    """Run neural network experiments with different activation functions."""
    # Load and prepare data
    X_train, X_test, y_train, y_test, le_target, feature_names = load_and_prepare()

    # Run learning rate analysis
    print("\nRunning learning rate analysis...")
    analyze_learning_rates(X_train, y_train, X_test, y_test)
    
    # Run experiments with different activation functions
    train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

    # Run experiments for both activation functions
    results_relu = run_experiment(X_train, y_train, X_test, y_test, 'relu', train_sizes)
    results_sigmoid = run_experiment(X_train, y_train, X_test, y_test, 'logistic', train_sizes)

    # Plot and save results
    plot_learning_curves(results_relu, results_sigmoid, train_sizes)

    print("\nExperiment completed! Check figures/nncp2/ for results.")


if __name__ == "__main__":
    main()
