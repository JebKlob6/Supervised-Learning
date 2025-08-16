from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score)
from sklearn.model_selection import (train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (LabelEncoder, StandardScaler, OneHotEncoder)

# Set random seeds for reproducibility
RANDOM_STATE = 42343289
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def create_output_directories():
    """Create necessary output directories."""
    directories = ["figures", "figures/nncp1", "results", "data_analysis"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_and_prepare():
    """Load cancer dataset with improved preprocessing strategy."""
    csv_path = Path("data/global_cancer_patients_2015_2024.csv")
    target_col = "Cancer_Stage"

    print(f"\n▶ Loading cancer dataset from {csv_path} …")
    df = pd.read_csv(csv_path).dropna()
    print(f"  Final shape after dropna: {df.shape}\n")

    # Print column info to understand the data structure
    print("Dataset columns and types:")
    print("-" * 60)
    for col in df.columns:
        print(f"  {col:<25}: {str(df[col].dtype):<10} (unique: {df[col].nunique():>4})")
        if df[col].nunique() <= 20:  # Show unique values for categorical-like columns
            unique_vals = sorted(df[col].unique()) if pd.api.types.is_numeric_dtype(df[col]) else list(df[col].unique())
            print(f"    Values: {unique_vals}")
    print()

    # Define columns to drop (identifiers and non-predictive features)
    drop_cols = [
        "Patient_ID",  # Identifier
        "Gender",  # Fixed: added missing comma
        "Country_Region",
        "Year",
    ]

    # Only drop columns that exist in the dataset
    drop_cols = [col for col in drop_cols if col in df.columns]
    print(f"Dropping columns: {drop_cols}")
    df = df.drop(columns=drop_cols)

    # Create binary target:  (0, I) vs Has (I, II, III, IV)
    df['Cancer_Stage_Binary'] = df[target_col].apply(
        lambda x: 'Early' if x in ['Stage 0', "Stage I", "Stage II"] else 'Advanced'
    )

    # Separate features and target
    X_raw = df.drop(columns=[target_col, 'Cancer_Stage_Binary'])
    y_raw = df['Cancer_Stage_Binary']

    # Check class distribution
    print("\nClass balance (Binary):")
    class_counts = y_raw.value_counts(normalize=True).sort_index()
    print(class_counts)
    print(f"Class imbalance ratio: {class_counts.max() / class_counts.min():.2f}")
    print()

    # Label encode target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y_raw)

    # Train/test split with stratification
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y_encoded,
        test_size=0.20,
        stratify=y_encoded,
        random_state=RANDOM_STATE,
    )

    # Define column types based on the dataset
    numeric_cols = [
        "Genetic_Risk", "Air_Pollution", "Alcohol_Use",
        "Smoking", "Obesity_Level", "Target_Severity_Score", "Age",
        "Survival_Years"
    ]

    categorical_cols = []  # Add any categorical columns here if they exist

    # Keep only columns that exist in the data
    numeric_cols = [col for col in numeric_cols if col in X_raw.columns]
    categorical_cols = [col for col in categorical_cols if col in X_raw.columns]

    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print()

    # Validate and convert data types
    for col in numeric_cols:
        if X_train_raw[col].dtype == 'object':
            print(f"WARNING: {col} is object type but should be numeric!")
            X_train_raw[col] = pd.to_numeric(X_train_raw[col], errors='coerce')
            X_test_raw[col] = pd.to_numeric(X_test_raw[col], errors='coerce')

    # Build preprocessing pipeline
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols))

    preprocess = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    # Fit preprocessing on training data
    X_train_processed = preprocess.fit_transform(X_train_raw)
    X_test_processed = preprocess.transform(X_test_raw)

    # Get feature names
    feature_names = []
    if numeric_cols:
        feature_names.extend(numeric_cols)
    if categorical_cols:
        try:
            cat_feature_names = list(preprocess.named_transformers_['cat'].get_feature_names_out(categorical_cols))
            feature_names.extend(cat_feature_names)
        except Exception as e:
            print(f"Warning: Could not get categorical feature names: {e}")

    print(f"\n▶ After preprocessing:")
    print(f"  X_train shape: {X_train_processed.shape}")
    print(f"  X_test  shape: {X_test_processed.shape}")
    print(f"  Number of features: {len(feature_names)}")
    print(f"  Number of classes: {len(le_target.classes_)} → {le_target.classes_}")
    print()

    return X_train_processed, X_test_processed, y_train, y_test, le_target, feature_names

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
        
        print(f"\nResults for {train_size*100}% training data:")
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
        plt.plot(loss_curve, label=f'{train_sizes[i]*100}% data')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curves (ReLU)')
    plt.legend()
    plt.grid(True)
    
    # Plot loss curves for Sigmoid
    plt.subplot(2, 2, 4)
    for i, loss_curve in enumerate(results_sigmoid['train_losses']):
        plt.plot(loss_curve, label=f'{train_sizes[i]*100}% data')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curves (Sigmoid)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('figures/nncp1/nn_learning_curves.png')
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
    plt.savefig('figures/nncp1/nn_learning_rate_analysis.png')
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

    print("\nExperiment completed! Check figures/nncp1/ for results.")

if __name__ == "__main__":
    main()