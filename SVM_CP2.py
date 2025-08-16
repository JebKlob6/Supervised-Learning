"""
Enhanced SVM Analysis for Company Bankruptcy Dataset
===================================================
Comprehensive analysis of SVM performance across different parameters with detailed documentation.

Key Features:
1. Multiple kernel analysis (Linear, Polynomial)
2. Parameter optimization (C, degree)
3. Learning and validation curves
4. Comprehensive visualization and documentation
"""
from __future__ import annotations

import os
import random
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import (train_test_split,
                                     StratifiedKFold, validation_curve, learning_curve)
from sklearn.preprocessing import (StandardScaler)
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

RANDOM_STATE = 42343289

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def create_output_directories():
    """Create necessary output directories."""
    directories = ["figures", "figures/svmcp2", "results", "data_analysis"]
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
    X = data[selected_features].copy()  # Create a copy to avoid SettingWithCopyWarning
    y = data[target].copy()

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
        X_scaled, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
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

def svm_learning_curve(X_train, X_test, y_train, y_test, kernel, C, gamma=None, degree=None):
    """Plot learning curve for SVM with given kernel and parameters."""
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Create the appropriate estimator based on kernel
    if kernel == 'linear':
        estimator = LinearSVC(C=C, class_weight='balanced', random_state=RANDOM_STATE, max_iter=250000, dual=False)
    else:
        # Ensure degree is set for polynomial kernel
        degree_param = degree if kernel == 'poly' else 3
        estimator = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree_param, 
                       random_state=RANDOM_STATE, class_weight='balanced')
    
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator,
        X_train, y_train,
        train_sizes=train_sizes,
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', label='Training score')
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes_abs, test_mean, 's-', label='Cross-validation score')
    plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title(f'SVM Learning Curves (kernel={kernel}, C={C})')
    plt.legend(loc='best')
    plt.grid(True)
    os.makedirs("figures/svmcp2", exist_ok=True)
    plt.savefig(f'figures/svmcp2/learning_curve_{kernel}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Also plot wall times vs. fraction
    plt.figure(figsize=(10, 6))
    wall_times = []
    for size in train_sizes_abs:
        start_time = time.time()
        if kernel == 'linear':
            model = LinearSVC(C=C, class_weight='balanced', random_state=RANDOM_STATE, max_iter=250000, dual=False)
        else:
            # Ensure degree is set for polynomial kernel
            degree_param = degree if kernel == 'poly' else 3
            model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree_param, 
                       random_state=RANDOM_STATE, class_weight='balanced')
        model.fit(X_train[:int(size)], y_train[:int(size)])
        wall_times.append(time.time() - start_time)
    
    plt.plot(train_sizes_abs, wall_times, 'o-', color='purple')
    plt.xlabel('Training Examples')
    plt.ylabel('Wall‐Clock Time (s)')
    plt.title(f'SVM Training Time ({kernel} kernel)')
    plt.grid(True)
    os.makedirs("figures/svmcp2", exist_ok=True)
    plt.savefig(f'figures/svmcp2/time_curve_{kernel}.png', dpi=300, bbox_inches='tight')
    plt.close()

def svm_validation_curve(X_train, y_train, kernel, param_name, param_range, C=1.0, gamma='scale', degree=3):
    """Plot validation curve for SVM, sweeping over 'param_name' in param_range."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # Build estimator with correct fixed parameters
    if kernel == 'linear':
        base_estimator = LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, max_iter=250000, dual=False)
    elif kernel == 'rbf':
        base_estimator = SVC(kernel='rbf', class_weight='balanced', random_state=RANDOM_STATE, gamma=gamma, C=C)
    elif kernel == 'poly':
        base_estimator = SVC(kernel='poly', class_weight='balanced', random_state=RANDOM_STATE, degree=degree, gamma=gamma, C=C)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    train_scores_raw, test_scores_raw = validation_curve(
        estimator   = base_estimator,
        X           = X_train,
        y           = y_train,
        param_name  = param_name,
        param_range = param_range,
        cv          = skf,
        scoring     = 'accuracy',
        n_jobs      = -1
    )

    train_scores = np.mean(train_scores_raw, axis=1)
    test_scores  = np.mean(test_scores_raw , axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(param_range, 1 - train_scores, 'o-', color='crimson', label='Train Error')
    plt.plot(param_range, 1 - test_scores , 'o-', color='goldenrod', label='CV  Error')
    plt.xlabel(param_name)
    plt.ylabel("Error Rate")
    plt.title(f"SVM Validation Curve ({kernel} kernel, {param_name})")
    if param_name in ['C', 'gamma']:
        plt.xscale('log')
    plt.legend()
    plt.grid(alpha=0.3)
    os.makedirs("figures/svmcp2", exist_ok=True)
    plt.savefig(f"figures/svmcp2/validation_curve_{kernel}_{param_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run the comprehensive SVM analysis pipeline for company bankruptcy prediction."""
    print("🔬 Enhanced SVM Company Bankruptcy Prediction Analysis")
    print("=" * 60)
    print("Comprehensive testing of SVM parameters:")
    print("  • Kernels: Linear, Polynomial")
    print("  • C values: [0.01, 0.1, 1, 10, 100, 1000]")
    print("  • Polynomial degrees: [2, 3, 4, 5]")
    print("=" * 60)

    # Create output directories
    create_output_directories()

    # 1. Load and prepare data
    print("\n🔄 STEP 1: Data Loading and Preprocessing")
    X_train, X_test, y_train, y_test, label_encoder, feature_names = load_and_prepare()

    # 30% subsample for speed:
    subsample_frac = 0.3
    n_samples = len(X_train)
    sub_idx = np.random.choice(n_samples, int(n_samples * subsample_frac), replace=False)
    X_train_sub = X_train.iloc[sub_idx].copy()
    y_train_sub = y_train.iloc[sub_idx].copy()

    print("Class balance in full training set:", np.bincount(y_train))
    print("Class balance in subsample:", np.bincount(y_train_sub))
    print("Label encoding:", label_encoder.classes_ if label_encoder else "None")

    print("\nGenerating learning curves...")
    # 1) Learning curves for linear & polynomial
    svm_learning_curve(X_train_sub, X_test, y_train_sub, y_test, kernel='linear', C=1)
    svm_learning_curve(X_train_sub, X_test, y_train_sub, y_test, kernel='poly', C=1, gamma=0.1, degree=2)

    print("\nGenerating validation curves...")
    # 2) Validation curves
    C_range     = [0.01, 0.1, 1, 10, 100, 1000]
    degree_range= [2, 3, 4, 5]

    # Linear kernel: sweep C
    svm_validation_curve(X_train_sub, y_train_sub, kernel='linear', param_name='C', param_range=C_range)

    # Poly kernel: sweep C (fix deg=2) and sweep degree (fix C=1, gamma=0.1)
    svm_validation_curve(X_train_sub, y_train_sub, kernel='poly', param_name='C', param_range=C_range, gamma=0.1, degree=2)
    svm_validation_curve(X_train_sub, y_train_sub, kernel='poly', param_name='degree', param_range=degree_range, C=1, gamma=0.1)

    print("\nSVM experiment complete. See figures/svmcp2/ for plots.")

    print("\nTesting different parameters with detailed analysis...")
    # Test different parameters with detailed analysis
    best_C = 1.0  # Start with a moderate C value
    for d in [2, 3, 4, 5]:
        print(f"\n=== Testing polynomial degree = {d} ===")
        model = SVC(kernel='poly',
                    C=best_C,
                    gamma=0.1,
                    degree=d,
                    class_weight='balanced',
                    random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # 1) Show which classes are in y_test vs. preds
        print("True-label counts:", np.bincount(y_test))
        print("Predicted-label counts:", np.bincount(preds))

        # 2) Show confusion matrix & classification report
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
        print("Classification Report:\n",
              classification_report(y_test, preds, target_names=['Non-Bankrupt', 'Bankrupt']))

        # 3) Compute and print F1 explicitly for both classes
        f1_positive = f1_score(y_test, preds, pos_label=1)  # F1 for bankrupt (1)
        f1_negative = f1_score(y_test, preds, pos_label=0)  # F1 for non-bankrupt (0)
        print(f"F1 (pos_label=1 'Bankrupt'): {f1_positive:.4f}")
        print(f"F1 (pos_label=0 'Non-Bankrupt'): {f1_negative:.4f}")
        print(f"Macro F1: {f1_score(y_test, preds, average='macro'):.4f}")

        # If we're predicting only one class, try adjusting C
        if len(np.unique(preds)) == 1:
            print(f"WARNING: Model is predicting only one class! Trying different C values...")
            for c in [0.01, 0.1, 10, 100]:
                print(f"\nTrying C = {c}")
                model = SVC(kernel='poly',
                            C=c,
                            gamma=0.1,
                            degree=d,
                            class_weight='balanced',
                            random_state=RANDOM_STATE)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                print("Predicted-label counts:", np.bincount(preds))
                print("F1 (pos_label=1 'Bankrupt'):", f1_score(y_test, preds, pos_label=1))
                print("F1 (pos_label=0 'Non-Bankrupt'):", f1_score(y_test, preds, pos_label=0))

if __name__ == "__main__":
    main()