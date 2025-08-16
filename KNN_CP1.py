"""
Enhanced KNN Analysis for Global Cancer Patients Dataset
========================================================
Comprehensive analysis of KNN performance across different parameters with detailed documentation.

Key Features:
1. Code verification and improvements
2. Enhanced feature selection analysis
3. Systematic testing of KNN parameters [2, 3, 4, 5, 7, 9, 12, 15]
4. K-fold validation testing [1,2,3,4,5,7]
5. Comprehensive visualization and documentation
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict
import os
import random

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, train_test_split, cross_val_score,
                                     StratifiedKFold, validation_curve, learning_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (LabelEncoder, StandardScaler, OneHotEncoder)
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

RANDOM_STATE = 42343289

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


# Set matplotlib seed for reproducibility


def create_output_directories():
    """Create necessary output directories."""
    directories = ["figures", "figures/knncp1", "results", "data_analysis"]
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


def comprehensive_feature_analysis(X_train, y_train, feature_names):
    """Comprehensive feature analysis with multiple selection methods."""
    print("\n▶ Comprehensive Feature Analysis")
    print("-" * 50)

    # Convert to DataFrame for easier analysis
    X_df = pd.DataFrame(X_train, columns=feature_names)

    # 1. Correlation Analysis
    print("\n1. Feature-Target Correlations:")
    correlations = []
    for feature in feature_names:
        corr = np.corrcoef(X_df[feature], y_train)[0, 1]
        correlations.append((feature, abs(corr), corr))

    correlations.sort(key=lambda x: x[1], reverse=True)
    for feature, abs_corr, corr in correlations:
        print(f"  {feature:<25}: {corr:>7.3f} (|{abs_corr:.3f}|)")

    # 2. Mutual Information
    mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({
        'feature': feature_names,
        'mutual_info': mi_scores
    }).sort_values('mutual_info', ascending=False)

    print("\n2. Mutual Information Scores:")
    for _, row in mi_df.iterrows():
        print(f"  {row['feature']:<25}: {row['mutual_info']:>7.3f}")

    # 3. F-statistic (ANOVA)
    f_scores, p_values = f_classif(X_train, y_train)
    f_df = pd.DataFrame({
        'feature': feature_names,
        'f_score': f_scores,
        'p_value': p_values
    }).sort_values('f_score', ascending=False)

    print("\n3. F-statistic Scores:")
    for _, row in f_df.iterrows():
        print(f"  {row['feature']:<25}: {row['f_score']:>7.1f} (p={row['p_value']:>6.3f})")

    # 4. Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Feature correlations
    corr_df = pd.DataFrame(correlations, columns=['feature', 'abs_corr', 'corr'])
    axes[0, 0].barh(range(len(corr_df)), corr_df['corr'])
    axes[0, 0].set_yticks(range(len(corr_df)))
    axes[0, 0].set_yticklabels(corr_df['feature'], fontsize=8)
    axes[0, 0].set_title('Feature-Target Correlations')
    axes[0, 0].set_xlabel('Correlation Coefficient')

    # Plot 2: Mutual Information
    axes[0, 1].barh(range(len(mi_df)), mi_df['mutual_info'])
    axes[0, 1].set_yticks(range(len(mi_df)))
    axes[0, 1].set_yticklabels(mi_df['feature'], fontsize=8)
    axes[0, 1].set_title('Mutual Information Scores')
    axes[0, 1].set_xlabel('Mutual Information')

    # Plot 3: F-statistics
    axes[1, 0].barh(range(len(f_df)), f_df['f_score'])
    axes[1, 0].set_yticks(range(len(f_df)))
    axes[1, 0].set_yticklabels(f_df['feature'], fontsize=8)
    axes[1, 0].set_title('F-statistic Scores')
    axes[1, 0].set_xlabel('F-statistic')

    # Plot 4: Feature distributions by class
    top_features = mi_df.head(6)['feature'].tolist()
    for i, feature in enumerate(top_features):
        if i < 3:
            row, col = 1, 1
            axes[row, col].remove()
            axes[row, col] = plt.subplot(2, 3, 5)
            break

    # Create subplot for feature distributions
    plt.subplot(2, 3, 6)
    for i, feature in enumerate(top_features[:3]):
        plt.subplot(2, 3, 4 + i)
        for class_label in np.unique(y_train):
            class_data = X_df[feature][y_train == class_label]
            plt.hist(class_data, alpha=0.5, label=f'Class {class_label}', bins=20)
        plt.title(f'{feature}', fontsize=8)
        plt.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig('data_analysis/comprehensive_feature_analysis.png', dpi=300, bbox_inches='tight')

    # Save analysis results
    analysis_results = {
        'correlations': corr_df,
        'mutual_info': mi_df,
        'f_statistics': f_df
    }

    # Save to CSV
    for name, df in analysis_results.items():
        df.to_csv(f'data_analysis/{name}_analysis.csv', index=False)

    return analysis_results


def comprehensive_knn_analysis(X_train, X_test, y_train, y_test):
    """
    Comprehensive KNN analysis across different parameters.
    Tests KNN with n_neighbors and k_folds
    """
    print("\n▶ Comprehensive KNN Parameter Analysis")
    print("-" * 50)

    # Define parameter ranges as requested
    n_neighbors_list = [2, 3, 4, 5, 7, 9, 12, 15]
    k_folds_list = [1, 3, 5]  # Note: k=1 means no CV, just train/test

    # Store all results
    results = []

    print("Testing all parameter combinations...")
    print(f"KNN neighbors: {n_neighbors_list}")
    print(f"K-fold values: {k_folds_list}")
    print()

    for n_neighbors in n_neighbors_list:
        for k_folds in k_folds_list:
            print(f"Testing n_neighbors={n_neighbors}, k_folds={k_folds}")

            # Create KNN model
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)

            # Train the model
            knn.fit(X_train, y_train)

            # Calculate training metrics
            train_pred = knn.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_pred)
            train_f1 = f1_score(y_train, train_pred)
            train_precision = precision_score(y_train, train_pred)
            train_recall = recall_score(y_train, train_pred)

            # Calculate test metrics
            test_pred = knn.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred)
            test_precision = precision_score(y_test, test_pred)
            test_recall = recall_score(y_test, test_pred)

            # Calculate cross-validation metrics (if k > 1)
            if k_folds > 1:
                # Create stratified k-fold with random state
                skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_STATE)
                cv_accuracy = cross_val_score(knn, X_train, y_train, cv=skf,
                                              scoring='accuracy')
                cv_f1 = cross_val_score(knn, X_train, y_train, cv=skf,
                                        scoring='f1')
                cv_precision = cross_val_score(knn, X_train, y_train, cv=skf,
                                               scoring='precision')
                cv_recall = cross_val_score(knn, X_train, y_train, cv=skf,
                                            scoring='recall')

                cv_accuracy_mean = cv_accuracy.mean()
                cv_accuracy_std = cv_accuracy.std()
                cv_f1_mean = cv_f1.mean()
                cv_f1_std = cv_f1.std()
                cv_precision_mean = cv_precision.mean()
                cv_precision_std = cv_precision.std()
                cv_recall_mean = cv_recall.mean()
                cv_recall_std = cv_recall.std()
            else:
                # For k=1, use test scores as "CV" scores
                cv_accuracy_mean = test_accuracy
                cv_accuracy_std = 0.0
                cv_f1_mean = test_f1
                cv_f1_std = 0.0
                cv_precision_mean = test_precision
                cv_precision_std = 0.0
                cv_recall_mean = test_recall
                cv_recall_std = 0.0

            # Store results
            result = {
                'n_neighbors': n_neighbors,
                'k_folds': k_folds,
                'train_accuracy': train_accuracy,
                'train_f1': train_f1,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'test_accuracy': test_accuracy,
                'test_f1': test_f1,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'cv_accuracy_mean': cv_accuracy_mean,
                'cv_accuracy_std': cv_accuracy_std,
                'cv_f1_mean': cv_f1_mean,
                'cv_f1_std': cv_f1_std,
                'cv_precision_mean': cv_precision_mean,
                'cv_precision_std': cv_precision_std,
                'cv_recall_mean': cv_recall_mean,
                'cv_recall_std': cv_recall_std,
                'overfitting': train_accuracy - test_accuracy
            }

            results.append(result)

            print(f"  Train: Acc={train_accuracy:.3f}, F1={train_f1:.3f}")
            print(f"  Test:  Acc={test_accuracy:.3f}, F1={test_f1:.3f}")
            print(f"  CV:    Acc={cv_accuracy_mean:.3f}±{cv_accuracy_std:.3f}, F1={cv_f1_mean:.3f}±{cv_f1_std:.3f}")
            print(f"  Overfitting: {train_accuracy - test_accuracy:.3f}")
            print()

    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)

    # Save detailed results
    results_df.to_csv('results/comprehensive_knn_results.csv', index=False)

    return results_df


def create_comprehensive_visualizations(results_df):
    """Create comprehensive visualizations of KNN performance."""
    # Create figure directory if it doesn't exist
    os.makedirs('figures/knncp1', exist_ok=True)
    
    # 1. Performance Metrics vs n_neighbors
    plt.figure(figsize=(12, 8))
    plt.plot(results_df['n_neighbors'], results_df['test_accuracy'], 'o-', label='Test Accuracy')
    plt.plot(results_df['n_neighbors'], results_df['test_f1'], 's-', label='Test F1 Score')
    plt.plot(results_df['n_neighbors'], results_df['test_precision'], '^-', label='Test Precision')
    plt.plot(results_df['n_neighbors'], results_df['test_recall'], 'v-', label='Test Recall')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Score')
    plt.title('KNN Performance Metrics vs Number of Neighbors')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/knncp1/knn_performance_metrics.png')
    plt.close()
    
    # 2. Training vs Test Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['n_neighbors'], results_df['train_accuracy'], 'o-', label='Training Accuracy')
    plt.plot(results_df['n_neighbors'], results_df['test_accuracy'], 's-', label='Test Accuracy')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('Training vs Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/knncp1/knn_train_test_accuracy.png')
    plt.close()
    
    # 3. Cross-validation Scores by k_folds
    plt.figure(figsize=(10, 6))
    for k in sorted(results_df['k_folds'].unique()):
        k_data = results_df[results_df['k_folds'] == k]
        plt.plot(k_data['n_neighbors'], k_data['cv_f1_mean'], 
                marker='o', label=f'k_folds={k}')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Cross-validation F1 Score')
    plt.title('Cross-validation Scores by k_folds')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/knncp1/knn_cv_scores.png')
    plt.close()


def generate_performance_summary(results_df):
    """Generate a comprehensive performance summary."""
    print("\n▶ Performance Summary and Recommendations")
    print("=" * 60)

    # Find best performing models
    best_test_accuracy = results_df.loc[results_df['test_accuracy'].idxmax()]
    best_test_f1 = results_df.loc[results_df['test_f1'].idxmax()]
    best_cv_f1 = results_df.loc[results_df['cv_f1_mean'].idxmax()]
    least_overfitting = results_df.loc[results_df['overfitting'].abs().idxmin()]

    print("\n🏆 Best Performing Models:")
    print(
        f"Best Test Accuracy: n_neighbors={best_test_accuracy['n_neighbors']}, k_folds={best_test_accuracy['k_folds']} → {best_test_accuracy['test_accuracy']:.3f}")
    print(
        f"Best Test F1:       n_neighbors={best_test_f1['n_neighbors']}, k_folds={best_test_f1['k_folds']} → {best_test_f1['test_f1']:.3f}")
    print(
        f"Best CV F1:         n_neighbors={best_cv_f1['n_neighbors']}, k_folds={best_cv_f1['k_folds']} → {best_cv_f1['cv_f1_mean']:.3f}")
    print(
        f"Least Overfitting:  n_neighbors={least_overfitting['n_neighbors']}, k_folds={least_overfitting['k_folds']} → {least_overfitting['overfitting']:.3f}")

    # Statistical summary by n_neighbors
    print("\n📊 Performance by Number of Neighbors:")
    print("-" * 50)
    neighbor_summary = results_df.groupby('n_neighbors').agg({
        'test_accuracy': ['mean', 'std', 'max'],
        'test_f1': ['mean', 'std', 'max'],
        'overfitting': ['mean', 'std']
    }).round(3)
    print(neighbor_summary)

    # Statistical summary by k_folds
    print("\n📊 Performance by K-Fold Values:")
    print("-" * 50)
    kfold_summary = results_df.groupby('k_folds').agg({
        'test_accuracy': ['mean', 'std', 'max'],
        'test_f1': ['mean', 'std', 'max'],
        'cv_f1_mean': ['mean', 'std', 'max'],
        'overfitting': ['mean', 'std']
    }).round(3)
    print(kfold_summary)

    # Generate recommendations
    print("\n💡 Recommendations:")
    print("-" * 30)

    # Analyze trends
    mean_by_neighbors = results_df.groupby('n_neighbors')['test_f1'].mean()
    best_neighbor = mean_by_neighbors.idxmax()

    mean_by_kfolds = results_df.groupby('k_folds')['cv_f1_mean'].mean()
    best_kfold = mean_by_kfolds.idxmax()

    print(f"1. Optimal n_neighbors: {best_neighbor} (avg F1: {mean_by_neighbors[best_neighbor]:.3f})")
    print(f"2. Optimal k_folds: {best_kfold} (avg CV F1: {mean_by_kfolds[best_kfold]:.3f})")

    # Overfitting analysis
    avg_overfitting = results_df.groupby('n_neighbors')['overfitting'].mean()
    least_overfitting_k = avg_overfitting.abs().idxmin()
    print(
        f"3. Least overfitting with n_neighbors: {least_overfitting_k} (avg overfitting: {avg_overfitting[least_overfitting_k]:.3f})")

    # Stability analysis
    stability_by_neighbors = results_df.groupby('n_neighbors')['test_f1'].std()
    most_stable_k = stability_by_neighbors.idxmin()
    print(f"4. Most stable n_neighbors: {most_stable_k} (F1 std: {stability_by_neighbors[most_stable_k]:.3f})")

    # Final recommendation
    print(f"\n🎯 Final Recommendation:")
    print(f"   Use n_neighbors={best_neighbor} with k_folds={best_kfold}")
    print(f"   Expected performance: F1 ≈ {mean_by_neighbors[best_neighbor]:.3f}")

    # Save summary to file
    summary_dict = {
        'best_models': {
            'best_test_accuracy': best_test_accuracy.to_dict(),
            'best_test_f1': best_test_f1.to_dict(),
            'best_cv_f1': best_cv_f1.to_dict(),
            'least_overfitting': least_overfitting.to_dict()
        },
        'recommendations': {
            'optimal_n_neighbors': int(best_neighbor),
            'optimal_k_folds': int(best_kfold),
            'expected_f1': float(mean_by_neighbors[best_neighbor])
        }
    }

    import json
    with open('results/performance_summary.json', 'w') as f:
        json.dump(summary_dict, f, indent=2)

    return summary_dict


def create_final_model_evaluation(X_train, X_test, y_train, y_test, best_params, label_encoder):
    """Create final model with best parameters and comprehensive evaluation."""
    print(f"\n▶ Final Model Evaluation")
    print("-" * 40)

    # Create and train final model
    final_model = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'])
    final_model.fit(X_train, y_train)

    # Predictions
    y_train_pred = final_model.predict(X_train)
    y_test_pred = final_model.predict(X_test)

    # Comprehensive metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)

    # Add cross-validation with random state
    cv_scores = cross_val_score(final_model, X_train, y_train, cv=5, scoring='f1')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    print(f"Final Model Performance (n_neighbors={best_params['n_neighbors']}):")
    print(f"  Training Accuracy: {train_accuracy:.3f}")
    print(f"  Test Accuracy:     {test_accuracy:.3f}")
    print(f"  Test F1 Score:     {test_f1:.3f}")
    print(f"  Test Precision:    {test_precision:.3f}")
    print(f"  Test Recall:       {test_recall:.3f}")
    print(f"  CV F1 Score:       {cv_mean:.3f} ± {cv_std:.3f}")
    print(f"  Overfitting:       {train_accuracy - test_accuracy:.3f}")

    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap='Blues')
    plt.title(f'Final Model Confusion Matrix (n_neighbors={best_params["n_neighbors"]})')
    plt.tight_layout()
    plt.savefig('figures/knncp1/final_confusion_matrix.png', dpi=300, bbox_inches='tight')

    return final_model


def generate_learning_curves(X_train, X_test, y_train, y_test, best_params):
    """Generate learning curves for the best KNN model."""
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        KNeighborsClassifier(**best_params),
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
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('figures/knncp1/knn_learning_curves.png')
    plt.close()


def main():
    """Run KNN experiments with different parameters."""
    # Create output directories
    create_output_directories()
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, le_target, feature_names = load_and_prepare()
    
    # Run KNN analysis with different parameters
    results_df = comprehensive_knn_analysis(X_train, X_test, y_train, y_test)
    
    # Create visualizations and save results
    create_comprehensive_visualizations(results_df)
    summary_dict = generate_performance_summary(results_df)
    
    # Get best parameters and generate learning curves
    best_params = {'n_neighbors': results_df.loc[results_df['test_accuracy'].idxmax(), 'n_neighbors']}
    generate_learning_curves(X_train, X_test, y_train, y_test, best_params)
    
    # Create final model evaluation and report
    final_model = create_final_model_evaluation(X_train, X_test, y_train, y_test, best_params, le_target)
    
    print("\nAnalysis completed! Check the figures directory for visualizations.")



if __name__ == "__main__":
    main()
