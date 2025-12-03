"""
Model definitions and training utilities for forest element tier prediction.

This module contains various model architectures and training functions.
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings

warnings.filterwarnings('ignore')


def get_all_models() -> dict:
    """
    Get a dictionary of all models to evaluate.
    
    Returns:
        Dictionary mapping model name to model instance
    """
    return {
        # Random Forest variants
        'RandomForest (depth=5, regularized)': RandomForestClassifier(
            n_estimators=100, 
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        ),
        'RandomForest (depth=10)': RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        ),
        'RandomForest (depth=15)': RandomForestClassifier(
            n_estimators=200, 
            max_depth=15, 
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced'
        ),
        
        # Extra Trees
        'ExtraTrees (depth=10)': ExtraTreesClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        ),
        
        # Gradient Boosting variants
        'GradientBoosting (depth=3)': GradientBoostingClassifier(
            n_estimators=100, 
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=5,
            random_state=42
        ),
        'GradientBoosting (depth=5)': GradientBoostingClassifier(
            n_estimators=100, 
            max_depth=5, 
            learning_rate=0.1,
            random_state=42
        ),
        'GradientBoosting (depth=7)': GradientBoostingClassifier(
            n_estimators=150, 
            max_depth=7, 
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),
        
        # Histogram-based Gradient Boosting (faster, handles missing values)
        'HistGradientBoosting': HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=42
        ),
        
        # AdaBoost
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),
        
        # Bagging
        'Bagging (DecisionTree)': BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=10),
            n_estimators=50,
            random_state=42
        ),
        
        # SVM variants
        'SVM (RBF kernel)': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        ),
        'SVM (polynomial kernel)': SVC(
            kernel='poly',
            degree=3,
            C=1.0,
            class_weight='balanced',
            probability=True,
            random_state=42
        ),
        
        # K-Nearest Neighbors
        'KNN (k=5)': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='euclidean'
        ),
        'KNN (k=10)': KNeighborsClassifier(
            n_neighbors=10,
            weights='distance',
            metric='euclidean'
        ),
        
        # Neural Network variants
        'MLP (64-32)': MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            random_state=42
        ),
        'MLP (128-64-32)': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            random_state=42
        ),
        'MLP (256-128-64)': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            random_state=42
        ),
        
        # Linear models
        'LogisticRegression': LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        'RidgeClassifier': RidgeClassifier(
            alpha=1.0,
            class_weight='balanced',
            random_state=42
        ),
        
        # Naive Bayes
        'GaussianNB': GaussianNB(),
        
        # Decision Tree (baseline)
        'DecisionTree (depth=10)': DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ),
    }


def train_and_evaluate_models(X: np.ndarray, y: np.ndarray, feature_names: list) -> tuple:
    """
    Train and evaluate multiple models with train/validation/test splits.
    Includes overfitting detection.
    
    Args:
        X: Feature array
        y: Target array
        feature_names: List of feature names
    
    Returns:
        Tuple of (best_model, scaler, best_model_name, all_results)
    """
    # First split: separate test set (held out completely)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Second split: separate validation from training
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42
    )
    
    print("=" * 70)
    print("DATA SPLITS (for overfitting detection)")
    print("=" * 70)
    print(f"Training set:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Get all models
    models = get_all_models()
    
    best_model = None
    best_val_accuracy = 0
    best_model_name = ""
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING AND OVERFITTING ANALYSIS")
    print("=" * 70)
    print(f"\n{'Model':<35} {'Train':>8} {'Val':>8} {'Δ':>8} {'Status':>10}")
    print("-" * 75)
    
    results = []
    
    for name, model in models.items():
        try:
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Predict on all sets
            y_train_pred = model.predict(X_train_scaled)
            y_val_pred = model.predict(X_val_scaled)
            
            # Calculate accuracies
            train_acc = accuracy_score(y_train, y_train_pred)
            val_acc = accuracy_score(y_val, y_val_pred)
            
            # Calculate overfitting gap
            overfit_gap = train_acc - val_acc
            
            # Determine status
            if overfit_gap > 0.15:
                status = "⚠ OVERFIT"
            elif overfit_gap > 0.08:
                status = "⚡ SLIGHT"
            else:
                status = "✓ GOOD"
            
            print(f"{name:<35} {train_acc:>8.4f} {val_acc:>8.4f} {overfit_gap:>+8.4f} {status:>10}")
            
            results.append({
                'name': name,
                'model': model,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'overfit_gap': overfit_gap
            })
            
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model = model
                best_model_name = name
                
        except Exception as e:
            print(f"{name:<35} FAILED: {str(e)[:30]}")
    
    # Sort results by validation accuracy
    results.sort(key=lambda x: x['val_acc'], reverse=True)
    
    # Show top 5 models
    print("\n" + "-" * 70)
    print("TOP 5 MODELS BY VALIDATION ACCURACY:")
    print("-" * 70)
    for i, result in enumerate(results[:5], 1):
        print(f"  {i}. {result['name']}: {result['val_acc']:.4f}")
    
    # Cross-validation on training data for top models
    print("\n" + "-" * 70)
    print("Cross-Validation Results (on training set, top 5 models):")
    print("-" * 70)
    
    for result in results[:5]:
        n_folds = min(5, min(np.unique(y_train, return_counts=True)[1]))
        n_folds = max(2, n_folds)
        try:
            cv_scores = cross_val_score(result['model'], X_train_scaled, y_train, cv=n_folds)
            print(f"  {result['name']:<35} CV({n_folds}-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        except Exception as e:
            print(f"  {result['name']:<35} CV failed: {str(e)[:30]}")
    
    # Final evaluation on test set (only for best model)
    print("\n" + "=" * 70)
    print(f"FINAL EVALUATION - Best Model: {best_model_name}")
    print("=" * 70)
    
    y_test_pred = best_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # Get train and val accuracy for best model
    best_result = next(r for r in results if r['name'] == best_model_name)
    
    print(f"\n{'Set':<15} {'Accuracy':>10}")
    print("-" * 27)
    print(f"{'Training':<15} {best_result['train_acc']:>10.4f}")
    print(f"{'Validation':<15} {best_result['val_acc']:>10.4f}")
    print(f"{'Test':<15} {test_acc:>10.4f}")
    
    # Overfitting analysis summary
    train_test_gap = best_result['train_acc'] - test_acc
    print(f"\n{'Overfitting Analysis':}")
    print(f"  Train-Val gap:  {best_result['overfit_gap']:+.4f}")
    print(f"  Train-Test gap: {train_test_gap:+.4f}")
    
    if train_test_gap > 0.10:
        print("  ⚠  WARNING: Significant overfitting detected!")
        print("      Consider more regularization or more training data.")
    elif train_test_gap > 0.05:
        print("  ⚡ NOTICE: Slight overfitting present.")
        print("      Model should still generalize reasonably well.")
    else:
        print("  ✓  GOOD: No significant overfitting detected.")
        print("      Model generalizes well to unseen data.")
    
    # Feature importance for the best model
    if hasattr(best_model, 'feature_importances_'):
        print("\n" + "-" * 40)
        print("Feature Importances:")
        print("-" * 40)
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in indices:
            print(f"  {feature_names[i]}: {importances[i]:.4f}")
    
    # Detailed classification report for best model on test set
    print("\n" + "-" * 40)
    print("Classification Report (Test Set):")
    print("-" * 40)
    print(classification_report(y_test, y_test_pred))
    
    # Confusion matrix
    print("\n" + "-" * 40)
    print("Confusion Matrix (Test Set):")
    print("-" * 40)
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    return best_model, scaler, best_model_name, results


def save_model(model, scaler, label_encoder, feature_names: list, model_name: str, 
               save_path: str = 'tier_predictor_model.joblib') -> str:
    """
    Save the trained model and related components.
    
    Args:
        model: Trained model
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder for element types
        feature_names: List of feature names
        model_name: Name of the model
        save_path: Path to save the model
    
    Returns:
        Path where model was saved
    """
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'model_name': model_name
    }
    
    joblib.dump(model_data, save_path)
    print(f"\nModel saved to: {save_path}")
    
    return save_path


def load_model(load_path: str = 'tier_predictor_model.joblib') -> dict:
    """
    Load a trained model and related components.
    
    Args:
        load_path: Path to the saved model
    
    Returns:
        Dictionary with model, scaler, label_encoder, feature_names, and model_name
    """
    return joblib.load(load_path)

