"""
Regression model definitions for forest element diameter prediction.

This module contains various regression model architectures and training functions.
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    HuberRegressor,
    BayesianRidge
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings

warnings.filterwarnings('ignore')


def get_all_regression_models() -> dict:
    """
    Get a dictionary of all regression models to evaluate.
    
    Returns:
        Dictionary mapping model name to model instance
    """
    return {
        # Random Forest variants
        'RandomForest (depth=5)': RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42
        ),
        'RandomForest (depth=10)': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42
        ),
        'RandomForest (depth=15)': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=1,
            random_state=42
        ),
        
        # Extra Trees
        'ExtraTrees (depth=10)': ExtraTreesRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42
        ),
        
        # Gradient Boosting variants
        'GradientBoosting (depth=3)': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=5,
            random_state=42
        ),
        'GradientBoosting (depth=5)': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        'GradientBoosting (depth=7)': GradientBoostingRegressor(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),
        
        # Histogram-based Gradient Boosting
        'HistGradientBoosting': HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=42
        ),
        
        # AdaBoost
        'AdaBoost': AdaBoostRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),
        
        # Bagging
        'Bagging (DecisionTree)': BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=10),
            n_estimators=50,
            random_state=42
        ),
        
        # SVR variants
        'SVR (RBF kernel)': SVR(
            kernel='rbf',
            C=1.0,
            gamma='scale'
        ),
        'SVR (polynomial kernel)': SVR(
            kernel='poly',
            degree=3,
            C=1.0
        ),
        
        # K-Nearest Neighbors
        'KNN (k=5)': KNeighborsRegressor(
            n_neighbors=5,
            weights='distance',
            metric='euclidean'
        ),
        'KNN (k=10)': KNeighborsRegressor(
            n_neighbors=10,
            weights='distance',
            metric='euclidean'
        ),
        
        # Neural Network variants
        'MLP (64-32)': MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            random_state=42
        ),
        'MLP (128-64-32)': MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            random_state=42
        ),
        'MLP (256-128-64)': MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            random_state=42
        ),
        
        # Linear models
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.1, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'HuberRegressor': HuberRegressor(max_iter=1000),
        'BayesianRidge': BayesianRidge(),
        
        # Decision Tree (baseline)
        'DecisionTree (depth=10)': DecisionTreeRegressor(
            max_depth=10,
            min_samples_leaf=2,
            random_state=42
        ),
    }


def train_and_evaluate_regression_models(X: np.ndarray, y: np.ndarray, 
                                          feature_names: list) -> tuple:
    """
    Train and evaluate multiple regression models with train/validation/test splits.
    
    Args:
        X: Feature array
        y: Target array (diameter)
        feature_names: List of feature names
    
    Returns:
        Tuple of (best_model, scaler, best_model_name, all_results)
    """
    # First split: separate test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Second split: separate validation from training
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42
    )
    
    print("=" * 80)
    print("DATA SPLITS (for overfitting detection)")
    print("=" * 80)
    print(f"Training set:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Get all models
    models = get_all_regression_models()
    
    best_model = None
    best_val_r2 = -float('inf')
    best_model_name = ""
    
    print("\n" + "=" * 80)
    print("MODEL TRAINING AND OVERFITTING ANALYSIS")
    print("=" * 80)
    print(f"\n{'Model':<35} {'Train R²':>10} {'Val R²':>10} {'Val MAE':>10} {'Status':>10}")
    print("-" * 80)
    
    results = []
    
    for name, model in models.items():
        try:
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Predict on all sets
            y_train_pred = model.predict(X_train_scaled)
            y_val_pred = model.predict(X_val_scaled)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            val_mae = mean_absolute_error(y_val, y_val_pred)
            
            # Calculate overfitting gap
            overfit_gap = train_r2 - val_r2
            
            # Determine status
            if overfit_gap > 0.15:
                status = "⚠ OVERFIT"
            elif overfit_gap > 0.08:
                status = "⚡ SLIGHT"
            else:
                status = "✓ GOOD"
            
            print(f"{name:<35} {train_r2:>10.4f} {val_r2:>10.4f} {val_mae:>10.2f} {status:>10}")
            
            results.append({
                'name': name,
                'model': model,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'val_mae': val_mae,
                'overfit_gap': overfit_gap
            })
            
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_model = model
                best_model_name = name
                
        except Exception as e:
            print(f"{name:<35} FAILED: {str(e)[:30]}")
    
    # Sort results by validation R²
    results.sort(key=lambda x: x['val_r2'], reverse=True)
    
    # Show top 5 models
    print("\n" + "-" * 80)
    print("TOP 5 MODELS BY VALIDATION R²:")
    print("-" * 80)
    for i, result in enumerate(results[:5], 1):
        print(f"  {i}. {result['name']}: R²={result['val_r2']:.4f}, MAE={result['val_mae']:.2f}")
    
    # Cross-validation on training data for top models
    print("\n" + "-" * 80)
    print("Cross-Validation Results (on training set, top 5 models):")
    print("-" * 80)
    
    for result in results[:5]:
        try:
            cv_scores = cross_val_score(result['model'], X_train_scaled, y_train, 
                                        cv=5, scoring='r2')
            print(f"  {result['name']:<35} CV(5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        except Exception as e:
            print(f"  {result['name']:<35} CV failed: {str(e)[:30]}")
    
    # Final evaluation on test set
    print("\n" + "=" * 80)
    print(f"FINAL EVALUATION - Best Model: {best_model_name}")
    print("=" * 80)
    
    y_test_pred = best_model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Get train and val metrics for best model
    best_result = next(r for r in results if r['name'] == best_model_name)
    
    print(f"\n{'Set':<15} {'R²':>10} {'MAE':>10} {'RMSE':>10}")
    print("-" * 47)
    print(f"{'Training':<15} {best_result['train_r2']:>10.4f} {'-':>10} {'-':>10}")
    print(f"{'Validation':<15} {best_result['val_r2']:>10.4f} {best_result['val_mae']:>10.2f} {'-':>10}")
    print(f"{'Test':<15} {test_r2:>10.4f} {test_mae:>10.2f} {test_rmse:>10.2f}")
    
    # Overfitting analysis summary
    train_test_gap = best_result['train_r2'] - test_r2
    print(f"\nOverfitting Analysis:")
    print(f"  Train-Val R² gap:  {best_result['overfit_gap']:+.4f}")
    print(f"  Train-Test R² gap: {train_test_gap:+.4f}")
    
    if train_test_gap > 0.10:
        print("  ⚠  WARNING: Significant overfitting detected!")
    elif train_test_gap > 0.05:
        print("  ⚡ NOTICE: Slight overfitting present.")
    else:
        print("  ✓  GOOD: No significant overfitting detected.")
    
    # Feature importance for the best model
    if hasattr(best_model, 'feature_importances_'):
        print("\n" + "-" * 40)
        print("Feature Importances:")
        print("-" * 40)
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in indices:
            print(f"  {feature_names[i]}: {importances[i]:.4f}")
    
    # Prediction error distribution
    print("\n" + "-" * 40)
    print("Prediction Error Distribution (Test Set):")
    print("-" * 40)
    errors = y_test - y_test_pred
    print(f"  Mean Error:   {np.mean(errors):>8.2f}")
    print(f"  Std Error:    {np.std(errors):>8.2f}")
    print(f"  Min Error:    {np.min(errors):>8.2f}")
    print(f"  Max Error:    {np.max(errors):>8.2f}")
    print(f"  Median Error: {np.median(errors):>8.2f}")
    
    return best_model, scaler, best_model_name, results


def save_regression_model(model, scaler, label_encoder, feature_names: list, 
                          model_name: str, 
                          save_path: str = 'diameter_predictor_model.joblib') -> str:
    """
    Save the trained regression model and related components.
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


def load_regression_model(load_path: str = 'diameter_predictor_model.joblib') -> dict:
    """
    Load a trained regression model and related components.
    """
    return joblib.load(load_path)

