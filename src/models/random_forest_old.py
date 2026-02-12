"""
Random Forest model training and hyperparameter tuning.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import numpy as np
from typing import Dict, Tuple
import joblib
import os


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict = None,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        params: Model parameters
        random_state: Random seed
    
    Returns:
        Trained Random Forest model
    
    Example:
        >>> model = train_random_forest(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    if params is None:
        params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': random_state,
            'n_jobs': -1,
        }
    
    print("\nTraining Random Forest...")
    print(f"  Parameters: {params}")
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    print("✓ Training complete!")
    
    return model


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict = None,
    method: str = 'grid',
    cv: int = 5,
    random_state: int = 42,
    verbose: int = 1
) -> Tuple[RandomForestClassifier, Dict]:
    """
    Perform hyperparameter tuning using Grid or Random Search.
    
    Args:
        X_train: Training features
        y_train: Training labels
        param_grid: Parameter grid for search
        method: 'grid' or 'random'
        cv: Number of cross-validation folds
        random_state: Random seed
        verbose: Verbosity level
    
    Returns:
        Tuple of (best_model, best_params)
    
    Example:
        >>> param_grid = {
        >>>     'n_estimators': [100, 200, 300],
        >>>     'max_depth': [10, 20, 30],
        >>> }
        >>> model, params = tune_hyperparameters(X_train, y_train, param_grid)
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
    
    print(f"\nHyperparameter Tuning ({method.upper()} search)...")
    print(f"  Parameter grid: {param_grid}")
    print(f"  Cross-validation folds: {cv}")
    
    base_model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1
    )
    
    if method == 'grid':
        search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=verbose
        )
    elif method == 'random':
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=20,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            random_state=random_state,
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    search.fit(X_train, y_train)
    
    print("\n✓ Tuning complete!")
    print(f"  Best score (F1): {search.best_score_:.4f}")
    print(f"  Best parameters: {search.best_params_}")
    
    return search.best_estimator_, search.best_params_


def evaluate_cross_validation(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5
) -> Dict:
    """
    Perform cross-validation and return scores.
    
    Args:
        model: Trained model
        X: Features
        y: Labels
        cv: Number of folds
    
    Returns:
        Dictionary with CV scores
    """
    print(f"\nPerforming {cv}-fold cross-validation...")
    
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_scores = {}
    
    for metric in scoring_metrics:
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
        cv_scores[metric] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
        print(f"  {metric:12s}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    return cv_scores


def get_feature_importance(
    model: RandomForestClassifier,
    feature_names: list = None,
    top_n: int = 20
) -> Tuple[np.ndarray, list]:
    """
    Get feature importances from trained Random Forest.
    
    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        top_n: Number of top features to return
    
    Returns:
        Tuple of (importances, feature_names)
    """
    importances = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    top_importances = importances[indices]
    top_features = [feature_names[i] for i in indices]
    
    return top_importances, top_features


def save_model(
    model: RandomForestClassifier,
    output_path: str,
    create_dir: bool = True
):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        output_path: Path to save model (e.g., 'results/models/rf_model.joblib')
        create_dir: Whether to create output directory if it doesn't exist
    """
    if create_dir:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    joblib.dump(model, output_path)
    print(f"✓ Model saved to {output_path}")


def load_model(model_path: str) -> RandomForestClassifier:
    """
    Load trained model from disk.
    
    Args:
        model_path: Path to saved model
    
    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    print(f"✓ Model loaded from {model_path}")
    return model


if __name__ == "__main__":
    # Test with dummy data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Test training
    model = train_random_forest(X, y)
    
    # Test hyperparameter tuning (small grid for quick test)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
    }
    tuned_model, best_params = tune_hyperparameters(X, y, param_grid, method='grid', cv=3)
    
    # Test cross-validation
    cv_scores = evaluate_cross_validation(tuned_model, X, y, cv=3)
    
    # Test feature importance
    importances, features = get_feature_importance(tuned_model, top_n=10)
    print(f"\nTop 10 features:")
    for feat, imp in zip(features, importances):
        print(f"  {feat}: {imp:.4f}")
