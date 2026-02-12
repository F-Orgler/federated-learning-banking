"""
Model performance metrics evaluation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict
import pandas as pd


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for ROC-AUC)
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    
    return metrics


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    detailed: bool = True
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        detailed: Whether to include confusion matrix and classification report
    
    Returns:
        Dictionary with all evaluation results
    """
    print("\nEvaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Probabilities (if available)
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    
    # Print metrics
    print("\nPerformance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    results = {
        'metrics': metrics,
        'predictions': y_pred,
    }
    
    if y_prob is not None:
        results['probabilities'] = y_prob
    
    if detailed:
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        results['confusion_matrix'] = cm
        
        print("\nConfusion Matrix:")
        print(f"  TN: {cm[0, 0]:5d}  FP: {cm[0, 1]:5d}")
        print(f"  FN: {cm[1, 0]:5d}  TP: {cm[1, 1]:5d}")
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        results['classification_report'] = report
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    return results


def evaluate_per_group(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensitive_feature: np.ndarray,
    group_names: Dict = None
) -> Dict:
    """
    Evaluate model performance separately for different groups.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        sensitive_feature: Sensitive attribute values (e.g., gender)
        group_names: Mapping of values to group names
    
    Returns:
        Dictionary with per-group metrics
    
    Example:
        >>> # Evaluate by gender
        >>> results = evaluate_per_group(
        >>>     model, X_test, y_test, 
        >>>     gender_values, 
        >>>     group_names={0: 'Female', 1: 'Male'}
        >>> )
    """
    if group_names is None:
        group_names = {val: f"Group_{val}" for val in np.unique(sensitive_feature)}
    
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    
    per_group_results = {}
    
    print("\nPer-Group Performance:")
    print("-" * 60)
    
    for group_val, group_name in group_names.items():
        # Get indices for this group
        group_mask = (sensitive_feature == group_val)
        
        if not np.any(group_mask):
            print(f"{group_name}: No samples")
            continue
        
        # Calculate metrics for this group
        y_true_group = y_test[group_mask]
        y_pred_group = y_pred[group_mask]
        y_prob_group = y_prob[group_mask] if y_prob is not None else None
        
        metrics = calculate_metrics(y_true_group, y_pred_group, y_prob_group)
        
        per_group_results[group_name] = {
            'n_samples': np.sum(group_mask),
            'metrics': metrics
        }
        
        print(f"\n{group_name} (n={np.sum(group_mask)}):")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print("-" * 60)
    
    return per_group_results


def compare_models(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare metrics across multiple models.
    
    Args:
        results_dict: Dictionary of {model_name: results_dict}
    
    Returns:
        DataFrame with comparison
    
    Example:
        >>> results = {
        >>>     'Baseline': baseline_results,
        >>>     'Fair Model': fair_model_results,
        >>>     'FL Model': fl_model_results,
        >>> }
        >>> comparison = compare_models(results)
    """
    comparison_data = []
    
    for model_name, results in results_dict.items():
        metrics = results.get('metrics', results)
        row = {'Model': model_name}
        row.update(metrics)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Reorder columns
    cols = ['Model', 'accuracy', 'precision', 'recall', 'f1']
    if 'roc_auc' in df.columns:
        cols.append('roc_auc')
    
    df = df[cols]
    
    return df


if __name__ == "__main__":
    # Test with dummy data
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(n_samples=1000, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Test evaluation
    results = evaluate_model(model, X_test, y_test, detailed=True)
    
    # Test per-group evaluation
    sensitive = np.random.randint(0, 2, size=len(y_test))
    per_group = evaluate_per_group(
        model, X_test, y_test, sensitive,
        group_names={0: 'Group A', 1: 'Group B'}
    )
