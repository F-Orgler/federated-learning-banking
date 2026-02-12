"""
Model explainability using SHAP (SHapley Additive exPlanations).
"""

import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import List, Tuple
import os


def explain_model_shap(
    model,
    X_test: np.ndarray,
    feature_names: List[str] = None,
    max_samples: int = 1000
) -> Tuple[shap.Explainer, np.ndarray]:
    """
    Generate SHAP explanations for a model.
    
    Args:
        model: Trained model (preferably tree-based for speed)
        X_test: Test data for explanation
        feature_names: List of feature names
        max_samples: Maximum samples to explain (for speed)
    
    Returns:
        Tuple of (explainer, shap_values)
    
    Example:
        >>> explainer, shap_values = explain_model_shap(model, X_test, feature_names)
        >>> plot_shap_summary(shap_values, X_test, feature_names)
    """
    print(f"\nGenerating SHAP explanations...")
    print(f"  Samples: {min(len(X_test), max_samples)}")
    
    # Subsample if needed (for speed)
    if len(X_test) > max_samples:
        indices = np.random.choice(len(X_test), max_samples, replace=False)
        X_explain = X_test[indices]
    else:
        X_explain = X_test
    
    # Create explainer (Tree explainer for tree-based models)
    try:
        explainer = shap.TreeExplainer(model)
        print("  Using TreeExplainer (fast)")
    except:
        # Fallback to KernelExplainer for other models
        explainer = shap.KernelExplainer(model.predict, X_explain[:100])
        print("  Using KernelExplainer (slower)")
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_explain)
    
    # For binary classification, shap_values might be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Take positive class
    
    print(f"✓ SHAP values computed: shape {shap_values.shape}")
    
    return explainer, shap_values


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str] = None,
    max_display: int = 20,
    output_path: str = None,
    show: bool = True
):
    """
    Create SHAP summary plot.
    
    Args:
        shap_values: SHAP values from explainer
        X: Feature data used for explanation
        feature_names: Feature names
        max_display: Maximum features to display
        output_path: Path to save plot
        show: Whether to display plot
    """
    plt.figure(figsize=(10, 8))
    
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ SHAP summary plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: List[str] = None,
    max_display: int = 20,
    output_path: str = None,
    show: bool = True
):
    """
    Create SHAP bar plot (mean absolute SHAP values).
    
    Args:
        shap_values: SHAP values from explainer
        feature_names: Feature names
        max_display: Maximum features to display
        output_path: Path to save plot
        show: Whether to display plot
    """
    plt.figure(figsize=(10, 8))
    
    shap.summary_plot(
        shap_values,
        features=None,
        feature_names=feature_names,
        max_display=max_display,
        plot_type='bar',
        show=False
    )
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ SHAP bar plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_shap_dependence(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    feature_idx: int,
    interaction_idx: str = 'auto',
    output_path: str = None,
    show: bool = True
):
    """
    Create SHAP dependence plot for a specific feature.
    
    Args:
        shap_values: SHAP values from explainer
        X: Feature data
        feature_names: Feature names
        feature_idx: Index of feature to plot
        interaction_idx: Index of interaction feature ('auto' for automatic)
        output_path: Path to save plot
        show: Whether to display plot
    """
    plt.figure(figsize=(10, 6))
    
    shap.dependence_plot(
        feature_idx,
        shap_values,
        X,
        feature_names=feature_names,
        interaction_index=interaction_idx,
        show=False
    )
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ SHAP dependence plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def get_feature_importance_from_shap(
    shap_values: np.ndarray,
    feature_names: List[str] = None,
    top_n: int = 20
) -> Tuple[np.ndarray, List[str]]:
    """
    Get feature importance from SHAP values (mean absolute).
    
    Args:
        shap_values: SHAP values
        feature_names: Feature names
        top_n: Number of top features to return
    
    Returns:
        Tuple of (importance_values, feature_names)
    """
    # Calculate mean absolute SHAP value for each feature
    importance = np.abs(shap_values).mean(axis=0)
    
    # Sort by importance
    indices = np.argsort(importance)[::-1][:top_n]
    
    top_importance = importance[indices]
    
    if feature_names is not None:
        top_features = [feature_names[i] for i in indices]
    else:
        top_features = [f"Feature {i}" for i in indices]
    
    return top_importance, top_features


def explain_single_prediction(
    explainer: shap.Explainer,
    X_single: np.ndarray,
    feature_names: List[str] = None,
    output_path: str = None,
    show: bool = True
):
    """
    Create waterfall plot for a single prediction.
    
    Args:
        explainer: SHAP explainer object
        X_single: Single sample (1D array or 2D with shape (1, n_features))
        feature_names: Feature names
        output_path: Path to save plot
        show: Whether to display plot
    """
    # Ensure 2D
    if X_single.ndim == 1:
        X_single = X_single.reshape(1, -1)
    
    # Get SHAP values for this sample
    shap_values = explainer.shap_values(X_single)
    
    # For binary classification, might be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Create waterfall plot
    plt.figure(figsize=(10, 6))
    
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value if isinstance(explainer.expected_value, float) else explainer.expected_value[1],
            data=X_single[0],
            feature_names=feature_names
        ),
        show=False
    )
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Waterfall plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_waterfall_plots(
    explainer: shap.Explainer,
    X_test: np.ndarray,
    y_test: np.ndarray = None,
    y_pred: np.ndarray = None,
    feature_names: List[str] = None,
    n_examples: int = 3,
    output_dir: str = None,
    show: bool = False
):
    """
    Create waterfall plots for multiple example predictions.
    
    Selects examples to show different types of predictions:
    - Correct positive prediction
    - Correct negative prediction  
    - Incorrect prediction (if available)
    
    Args:
        explainer: SHAP explainer object
        X_test: Test data
        y_test: True labels (optional, for selecting examples)
        y_pred: Predicted labels (optional, for selecting examples)
        feature_names: Feature names
        n_examples: Number of examples to create (default 3)
        output_dir: Directory to save plots
        show: Whether to display plots
    
    Returns:
        List of indices of explained samples
    """
    print(f"\nCreating {n_examples} waterfall plot examples...")
    
    # Select example indices
    indices = []
    
    if y_test is not None and y_pred is not None:
        # Try to get diverse examples
        correct_pos = np.where((y_test == 1) & (y_pred == 1))[0]
        correct_neg = np.where((y_test == 0) & (y_pred == 0))[0]
        incorrect = np.where(y_test != y_pred)[0]
        
        # Select one from each category if possible
        if len(correct_pos) > 0:
            indices.append(np.random.choice(correct_pos))
        if len(correct_neg) > 0:
            indices.append(np.random.choice(correct_neg))
        if len(incorrect) > 0 and len(indices) < n_examples:
            indices.append(np.random.choice(incorrect))
        
        # Fill remaining with random samples
        while len(indices) < n_examples:
            idx = np.random.randint(0, len(X_test))
            if idx not in indices:
                indices.append(idx)
    else:
        # Random selection
        indices = np.random.choice(len(X_test), min(n_examples, len(X_test)), replace=False)
    
    # Create waterfall plots
    for i, idx in enumerate(indices):
        if output_dir:
            output_path = os.path.join(output_dir, f'waterfall_example_{i+1}.png')
        else:
            output_path = None
        
        explain_single_prediction(
            explainer,
            X_test[idx],
            feature_names=feature_names,
            output_path=output_path,
            show=show
        )
        
        if y_test is not None and y_pred is not None:
            true_label = y_test[idx]
            pred_label = y_pred[idx]
            print(f"  Example {i+1}: True={true_label}, Pred={pred_label}, " + 
                  ("✓ Correct" if true_label == pred_label else "✗ Incorrect"))
    
    print(f"✓ Created {len(indices)} waterfall plots")
    return indices

if __name__ == "__main__":
    # Test with dummy data
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    feature_names = [f"Feature_{i}" for i in range(20)]
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X[:800], y[:800])
    
    # Test SHAP
    explainer, shap_values = explain_model_shap(
        model, X[800:], feature_names, max_samples=100
    )
    
    # Test plots
    plot_shap_summary(shap_values, X[800:900], feature_names, show=False)
    plot_shap_bar(shap_values, feature_names, show=False)
    
    # Test feature importance
    importance, features = get_feature_importance_from_shap(shap_values, feature_names)
    print("\nTop 10 features by SHAP:")
    for feat, imp in zip(features[:10], importance[:10]):
        print(f"  {feat}: {imp:.4f}")
