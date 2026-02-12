"""
Visualization utilities for plots and charts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str] = None,
    title: str = "Confusion Matrix",
    output_path: str = None,
    show: bool = True
):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix
        labels: Class labels
        title: Plot title
        output_path: Path to save plot
        show: Whether to display plot
    """
    if labels is None:
        labels = ["<=50K", ">50K"]
    
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_metrics_comparison(
    results_dict: Dict[str, Dict],
    metrics: List[str] = None,
    title: str = "Model Comparison",
    output_path: str = None,
    show: bool = True
):
    """
    Plot bar chart comparing metrics across models.
    
    Args:
        results_dict: Dictionary of {model_name: metrics_dict}
        metrics: List of metrics to plot
        title: Plot title
        output_path: Path to save plot
        show: Whether to display plot
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Prepare data
    models = list(results_dict.keys())
    data = {metric: [] for metric in metrics}
    
    for model_name in models:
        model_metrics = results_dict[model_name].get('metrics', results_dict[model_name])
        for metric in metrics:
            data[metric].append(model_metrics.get(metric, 0))
    
    # Create plot
    x = np.arange(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        offset = width * (i - len(metrics)/2 + 0.5)
        ax.bar(x + offset, data[metric], width, label=metric.capitalize())
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Metrics comparison saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_fairness_comparison(
    fairness_results: Dict[str, Dict],
    title: str = "Fairness Comparison",
    output_path: str = None,
    show: bool = True
):
    """
    Plot fairness metrics comparison.
    
    Args:
        fairness_results: Dictionary of {model_name: fairness_metrics}
        title: Plot title
        output_path: Path to save plot
        show: Whether to display plot
    """
    metrics_to_plot = [
        'disparate_impact',
        'statistical_parity_difference',
        'equal_opportunity_difference',
        'average_odds_difference'
    ]
    
    # Prepare data
    models = list(fairness_results.keys())
    data = {metric: [] for metric in metrics_to_plot}
    
    for model_name in models:
        for metric in metrics_to_plot:
            data[metric].append(fairness_results[model_name].get(metric, 0))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        colors = ['green' if abs(v) < 0.1 else 'orange' if abs(v) < 0.2 else 'red' for v in data[metric]]
        
        ax.bar(models, data[metric], color=colors, alpha=0.7)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        # Add threshold lines for some metrics
        if 'difference' in metric:
            ax.axhline(y=0.1, color='orange', linestyle=':', linewidth=1, alpha=0.5)
            ax.axhline(y=-0.1, color='orange', linestyle=':', linewidth=1, alpha=0.5)
        elif metric == 'disparate_impact':
            ax.axhline(y=0.8, color='orange', linestyle=':', linewidth=1, alpha=0.5)
        
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(models, rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Fairness comparison saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_history(
    history: Dict[str, List],
    title: str = "Training History",
    output_path: str = None,
    show: bool = True
):
    """
    Plot training history (loss, accuracy over rounds/epochs).
    
    Args:
        history: Dictionary with 'loss', 'accuracy', etc. as keys
        title: Plot title
        output_path: Path to save plot
        show: Whether to display plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Loss plot
    if 'loss' in history:
        axes[0].plot(history['loss'], marker='o', label='Loss')
        axes[0].set_xlabel('Round/Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss over Time')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    
    # Accuracy/F1 plot
    ax1 = axes[1]
    if 'accuracy' in history:
        ax1.plot(history['accuracy'], marker='o', label='Accuracy', color='blue')
    if 'f1' in history:
        ax1.plot(history['f1'], marker='s', label='F1-Score', color='green')
    
    ax1.set_xlabel('Round/Epoch')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Test visualizations
    
    # Test confusion matrix
    cm = np.array([[80, 20], [15, 85]])
    plot_confusion_matrix(cm, show=False)
    
    # Test metrics comparison
    results = {
        'Model A': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'f1': 0.85},
        'Model B': {'accuracy': 0.87, 'precision': 0.84, 'recall': 0.90, 'f1': 0.87},
        'Model C': {'accuracy': 0.83, 'precision': 0.80, 'recall': 0.85, 'f1': 0.82},
    }
    plot_metrics_comparison(results, show=False)
    
    print("Visualization utility test complete!")
