"""Evaluation utilities for model performance, fairness, and explainability."""

from .metrics import evaluate_model, calculate_metrics
from .fairness import evaluate_fairness, mitigate_bias
from .explainability import explain_model_shap, plot_shap_summary

__all__ = [
    'evaluate_model',
    'calculate_metrics',
    'evaluate_fairness',
    'mitigate_bias',
    'explain_model_shap',
    'plot_shap_summary',
    'create_waterfall_plots',
]
