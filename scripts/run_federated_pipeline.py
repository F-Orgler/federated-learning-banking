#!/usr/bin/env python3
# scripts/run_fl_experiments.py
"""
Federated Learning Experiments - Business Case Analysis

Compares:
- Centralized Baseline (all data together)
- Federated Learning (3 aggregation methods)
- Local Models (each bank trains only on own data)

Shows: Is FL worth the cost and risk for each bank?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from typing import Dict, List, Tuple
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data import load_all_banks, preprocess_data
from src.training.federated import (
    FederatedRandomForestClient,
    create_federated_ensemble
)
from src.evaluation import evaluate_fairness
from src.utils import load_config

# Convert numpy types to Python types
def convert_numpy_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(i) for i in obj]
    return obj

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def split_data_by_bank(df: pd.DataFrame, test_size: float = 0.2, 
                       random_state: int = 42) -> Dict[str, Tuple]:
    """Split data into train/test sets per bank."""
    bank_splits = {}
    
    for bank in df['institute'].unique():
        bank_df = df[df['institute'] == bank].copy()
        bank_df = bank_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        split_idx = int(len(bank_df) * (1 - test_size))
        train_df = bank_df.iloc[:split_idx]
        test_df = bank_df.iloc[split_idx:]
        
        bank_splits[bank] = (train_df, test_df)
        print(f"  {bank}: Train={len(train_df)}, Test={len(test_df)}")
    
    return bank_splits


def prepare_federated_data(bank_splits: Dict[str, Tuple], 
                          missing_strategy: str = 'combined') -> Tuple[Dict, Dict, Dict]:
    """Preprocess data - each bank keeps its own test set."""
    client_data = {}
    test_data_per_bank = {}
    
    # Fit preprocessing on ALL training data
    print("\nFitting preprocessing on combined training data...")
    all_train_dfs = [train_df for train_df, _ in bank_splits.values()]
    combined_train_df = pd.concat(all_train_dfs, ignore_index=True)
    
    X_train_combined, y_train_combined, artifacts = preprocess_data(
        combined_train_df,
        missing_strategy=missing_strategy,
        fit_transform=True
    )
    
    print(f"✓ Preprocessing fitted on {len(X_train_combined)} samples")
    print(f"✓ Feature dimension: {X_train_combined.shape[1]}")
    
    # Transform each bank's data
    for bank, (train_df, test_df) in bank_splits.items():
        print(f"\nProcessing {bank}...")
        
        X_train, y_train, _ = preprocess_data(
            train_df, missing_strategy=missing_strategy,
            scaler=artifacts['scaler'], encoder=artifacts['encoder'],
            fit_transform=False
        )
        
        X_test, y_test, _ = preprocess_data(
            test_df, missing_strategy=missing_strategy,
            scaler=artifacts['scaler'], encoder=artifacts['encoder'],
            fit_transform=False
        )
        
        client_data[bank] = (X_train, y_train, X_test, y_test)
        test_data_per_bank[bank] = (X_test, y_test, len(X_test))
        
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    return client_data, test_data_per_bank, artifacts


def load_baseline_results(baseline_results_path: str) -> Dict:
    """Load baseline model metrics."""
    try:
        with open(baseline_results_path, 'r') as f:
            data = json.load(f)
        
        if 'baseline' in data:
            return data['baseline']
        elif 'metrics' in data:
            return data['metrics']
        else:
            return data
            
    except FileNotFoundError:
        print(f"⚠ Warning: Baseline not found")
        return None


def train_local_models(client_data: Dict, model_params: dict, 
                       verbose: bool = True) -> Dict:
    """
    Train local models - each bank only on own data.
    This is the "no FL" baseline.
    """
    if verbose:
        print("\n" + "="*80)
        print("TRAINING LOCAL MODELS (No Federation)")
        print("="*80)
    
    local_results = {}
    
    for bank, (X_train, y_train, X_test, y_test) in client_data.items():
        if verbose:
            print(f"\n{bank}: Training on {len(X_train)} samples...")
        
        # Train local model
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        
        # Evaluate on own test set
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0,
            'test_size': len(X_test),
            'model': model
        }
        
        local_results[bank] = metrics
        
        if verbose:
            print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    
    return local_results


def train_federated_ensemble(client_data: Dict, model_params: dict,
                             aggregation_method: str, local_results: Dict = None,
                             verbose: bool = True) -> Tuple:
    """Train federated ensemble with specified aggregation method."""
    if verbose:
        print(f"\n{'='*80}")
        print(f"FEDERATED TRAINING ({aggregation_method.upper()} Aggregation)")
        print("="*80)
    
    # Train client models
    clients = []
    for bank, (X_train, y_train, X_test, y_test) in client_data.items():
        client = FederatedRandomForestClient(
            client_id=bank, X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test, model_params=model_params
        )
        client.train_local_model(verbose=verbose)
        clients.append(client)
    
    # Create ensemble with specified aggregation
    local_accuracies = None
    if aggregation_method == 'performance' and local_results:
        local_accuracies = [local_results[client.client_id]['accuracy'] for client in clients]
    
    ensemble = create_federated_ensemble(
        clients, 
        aggregation_method=aggregation_method,
        local_accuracies=local_accuracies
    )
    
    return ensemble, clients


def evaluate_ensemble_per_bank(ensemble, test_data_per_bank: Dict,
                               verbose: bool = True) -> Dict:
    """Evaluate ensemble on each bank's test set."""
    per_bank_metrics = {}
    all_metrics = []
    all_weights = []
    
    if verbose:
        print(f"\nEvaluating ensemble per bank...")
    
    for bank, (X_test, y_test, size) in test_data_per_bank.items():
        y_pred = ensemble.predict(X_test)
        y_proba = ensemble.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0,
            'test_size': size
        }
        
        per_bank_metrics[bank] = metrics
        all_metrics.append(metrics)
        all_weights.append(size)
        
        if verbose:
            print(f"  {bank}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    # Weighted average
    total_weight = sum(all_weights)
    weighted_avg = {
        'accuracy': sum(m['accuracy'] * w for m, w in zip(all_metrics, all_weights)) / total_weight,
        'precision': sum(m['precision'] * w for m, w in zip(all_metrics, all_weights)) / total_weight,
        'recall': sum(m['recall'] * w for m, w in zip(all_metrics, all_weights)) / total_weight,
        'f1': sum(m['f1'] * w for m, w in zip(all_metrics, all_weights)) / total_weight,
        'roc_auc': sum(m['roc_auc'] * w for m, w in zip(all_metrics, all_weights)) / total_weight,
    }
    
    return {'per_bank': per_bank_metrics, 'weighted_average': weighted_avg}

# NEUE FUNKTION
def apply_precomputed_thresholds(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensitive_feature: np.ndarray,
    threshold_privileged: float,
    threshold_unprivileged: float
) -> Dict:
    """
    Apply precomputed thresholds directly (no optimization needed).
    
    Args:
        model: Trained model (sklearn or FL ensemble)
        X_test: Test features
        y_test: True labels
        sensitive_feature: Sensitive attribute (e.g., gender_test)
        threshold_privileged: Threshold for privileged group (e.g., Male)
        threshold_unprivileged: Threshold for unprivileged group (e.g., Female)
    
    Returns:
        Dict with predictions and metrics (same format as optimize_threshold_fairness)
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Get probabilities
    y_proba_raw = model.predict_proba(X_test)
    
    # Handle different predict_proba formats:
    # - sklearn: (n_samples, 2) - column 0 = class 0, column 1 = class 1
    # - FL ensemble: (n_samples,) - directly probabilities for class 1
    if y_proba_raw.ndim == 2:
        # sklearn format
        y_proba = y_proba_raw[:, 1]
    else:
        # FL ensemble format (already 1D)
        y_proba = y_proba_raw
    
    # Apply different thresholds per group
    y_pred = np.zeros(len(y_test), dtype=int)
    privileged_mask = (sensitive_feature == 1)
    unprivileged_mask = (sensitive_feature == 0)
    
    y_pred[privileged_mask] = (y_proba[privileged_mask] >= threshold_privileged).astype(int)
    y_pred[unprivileged_mask] = (y_proba[unprivileged_mask] >= threshold_unprivileged).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    return {
        'y_pred': y_pred,
        'threshold_privileged': threshold_privileged,
        'threshold_unprivileged': threshold_unprivileged,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_business_case_comparison(baseline_metrics: Dict, 
                                  fl_results: Dict[str, Dict],
                                  local_results: Dict,
                                  output_path: str):
    """
    Create comprehensive business case comparison plot.
    
    Shows: Centralized vs FL (3 methods) vs Local (per bank)
    """
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Prepare data
    model_names = ['Centralized\nBaseline']
    all_values = {}
    
    for metric in metrics_to_plot:
        values = [baseline_metrics.get(metric, 0) if baseline_metrics else 0]
        
        # FL methods
        for fl_name, fl_data in fl_results.items():
            if metric == metrics_to_plot[0]:  # Only add name once
                model_names.append(f'FL\n{fl_name}')
            values.append(fl_data['weighted_average'].get(metric, 0))
        
        # Local models
        for bank, local_data in local_results.items():
            if metric == metrics_to_plot[0]:  # Only add name once
                model_names.append(f'Local\n{bank}')
            values.append(local_data.get(metric, 0))
        
        all_values[metric] = values
    
    # Create plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    x = np.arange(len(model_names))
    width = 0.6
    
    colors = ['#2ecc71'] + ['#3498db']*3 + ['#e74c3c']*3  # Green=Baseline, Blue=FL, Red=Local
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = all_values[metric]
        
        bars = ax.bar(x, values, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel(metric.upper().replace('_', ' '), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.upper().replace("_", " ")}', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=9, rotation=0)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0.8, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Remove last subplot (we have 5 metrics, 6 subplots)
    fig.delaxes(axes[5])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Centralized (All Data)'),
        Patch(facecolor='#3498db', label='Federated Learning'),
        Patch(facecolor='#e74c3c', label='Local Only (Own Data)')
    ]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=11, frameon=True)
    
    plt.suptitle('Business Case Analysis: Is Federated Learning Worth It?', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Business case plot saved: {output_path}")


def evaluate_fairness_all_models(baseline_metrics, fl_results, local_results,
                                 client_data, bank_splits, output_dir):
    """Evaluate fairness for all models."""
    print("\n" + "="*80)
    print("FAIRNESS EVALUATION (All Models)")
    print("="*80)
    
    # Combine all test data for fairness evaluation
    all_test_dfs = [test_df for _, test_df in bank_splits.values()]
    combined_test_df = pd.concat(all_test_dfs, ignore_index=True)
    
    # Get sensitive features
    gender_map = {'Male': 1, 'Female': 0}
    gender_test = combined_test_df['gender'].map(gender_map).fillna(0).values
    race_test = np.where(combined_test_df['race'] == 'White', 1, 0)
    
    # Combined test data
    X_test_combined = np.vstack([X_test for _, _, X_test, _ in client_data.values()])
    y_test_combined = np.concatenate([y_test for _, _, _, y_test in client_data.values()])
    
    fairness_results = {}
    
    # Note: Baseline predictions not available here, would need to load model
    # FL predictions
    for fl_name, fl_data in fl_results.items():
        print(f"\nEvaluating fairness: FL {fl_name}...")
        ensemble = fl_data['ensemble']
        y_pred = ensemble.predict(X_test_combined)
        
        try:
            fairness_gender = evaluate_fairness(
                y_test_combined, y_pred,
                sensitive_features={'gender': gender_test},
                privileged_groups=[{'gender': 1}],
                unprivileged_groups=[{'gender': 0}]
            )
            fairness_results[f'FL_{fl_name}_gender'] = fairness_gender
            print(f"  Disparate Impact: {fairness_gender.get('disparate_impact', 0):.4f}")
        except Exception as e:
            print(f"  Warning: {e}")
    
    # Save fairness results
    fairness_path = os.path.join(output_dir, 'fairness_results.json')
    with open(fairness_path, 'w') as f:        
        json.dump(convert_numpy_to_python(fairness_results), f, indent=2)
    
    return fairness_results


def run_experiments(config_path: str):
    """Main experiment runner."""
    print("\n" + "="*80)
    print("FEDERATED LEARNING BUSINESS CASE ANALYSIS")
    print("="*80)
    
    config = load_config(config_path)
    print(f"✓ Loaded config")
    
    # Create directories
    output_dir = config['output']['models_dir']
    plots_dir = config['output']['plots_dir']
    for directory in [output_dir, plots_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Load data
    print("\n" + "="*80)
    print("DATA PREPARATION")
    print("="*80)
    
    df, _ = load_all_banks(
        banks=config['data']['banks'],
        data_path=config['data']['raw_path']
    )
    print(f"✓ Loaded {len(df)} samples")
    
    # Split by bank
    print("\nSplitting data by bank...")
    bank_splits = split_data_by_bank(df, test_size=config['data']['test_size'],
                                    random_state=config['data']['random_state'])
    
    # Preprocess
    client_data, test_data_per_bank, artifacts = prepare_federated_data(
        bank_splits, missing_strategy='combined'
    )
    
    # Load baseline
    baseline_metrics = load_baseline_results('results/reports/baseline_metrics.json')
    if baseline_metrics:
        print(f"\n✓ Baseline: Acc={baseline_metrics.get('accuracy', 0):.4f}, "
              f"F1={baseline_metrics.get('f1', 0):.4f}")
    
    # Model parameters (from baseline tuning)
    model_params = {
        'n_estimators': 300, 'max_depth': 30, 'min_samples_split': 5,
        'min_samples_leaf': 1, 'max_features': 'sqrt', 
        'random_state': 42, 'n_jobs': -1
    }

    # 1. Train local models (no FL)
    local_results = train_local_models(client_data, model_params, verbose=True)
    
    # 2. Train FL ensembles (3 aggregation methods)
    aggregation_methods = ['weighted', 'uniform', 'performance']
    fl_results = {}
    
    for method in aggregation_methods:
        ensemble, clients = train_federated_ensemble(
            client_data, model_params, method, local_results, verbose=True
        )
        
        # Evaluate
        eval_results = evaluate_ensemble_per_bank(
            ensemble, test_data_per_bank, verbose=True
        )
        
        fl_results[method] = {
            'ensemble': ensemble,
            'weighted_average': eval_results['weighted_average'],
            'per_bank': eval_results['per_bank']
        }
        
        # Save ensemble
        ensemble_dir = os.path.join(output_dir, f'fed_ensemble_{method}')
        ensemble.save(ensemble_dir)
    
    # 3. Create business case comparison plot
    print("\n" + "="*80)
    print("CREATING BUSINESS CASE COMPARISON")
    print("="*80)
    
    # Original plot (without threshold optimization)
    plot_path_original = os.path.join(plots_dir, 'business_case_comparison_original.png')
    plot_business_case_comparison(
        baseline_metrics, fl_results, local_results, plot_path_original
    )
    print(f"\n✓ Original plot saved: {plot_path_original}")
    
    # ==============================================================================
    # THRESHOLD OPTIMIZATION (für alle Modelle)
    # ==============================================================================
    print("\n" + "="*80)
    print("APPLYING THRESHOLD OPTIMIZATION")
    print("="*80)
    
    # Check if threshold optimization is enabled
    fairness_config = config.get('fairness', {})
    threshold_config = fairness_config.get('threshold_optimization', {})
    
    if not threshold_config.get('enabled', False):
        print("⚠ Threshold optimization disabled in config")
        print("  Only original plot created")
    else:
        # Prepare gender_test for all banks combined
        print("\nExtracting gender features from test data...")
        all_test_dfs = []
        for bank, (_, test_df) in bank_splits.items():
            all_test_dfs.append(test_df)
        combined_test_df = pd.concat(all_test_dfs, ignore_index=True)
        gender_test = combined_test_df['gender'].map({'Male': 1, 'Female': 0}).fillna(0).values
        
        # Get combined test data
        all_X_test = []
        all_y_test = []
        for bank, (X_test, y_test, _) in test_data_per_bank.items():
            all_X_test.append(X_test)
            all_y_test.append(y_test)
        X_test_combined = np.vstack(all_X_test)
        y_test_combined = np.concatenate(all_y_test)
        
        print(f"✓ Test data: {X_test_combined.shape[0]} samples")
        print(f"✓ Gender distribution: Male={np.sum(gender_test==1)}, Female={np.sum(gender_test==0)}")
        
        use_precomputed = threshold_config.get('use_precomputed', False)
        
        # Import threshold function
        try:
            from src.evaluation.fairness import optimize_threshold_fairness
        except ImportError:
            print("✗ Could not import optimize_threshold_fairness")
            print("  Skipping threshold optimization")
            use_precomputed = False
            threshold_config['enabled'] = False
        
        if threshold_config.get('enabled', False):
            # Store optimized metrics
            baseline_metrics_opt = baseline_metrics.copy() if baseline_metrics else None
            fl_results_opt = {}
            local_results_opt = {}
            
            # 1. Optimize Centralized Baseline (if available)
            if baseline_metrics:
                print("\n--- Baseline Model ---")
                # Load baseline model
                baseline_model_path = 'results/models/baseline_random_forest.joblib'
                if os.path.exists(baseline_model_path):
                    import joblib
                    baseline_model = joblib.load(baseline_model_path)
                    
                    if use_precomputed:
                        print("Using precomputed thresholds...")
                        precomputed = threshold_config['precomputed_thresholds']['gender']
                        threshold_results = apply_precomputed_thresholds(
                            baseline_model, X_test_combined, y_test_combined, gender_test,
                            threshold_privileged=precomputed['privileged'],
                            threshold_unprivileged=precomputed['unprivileged']
                        )
                        print(f"  ✓ Applied: {precomputed['privileged']:.3f} / {precomputed['unprivileged']:.3f}")
                    else:
                        print("Optimizing thresholds...")
                        threshold_results = optimize_threshold_fairness(
                            baseline_model, X_test_combined, y_test_combined, gender_test,
                            objective='equalized_odds',
                            min_recall=threshold_config.get('min_recall', 0.6),
                            disparate_impact_target=threshold_config.get('disparate_impact_target', 0.8)
                        )
                    
                    # Update metrics
                    baseline_metrics_opt = {
                        'accuracy': threshold_results['accuracy'],
                        'precision': threshold_results['precision'],
                        'recall': threshold_results['recall'],
                        'f1': threshold_results['f1'],
                        'roc_auc': baseline_metrics.get('roc_auc', 0)  # Keep original
                    }
                    print(f"  Recall: {baseline_metrics['recall']:.4f} → {threshold_results['recall']:.4f}")
                else:
                    print("  ⚠ Baseline model not found, using original metrics")
                    baseline_metrics_opt = baseline_metrics
            
            # 2. Optimize FL Ensembles
            for method, fl_data in fl_results.items():
                print(f"\n--- FL {method.capitalize()} ---")
                ensemble = fl_data['ensemble']
                
                if use_precomputed:
                    print("Using precomputed thresholds...")
                    precomputed = threshold_config['precomputed_thresholds']['gender']
                    threshold_results = apply_precomputed_thresholds(
                        ensemble, X_test_combined, y_test_combined, gender_test,
                        threshold_privileged=precomputed['privileged'],
                        threshold_unprivileged=precomputed['unprivileged']
                    )
                    print(f"  ✓ Applied: {precomputed['privileged']:.3f} / {precomputed['unprivileged']:.3f}")
                else:
                    print("Optimizing thresholds...")
                    threshold_results = optimize_threshold_fairness(
                        ensemble, X_test_combined, y_test_combined, gender_test,
                        objective='equalized_odds',
                        min_recall=threshold_config.get('min_recall', 0.6),
                        disparate_impact_target=threshold_config.get('disparate_impact_target', 0.8)
                    )
                
                # Update weighted average metrics
                fl_results_opt[method] = {
                    'weighted_average': {
                        'accuracy': threshold_results['accuracy'],
                        'precision': threshold_results['precision'],
                        'recall': threshold_results['recall'],
                        'f1': threshold_results['f1'],
                        'roc_auc': fl_data['weighted_average'].get('roc_auc', 0)
                    },
                    'per_bank': fl_data['per_bank']  # Keep original per-bank
                }
                print(f"  Recall: {fl_data['weighted_average']['recall']:.4f} → {threshold_results['recall']:.4f}")
            
            # 3. Optimize Local Models
            for bank, local_data in local_results.items():
                print(f"\n--- Local {bank} ---")
                local_model = local_data['model']
                
                if use_precomputed:
                    print("Using precomputed thresholds...")
                    precomputed = threshold_config['precomputed_thresholds']['gender']
                    threshold_results = apply_precomputed_thresholds(
                        local_model, X_test_combined, y_test_combined, gender_test,
                        threshold_privileged=precomputed['privileged'],
                        threshold_unprivileged=precomputed['unprivileged']
                    )
                    print(f"  ✓ Applied: {precomputed['privileged']:.3f} / {precomputed['unprivileged']:.3f}")
                else:
                    print("Optimizing thresholds...")
                    threshold_results = optimize_threshold_fairness(
                        local_model, X_test_combined, y_test_combined, gender_test,
                        objective='equalized_odds',
                        min_recall=threshold_config.get('min_recall', 0.6),
                        disparate_impact_target=threshold_config.get('disparate_impact_target', 0.8)
                    )
                
                # Update metrics
                local_results_opt[bank] = {
                    'accuracy': threshold_results['accuracy'],
                    'precision': threshold_results['precision'],
                    'recall': threshold_results['recall'],
                    'f1': threshold_results['f1'],
                    'roc_auc': local_data.get('roc_auc', 0)
                }
                print(f"  Recall: {local_data['recall']:.4f} → {threshold_results['recall']:.4f}")
            
            # Create optimized plot
            print("\n" + "="*80)
            print("CREATING OPTIMIZED PLOT")
            print("="*80)
            
            plot_path_optimized = os.path.join(plots_dir, 'business_case_comparison_optimized.png')
            plot_business_case_comparison(
                baseline_metrics_opt, fl_results_opt, local_results_opt, plot_path_optimized
            )
            print(f"\n✓ Optimized plot saved: {plot_path_optimized}")
            
            print("\n✓ Created 2 plots:")
            print(f"  1. {plot_path_original}")
            print(f"  2. {plot_path_optimized}")
    
    # Keep original plot_path for backward compatibility
    plot_path = plot_path_original
    
    # 4. Evaluate fairness
    fairness_results = evaluate_fairness_all_models(
        baseline_metrics, fl_results, local_results,
        client_data, bank_splits, output_dir
    )
    
    # 5. Save all results
    results = {
        'baseline': baseline_metrics,
        'local_models': {k: {m: v for m, v in v.items() if m != 'model'} 
                        for k, v in local_results.items()},
        'federated': {k: {'weighted_average': v['weighted_average'], 
                         'per_bank': v['per_bank']}
                     for k, v in fl_results.items()},
        'fairness': fairness_results
    }
    
    results_path = os.path.join(output_dir, 'complete_results.json')
    with open(results_path, 'w') as f:
        
        json.dump(convert_numpy_to_python(results), f, indent=2)
    
    print(f"\n✓ Results saved: {results_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("BUSINESS CASE SUMMARY")
    print("="*80)
    print(f"\nCentralized Baseline:     Acc={baseline_metrics.get('accuracy', 0):.4f}")
    print(f"\nFederated Learning:")
    for method, data in fl_results.items():
        print(f"  {method.capitalize():12s}: Acc={data['weighted_average']['accuracy']:.4f}")
    print(f"\nLocal Models (No FL):")
    for bank, data in local_results.items():
        print(f"  {bank:12s}: Acc={data['accuracy']:.4f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"📊 Plot: {plot_path}")
    print(f"📁 Results: {results_path}")


def main():
    parser = argparse.ArgumentParser(description='FL Business Case Analysis')
    parser.add_argument('--config', type=str, 
                       default='experiments/federated_config.yaml')
    args = parser.parse_args()
    
    try:
        run_experiments(args.config)
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
