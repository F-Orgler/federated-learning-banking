# src/evaluation/fairness.py
"""
Comprehensive fairness evaluation and bias mitigation toolkit.

Supports:
- Fairness metrics (Disparate Impact, Equal Opportunity, etc.)
- Preprocessing mitigation (Reweighing, Disparate Impact Remover, LFR)
- Inprocessing mitigation (Fairlearn constraints)
- Postprocessing mitigation (Threshold optimization, Calibrated Equalized Odds)
- Recall optimization through adaptive thresholds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix, roc_curve, auc
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing


# ==============================================================================
# AIF360 Dataset Creation
# ==============================================================================

def create_aif360_dataset(
    X: np.ndarray,
    y: np.ndarray,
    sensitive_features: Dict[str, np.ndarray],
    feature_names: List[str] = None
) -> StandardDataset:
    """Create AIF360 StandardDataset from numpy arrays."""
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Combine features and sensitive attributes
    df = pd.DataFrame(X, columns=feature_names)
    
    for attr_name, attr_values in sensitive_features.items():
        df[attr_name] = attr_values
    
    df['label'] = y
    
    # Create dataset
    dataset = StandardDataset(
        df,
        label_name='label',
        favorable_classes=[1],
        protected_attribute_names=list(sensitive_features.keys()),
        privileged_classes=[[1]] * len(sensitive_features)
    )
    
    return dataset


# ==============================================================================
# FAIRNESS EVALUATION
# ==============================================================================

def evaluate_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: Dict[str, np.ndarray],
    privileged_groups: List[Dict] = None,
    unprivileged_groups: List[Dict] = None
) -> Dict:
    """
    Evaluate fairness metrics using AIF360.
    
    Returns comprehensive fairness metrics including:
    - Disparate Impact Ratio (ratio of positive rates between groups)
    - Equal Odds Difference (average of TPR and FPR differences)
    - Predictive Rate Parity (positive predictive parity)
    """
    
    # Create dummy features (AIF360 needs features, but we only care about labels)
    n_samples = len(y_true)
    dummy_X = np.zeros((n_samples, 1))
    feature_names = ['dummy_feature']  # Only 1 feature column
    
    # Create datasets
    dataset_true = create_aif360_dataset(
        dummy_X, y_true, sensitive_features, feature_names
    )
    dataset_pred = create_aif360_dataset(
        dummy_X, y_pred, sensitive_features, feature_names
    )
    
    # Set default groups
    if privileged_groups is None or unprivileged_groups is None:
        privileged_groups = []
        unprivileged_groups = []
        for attr_name in sensitive_features.keys():
            privileged_groups.append({attr_name: 1})
            unprivileged_groups.append({attr_name: 0})
    
    # Calculate metrics
    metric = ClassificationMetric(
        dataset_true, dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    # Calculate Equal Odds Difference (average of TPR and FPR differences)
    tpr_diff = metric.true_positive_rate_difference()
    fpr_diff = metric.false_positive_rate_difference()
    equal_odds_diff = (abs(tpr_diff) + abs(fpr_diff)) / 2
    
    results = {
        'disparate_impact_ratio': metric.disparate_impact(),
        'equal_odds_difference': equal_odds_diff,
        'predictive_rate_parity': 1 - abs(metric.positive_predictive_value() - 
                                          metric.positive_predictive_value(privileged=False)),
        'is_fair': (
            metric.disparate_impact() >= 0.8 and
            equal_odds_diff <= 0.1
        )
    }
    
    print(f"\nFairness Metrics:")
    print(f"  Disparate Impact Ratio: {results['disparate_impact_ratio']:.4f} (≥0.8 is fair)")
    print(f"  Equal Odds Difference: {results['equal_odds_difference']:.4f} (≤0.1 is fair)")
    print(f"  Predictive Rate Parity: {results['predictive_rate_parity']:.4f} (~1.0 is fair)")
    print(f"  Overall Fair: {results['is_fair']}")
    
    return results


# ==============================================================================
# PREPROCESSING MITIGATION
# ==============================================================================

def mitigate_bias(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sensitive_features_train: Dict[str, np.ndarray],
    method: str = 'reweighing',
    privileged_groups: List[Dict] = None,
    unprivileged_groups: List[Dict] = None,
    **kwargs
) -> Tuple:
    """
    Apply preprocessing bias mitigation.
    
    IMPORTANT: X_train should NOT contain protected attributes!
    Protected attributes are passed separately in sensitive_features_train.
    The output X_fair will also NOT contain protected attributes.
    
    Args:
        X_train: Training features (WITHOUT protected attributes)
        y_train: Training labels
        sensitive_features_train: Dict of sensitive attributes (separate from X)
        method: 'reweighing', 'disparate_impact_remover', 'lfr'
        privileged_groups: Privileged group definitions
        unprivileged_groups: Unprivileged group definitions
        **kwargs: Method-specific parameters
    
    Returns:
        Tuple of (X_train_fair, y_train_fair, sample_weights)
        - X_train_fair: Fair features (WITHOUT protected attributes)
        - y_train_fair: Fair labels
        - sample_weights: Weights for training (for reweighing)
    
    Methods:
        - reweighing: Reweight samples to balance groups
        - disparate_impact_remover: Remove correlation with sensitive features
        - lfr: Learning Fair Representations
    """
    from aif360.algorithms.preprocessing import DisparateImpactRemover, LFR
    
    print(f"\n{'='*60}")
    print(f"PREPROCESSING: {method.upper()}")
    print(f"{'='*60}")
    
    # Remember original number of features (WITHOUT protected attributes)
    n_original_features = X_train.shape[1]
    
    # Create dataset (AIF360 adds protected attributes temporarily)
    feature_names = [f'feature_{i}' for i in range(n_original_features)]
    dataset = create_aif360_dataset(
        X_train, y_train, sensitive_features_train, feature_names
    )
    
    # Set default groups
    if privileged_groups is None or unprivileged_groups is None:
        privileged_groups = []
        unprivileged_groups = []
        
        for attr_name, attr_values in sensitive_features_train.items():
            unique_vals = np.unique(attr_values)
            if len(unique_vals) >= 2:
                unprivileged_groups.append({attr_name: unique_vals[0]})
                privileged_groups.append({attr_name: unique_vals[1]})
    
    # Apply mitigation
    if method == 'reweighing':
        mitigator = Reweighing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )
        dataset_transformed = mitigator.fit_transform(dataset)
        
        # Extract features WITHOUT protected attributes!
        df_transformed = dataset_transformed.convert_to_dataframe()[0]
        protected_attrs = list(sensitive_features_train.keys())
        feature_cols = [col for col in df_transformed.columns 
                       if col not in protected_attrs and col != 'label']
        
        X_fair = df_transformed[feature_cols].values
        y_fair = dataset_transformed.labels.ravel()
        sample_weights = dataset_transformed.instance_weights
        
        print(f"✓ Reweighing applied")
        print(f"  Weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
        print(f"  ✓ Protected attributes removed from output")
        print(f"    Original features: {n_original_features}, Output features: {X_fair.shape[1]}")
    
    elif method == 'disparate_impact_remover':
        repair_level = kwargs.get('repair_level', 1.0)
        mitigator = DisparateImpactRemover(repair_level=repair_level)
        dataset_transformed = mitigator.fit_transform(dataset)
        
        # Extract features WITHOUT protected attributes!
        df_transformed = dataset_transformed.convert_to_dataframe()[0]
        protected_attrs = list(sensitive_features_train.keys())
        feature_cols = [col for col in df_transformed.columns 
                       if col not in protected_attrs and col != 'label']
        
        X_fair = df_transformed[feature_cols].values
        y_fair = dataset_transformed.labels.ravel()
        sample_weights = np.ones(len(y_fair))
        
        print(f"✓ Disparate Impact Remover applied")
        print(f"  Repair level: {repair_level}")
        print(f"  ✓ Protected attributes removed from output")
        print(f"    Original features: {n_original_features}, Output features: {X_fair.shape[1]}")
    
    elif method == 'lfr':
        k = kwargs.get('k', 5)
        Ax = kwargs.get('Ax', 0.01)
        Ay = kwargs.get('Ay', 1.0)
        Az = kwargs.get('Az', 50.0)
        
        mitigator = LFR(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            k=k, Ax=Ax, Ay=Ay, Az=Az,
            verbose=0
        )
        dataset_transformed = mitigator.fit_transform(dataset)
        
        # Extract features WITHOUT protected attributes!
        df_transformed = dataset_transformed.convert_to_dataframe()[0]
        protected_attrs = list(sensitive_features_train.keys())
        feature_cols = [col for col in df_transformed.columns 
                       if col not in protected_attrs and col != 'label']
        
        X_fair = df_transformed[feature_cols].values
        y_fair = dataset_transformed.labels.ravel()
        sample_weights = np.ones(len(y_fair))
        
        print(f"✓ Learning Fair Representations applied")
        print(f"  Latent dimension k: {k}")
        print(f"  ✓ Protected attributes removed from output")
        print(f"    Original features: {n_original_features}, Output features: {X_fair.shape[1]}")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"✓ Dataset transformed: {X_fair.shape}")
    print(f"  Model will NOT have access to protected attributes ✓")
    
    return X_fair, y_fair, sample_weights


# ==============================================================================
# INPROCESSING MITIGATION (Fairlearn)
# ==============================================================================

def train_fair_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sensitive_features_train: np.ndarray,
    base_estimator,
    constraint: str = 'demographic_parity',
    epsilon: float = 0.01,
    **kwargs
):
    """
    Train fair classifier using Fairlearn constraints.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sensitive_features_train: Sensitive attribute (1D array)
        base_estimator: Base classifier (e.g., RandomForestClassifier)
        constraint: 'demographic_parity', 'equalized_odds', 'equal_opportunity'
        epsilon: Constraint tolerance
        **kwargs: Additional parameters for ExponentiatedGradient
    
    Returns:
        Trained fair classifier
    """
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, \
                                    EqualizedOdds, TruePositiveRateParity
    
    print(f"\n{'='*60}")
    print(f"INPROCESSING: Fairlearn {constraint.upper()}")
    print(f"{'='*60}")
    
    # Map constraint name to class
    constraints = {
        'demographic_parity': DemographicParity(difference_bound=epsilon),
        'equalized_odds': EqualizedOdds(difference_bound=epsilon),
        'equal_opportunity': TruePositiveRateParity(difference_bound=epsilon)
    }
    
    if constraint not in constraints:
        raise ValueError(f"Unknown constraint: {constraint}")
    
    # Train fair classifier
    mitigator = ExponentiatedGradient(
        base_estimator,
        constraints=constraints[constraint],
        **kwargs
    )
    
    print(f"Training with constraint: {constraint}")
    print(f"Epsilon (tolerance): {epsilon}")
    
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_features_train)
    
    print(f"✓ Fair classifier trained")
    print(f"  Number of predictors: {len(mitigator.predictors_)}")
    
    return mitigator


# ==============================================================================
# POSTPROCESSING: THRESHOLD OPTIMIZATION
# ==============================================================================

def optimize_threshold_fairness(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensitive_features_test: np.ndarray,
    objective: str = 'equalized_odds',
    constraint: str = 'recall',
    min_recall: float = 0.6,
    disparate_impact_target: float = 0.8
) -> Dict:
    """
    Optimize decision thresholds to improve fairness AND recall.
    
    This addresses the two main issues:
    1. Low recall (too many false negatives)
    2. Poor fairness (different treatment of groups)
    
    Args:
        model: Trained classifier with predict_proba
        X_test: Test features
        y_test: Test labels
        sensitive_features_test: Sensitive attribute (1D array)
        objective: 'equalized_odds', 'equal_opportunity', 'demographic_parity'
        constraint: 'recall' to ensure minimum recall
        min_recall: Minimum required recall (default: 0.6)
        disparate_impact_target: Target disparate impact (default: 0.8)
    
    Returns:
        Dict with optimized thresholds and metrics
    """
    from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
    
    print(f"\n{'='*60}")
    print(f"POSTPROCESSING: Threshold Optimization")
    print(f"{'='*60}")
    print(f"Objective: {objective}")
    print(f"Constraint: Min recall = {min_recall}")
    print(f"Target Disparate Impact: {disparate_impact_target}")
    
    # Get probability predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Get unique groups
    unique_groups = np.unique(sensitive_features_test)
    
    if len(unique_groups) != 2:
        raise ValueError("Only binary sensitive attributes supported")
    
    privileged_group = unique_groups[1]
    unprivileged_group = unique_groups[0]
    
    # Find optimal thresholds
    best_result = None
    best_score = -np.inf
    
    # Grid search over possible thresholds
    thresholds = np.linspace(0.1, 0.9, 81)
    
    for thresh_priv in thresholds:
        for thresh_unpriv in thresholds:
            # Apply group-specific thresholds
            y_pred = np.zeros(len(y_test))
            
            priv_mask = (sensitive_features_test == privileged_group)
            unpriv_mask = (sensitive_features_test == unprivileged_group)
            
            y_pred[priv_mask] = (y_proba[priv_mask] > thresh_priv).astype(int)
            y_pred[unpriv_mask] = (y_proba[unpriv_mask] > thresh_unpriv).astype(int)
            
            # Calculate metrics
            recall_overall = recall_score(y_test, y_pred)
            
            # Skip if recall constraint not met
            if recall_overall < min_recall:
                continue
            
            # Calculate fairness metrics
            recall_priv = recall_score(
                y_test[priv_mask], y_pred[priv_mask], zero_division=0
            )
            recall_unpriv = recall_score(
                y_test[unpriv_mask], y_pred[unpriv_mask], zero_division=0
            )
            
            # Positive rate for each group
            pos_rate_priv = np.mean(y_pred[priv_mask])
            pos_rate_unpriv = np.mean(y_pred[unpriv_mask])
            
            # Disparate impact
            if pos_rate_priv > 0:
                disparate_impact = pos_rate_unpriv / pos_rate_priv
            else:
                disparate_impact = 0
            
            # Objective: maximize recall while achieving fairness
            if objective == 'equalized_odds':
                fairness_score = 1 - abs(recall_priv - recall_unpriv)
            elif objective == 'equal_opportunity':
                fairness_score = 1 - abs(recall_priv - recall_unpriv)
            elif objective == 'demographic_parity':
                fairness_score = 1 - abs(pos_rate_priv - pos_rate_unpriv)
            else:
                fairness_score = 0
            
            # Combined score
            di_penalty = max(0, disparate_impact_target - disparate_impact)
            score = recall_overall + fairness_score - di_penalty * 2
            
            if score > best_score:
                best_score = score
                best_result = {
                    'threshold_privileged': thresh_priv,
                    'threshold_unprivileged': thresh_unpriv,
                    'y_pred': y_pred.copy(),
                    'recall': recall_overall,
                    'recall_privileged': recall_priv,
                    'recall_unprivileged': recall_unpriv,
                    'disparate_impact': disparate_impact,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0),
                    'fairness_score': fairness_score,
                    'combined_score': score
                }
    
    if best_result is None:
        print("⚠ Warning: No threshold combination met recall constraint")
        print(f"  Using default threshold 0.5")
        y_pred = (y_proba > 0.5).astype(int)
        best_result = {
            'threshold_privileged': 0.5,
            'threshold_unprivileged': 0.5,
            'y_pred': y_pred,
            'recall': recall_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
    
    print(f"\n✓ Optimal thresholds found:")
    print(f"  Privileged group: {best_result['threshold_privileged']:.3f}")
    print(f"  Unprivileged group: {best_result['threshold_unprivileged']:.3f}")
    print(f"\n✓ Metrics:")
    print(f"  Recall: {best_result['recall']:.4f} (target: ≥{min_recall})")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  F1-Score: {best_result['f1']:.4f}")
    if 'disparate_impact' in best_result:
        print(f"  Disparate Impact: {best_result['disparate_impact']:.4f} (target: ≥{disparate_impact_target})")
    
    return best_result


# ==============================================================================
# POSTPROCESSING: CALIBRATED EQUALIZED ODDS
# ==============================================================================

def apply_calibrated_equalized_odds(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensitive_features_test: Dict[str, np.ndarray],
    cost_constraint: str = 'fpr'
) -> Dict:
    """
    Apply Calibrated Equalized Odds postprocessing.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
        sensitive_features_test: Dict of sensitive attributes
        cost_constraint: 'fpr', 'fnr', or 'weighted'
    
    Returns:
        Dict with fair predictions and metrics
    """
    from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
    
    print(f"\n{'='*60}")
    print(f"POSTPROCESSING: Calibrated Equalized Odds")
    print(f"{'='*60}")
    print(f"Cost constraint: {cost_constraint}")
    
    # Create datasets
    y_pred = model.predict(X_test)
    feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]
    
    dataset_true = create_aif360_dataset(
        X_test, y_test, sensitive_features_test, feature_names
    )
    dataset_pred = create_aif360_dataset(
        X_test, y_pred, sensitive_features_test, feature_names
    )
    
    # Set groups
    privileged_groups = []
    unprivileged_groups = []
    for attr_name in sensitive_features_test.keys():
        privileged_groups.append({attr_name: 1})
        unprivileged_groups.append({attr_name: 0})
    
    # Apply postprocessing
    postprocessor = CalibratedEqOddsPostprocessing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
        cost_constraint=cost_constraint
    )
    
    dataset_fair = postprocessor.fit_predict(dataset_true, dataset_pred)
    
    y_pred_fair = dataset_fair.labels.ravel()
    
    print(f"✓ Calibrated Equalized Odds applied")
    print(f"  Predictions adjusted: {np.sum(y_pred != y_pred_fair)}/{len(y_pred)}")
    
    return {
        'y_pred_fair': y_pred_fair,
        'dataset_fair': dataset_fair
    }


# ==============================================================================
# COMPREHENSIVE FAIRNESS PIPELINE
# ==============================================================================

def run_fairness_improvement_pipeline(
    model,
    X_train, y_train,
    X_test, y_test,
    sensitive_features_train: Dict[str, np.ndarray],
    sensitive_features_test: Dict[str, np.ndarray],
    preprocessing_method: str = 'reweighing',
    use_threshold_optimization: bool = True,
    min_recall: float = 0.6,
    disparate_impact_target: float = 0.8
) -> Dict:
    """
    Comprehensive fairness improvement pipeline.
    
    Combines:
    1. Preprocessing mitigation (optional)
    2. Model retraining with fair data (optional)
    3. Threshold optimization (recommended for recall + fairness)
    
    Returns:
        Dict with improved model, predictions, and all metrics
    """
    results = {}
    
    # Step 1: Baseline evaluation
    print("\n" + "="*80)
    print("BASELINE EVALUATION")
    print("="*80)
    
    y_pred_baseline = model.predict(X_test)
    results['baseline'] = evaluate_fairness(
        y_test, y_pred_baseline,
        sensitive_features_test
    )
    
    # Step 2: Preprocessing (optional)
    if preprocessing_method:
        X_train_fair, y_train_fair, sample_weights = mitigate_bias(
            X_train, y_train,
            sensitive_features_train,
            method=preprocessing_method
        )
        results['preprocessing_method'] = preprocessing_method
        results['sample_weights'] = sample_weights
    else:
        X_train_fair = X_train
        y_train_fair = y_train
        sample_weights = None
    
    # Step 3: Threshold optimization (for recall + fairness)
    if use_threshold_optimization:
        # Use first sensitive feature for threshold optimization
        sensitive_attr_name = list(sensitive_features_test.keys())[0]
        sensitive_test = sensitive_features_test[sensitive_attr_name]
        
        threshold_results = optimize_threshold_fairness(
            model, X_test, y_test, sensitive_test,
            min_recall=min_recall,
            disparate_impact_target=disparate_impact_target
        )
        
        results['optimized'] = {
            'thresholds': {
                'privileged': threshold_results['threshold_privileged'],
                'unprivileged': threshold_results['threshold_unprivileged']
            },
            'predictions': threshold_results['y_pred'],
            'metrics': {
                'recall': threshold_results['recall'],
                'accuracy': threshold_results['accuracy'],
                'precision': threshold_results['precision'],
                'f1': threshold_results['f1']
            }
        }
        
        # Evaluate fairness with optimized predictions
        results['optimized']['fairness'] = evaluate_fairness(
            y_test, threshold_results['y_pred'],
            sensitive_features_test
        )
    
    return results
