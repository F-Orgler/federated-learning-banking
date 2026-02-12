#!/usr/bin/env python
"""
Main pipeline script - runs complete baseline training workflow.

Usage:
    python scripts/run_pipeline.py [--config path/to/config.yaml]
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import load_all_banks, preprocess_data, create_federated_splits, save_preprocessing_artifacts
from src.models import train_random_forest, tune_hyperparameters
from src.evaluation import evaluate_model, evaluate_fairness, explain_model_shap, plot_shap_summary
from src.evaluation.explainability import create_waterfall_plots
from src.utils import load_config, save_results, create_output_dirs
from src.utils.visualization import plot_confusion_matrix, plot_metrics_comparison
from sklearn.model_selection import train_test_split
import joblib
import json
import numpy as np
import pandas as pd


def run_baseline_pipeline(config_path: str = "experiments/baseline_config.yaml"):
    """
    Run complete baseline training pipeline.
    
    Steps:
        1. Load and preprocess data
        2. Train Random Forest (with optional hyperparameter tuning)
        3. Evaluate performance
        4. Evaluate fairness
        5. Apply bias mitigation (if needed)
        6. Generate SHAP explanations
        7. Save all results
    """
    print("="*80)
    print("BASELINE TRAINING PIPELINE")
    print("="*80)
    
    # Load configuration
    config = load_config(config_path)
    create_output_dirs(config)
    
    # ========================================
    # 1. LOAD DATA
    # ========================================
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    banks = config['data'].get('banks', ['BankA', 'BankB', 'BankC'])
    data_path = config['data'].get('raw_path', 'data/raw')
    
    combined_df, bank_dfs = load_all_banks(banks, data_path)
    
    # ========================================
    # 2. PREPROCESS DATA
    # ========================================
    print("\n" + "="*80)
    print("STEP 2: PREPROCESSING")
    print("="*80)
    
    # Split into train/test
    test_size = config['data'].get('test_size', 0.2)
    random_state = config['data'].get('random_state', 42)
    
    train_df, test_df = train_test_split(
        combined_df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=combined_df['income']
    )
    
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Preprocess
    X_train, y_train, artifacts = preprocess_data(train_df, missing_strategy='combined', fit_transform=True)
    X_test, y_test, _ = preprocess_data(
        test_df,
        missing_strategy='combined',
        scaler=artifacts['scaler'],
        encoder=artifacts['encoder'],
        fit_transform=False
    )
    
    # Save preprocessing artifacts
    save_preprocessing_artifacts(artifacts, config['data'].get('processed_path', 'data/processed'))
    
    # ========================================
    # 3. TRAIN MODEL
    # ========================================
    print("\n" + "="*80)
    print("STEP 3: TRAINING RANDOM FOREST")
    print("="*80)
    
    model_config = config['model']
    
    if model_config.get('hyperparameter_tuning', {}).get('enabled', False):
        print("Performing hyperparameter tuning...")
        tuning_config = model_config['hyperparameter_tuning']
        
        model, best_params = tune_hyperparameters(
            X_train, y_train,
            param_grid=tuning_config.get('param_grid'),
            method=tuning_config.get('method', 'grid'),
            cv=tuning_config.get('cv_folds', 5),
            random_state=random_state
        )
        
        # Save best parameters
        best_params_path = os.path.join(config['output']['reports_dir'], 'best_hyperparameters.json')
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
    else:
        model = train_random_forest(
            X_train, y_train,
            params=model_config.get('params'),
            random_state=random_state
        )
    
    # ========================================
    # 4. EVALUATE PERFORMANCE
    # ========================================
    print("\n" + "="*80)
    print("STEP 4: EVALUATING PERFORMANCE")
    print("="*80)
    
    results = evaluate_model(model, X_test, y_test, detailed=True)
    
    # Plot confusion matrix
    cm_path = os.path.join(config['output']['plots_dir'], 'confusion_matrix.png')
    plot_confusion_matrix(
        results['confusion_matrix'],
        title="Baseline Model - Confusion Matrix",
        output_path=cm_path,
        show=False
    )
    
    # Save metrics
    metrics_path = os.path.join(config['output']['reports_dir'], 'baseline_metrics.json')
    save_results({'baseline': results['metrics']}, metrics_path, format='json')
    
    # ========================================
    # 5. FAIRNESS EVALUATION & IMPROVEMENT
    # ========================================
    if config.get('fairness', {}).get('enabled', True):
        print("\n" + "="*80)
        print("STEP 5: FAIRNESS EVALUATION & IMPROVEMENT")
        print("="*80)
        
        # Extract sensitive features
        test_df_original = test_df.copy()
        train_df_original = train_df.copy()
        
        gender_map = {'Male': 1, 'Female': 0}
        
        gender_train = train_df_original['gender'].map(gender_map).fillna(0).values
        race_train = np.where(train_df_original['race'] == 'White', 1, 0)
        gender_test = test_df_original['gender'].map(gender_map).fillna(0).values
        race_test = np.where(test_df_original['race'] == 'White', 1, 0)
        
        # 5a. Baseline Fairness
        print("\n--- Baseline Fairness Gender ---")
        fairness_results_gender = evaluate_fairness(
            y_test, results['predictions'],
            sensitive_features={'gender': gender_test},
            privileged_groups=[{'gender': 1}],
            unprivileged_groups=[{'gender': 0}]
        )
        
        print("\n--- Baseline Fairness Race ---")
        fairness_results_race = evaluate_fairness(
            y_test, results['predictions'],
            sensitive_features={'race': race_test},
            privileged_groups=[{'race': 1}],
            unprivileged_groups=[{'race': 0}]
        )

        fairness_results = {
            'baseline': {
                'gender': fairness_results_gender,
                'race': fairness_results_race,
                'metrics': results['metrics']
            }
        }
        
        # 5b. Fairness Improvement
        if config.get('fairness', {}).get('improvement', {}).get('enabled', True):
            from src.evaluation.fairness import (
                mitigate_bias, optimize_threshold_fairness,
                run_fairness_improvement_pipeline
            )
            
            print("\n" + "-"*60)
            print("FAIRNESS IMPROVEMENT")
            print("-"*60)
            
            improvement_config = config['fairness']['improvement']
            methods = improvement_config.get('methods', ['threshold_optimization'])
            
            # Try each improvement method
            for method in methods:
                print(f"\n{'='*60}")
                print(f"Method: {method.upper()}")
                print(f"{'='*60}")
                
                if method == 'threshold_optimization':
                    # Optimize thresholds for recall + fairness
                    threshold_results = optimize_threshold_fairness(
                        model, X_test, y_test, gender_test,
                        objective='equalized_odds',
                        min_recall=improvement_config.get('min_recall', 0.6),
                        disparate_impact_target=improvement_config.get('disparate_impact_target', 0.8)
                    )
                    
                    # Evaluate fairness with optimized thresholds
                    fairness_optimized = evaluate_fairness(
                        y_test, threshold_results['y_pred'],
                        sensitive_features={'gender': gender_test}
                    )
                    
                    fairness_results['threshold_optimized'] = {
                        'thresholds': {
                            'privileged': threshold_results['threshold_privileged'],
                            'unprivileged': threshold_results['threshold_unprivileged']
                        },
                        'metrics': {
                            'accuracy': threshold_results['accuracy'],
                            'precision': threshold_results['precision'],
                            'recall': threshold_results['recall'],
                            'f1': threshold_results['f1']
                        },
                        'fairness_gender': fairness_optimized
                    }
                    
                    print(f"\n✓ Threshold Optimization:")
                    print(f"  Recall: {results['metrics']['recall']:.4f} → {threshold_results['recall']:.4f}")
                    print(f"  Disparate Impact Ratio: {fairness_results_gender['disparate_impact_ratio']:.4f} → {fairness_optimized['disparate_impact_ratio']:.4f}")
                
                elif method in ['reweighing', 'disparate_impact_remover', 'lfr']:
                    # Preprocessing + Retrain
                    X_fair, y_fair, sample_weights = mitigate_bias(
                        X_train, y_train,
                        sensitive_features_train={'gender': gender_train},
                        method=method,
                        repair_level=improvement_config.get('repair_level', 1.0) if method == 'disparate_impact_remover' else None
                    )
                    
                    # Retrain model
                    print("\nRetraining model with fair data...")
                    model_fair = train_random_forest(
                        X_fair, y_fair,
                        params=model_config.get('params'),
                        random_state=random_state,
                        sample_weight=sample_weights if method == 'reweighing' else None
                    )
                    
                    # Evaluate
                    results_fair = evaluate_model(model_fair, X_test, y_test, detailed=True)
                    fairness_fair_gender = evaluate_fairness(
                        y_test, results_fair['predictions'],
                        sensitive_features={'gender': gender_test}
                    )
                    
                    fairness_results[method] = {
                        'metrics': results_fair['metrics'],
                        'fairness_gender': fairness_fair_gender
                    }
                    
                    print(f"\n✓ {method.capitalize()}:")
                    print(f"  Accuracy: {results['metrics']['accuracy']:.4f} → {results_fair['metrics']['accuracy']:.4f}")
                    print(f"  Disparate Impact Ratio: {fairness_results_gender['disparate_impact_ratio']:.4f} → {fairness_fair_gender['disparate_impact_ratio']:.4f}")
                    
                    # Save fair model if significantly better
                    if (fairness_fair_gender['disparate_impact_ratio'] > fairness_results_gender['disparate_impact_ratio'] * 1.1 and
                        results_fair['metrics']['accuracy'] >= results['metrics']['accuracy'] * 0.95):
                        fair_model_path = os.path.join(
                            config['output']['models_dir'],
                            f'fair_model_{method}.joblib'
                        )
                        joblib.dump(model_fair, fair_model_path)
                        print(f"  ✓ Fair model saved: {fair_model_path}")
            
            # Save comprehensive results
            fairness_path = os.path.join(config['output']['reports_dir'], 'fairness_comprehensive.json')
        else:
            # Save baseline only
            fairness_path = os.path.join(config['output']['reports_dir'], 'baseline_fairness.json')
        
        save_results(fairness_results, fairness_path, format='json')
        print(f"\n✓ Fairness results saved: {fairness_path}")
    
    # ========================================
    # 6. EXPLAINABILITY
    # ========================================
    if config.get('explainability', {}).get('enabled', {True}):
        print("\n" + "="*80)
        print("STEP 6: EXPLAINABILITY (SHAP)")
        print("="*80)
        
        explainer, shap_values = explain_model_shap(
            model,
            X_test,
            feature_names=artifacts['feature_names'],
            max_samples=config['explainability']['shap'].get('max_samples', 1000)
        )
        
        # Plot SHAP summary
        shap_summary_path = os.path.join(config['output']['plots_dir'], 'shap_summary.png')
        plot_shap_summary(
            shap_values,
            X_test[:len(shap_values)],
            feature_names=artifacts['feature_names'],
            output_path=shap_summary_path,
            show=False
        )
        
        # Create waterfall plots for example predictions
        print("\nCreating SHAP waterfall plots for example predictions...")
        create_waterfall_plots(
            explainer,
            X_test,
            y_test=y_test,
            y_pred=results['predictions'],
            feature_names=artifacts['feature_names'],
            n_examples=3,
            output_dir=config['output']['plots_dir'],
            show=False
        )
    
    # ========================================
    # 7. SAVE MODEL
    # ========================================
    if config['output'].get('save_model', True):
        print("\n" + "="*80)
        print("STEP 7: SAVING MODEL")
        print("="*80)
        
        model_path = os.path.join(
            config['output']['models_dir'],
            config['output'].get('model_name', 'baseline_random_forest.joblib')
        )
        joblib.dump(model, model_path)
        print(f"✓ Model saved to {model_path}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  Models: {config['output']['models_dir']}")
    print(f"  Plots: {config['output']['plots_dir']}")
    print(f"  Reports: {config['output']['reports_dir']}")
    
    return {
        'model': model,
        'results': results,
        'artifacts': artifacts,
        'fairness': fairness_results if config.get('fairness', {}).get('enabled') else None
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline training pipeline")
    parser.add_argument(
        '--config',
        type=str,
        #default='experiments/baseline_config.yaml',
        default='experiments/baseline_config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_baseline_pipeline(args.config)
        print("\n✓ Pipeline executed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
