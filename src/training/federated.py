# src/training/federated.py

"""
Federated Learning implementation for Random Forest models.

Since Random Forest doesn't use gradient descent, we implement
Federated Random Forest as an ensemble approach:
1. Each bank trains a local Random Forest model
2. Predictions are combined via weighted voting
3. Weights are based on training data size (FedProx-inspired)
"""

import numpy as np
import joblib
import os
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class FederatedRandomForestClient:
    """
    A Federated Learning client that trains a Random Forest model
    on local bank data.
    """
    
    def __init__(self, client_id: str, X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray, model_params: dict):
        """
        Initialize FL client.
        
        Args:
            client_id: Bank identifier (e.g., "BankA")
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            model_params: Random Forest hyperparameters
        """
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_params = model_params
        self.model = None
        self.training_size = len(X_train)
        
    def train_local_model(self, verbose: bool = True) -> Dict:
        """
        Train local Random Forest model on bank's private data.
        
        Returns:
            dict: Training metrics
        """
        if verbose:
            print(f"  [{self.client_id}] Training local model on {self.training_size} samples...")
        
        # Train Random Forest
        self.model = RandomForestClassifier(**self.model_params)
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate on local test set
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        metrics = {
            'client_id': self.client_id,
            'training_size': self.training_size,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_test, y_proba) if len(np.unique(self.y_test)) > 1 else 0.0
        }
        
        if verbose:
            print(f"  [{self.client_id}] Local metrics: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        
        return metrics
    
    def get_model(self) -> RandomForestClassifier:
        """Return the trained local model."""
        return self.model
    
    def save_model(self, filepath: str):
        """Save local model to disk."""
        joblib.dump(self.model, filepath)
        print(f"  [{self.client_id}] Model saved to {filepath}")


class FederatedRandomForestEnsemble:
    """
    Federated Random Forest ensemble that combines predictions
    from multiple bank models via weighted voting.
    
    Supports multiple aggregation methods:
    - weighted: by data size (standard FL)
    - uniform: equal weights for all banks
    - performance: by local model accuracy
    """
    
    def __init__(self, client_models: List[RandomForestClassifier], 
                 client_weights: List[float], client_ids: List[str],
                 aggregation_method: str = 'weighted'):
        """
        Initialize federated ensemble.
        
        Args:
            client_models: List of trained local RF models
            client_weights: Voting weights
            client_ids: Bank identifiers
            aggregation_method: 'weighted', 'uniform', or 'performance'
        """
        self.client_models = client_models
        self.client_ids = client_ids
        self.n_clients = len(client_models)
        self.aggregation_method = aggregation_method
        
        # Set weights based on aggregation method
        if aggregation_method == 'uniform':
            self.client_weights = np.ones(len(client_weights)) / len(client_weights)
        else:
            self.client_weights = np.array(client_weights) / np.sum(client_weights)
        
        print(f"\n✓ Federated Ensemble created ({aggregation_method} aggregation)")
        print(f"  Clients: {self.n_clients}")
        for cid, weight in zip(client_ids, self.client_weights):
            print(f"  - {cid}: weight={weight:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using weighted voting from all client models.
        
        Args:
            X: Features
            
        Returns:
            Predictions (0 or 1)
        """
        # Get predictions from all clients
        predictions = np.array([model.predict(X) for model in self.client_models])
        
        # Weighted voting
        weighted_votes = np.sum(predictions * self.client_weights[:, np.newaxis], axis=0)
        final_predictions = (weighted_votes >= 0.5).astype(int)
        
        return final_predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using weighted average of client predictions.
        
        Args:
            X: Features
            
        Returns:
            Probability of class 1
        """
        # Get probabilities from all clients
        probabilities = np.array([model.predict_proba(X)[:, 1] for model in self.client_models])
        
        # Weighted average
        weighted_proba = np.sum(probabilities * self.client_weights[:, np.newaxis], axis=0)
        
        return weighted_proba
    
    def save(self, dirpath: str):
        """Save ensemble to directory."""
        os.makedirs(dirpath, exist_ok=True)
        
        # Save individual models
        for i, (model, cid) in enumerate(zip(self.client_models, self.client_ids)):
            filepath = os.path.join(dirpath, f"{cid}_model.joblib")
            joblib.dump(model, filepath)
        
        # Save metadata
        metadata = {
            'client_ids': self.client_ids,
            'client_weights': self.client_weights.tolist(),
            'n_clients': self.n_clients
        }
        joblib.dump(metadata, os.path.join(dirpath, 'ensemble_metadata.joblib'))
        
        print(f"✓ Federated ensemble saved to {dirpath}")
    
    @classmethod
    def load(cls, dirpath: str):
        """Load ensemble from directory."""
        # Load metadata
        metadata = joblib.load(os.path.join(dirpath, 'ensemble_metadata.joblib'))
        
        # Load individual models
        client_models = []
        for cid in metadata['client_ids']:
            filepath = os.path.join(dirpath, f"{cid}_model.joblib")
            client_models.append(joblib.load(filepath))
        
        return cls(client_models, metadata['client_weights'], metadata['client_ids'])


def create_federated_ensemble(clients: List[FederatedRandomForestClient], 
                             aggregation_method: str = 'weighted',
                             local_accuracies: List[float] = None) -> FederatedRandomForestEnsemble:
    """
    Create federated ensemble from trained clients.
    
    Args:
        clients: List of trained FL clients
        aggregation_method: 'weighted' (by data size), 'uniform', or 'performance' (by accuracy)
        local_accuracies: Local model accuracies (required if aggregation_method='performance')
        
    Returns:
        FederatedRandomForestEnsemble
    """
    models = [client.get_model() for client in clients]
    ids = [client.client_id for client in clients]
    
    # Determine weights based on aggregation method
    if aggregation_method == 'weighted':
        weights = [client.training_size for client in clients]
    elif aggregation_method == 'uniform':
        weights = [1.0] * len(clients)  # Equal weights
    elif aggregation_method == 'performance':
        if local_accuracies is None:
            raise ValueError("local_accuracies required for performance-based aggregation")
        weights = local_accuracies
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    return FederatedRandomForestEnsemble(models, weights, ids, aggregation_method)


def run_federated_training(
    client_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    model_params: dict,
    verbose: bool = True
) -> Tuple[FederatedRandomForestEnsemble, Dict[str, Dict]]:
    """
    Run federated training across multiple banks.
    
    Args:
        client_data: Dict mapping client_id -> (X_train, y_train, X_test, y_test)
        model_params: Random Forest hyperparameters
        verbose: Print training progress
        
    Returns:
        ensemble: Trained federated ensemble
        local_metrics: Per-client training metrics
    """
    if verbose:
        print("\n" + "="*80)
        print("FEDERATED TRAINING")
        print("="*80)
        print(f"Number of clients: {len(client_data)}")
        print(f"Model: Random Forest")
        print(f"Parameters: {model_params}")
    
    # Create and train clients
    clients = []
    local_metrics = {}
    
    for client_id, (X_train, y_train, X_test, y_test) in client_data.items():
        client = FederatedRandomForestClient(
            client_id=client_id,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_params=model_params
        )
        
        # Train local model
        metrics = client.train_local_model(verbose=verbose)
        local_metrics[client_id] = metrics
        
        clients.append(client)
    
    # Create ensemble
    ensemble = create_federated_ensemble(clients)
    
    if verbose:
        print("\n✓ Federated training complete!")
    
    return ensemble, local_metrics


def evaluate_federated_model(
    ensemble: FederatedRandomForestEnsemble,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True
) -> Dict:
    """
    Evaluate federated ensemble on global test set.
    
    Args:
        ensemble: Trained federated ensemble
        X_test: Test features
        y_test: Test labels
        verbose: Print results
        
    Returns:
        dict: Performance metrics
    """
    # Get predictions
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
    }
    
    if verbose:
        print("\n" + "="*80)
        print("FEDERATED MODEL EVALUATION (Global Test Set)")
        print("="*80)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


def evaluate_per_bank(
    clients: List[FederatedRandomForestClient],
    ensemble: FederatedRandomForestEnsemble,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Evaluate both local and global models on each bank's test set.
    
    Args:
        clients: List of trained FL clients
        ensemble: Trained federated ensemble
        verbose: Print results
        
    Returns:
        dict: Per-bank metrics for local and global models
    """
    results = {}
    
    if verbose:
        print("\n" + "="*80)
        print("PER-BANK EVALUATION")
        print("="*80)
    
    for client in clients:
        cid = client.client_id
        
        # Local model predictions
        y_pred_local = client.model.predict(client.X_test)
        y_proba_local = client.model.predict_proba(client.X_test)[:, 1]
        
        # Global ensemble predictions
        y_pred_global = ensemble.predict(client.X_test)
        y_proba_global = ensemble.predict_proba(client.X_test)
        
        # Metrics
        local_metrics = {
            'accuracy': accuracy_score(client.y_test, y_pred_local),
            'precision': precision_score(client.y_test, y_pred_local, zero_division=0),
            'recall': recall_score(client.y_test, y_pred_local, zero_division=0),
            'f1': f1_score(client.y_test, y_pred_local, zero_division=0),
            'roc_auc': roc_auc_score(client.y_test, y_proba_local) if len(np.unique(client.y_test)) > 1 else 0.0
        }
        
        global_metrics = {
            'accuracy': accuracy_score(client.y_test, y_pred_global),
            'precision': precision_score(client.y_test, y_pred_global, zero_division=0),
            'recall': recall_score(client.y_test, y_pred_global, zero_division=0),
            'f1': f1_score(client.y_test, y_pred_global, zero_division=0),
            'roc_auc': roc_auc_score(client.y_test, y_proba_global) if len(np.unique(client.y_test)) > 1 else 0.0
        }
        
        results[cid] = {
            'local': local_metrics,
            'global': global_metrics
        }
        
        if verbose:
            print(f"\n{cid}:")
            print(f"  Local Model  - Acc: {local_metrics['accuracy']:.4f}, F1: {local_metrics['f1']:.4f}")
            print(f"  Global Model - Acc: {global_metrics['accuracy']:.4f}, F1: {global_metrics['f1']:.4f}")
    
    return results
