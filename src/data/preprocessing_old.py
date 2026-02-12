"""
Data preprocessing utilities for Adult dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import joblib
import os


# Column definitions (Adult dataset standard)
NUMERICAL_COLS = [
    'age', 'educational-num', 'capital-gain', 
    'capital-loss', 'hours-per-week'
]

CATEGORICAL_COLS = [
    'workclass', 'education', 'marital-status', 
    'occupation', 'relationship', 'race', 'gender', 'native-country'
]

# Sensitive attributes for fairness analysis
SENSITIVE_ATTRIBUTES = ['gender', 'race']

# Target column
TARGET_COL = 'income'


def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values (marked as '?').
    
    Args:
        df: Input DataFrame
        strategy: 'drop' or 'mode' for categorical and 'median' for numerical or 'combined'
    
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    
    if strategy == 'drop':
        n_before = len(df)
        df = df.dropna()
        n_after = len(df)
        print(f"  Dropped {n_before - n_after} rows with missing values")
    
    elif strategy == 'combined':
        # Fill categorical with mode
        for col in CATEGORICAL_COLS:
            if col in df.columns and df[col].isna().any():
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)
                print(f"  Filled {col} with mode: {mode_value}")
        # Fill numerical with median
        for col in NUMERICAL_COLS:
            if col in df.columns and df[col].isna().any():
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                print(f"  Filled {col} with median: {median_value}")
    
    elif strategy == 'mode':
        # Fill categorical with mode
        for col in CATEGORICAL_COLS:
            if col in df.columns and df[col].isna().any():
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)
                print(f"  Filled {col} with mode: {mode_value}")
    
    elif strategy == 'median':
        # Fill numerical with median
        for col in NUMERICAL_COLS:
            if col in df.columns and df[col].isna().any():
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                print(f"  Filled {col} with median: {median_value}")
    
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode target variable: >50K = 1, <=50K = 0
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with encoded target
    """
    df = df.copy()
    
    # Clean target strings (remove dots and whitespace)
    df[TARGET_COL] = df[TARGET_COL].str.strip().str.rstrip('.')
    
    # Encode: >50K = 1, <=50K = 0
    df[TARGET_COL] = (df[TARGET_COL] == '>50K').astype(int)
    
    return df


def preprocess_data(
    df: pd.DataFrame,
    missing_strategy: str = 'drop',
    scaling: str = 'standard',
    scaler=None,
    encoder=None,
    fit_transform: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Complete preprocessing pipeline.
    
    Args:
        df: Input DataFrame
        missing_strategy: How to handle missing values
        scaling: Scaling method ('standard', 'minmax', 'none')
        scaler: Fitted scaler (if fit_transform=False)
        encoder: Fitted encoder (if fit_transform=False)
        fit_transform: If True, fit and transform. If False, only transform.
    
    Returns:
        Tuple of:
            - X: Feature array (scaled and encoded)
            - y: Target array
            - artifacts: Dict with scaler, encoder, feature_names
    
    Example:
        >>> # Training
        >>> X_train, y_train, artifacts = preprocess_data(train_df, fit_transform=True)
        >>> 
        >>> # Testing (use fitted transformers)
        >>> X_test, y_test, _ = preprocess_data(
        >>>     test_df, 
        >>>     scaler=artifacts['scaler'],
        >>>     encoder=artifacts['encoder'],
        >>>     fit_transform=False
        >>> )
    """
    df = df.copy()
    
    print("\nPreprocessing data...")
    
    # 1. Handle missing values
    df = handle_missing_values(df, strategy=missing_strategy)
    
    # 2. Encode target
    df = encode_target(df)
    
    # 3. Separate features and target
    # Drop 'institute' column (not a feature)
    feature_cols = [col for col in df.columns if col not in [TARGET_COL, 'institute']]
    
    X_df = df[feature_cols]
    y = df[TARGET_COL].values
    
    # 4. Separate numerical and categorical
    num_features = [col for col in NUMERICAL_COLS if col in X_df.columns]
    cat_features = [col for col in CATEGORICAL_COLS if col in X_df.columns]
    
    # 5. Scale numerical features
    X_num = X_df[num_features].values
    
    if scaling == 'standard':
        if fit_transform:
            scaler = StandardScaler()
            X_num_scaled = scaler.fit_transform(X_num)
            print(f"  ✓ Fitted StandardScaler on {len(num_features)} numerical features")
        else:
            X_num_scaled = scaler.transform(X_num)
    elif scaling == 'none':
        X_num_scaled = X_num
        scaler = None
    
    # 6. One-hot encode categorical features
    X_cat = X_df[cat_features].astype(str)
    
    if fit_transform:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat_encoded = encoder.fit_transform(X_cat)
        cat_feature_names = encoder.get_feature_names_out(cat_features)
        print(f"  ✓ Fitted OneHotEncoder: {len(cat_features)} → {len(cat_feature_names)} features")
    else:
        X_cat_encoded = encoder.transform(X_cat)
        cat_feature_names = encoder.get_feature_names_out(cat_features)
    
    # 7. Combine features
    X = np.hstack([X_num_scaled, X_cat_encoded])
    
    # 8. Create feature names
    feature_names = list(num_features) + list(cat_feature_names)
    
    print(f"  ✓ Final shape: {X.shape}")
    print(f"  ✓ Class distribution: {np.bincount(y)}")
    
    artifacts = {
        'scaler': scaler,
        'encoder': encoder,
        'feature_names': feature_names,
        'num_features': num_features,
        'cat_features': cat_features,
    }
    
    return X, y, artifacts


def create_federated_splits(
    bank_dfs: Dict[str, pd.DataFrame],
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict:
    """
    Create train/test splits for federated learning.
    
    Each bank keeps its data separate (simulating federated setting).
    
    Args:
        bank_dfs: Dictionary of bank DataFrames
        test_size: Fraction for test set
        random_state: Random seed
    
    Returns:
        Dictionary with preprocessed data for each bank:
        {
            'BankA': {'X_train': ..., 'y_train': ..., 'X_test': ..., 'y_test': ...},
            'BankB': {...},
            'BankC': {...},
            'artifacts': {...}  # Shared preprocessing artifacts
        }
    """
    print("\nCreating federated data splits...")
    
    federated_data = {}
    shared_artifacts = None
    
    for bank_name, df in bank_dfs.items():
        print(f"\n{bank_name}:")
        
        # Split into train/test
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df[TARGET_COL]
        )
        
        print(f"  Train: {len(train_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        
        # Preprocess training data (fit transformers on first bank)
        if shared_artifacts is None:
            # First bank: fit transformers
            X_train, y_train, artifacts = preprocess_data(
                train_df, fit_transform=True
            )
            shared_artifacts = artifacts
        else:
            # Other banks: use fitted transformers
            X_train, y_train, _ = preprocess_data(
                train_df,
                scaler=shared_artifacts['scaler'],
                encoder=shared_artifacts['encoder'],
                fit_transform=False
            )
        
        # Preprocess test data (always use fitted transformers)
        X_test, y_test, _ = preprocess_data(
            test_df,
            scaler=shared_artifacts['scaler'],
            encoder=shared_artifacts['encoder'],
            fit_transform=False
        )
        
        federated_data[bank_name] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
        }
    
    federated_data['artifacts'] = shared_artifacts
    
    print("\n✓ Federated splits created!")
    return federated_data


def save_preprocessing_artifacts(artifacts: Dict, output_dir: str = "data/processed"):
    """
    Save preprocessing artifacts (scaler, encoder) to disk.
    
    Args:
        artifacts: Dictionary with scaler, encoder, etc.
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if artifacts['scaler'] is not None:
        scaler_path = os.path.join(output_dir, 'scaler.joblib')
        joblib.dump(artifacts['scaler'], scaler_path)
        print(f"✓ Saved scaler to {scaler_path}")
    
    if artifacts['encoder'] is not None:
        encoder_path = os.path.join(output_dir, 'encoder.joblib')
        joblib.dump(artifacts['encoder'], encoder_path)
        print(f"✓ Saved encoder to {encoder_path}")
    
    # Save feature names
    feature_names_path = os.path.join(output_dir, 'feature_names.txt')
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(artifacts['feature_names']))
    print(f"✓ Saved feature names to {feature_names_path}")


def load_preprocessing_artifacts(input_dir: str = "data/processed") -> Dict:
    """
    Load preprocessing artifacts from disk.
    
    Args:
        input_dir: Input directory
    
    Returns:
        Dictionary with loaded artifacts
    """
    artifacts = {}
    
    scaler_path = os.path.join(input_dir, 'scaler.joblib')
    if os.path.exists(scaler_path):
        artifacts['scaler'] = joblib.load(scaler_path)
        print(f"✓ Loaded scaler from {scaler_path}")
    
    encoder_path = os.path.join(input_dir, 'encoder.joblib')
    if os.path.exists(encoder_path):
        artifacts['encoder'] = joblib.load(encoder_path)
        print(f"✓ Loaded encoder from {encoder_path}")
    
    feature_names_path = os.path.join(input_dir, 'feature_names.txt')
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            artifacts['feature_names'] = [line.strip() for line in f]
        print(f"✓ Loaded feature names from {feature_names_path}")
    
    return artifacts


if __name__ == "__main__":
    # Test preprocessing
    from data_loader import load_all_banks
    
    combined_df, bank_dfs = load_all_banks()
    
    # Test combined preprocessing
    X, y, artifacts = preprocess_data(combined_df)
    print(f"\nCombined preprocessing test:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Features: {len(artifacts['feature_names'])}")
    
    # Test federated splits
    federated_data = create_federated_splits(bank_dfs)
    print(f"\nFederated splits test:")
    for bank in ['BankA', 'BankB', 'BankC']:
        print(f"  {bank}: {federated_data[bank]['X_train'].shape}")
