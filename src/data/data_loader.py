"""
Data loading utilities for bank datasets.
"""

import os
import pandas as pd
from typing import List, Tuple, Dict
import numpy as np


def load_bank_data(bank_name: str, data_path: str = "data/raw") -> pd.DataFrame:
    """
    Load data for a single bank.
    
    Args:
        bank_name: Name of the bank (e.g., 'BankA', 'BankB', 'BankC')
        data_path: Path to raw data directory
    
    Returns:
        DataFrame with bank data
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    filepath = os.path.join(data_path, f"{bank_name}.csv")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    print(f"✓ Loaded {bank_name}: {len(df)} samples")
    return df


def load_all_banks(banks: List[str] = None, data_path: str = "data/raw") -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load data from all banks.
    
    Args:
        banks: List of bank names. If None, defaults to ['BankA', 'BankB', 'BankC']
        data_path: Path to raw data directory
    
    Returns:
        Tuple of:
            - Combined DataFrame (all banks concatenated)
            - Dictionary of individual bank DataFrames
    
    Example:
        >>> combined_df, bank_dfs = load_all_banks()
        >>> print(f"Total samples: {len(combined_df)}")
        >>> print(f"Bank A samples: {len(bank_dfs['BankA'])}")
    """
    if banks is None:
        banks = ['BankA', 'BankB', 'BankC']
    
    bank_dfs = {}
    all_dfs = []
    
    print("Loading bank data...")
    for bank in banks:
        df = load_bank_data(bank, data_path)
        bank_dfs[bank] = df
        all_dfs.append(df)
    
    # Concatenate all banks
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\n✓ Total combined: {len(combined_df)} samples")
    print(f"  Distribution: {dict(combined_df['institute'].value_counts())}")
    
    return combined_df, bank_dfs


def get_data_statistics(df: pd.DataFrame) -> Dict:
    """
    Get basic statistics about the dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'n_samples': len(df),
        'n_features': len(df.columns) - 2,  # Exclude 'institute' and 'income'
        'class_distribution': df['income'].value_counts().to_dict(),
        'missing_values': (df == '?').sum().to_dict(),
        'numeric_features': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_features': df.select_dtypes(include=['object']).columns.tolist(),
    }
    
    # Class balance
    class_counts = df['income'].value_counts()
    stats['class_balance'] = {
        'ratio': class_counts.min() / class_counts.max(),
        'percentage': (class_counts / len(df) * 100).to_dict()
    }
    
    return stats


def print_data_info(df: pd.DataFrame, title: str = "Dataset Info"):
    """
    Print formatted data information.
    
    Args:
        df: Input DataFrame
        title: Title for the output
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    stats = get_data_statistics(df)
    
    print(f"\nSamples: {stats['n_samples']}")
    print(f"Features: {stats['n_features']}")
    
    print(f"\nClass Distribution:")
    for label, count in stats['class_distribution'].items():
        pct = stats['class_balance']['percentage'][label]
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    print(f"\nMissing Values (marked as '?'):")
    missing = {k: v for k, v in stats['missing_values'].items() if v > 0}
    if missing:
        for col, count in sorted(missing.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {col}: {count}")
    else:
        print("  None")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test loading
    combined_df, bank_dfs = load_all_banks()
    
    print_data_info(combined_df, "Combined Dataset")
    
    for bank_name, df in bank_dfs.items():
        print_data_info(df, f"{bank_name} Dataset")
