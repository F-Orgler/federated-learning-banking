"""
Configuration loading and result saving utilities.
"""

import yaml
import json
import os
from typing import Dict, Any
import pandas as pd


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    
    Example:
        >>> config = load_config('experiments/baseline_config.yaml')
        >>> model_params = config['model']['params']
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Loaded config from {config_path}")
    
    return config


def save_results(
    results: Dict,
    output_path: str,
    format: str = 'json'
):
    """
    Save results to file.
    
    Args:
        results: Results dictionary
        output_path: Path to save results
        format: Output format ('json' or 'csv')
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✓ Results saved to {output_path}")
    
    elif format == 'csv':
        # Convert dict to DataFrame if possible
        if isinstance(results, dict) and all(isinstance(v, (list, dict)) for v in results.values()):
            df = pd.DataFrame(results)
        else:
            df = pd.Series(results).to_frame()
        
        df.to_csv(output_path)
        print(f"✓ Results saved to {output_path}")
    
    else:
        raise ValueError(f"Unknown format: {format}")


def create_output_dirs(config: Dict):
    """
    Create output directories based on config.
    
    Args:
        config: Configuration dictionary
    """
    output_config = config.get('output', {})
    
    dirs_to_create = [
        output_config.get('models_dir', 'results/models'),
        output_config.get('plots_dir', 'results/plots'),
        output_config.get('reports_dir', 'results/reports'),
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    
    print("✓ Output directories created")


def merge_configs(*configs: Dict) -> Dict:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Configuration dictionaries to merge
    
    Returns:
        Merged configuration
    """
    merged = {}
    
    for config in configs:
        merged = _deep_update(merged, config)
    
    return merged


def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """Recursively update nested dictionary."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


if __name__ == "__main__":
    # Test config loading
    test_config = {
        'model': {
            'type': 'random_forest',
            'params': {
                'n_estimators': 100,
                'max_depth': 10
            }
        },
        'data': {
            'test_size': 0.2
        }
    }
    
    # Test saving
    save_results(test_config, 'test_config.json', format='json')
    
    # Test loading (would need actual file)
    # loaded = load_config('test_config.yaml')
    
    print("Config utility test complete!")
