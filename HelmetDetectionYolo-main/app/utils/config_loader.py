"""
Configuration loader utility
"""
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "app/config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config YAML file
    
    Returns:
        Dictionary containing configuration
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

