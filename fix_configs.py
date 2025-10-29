#!/usr/bin/env python3
"""Remove quantization_config from exported model configs to make them loadable by transformers."""

import json
import sys
from pathlib import Path

def fix_config(model_dir):
    """Remove quantization_config from config.json if it exists."""
    config_path = Path(model_dir) / "config.json"
    
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if 'quantization_config' in config:
        print(f"Removing quantization_config from {config_path}")
        del config['quantization_config']
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Fixed {config_path}")
        return True
    else:
        print(f"No quantization_config found in {config_path}")
        return False

if __name__ == "__main__":
    models = [
        "/resource/llama-quant-nosearch-aligned",
        "/resource/llama-quant-search-aligned",
    ]
    
    for model_dir in models:
        print(f"\nProcessing {model_dir}...")
        fix_config(model_dir)

