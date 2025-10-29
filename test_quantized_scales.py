#!/usr/bin/env python3
"""
Compare the actual quantized weights and scales between nosearch and search models.
This tests if scale search is producing different scales (not just different reconstruction).
"""

import torch
from safetensors import safe_open
from pathlib import Path
import json
import numpy as np

def load_scales_and_weights(model_path):
    """Load quantized weights and their scales."""
    model_path = Path(model_path)
    
    # Load the index
    index_file = model_path / "model.safetensors.index.json"
    with open(index_file, 'r') as f:
        index = json.load(f)
    
    weight_map = index['weight_map']
    
    # Group by shard
    shard_to_keys = {}
    for key, shard in weight_map.items():
        if shard not in shard_to_keys:
            shard_to_keys[shard] = []
        shard_to_keys[shard].append(key)
    
    scales = {}
    weights = {}
    
    for shard_name, keys in shard_to_keys.items():
        shard_path = model_path / shard_name
        
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in keys:
                tensor = f.get_tensor(key)
                
                if '.weight_scale' in key and not key.endswith('_scale_2'):
                    scales[key] = tensor
                elif '.weight' in key and not any(x in key for x in ['_scale', 'embed', 'norm']):
                    weights[key] = tensor
    
    return weights, scales

def compare_scales(nosearch_path, search_path):
    """Compare scales between two models."""
    
    print("=" * 80)
    print("COMPARING QUANTIZED SCALES (NOT RECONSTRUCTED WEIGHTS)")
    print("=" * 80)
    print()
    
    print(f"Loading nosearch scales from {nosearch_path}...")
    nosearch_weights, nosearch_scales = load_scales_and_weights(nosearch_path)
    print(f"  Found {len(nosearch_scales)} scale tensors")
    print(f"  Found {len(nosearch_weights)} quantized weights\n")
    
    print(f"Loading search scales from {search_path}...")
    search_weights, search_scales = load_scales_and_weights(search_path)
    print(f"  Found {len(search_scales)} scale tensors")
    print(f"  Found {len(search_weights)} quantized weights\n")
    
    print("=" * 80)
    print("SCALE COMPARISON")
    print("=" * 80)
    print()
    
    identical_scales = 0
    different_scales = 0
    scale_differences = []
    
    for scale_name in sorted(nosearch_scales.keys()):
        if scale_name not in search_scales:
            continue
        
        nosearch_scale = nosearch_scales[scale_name]
        search_scale = search_scales[scale_name]
        
        # Check if scales are identical
        if torch.equal(nosearch_scale, search_scale):
            identical_scales += 1
        else:
            different_scales += 1
            
            # Measure difference
            diff = torch.abs(nosearch_scale.float() - search_scale.float()).mean().item()
            max_diff = torch.abs(nosearch_scale.float() - search_scale.float()).max().item()
            scale_differences.append({
                'name': scale_name,
                'mean_diff': diff,
                'max_diff': max_diff
            })
    
    print(f"Identical scales: {identical_scales} / {len(nosearch_scales)}")
    print(f"Different scales: {different_scales} / {len(nosearch_scales)}")
    print()
    
    if different_scales > 0:
        print(f"✓ SCALES ARE DIFFERENT - Scale search is producing different scales!")
        print()
        print("Top 10 layers with largest scale differences:")
        print("-" * 80)
        scale_differences.sort(key=lambda x: x['max_diff'], reverse=True)
        for i, diff in enumerate(scale_differences[:10]):
            print(f"{i+1}. {diff['name']}")
            print(f"   Mean absolute difference: {diff['mean_diff']:.8f}")
            print(f"   Max absolute difference:  {diff['max_diff']:.8f}")
            print()
    else:
        print("✗ ALL SCALES ARE IDENTICAL!")
        print("  Scale search is NOT producing different scales")
        print("  This means scale search is not working or not being saved")
    
    print()
    print("=" * 80)
    print("QUANTIZED WEIGHT COMPARISON")
    print("=" * 80)
    print()
    
    identical_weights = 0
    different_weights = 0
    
    for weight_name in sorted(nosearch_weights.keys()):
        if weight_name not in search_weights:
            continue
        
        nosearch_weight = nosearch_weights[weight_name]
        search_weight = search_weights[weight_name]
        
        if torch.equal(nosearch_weight, search_weight):
            identical_weights += 1
        else:
            different_weights += 1
    
    print(f"Identical quantized weights: {identical_weights} / {len(nosearch_weights)}")
    print(f"Different quantized weights: {different_weights} / {len(nosearch_weights)}")
    print()
    
    if different_weights > 0:
        print("✓ QUANTIZED WEIGHTS ARE DIFFERENT!")
    else:
        print("✗ QUANTIZED WEIGHTS ARE IDENTICAL!")
        print("  Even the packed FP4 values are the same")
    
    print("=" * 80)
    
    return identical_scales, different_scales, identical_weights, different_weights


if __name__ == "__main__":
    nosearch_path = "/resource/llama-quant-nosearch-aligned/"
    search_path = "/resource/llama-quant-search-aligned/"
    
    identical_scales, different_scales, identical_weights, different_weights = compare_scales(
        nosearch_path,
        search_path
    )
    
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    if different_scales > 0:
        print("✓ Scale search IS working - scales are different")
        if different_weights > 0:
            print("✓ Different scales lead to different quantized weights")
        else:
            print("⚠ Scales are different but quantized weights are the same (unexpected!)")
    else:
        print("✗ Scale search NOT working - all scales are identical")
        print("  Possible causes:")
        print("  1. Scale search not running during export")
        print("  2. Scale search producing same results as greedy")
        print("  3. Searched scales not being saved correctly")
    
    print("=" * 80)

