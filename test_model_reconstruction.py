#!/usr/bin/env python3
"""
Compare reconstruction quality of exported quantized models.
Tests if the 'search' model has better weight reconstruction than 'nosearch' model.
Uses NVFP4QTensor dequantizer to properly load quantized weights.
"""

import torch
from transformers import AutoModelForCausalLM
from safetensors import safe_open
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

def load_quantized_weights(model_path):
    """Load quantized weights from ModelOpt checkpoint and dequantize them."""
    model_path = Path(model_path)
    
    # Load the index to find which shard contains which weights
    index_file = model_path / "model.safetensors.index.json"
    with open(index_file, 'r') as f:
        index = json.load(f)
    
    weight_map = index['weight_map']
    
    # Group weights by shard file
    shard_to_weights = {}
    for weight_name, shard_name in weight_map.items():
        if shard_name not in shard_to_weights:
            shard_to_weights[shard_name] = []
        shard_to_weights[shard_name].append(weight_name)
    
    # Load and dequantize weights from all shards
    all_weights = {}
    
    for shard_name, weight_names in shard_to_weights.items():
        shard_path = model_path / shard_name
        
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for weight_name in weight_names:
                # Only process actual weights (not scales)
                if weight_name.endswith(('.weight_scale', '.weight_scale_2', '.input_scale')):
                    continue
                
                if not weight_name.endswith('.weight'):
                    all_weights[weight_name] = f.get_tensor(weight_name)
                    continue
                
                # This is a quantized weight - need to dequantize
                weight_data = f.get_tensor(weight_name)
                
                # Get corresponding scales
                scale_name = weight_name.replace('.weight', '.weight_scale')
                scale2_name = weight_name.replace('.weight', '.weight_scale_2')
                
                if scale_name in weight_map and scale2_name in weight_map:
                    # Load scales
                    scale1 = f.get_tensor(scale_name) if scale_name in weight_names else None
                    scale2 = f.get_tensor(scale2_name) if scale2_name in weight_names else None
                    
                    if scale1 is None or scale2 is None:
                        # Scales in different shard - load them
                        scale1_shard = model_path / weight_map[scale_name]
                        scale2_shard = model_path / weight_map[scale2_name]
                        
                        with safe_open(scale1_shard, framework="pt", device="cpu") as f1:
                            scale1 = f1.get_tensor(scale_name)
                        with safe_open(scale2_shard, framework="pt", device="cpu") as f2:
                            scale2 = f2.get_tensor(scale2_name)
                    
                    # Dequantize using NVFP4QTensor
                    # The weight_data is already the packed quantized tensor
                    # Constructor signature: NVFP4QTensor(shape, dtype, quantized_data)
                    unpacked_shape = weight_data.shape[:-1] + (weight_data.shape[-1] * 2,)
                    qtensor = NVFP4QTensor(
                        unpacked_shape,  # Original (unpacked) shape
                        torch.bfloat16,  # dtype
                        weight_data      # packed quantized data
                    )
                    
                    # Dequantize
                    dequantized = qtensor.dequantize(
                        dtype=torch.bfloat16,
                        scale=scale1,
                        double_scale=scale2,
                        block_sizes={-1: 16}  # NVFP4 uses block size 16
                    )
                    
                    all_weights[weight_name] = dequantized
                else:
                    # Not quantized, use as-is
                    all_weights[weight_name] = weight_data
    
    return all_weights

def get_baseline_weights(model_path):
    """Load baseline model weights."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu"  # Load to CPU to save GPU memory
    )
    
    weights = {}
    for name, param in model.named_parameters():
        if '.weight' in name:
            weights[name] = param.data.clone()
    
    del model
    torch.cuda.empty_cache()
    
    return weights

def compare_models(baseline_path, nosearch_path, search_path):
    """Compare reconstruction quality of two quantized models against baseline."""
    
    print("=" * 80)
    print("COMPARING EXPORTED MODEL RECONSTRUCTION QUALITY")
    print("=" * 80)
    print()
    
    # Load baseline model (BF16)
    print(f"Loading baseline weights from {baseline_path}...")
    baseline_weights = get_baseline_weights(baseline_path)
    print(f"  Loaded {len(baseline_weights)} weight tensors\n")
    
    # Load nosearch model (quantized)
    print(f"Loading and dequantizing nosearch model from {nosearch_path}...")
    nosearch_weights = load_quantized_weights(nosearch_path)
    print(f"  Loaded {len(nosearch_weights)} weight tensors\n")
    
    # Load search model (quantized)
    print(f"Loading and dequantizing search model from {search_path}...")
    search_weights = load_quantized_weights(search_path)
    print(f"  Loaded {len(search_weights)} weight tensors\n")
    
    print("=" * 80)
    print("WEIGHT-BY-WEIGHT COMPARISON")
    print("=" * 80)
    print()
    
    nosearch_errors = []
    search_errors = []
    improvements = []
    layer_results = []
    
    # Find common quantized weights (those that have scales)
    quantized_weights = set()
    for name in baseline_weights.keys():
        if name in nosearch_weights and name in search_weights:
            # Check if this weight was actually quantized (not embedding/lm_head)
            if name.endswith('.weight') and 'embed' not in name.lower() and 'lm_head' not in name.lower():
                quantized_weights.add(name)
    
    print(f"Found {len(quantized_weights)} quantized weights to compare\n")
    
    # Compare each weight tensor
    for name in tqdm(sorted(quantized_weights), desc="Comparing weights"):
        baseline_weight = baseline_weights[name].float()
        nosearch_weight = nosearch_weights[name].float()
        search_weight = search_weights[name].float()
        
        # Calculate MSE
        mse_nosearch = torch.nn.functional.mse_loss(nosearch_weight, baseline_weight).item()
        mse_search = torch.nn.functional.mse_loss(search_weight, baseline_weight).item()
        
        nosearch_errors.append(mse_nosearch)
        search_errors.append(mse_search)
        
        if mse_nosearch > 0:
            improvement = ((mse_nosearch - mse_search) / mse_nosearch) * 100
        else:
            improvement = 0
        
        improvements.append(improvement)
        
        layer_results.append({
            'name': name,
            'nosearch_mse': mse_nosearch,
            'search_mse': mse_search,
            'improvement': improvement,
            'better': 'search' if mse_search < mse_nosearch else 'nosearch'
        })
    
    # Sort by improvement to show best improvements first
    layer_results.sort(key=lambda x: x['improvement'], reverse=True)
    
    # Show top 10 improvements
    print("\nTop 10 layers with best scale search improvement:")
    print("-" * 80)
    for i, result in enumerate(layer_results[:10]):
        print(f"{i+1}. {result['name']}")
        print(f"   No Search MSE:   {result['nosearch_mse']:.8f}")
        print(f"   With Search MSE: {result['search_mse']:.8f}")
        print(f"   Improvement:     {result['improvement']:+.2f}%")
        print()
    
    # Show top 10 regressions (if any)
    bottom_results = [r for r in layer_results if r['improvement'] < 0]
    if bottom_results:
        print("\nTop 10 layers where scale search performed worse:")
        print("-" * 80)
        for i, result in enumerate(bottom_results[:10]):
            print(f"{i+1}. {result['name']}")
            print(f"   No Search MSE:   {result['nosearch_mse']:.8f}")
            print(f"   With Search MSE: {result['search_mse']:.8f}")
            print(f"   Regression:      {result['improvement']:+.2f}%")
            print()
    
    # Overall statistics
    avg_nosearch = np.mean(nosearch_errors)
    avg_search = np.mean(search_errors)
    avg_improvement = np.mean(improvements)
    
    wins_search = sum(1 for r in layer_results if r['better'] == 'search')
    wins_nosearch = sum(1 for r in layer_results if r['better'] == 'nosearch')
    ties = sum(1 for r in layer_results if r['nosearch_mse'] == r['search_mse'])
    
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total layers compared: {len(layer_results)}")
    print()
    print(f"Average MSE (nosearch): {avg_nosearch:.8f}")
    print(f"Average MSE (search):   {avg_search:.8f}")
    print(f"Average improvement:    {avg_improvement:+.2f}%")
    print()
    print(f"Search wins:    {wins_search} / {len(layer_results)} ({wins_search/len(layer_results)*100:.1f}%)")
    print(f"No search wins: {wins_nosearch} / {len(layer_results)} ({wins_nosearch/len(layer_results)*100:.1f}%)")
    print(f"Ties:           {ties} / {len(layer_results)} ({ties/len(layer_results)*100:.1f}%)")
    print()
    
    if avg_search < avg_nosearch:
        improvement_pct = ((avg_nosearch - avg_search) / avg_nosearch) * 100
        print(f"✓ SCALE SEARCH IS WORKING!")
        print(f"  {improvement_pct:.2f}% better reconstruction quality on average")
    elif avg_search == avg_nosearch:
        print("✗ MODELS ARE IDENTICAL!")
        print("  Both models have the same weights - scale search was not applied")
    else:
        regression_pct = ((avg_search - avg_nosearch) / avg_nosearch) * 100
        print(f"✗ SCALE SEARCH IS WORSE!")
        print(f"  {regression_pct:.2f}% worse reconstruction quality on average")
    
    print("=" * 80)
    
    return layer_results, avg_nosearch, avg_search, avg_improvement


if __name__ == "__main__":
    import sys
    
    # Paths to models
    baseline_path = "Qwen/Qwen3-8B"
    nosearch_path = "/resource/llama-quant-nosearch-aligned/"
    search_path = "/resource/llama-quant-search-aligned/"
    
    # Allow override from command line
    if len(sys.argv) > 1:
        baseline_path = sys.argv[1]
    if len(sys.argv) > 2:
        nosearch_path = sys.argv[2]
    if len(sys.argv) > 3:
        search_path = sys.argv[3]
    
    print(f"Baseline:  {baseline_path}")
    print(f"No Search: {nosearch_path}")
    print(f"Search:    {search_path}")
    print()
    
    results, avg_nosearch, avg_search, avg_improvement = compare_models(
        baseline_path, 
        nosearch_path, 
        search_path
    )

