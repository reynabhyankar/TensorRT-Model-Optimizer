#!/usr/bin/env python3
"""Test script to verify NVFP4 scale search produces better reconstruction."""

import torch
import os
import numpy as np

# Import the NVFP4 quantization module
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

def test_scale_search_reconstruction(block_size=16, num_tests=10):
    """Test that scale search improves reconstruction quality."""
    
    print("=" * 80)
    print("Testing NVFP4 Scale Search vs Standard Quantization")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Block size: {block_size}")
    print(f"Number of test tensors: {num_tests}\n")
    
    nosearch_errors = []
    search_errors = []
    
    for test_idx in range(num_tests):
        # Create a random tensor (simulating a weight matrix)
        shape = (4096, 4096)  # Typical LLM weight shape
        original = torch.randn(shape, dtype=torch.bfloat16, device=device)
        
        # Test 1: Without scale search
        os.environ["USE_SCALE_SEARCH"] = "0"
        qtensor_nosearch, scale1_nosearch, scale2_nosearch = NVFP4QTensor.quantize(
            original, 
            block_size=block_size
        )
        reconstructed_nosearch = qtensor_nosearch.dequantize(
            dtype=torch.bfloat16,
            scale=scale1_nosearch,
            double_scale=scale2_nosearch,
            block_sizes={-1: block_size}
        )
        
        # Test 2: With scale search
        os.environ["USE_SCALE_SEARCH"] = "1"
        qtensor_search, scale1_search, scale2_search = NVFP4QTensor.quantize(
            original,
            block_size=block_size
        )
        reconstructed_search = qtensor_search.dequantize(
            dtype=torch.bfloat16,
            scale=scale1_search,
            double_scale=scale2_search,
            block_sizes={-1: block_size}
        )
        
        # Calculate reconstruction errors
        error_nosearch = torch.nn.functional.mse_loss(
            reconstructed_nosearch.float(), 
            original.float()
        ).item()
        
        error_search = torch.nn.functional.mse_loss(
            reconstructed_search.float(),
            original.float()
        ).item()
        
        nosearch_errors.append(error_nosearch)
        search_errors.append(error_search)
        
        improvement = ((error_nosearch - error_search) / error_nosearch) * 100
        
        print(f"Test {test_idx + 1}:")
        print(f"  No Search MSE:   {error_nosearch:.6f}")
        print(f"  With Search MSE: {error_search:.6f}")
        print(f"  Improvement:     {improvement:+.2f}%")
        print(f"  Better:          {'✓ SEARCH' if error_search < error_nosearch else '✗ NO SEARCH'}")
        print()
    
    # Summary statistics
    avg_nosearch = np.mean(nosearch_errors)
    avg_search = np.mean(search_errors)
    avg_improvement = ((avg_nosearch - avg_search) / avg_nosearch) * 100
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Average MSE without search: {avg_nosearch:.6f}")
    print(f"Average MSE with search:    {avg_search:.6f}")
    print(f"Average improvement:        {avg_improvement:+.2f}%")
    print()
    
    wins_search = sum(1 for n, s in zip(nosearch_errors, search_errors) if s < n)
    wins_nosearch = sum(1 for n, s in zip(nosearch_errors, search_errors) if n < s)
    ties = sum(1 for n, s in zip(nosearch_errors, search_errors) if n == s)
    
    print(f"Search wins:    {wins_search}/{num_tests}")
    print(f"No search wins: {wins_nosearch}/{num_tests}")
    print(f"Ties:           {ties}/{num_tests}")
    print()
    
    if avg_search < avg_nosearch:
        print("✓ Scale search IS working correctly - lower reconstruction error!")
    else:
        print("✗ Scale search NOT working - higher or equal reconstruction error!")
        print("  This indicates a bug in the scale search implementation.")
    
    print("=" * 80)
    
    return avg_nosearch, avg_search, avg_improvement


def test_single_tensor_verbose():
    """Detailed test of a single tensor to inspect quantization behavior."""
    
    print("\n" + "=" * 80)
    print("DETAILED SINGLE TENSOR TEST")
    print("=" * 80 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block_size = 16
    
    # Create a small tensor for detailed inspection
    torch.manual_seed(42)
    original = torch.randn(128, 128, dtype=torch.bfloat16, device=device)
    
    print(f"Original tensor shape: {original.shape}")
    print(f"Original tensor stats:")
    print(f"  Min:  {original.min().item():.4f}")
    print(f"  Max:  {original.max().item():.4f}")
    print(f"  Mean: {original.mean().item():.4f}")
    print(f"  Std:  {original.std().item():.4f}")
    print()
    
    # Without scale search
    print("Quantizing WITHOUT scale search...")
    os.environ["USE_SCALE_SEARCH"] = "0"
    qtensor_nosearch, scale1_nosearch, scale2_nosearch = NVFP4QTensor.quantize(
        original, 
        block_size=block_size
    )
    recon_nosearch = qtensor_nosearch.dequantize(
        dtype=torch.bfloat16,
        scale=scale1_nosearch,
        double_scale=scale2_nosearch,
        block_sizes={-1: block_size}
    )
    
    print(f"  Scale1 shape: {scale1_nosearch.shape}, dtype: {scale1_nosearch.dtype}")
    print(f"  Scale2 shape: {scale2_nosearch.shape}, dtype: {scale2_nosearch.dtype}")
    print(f"  Quantized shape: {qtensor_nosearch._quantized_data.shape}")
    print()
    
    # With scale search
    print("Quantizing WITH scale search...")
    os.environ["USE_SCALE_SEARCH"] = "1"
    qtensor_search, scale1_search, scale2_search = NVFP4QTensor.quantize(
        original,
        block_size=block_size
    )
    recon_search = qtensor_search.dequantize(
        dtype=torch.bfloat16,
        scale=scale1_search,
        double_scale=scale2_search,
        block_sizes={-1: block_size}
    )
    
    print(f"  Scale1 shape: {scale1_search.shape}, dtype: {scale1_search.dtype}")
    print(f"  Scale2 shape: {scale2_search.shape}, dtype: {scale2_search.dtype}")
    print(f"  Quantized shape: {qtensor_search._quantized_data.shape}")
    print()
    
    # Compare errors
    error_nosearch = torch.nn.functional.mse_loss(recon_nosearch.float(), original.float()).item()
    error_search = torch.nn.functional.mse_loss(recon_search.float(), original.float()).item()
    
    print("Reconstruction Errors:")
    print(f"  No search:   {error_nosearch:.8f}")
    print(f"  With search: {error_search:.8f}")
    print(f"  Difference:  {error_nosearch - error_search:.8f}")
    print()
    
    # Element-wise error analysis
    abs_error_nosearch = (recon_nosearch - original).abs()
    abs_error_search = (recon_search - original).abs()
    
    print("Absolute Error Statistics:")
    print("  Without search:")
    print(f"    Mean: {abs_error_nosearch.mean().item():.6f}")
    print(f"    Max:  {abs_error_nosearch.max().item():.6f}")
    print("  With search:")
    print(f"    Mean: {abs_error_search.mean().item():.6f}")
    print(f"    Max:  {abs_error_search.max().item():.6f}")
    print()


if __name__ == "__main__":
    # Run detailed single tensor test
    test_single_tensor_verbose()
    
    # Run multiple tests for statistical significance
    test_scale_search_reconstruction(block_size=16, num_tests=10)

