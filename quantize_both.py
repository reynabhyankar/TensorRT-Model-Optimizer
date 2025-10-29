#!/usr/bin/env python3
"""Quantize model with and without scale search to compare."""

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import modelopt.torch.quantization as mtq
from tqdm import tqdm
import os
import sys

def get_cnn_dataloader(model_name, num_samples=512, batch_size=8, seed=42, max_length=512):
    """Create a calibration dataloader using only the CNN/DailyMail dataset."""
    cnn_ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train[:5%]")
    
    def normalize_cnn(ex):
        return {"text": ex["article"]}
    
    cnn_ds = cnn_ds.map(normalize_cnn, remove_columns=cnn_ds.column_names)
    cnn_ds = cnn_ds.shuffle(seed=seed).select(range(min(num_samples, len(cnn_ds))))
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    
    tokenized_ds = cnn_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    dataloader = DataLoader(
        tokenized_ds,
        batch_size=batch_size,
        shuffle=False,
    )
    
    return dataloader

def quantize_and_export(use_scale_search: bool):
    """Quantize model with or without scale search and export."""
    
    suffix = "search" if use_scale_search else "nosearch"
    export_dir = f"./llama-quant-{suffix}-aligned/"
    
    print("\n" + "=" * 80)
    print(f"Quantizing model {'WITH' if use_scale_search else 'WITHOUT'} scale search")
    print(f"Export directory: {export_dir}")
    print("=" * 80 + "\n")
    
    # Set environment variable
    os.environ["USE_SCALE_SEARCH"] = "1" if use_scale_search else "0"
    print(f"Environment variable USE_SCALE_SEARCH is set to: {os.getenv('USE_SCALE_SEARCH')}")
    
    # Setup the model and calibration set
    model_name = "Qwen/Qwen3-8B"
    device = torch.device("cuda")
    
    print(f"\nLoading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Creating calibration dataloader...")
    calib_set = get_cnn_dataloader(model_name)
    
    def forward_loop(model):
        """Runs the model forward over the calibration set."""
        model.eval()
        for batch in tqdm(calib_set, desc="Calibrating"):
            batch_cuda = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            model(**batch_cuda)
    
    # Quantize
    print("\nStarting quantization...")
    model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop=forward_loop)
    print("Quantization complete!")
    
    # Verify quantization
    has_quantizers = any("quantizer" in name for name, _ in model.named_modules())
    print(f"Model has quantizers: {has_quantizers}")
    
    # Export
    from modelopt.torch.export import export_hf_checkpoint
    print(f"\nExporting quantized model to {export_dir}...")
    
    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir=export_dir)
    
    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer.save_pretrained(export_dir)
    
    print(f"\n✓ Export complete to {export_dir}!")
    
    # Clean up to free memory
    del model
    torch.cuda.empty_cache()
    print("Cleared GPU memory\n")


if __name__ == "__main__":
    # Quantize WITHOUT scale search
    quantize_and_export(use_scale_search=False)
    
    # Quantize WITH scale search
    quantize_and_export(use_scale_search=True)
    
    print("\n" + "=" * 80)
    print("BOTH MODELS CREATED SUCCESSFULLY!")
    print("=" * 80)
    print("\nYou can now run: bash eval_vllm.sh")
    print("\nExpected results:")
    print("  - nosearch model: Higher perplexity")
    print("  - search model:   Lower perplexity (better!)")

