#!/usr/bin/env python3
"""Evaluate ModelOpt quantized checkpoints on wikitext."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
import sys

def evaluate_model(model_path, task="wikitext", batch_size=4):
    """Evaluate a ModelOpt quantized model on a task."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"Task: {task}")
    print(f"{'='*60}\n")
    
    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Create HF model wrapper for lm-eval
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    
    # Run evaluation
    task_manager = TaskManager()
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=[task],
        num_fewshot=0,
        batch_size=batch_size,
        task_manager=task_manager,
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Results for {model_path}:")
    print(f"{'='*60}")
    for task_name, task_results in results["results"].items():
        print(f"\nTask: {task_name}")
        for metric, value in task_results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    # Evaluate both quantized models
    models = [
        "/resource/llama-quant-nosearch-aligned/",
        "/resource/llama-quant-search-aligned/",
    ]
    
    all_results = {}
    for model_path in models:
        try:
            results = evaluate_model(model_path, task="wikitext", batch_size=4)
            all_results[model_path] = results
        except Exception as e:
            print(f"ERROR evaluating {model_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for model_path, results in all_results.items():
        print(f"\n{model_path}:")
        if "results" in results:
            for task_name, task_results in results["results"].items():
                for metric, value in task_results.items():
                    if "word_perplexity" in metric or "byte_perplexity" in metric:
                        if isinstance(value, float):
                            print(f"  {metric}: {value:.4f}")

