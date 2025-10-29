#!/bin/bash

cd examples/llm_eval

echo ""
echo "=== Evaluating No Scale Search (Pre-Quantized) ==="
python lm_eval_hf.py \
    --model hf \
    --model_args pretrained=/resource/llama-quant-nosearch-aligned/ \
    --tasks wikitext \
    --batch_size 4

echo ""
echo "=== Evaluating With Scale Search (Pre-Quantized) ==="
python lm_eval_hf.py \
    --model hf \
    --model_args pretrained=/resource/llama-quant-search-aligned/ \
    --tasks wikitext \
    --batch_size 4