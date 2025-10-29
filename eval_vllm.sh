#!/bin/bash

# Script to evaluate ModelOpt quantized models using vLLM serving
# Based on the README.md in examples/llm_eval

# Port configuration
PORT_NOSEARCH=8000
PORT_SEARCH=8001

# Use all 8 GPUs
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CUDA_VISIBLE_DEVICES=4,5,6,7
# CUDA_DEVICE_MAX_CONNECTIONS=7
export CUDA_VISIBLE_DEVICES

# Completely disable torch compile to avoid Triton CUDA compatibility issues
export VLLM_TORCH_COMPILE_LEVEL=0
export TORCH_COMPILE_DISABLE=1

# Use data parallelism across 8 GPUs (more efficient for evaluation throughput)
DP_SIZE=4

# Task to evaluate (wikitext is not gated, gpqa requires authentication)
TASK=wikitext

# Base model name for output directory
BASE_MODEL="Qwen3-8B"
# Increase max length for GPQA's long contexts (some examples have 12K+ tokens)
MAX_SEQ_LEN=16384

# Output directory structure: output/{TASK}/{BASE_MODEL}/{TYPE}
OUTPUT_BASE="output/${TASK}/${BASE_MODEL}"
mkdir -p "${OUTPUT_BASE}/baseline"
mkdir -p "${OUTPUT_BASE}/nosearch"
mkdir -p "${OUTPUT_BASE}/search"

# Function to wait for server to be ready
wait_for_server() {
    local port=$1
    echo "Waiting for vLLM server on port $port to be ready..."
    for i in {1..60}; do
        if curl -s http://localhost:$port/v1/models > /dev/null 2>&1; then
            echo "Server is ready!"
            return 0
        fi
        sleep 2
    done
    echo "Server failed to start within 120 seconds"
    return 1
}

# Evaluate baseline BF16 model
PORT_BASELINE=7999
echo "=========================================="
echo "Starting vLLM server for BASELINE BF16 model"
echo "=========================================="

vllm serve Qwen/Qwen3-8B \
    --port $PORT_BASELINE \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --pipeline-parallel-size 1 \
    --tensor-parallel-size 1 \
    --data-parallel-size $DP_SIZE \
    --max-model-len $MAX_SEQ_LEN \
    --enforce-eager \
    &

VLLM_PID_BASELINE=$!
wait_for_server $PORT_BASELINE

if [ $? -eq 0 ]; then
    echo ""
    echo "Running evaluation on BASELINE BF16 model..."
    
    OUTPUT_PATH="${OUTPUT_BASE}/baseline"
    lm_eval \
        --model local-completions \
        --model_args model=Qwen/Qwen3-8B,base_url=http://localhost:${PORT_BASELINE}/v1/completions,tokenized_requests=False,num_concurrent=32 \
        --batch_size auto \
        --tasks ${TASK} \
        --output_path ${OUTPUT_PATH}
    
    echo ""
    echo "Killing vLLM server (PID: $VLLM_PID_BASELINE)..."
    kill $VLLM_PID_BASELINE
    wait $VLLM_PID_BASELINE 2>/dev/null
else
    echo "Failed to start vLLM server for baseline model"
    kill $VLLM_PID_BASELINE 2>/dev/null
fi

sleep 5

# Evaluate model without scale search
echo "=========================================="
echo "Starting vLLM server for NO SCALE SEARCH model"
echo "=========================================="

# Disable torch compile to avoid Triton CUDA compatibility issues
vllm serve /resource/llama-quant-nosearch-aligned/ \
    --quantization modelopt \
    --port $PORT_NOSEARCH \
    --gpu-memory-utilization 0.9 \
    --pipeline-parallel-size 1 \
    --tensor-parallel-size 1 \
    --data-parallel-size $DP_SIZE \
    --max-model-len $MAX_SEQ_LEN \
    --enforce-eager \
    &

VLLM_PID_NOSEARCH=$!
wait_for_server $PORT_NOSEARCH

if [ $? -eq 0 ]; then
    echo ""
    echo "Running evaluation on NO SCALE SEARCH model..."
    
    OUTPUT_PATH="${OUTPUT_BASE}/nosearch"
    lm_eval \
        --model local-completions \
        --model_args model=/resource/llama-quant-nosearch-aligned/,base_url=http://localhost:${PORT_NOSEARCH}/v1/completions,tokenized_requests=False,num_concurrent=32 \
        --batch_size auto \
        --tasks ${TASK} \
        --output_path ${OUTPUT_PATH}
    
    echo ""
    echo "Killing vLLM server (PID: $VLLM_PID_NOSEARCH)..."
    kill $VLLM_PID_NOSEARCH
    wait $VLLM_PID_NOSEARCH 2>/dev/null
else
    echo "Failed to start vLLM server for nosearch model"
    kill $VLLM_PID_NOSEARCH 2>/dev/null
fi

sleep 5

# Evaluate model with scale search
echo ""
echo "=========================================="
echo "Starting vLLM server for WITH SCALE SEARCH model"
echo "=========================================="

vllm serve /resource/llama-quant-search-aligned/ \
    --quantization modelopt \
    --port $PORT_SEARCH \
    --gpu-memory-utilization 0.9 \
    --pipeline-parallel-size 1 \
    --tensor-parallel-size 1 \
    --data-parallel-size $DP_SIZE \
    --max-model-len $MAX_SEQ_LEN \
    --enforce-eager \
    &

VLLM_PID_SEARCH=$!
wait_for_server $PORT_SEARCH

if [ $? -eq 0 ]; then
    echo ""
    echo "Running evaluation on WITH SCALE SEARCH model..."
    
    OUTPUT_PATH="${OUTPUT_BASE}/search"
    lm_eval \
        --model local-completions \
        --model_args model=/resource/llama-quant-search-aligned/,base_url=http://localhost:${PORT_SEARCH}/v1/completions,tokenized_requests=False,num_concurrent=32 \
        --batch_size auto \
        --tasks ${TASK} \
        --output_path ${OUTPUT_PATH}
    
    echo ""
    echo "Killing vLLM server (PID: $VLLM_PID_SEARCH)..."
    kill $VLLM_PID_SEARCH
    wait $VLLM_PID_SEARCH 2>/dev/null
else
    echo "Failed to start vLLM server for search model"
    kill $VLLM_PID_SEARCH 2>/dev/null
fi

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="

echo ""
echo "Results saved to:"
echo "  Baseline:  ${OUTPUT_BASE}/baseline/"
echo "  No Search: ${OUTPUT_BASE}/nosearch/"
echo "  Search:    ${OUTPUT_BASE}/search/"
echo ""

# Print summary if results exist
echo "Summary:"
echo "--------"
for TYPE in baseline nosearch search; do
    RESULTS_FILE="${OUTPUT_BASE}/${TYPE}/results.json"
    if [ -f "$RESULTS_FILE" ]; then
        echo "${TYPE}:"
        python3 -c "import json; data = json.load(open('${RESULTS_FILE}')); print('  Word Perplexity: {:.4f}'.format(data['results']['wikitext']['word_perplexity']))" 2>/dev/null || echo "  (results file exists but couldn't parse)"
    else
        echo "${TYPE}: No results found"
    fi
done

