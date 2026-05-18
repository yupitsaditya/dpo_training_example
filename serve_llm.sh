#!/bin/bash
# serve_llm.sh
# 
# This script hosts a local LLM using vLLM on 8 GPUs. 
# It provides an OpenAI-compatible API endpoint which can be used by 
# litellm or the openai sdk for generating the DPO dataset.
#
# Requirements: pip install vllm
#
# Using a state-of-the-art reasoning model provides much better code-understanding to generate hard negatives.
# Gemma-4-31B will run very fast and comfortably across 8 GPUs using tensor parallelism.

MODEL_NAME="google/gemma-4-31b-it"

# Drop to 4 GPUs to ensure Tensor Parallel Size is a power of 2.
# Prime numbers like 7 cause vLLM's C++ PagedAttention kernels to segfault due to uneven head division!
export CUDA_VISIBLE_DEVICES="1,2,3,4"
TENSOR_PARALLEL_SIZE=4

echo "Starting vLLM OpenAI API Server with model: $MODEL_NAME on GPUs: $CUDA_VISIBLE_DEVICES"

vllm serve $MODEL_NAME \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096
