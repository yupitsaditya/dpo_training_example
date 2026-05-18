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

echo "Starting vLLM OpenAI API Server with model: $MODEL_NAME on 8 GPUs"

vllm serve $MODEL_NAME \
    --tensor-parallel-size 8 \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --enable-reasoning \
    --reasoning-parser gemma4
