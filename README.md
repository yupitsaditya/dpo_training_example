# DPO & ORPO Training Pipeline with Gemma 4

This repository contains a complete pipeline for generating hard-negative data and fine-tuning reasoning models using Direct Preference Optimization (DPO) and Odds Ratio Preference Optimization (ORPO) on an 8-GPU setup. 

It is fully configured to use **Gemma 4** (`google/gemma-4-31b-it`), utilizing its native reasoning capabilities ("thinking" tokens) both during local serving and via an interactive UI.

## Repository Contents

* `serve_llm.sh`: Bash script to host Gemma 4 using `vLLM` with tensor parallelism across 8 GPUs. It explicitly enables Gemma 4's reasoning parser.
* `generate_dpo_data.py`: An asynchronous script utilizing `litellm` to call the local vLLM API. It mutates SFT code snippets to generate rejected (hard negative) examples.
* `train_dpo.py`: Script to train Gemma 4 using DPO with 4-bit quantization (QLoRA).
* `train_orpo.py`: Script to train Gemma 4 using ORPO (Odds Ratio Preference Optimization) with QLoRA.
* `gradio_app.py`: A Gradio-based chat UI that connects to the vLLM server, natively extracting and rendering Gemma's "thinking" process in a collapsible markdown block.

## Environment Setup

We recommend using `conda` to manage the environment and ensure compatibility with CUDA and the latest versions of `vllm`, `transformers`, and `trl`.

```bash
# 1. Create the conda environment
conda create -n dpo_env python=3.10 -y

# 2. Activate the environment
conda activate dpo_env

# 3. Install PyTorch (adjust the CUDA version to match your system, e.g., cu121 or cu124)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install the requirements (ensures Gemma 4, DPO, and ORPO support)
pip install -r requirements.txt
```

## How to Run

### 1. Host the Model
Start the vLLM server on your 8 GPUs. This will expose an OpenAI-compatible API on `localhost:8000`.
```bash
bash serve_llm.sh
```

### 2. Generate the DPO Dataset
Ensure you have your base SFT data in `sft_data.jsonl` (format: `{"prompt": "...", "chosen": "..."}`).
```bash
python generate_dpo_data.py
```
*This will output `dpo_dataset.jsonl`.*

### 3. Run the Gradio UI (Optional)
If you want to interact with the model and see its thinking process:
```bash
python gradio_app.py
```
*Access the UI at `http://localhost:7860`.*

### 4. Train the Model
Once data generation is complete, stop the vLLM server to free up VRAM, then run your preferred training method:
```bash
python train_dpo.py
# OR
python train_orpo.py
```
