import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# ==========================================
# HYPERPARAMETERS & VARIABLES
# ==========================================
MODEL_ID = "google/gemma-4-31b-it"
DATASET_PATH = "dpo_dataset.jsonl"
OUTPUT_DIR = "./dpo_results"

# DPO beta parameter controls the strength of the preference penalty.
BETA = 0.1  

MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 512

# Training setup suitable for 8 GPUs
BATCH_SIZE = 4                  # Per device
GRAD_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-5            # LoRA typically uses slightly higher learning rates than full finetuning

# ==========================================
# MAIN TRAINING LOGIC
# ==========================================
def main():
    # 1. Load Dataset
    # Assuming dataset format is standard: {"prompt": "...", "chosen": "...", "rejected": "..."}
    # TRL's DPOTrainer will natively handle this format.
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # 2. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        # Llama 3 does not define a pad token by default
        tokenizer.pad_token = tokenizer.eos_token 

    # 3. Configure 4-bit Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # Recommended for newer GPUs (Ampere/Hopper)
    )

    # 4. Load Base Model
    print(f"Loading base model {MODEL_ID} with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto" # Automatically scales across the 8 GPUs
    )
    
    # Prepares the model for k-bit training (handles gradient checkpointing and freezing base weights)
    model = prepare_model_for_kbit_training(model)

    # 5. Define LoRA Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Specific modules requested
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 6. DPO Training Configuration
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=BETA,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        optim="paged_adamw_32bit", # Memory efficient optimizer for QLoRA
        logging_steps=10,
        save_steps=100,
        bf16=True, # BFloat16 precision, excellent for Ampere+ GPUs
        report_to="none" # Switch to "wandb" if you use Weights and Biases
    )

    # 7. Initialize DPOTrainer
    # Notice: We do NOT pass `ref_model`. Since we pass a `peft_config`, the DPOTrainer 
    # implicitly uses the frozen base model (by disabling adapters) as the reference model!
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # 8. Run Training
    print("Starting DPO training...")
    trainer.train()

    # 9. Save final adapters
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Training complete. Final adapters and tokenizer saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
