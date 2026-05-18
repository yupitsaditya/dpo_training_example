import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import ORPOTrainer, ORPOConfig
from datasets import load_dataset

# ==========================================
# HYPERPARAMETERS & VARIABLES
# ==========================================
MODEL_ID = "google/gemma-4-31b-it"
DATASET_PATH = "dpo_dataset.jsonl"
OUTPUT_DIR = "./orpo_results"

# In ORPO, beta is the penalty weight for the odds ratio loss. 0.1 is standard.
BETA = 0.1

MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 512

BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 4
# ORPO can sometimes benefit from a slightly smaller learning rate than DPO
LEARNING_RATE = 8e-6 

# ==========================================
# MAIN TRAINING LOGIC
# ==========================================
def main():
    # 1. Load Dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # 2. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    # 3. Configure 4-bit Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 4. Load Base Model
    print(f"Loading base model {MODEL_ID} with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Prepares the model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # 5. Define LoRA Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 6. ORPO Training Configuration
    training_args = ORPOConfig(
        output_dir=OUTPUT_DIR,
        beta=BETA,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_steps=100,
        bf16=True, 
        report_to="none"
    )

    # 7. Initialize ORPOTrainer
    # Note: ORPO does NOT use a reference model entirely, which heavily reduces VRAM
    # and compute costs compared to DPO.
    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # 8. Run Training
    print("Starting ORPO training...")
    trainer.train()

    # 9. Save final adapters
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Training complete. Final adapters and tokenizer saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
