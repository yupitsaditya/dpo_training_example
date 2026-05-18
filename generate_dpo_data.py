import json
import asyncio
from litellm import acompletion
import os

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "sft_data.jsonl"
OUTPUT_FILE = "dpo_dataset.jsonl"

# The vLLM API server endpoint exposed by `serve_llm.sh`
API_BASE = "http://localhost:8000/v1"
# For vLLM using the OpenAI spec, the model name used by litellm must match 
# exactly what is hosted (or prefix with openai/ for litellm)
MODEL_NAME = "openai/Qwen/Qwen3.6-Coder-32B-Instruct"

CONCURRENCY_LIMIT = 32  # Controls how many parallel requests hit the server

SYSTEM_PROMPT = """You are an expert code mutator. I will give you a valid Python snippet that orchestrates internal APIs. You must introduce exactly ONE realistic logical error: either hallucinate a plausible but non-existent parameter (like `timeout=30`), OR swap the execution order of two dependent API calls. Output strictly the mutated raw code without markdown or explanations."""

# ==========================================
# LOGIC
# ==========================================
async def generate_rejected_example(session_semaphore, prompt: str, chosen: str) -> str:
    """
    Calls the local LLM concurrently to generate a mutated (rejected) code snippet.
    """
    async with session_semaphore:
        try:
            # We use litellm.acompletion to call the vLLM OpenAI endpoint
            response = await acompletion(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": chosen}
                ],
                api_base=API_BASE,
                temperature=0.7, # Adds slight randomness for varied hallucinations
                max_tokens=512,
            )
            
            # Extract the raw response text
            rejected = response.choices[0].message.content.strip()
            
            # Defensive post-processing: remove markdown code block backticks if present
            if rejected.startswith("```python"):
                rejected = rejected[9:]
            if rejected.startswith("```"):
                rejected = rejected[3:]
            if rejected.endswith("```"):
                rejected = rejected[:-3]
                
            return rejected.strip()
            
        except Exception as e:
            print(f"Error generating negative for prompt: {prompt[:30]}... Error: {e}")
            return None

async def process_data():
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file {INPUT_FILE} not found.")
        print("Please ensure your SFT data exists with 'prompt' and 'chosen' keys per line.")
        return

    # Read successful SFT data
    data = []
    with open(INPUT_FILE, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} examples. Starting concurrent generation...")
    
    # Create batch of asynchronous tasks
    tasks = []
    for item in data:
        tasks.append(generate_rejected_example(semaphore, item["prompt"], item["chosen"]))
        
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Write the completed DPO format output
    valid_count = 0
    with open(OUTPUT_FILE, "w") as f:
        for i, rejected in enumerate(results):
            if rejected: # If generation didn't fail
                dpo_item = {
                    "prompt": data[i]["prompt"],
                    "chosen": data[i]["chosen"],
                    "rejected": rejected
                }
                f.write(json.dumps(dpo_item) + "\n")
                valid_count += 1
                
    print(f"Successfully generated {valid_count}/{len(data)} DPO pairs.")
    print(f"Saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    asyncio.run(process_data())
