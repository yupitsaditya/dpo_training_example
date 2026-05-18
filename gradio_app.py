import gradio as gr
from openai import AsyncOpenAI
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Connect to our local vLLM instance serving Gemma 4
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-no-key-required"
)

MODEL_NAME = "google/gemma-4-31b-it"

# ==========================================
# LOGIC
# ==========================================
async def chat_stream(message, history):
    """
    Streams the response from the vLLM server, separating the 'reasoning' tokens
    (Gemma 4's thinking process) from the final content.
    """
    # Format history for OpenAI API
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
        
    messages.append({"role": "user", "content": message})

    # Start the stream
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
        temperature=0.7,
        max_tokens=2048,
        # If required by specific vLLM configurations for reasoning override:
        # extra_body={"chat_template_kwargs": {"thinking": True}}
    )

    reasoning_text = ""
    content_text = ""
    
    # We will yield the combined Markdown string iteratively to Gradio
    async for chunk in stream:
        delta = chunk.choices[0].delta
        
        # In modern vLLM/OpenAI specs, the reasoning tokens are sent in a separate field.
        # We use getattr or dict access to safely handle it depending on the exact SDK version.
        delta_dict = delta.model_dump()
        
        # Extract reasoning tokens (vLLM usually sends this in `reasoning` or `reasoning_content`)
        chunk_reasoning = delta_dict.get("reasoning", "") or delta_dict.get("reasoning_content", "")
        if not chunk_reasoning and hasattr(delta, 'reasoning'):
            chunk_reasoning = getattr(delta, 'reasoning') or ""
            
        if chunk_reasoning:
            reasoning_text += chunk_reasoning
            
        # Extract normal content tokens
        chunk_content = delta.content or ""
        if chunk_content:
            content_text += chunk_content
            
        # Construct the UI string
        # We put the reasoning inside a native HTML/Markdown <details> block so the user can expand/collapse it.
        display_text = ""
        if reasoning_text:
            display_text += f"<details open><summary><b>🧠 Gemma's Thinking Process</b></summary>\n\n> {reasoning_text.replace(chr(10), chr(10) + '> ')}\n\n</details>\n\n---\n\n"
            
        if content_text:
            display_text += content_text
            
        if display_text:
            yield display_text


# ==========================================
# UI DEFINITION
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# Gemma 4 ({MODEL_NAME}) - Reasoning UI")
    gr.Markdown("This interface connects to your local vLLM server. It automatically parses and formats Gemma 4's chain-of-thought thinking tokens!")
    
    chat_interface = gr.ChatInterface(
        fn=chat_stream,
        fill_height=True,
        chatbot=gr.Chatbot(height=600, render_markdown=True),
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
