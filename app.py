import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# ----------------------------------------------------
# Model Download
# ----------------------------------------------------

REPO_ID = "axelblenna/model"
FILENAME = "llama-3.2-1b-instruct.Q4_K_M.gguf"

MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
print(f"Model downloaded to: {MODEL_PATH}")

# ----------------------------------------------------
# Load Model (CPU Space friendly)
# ----------------------------------------------------

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1024,
    n_threads=6,
    n_gpu_layers=0,
    n_batch=128,
    verbose=False,
)

SYSTEM_PROMPT = (
    "You are a helpful and friendly assistant fine-tuned from Llama 3.2 1B."
)

# ----------------------------------------------------
# Llama 3 Prompt Builder
# ----------------------------------------------------

def build_prompt(history, user_msg):
    prompt = (
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>\n"
    )

    # History is list of dictionaries:
    # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    for turn in history:
        if turn["role"] == "user":
            prompt += (
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{turn['content']}<|eot_id|>\n"
            )
        elif turn["role"] == "assistant":
            prompt += (
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{turn['content']}<|eot_id|>\n"
            )

    # Add final user message
    prompt += (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n "
    )

    return prompt

# ----------------------------------------------------
# Chat Response Generator
# ----------------------------------------------------

def chat_fn(message, history):
    prompt = build_prompt(history, message)

    stream = llm(
        prompt,
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
        stream=True,
    )

    response = ""
    for chunk in stream:
        if "choices" in chunk and "text" in chunk["choices"][0]:
            delta = chunk["choices"][0]["text"]
            response += delta
            yield response


# ----------------------------------------------------
# Gradio UI
# ----------------------------------------------------

interface = gr.ChatInterface(
    fn=chat_fn,
    chatbot=gr.Chatbot(height=500, type="messages"),
    title="Fine-Tuned Llama 3.2 1B Chatbot",
    description="Running on CPU using llama-cpp-python.",
)

if __name__ == "__main__":
    interface.launch()
