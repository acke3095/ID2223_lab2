import time
import threading
import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# ----------------------------------------------------
# Model registry
# ----------------------------------------------------

AVAILABLE_MODELS = {
    "Llama 3.2 1B Instruct": {
        "repo_id": "axelblenna/model",
        "filename": "llama-3.2-1b-instruct.Q4_K_M.gguf",
        "system_prompt": "You are a helpful and friendly assistant fine-tuned from Llama 3.2 1B.",
        "prompt_type": "llama3",
        "n_ctx": 2048,
    },
    "Llama 3.2 3B Instruct": {
        "repo_id": "axelblenna/model_llama3B_instruct",
        "filename": "llama-3.2-3b-instruct.Q4_K_M.gguf",
        "system_prompt": "You are a helpful and friendly assistant fine-tuned from Llama 3.2 3B.",
        "prompt_type": "llama3",
        "n_ctx": 2048,
    },
    "Qwen 2.5 1.5B Instruct": {
        "repo_id": "axelblenna/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct.Q4_K_M.gguf",
        "system_prompt": "You are a helpful and friendly assistant fine-tuned from Qwen 2.5.",
        "prompt_type": "qwen",
        "n_ctx": 2048,
    },
    "Llama 3.2 1B LoRA + Training": {
        "repo_id": "axelblenna/Llama-1B-LoRA-Rank-r32",
        "filename": "llama-3.2-1b-instruct.Q4_K_M.gguf",
        "system_prompt": "You are a helpful and friendly assistant fine-tuned from Llama 3.2 1B.",
        "prompt_type": "llama3",
        "n_ctx": 2048,
    },
}

# ----------------------------------------------------
# Global state
# ----------------------------------------------------

llm = None
llm_lock = threading.Lock()

SYSTEM_PROMPT = ""
CURRENT_MODEL_NAME = ""
CURRENT_PROMPT_TYPE = ""

MAX_TURNS = 4  # prevent context overflow

# ----------------------------------------------------
# Model loading (SAFE)
# ----------------------------------------------------

def load_model(model_name):
    global llm, SYSTEM_PROMPT, CURRENT_MODEL_NAME, CURRENT_PROMPT_TYPE

    model_info = AVAILABLE_MODELS[model_name]
    CURRENT_PROMPT_TYPE = model_info["prompt_type"]

    model_path = hf_hub_download(
        repo_id=model_info["repo_id"],
        filename=model_info["filename"],
    )

    with llm_lock:
        llm = Llama(
            model_path=model_path,
            n_ctx=model_info["n_ctx"],
            n_threads=6,
            n_gpu_layers=0,
            n_batch=128,
            verbose=False,
        )

    SYSTEM_PROMPT = model_info["system_prompt"]
    CURRENT_MODEL_NAME = model_name

    # IMPORTANT: reset chat history on model switch
    return f"**Current model:** {model_name}", []


# Load default model
DEFAULT_MODEL = "Llama 3.2 1B LoRA + Training"
load_model(DEFAULT_MODEL)

# ----------------------------------------------------
# Prompt building
# ----------------------------------------------------

def build_prompt(history, user_msg):
    history = history[-MAX_TURNS:]

    if CURRENT_PROMPT_TYPE == "qwen":
        return build_qwen_prompt(history, user_msg)
    else:
        return build_llama_prompt(history, user_msg)


def build_qwen_prompt(history, user_msg):
    prompt = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
    )

    for turn in history:
        role = turn["role"]
        prompt += (
            f"<|im_start|>{role}\n"
            f"{turn['content']}\n"
            "<|im_end|>\n"
        )

    prompt += (
        "<|im_start|>user\n"
        f"{user_msg}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    return prompt


def build_llama_prompt(history, user_msg):
    prompt = (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>\n"
    )

    for turn in history:
        prompt += (
            f"<|start_header_id|>{turn['role']}<|end_header_id|>\n\n"
            f"{turn['content']}<|eot_id|>\n"
        )

    prompt += (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    return prompt

# ----------------------------------------------------
# Chat handler (STABLE)
# ----------------------------------------------------

def chat_handler(message, history):
    model_name = CURRENT_MODEL_NAME

    # placeholder assistant message
    history = history + [{
        "role": "assistant",
        "content": f"**{model_name}** · generating..."
    }]

    prompt = build_prompt(history[:-1], message)

    start_time = time.time()
    response = ""

    with llm_lock:
        for chunk in llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop=["<|eot_id|>", "<|im_end|>"],
            stream=True,
        ):
            if "choices" in chunk and "text" in chunk["choices"][0]:
                response += chunk["choices"][0]["text"]
                history[-1]["content"] = (
                    f"**{model_name}** · generating...\n\n{response}"
                )
                yield history, history, ""

    latency = time.time() - start_time

    # finalize messages
    history[-1]["content"] = (
        f"**{model_name}** · {latency:.2f}s\n"
        "— — —\n\n"
        f"{response}"
    )

    history.insert(-1, {"role": "user", "content": message})

    yield history, history, ""

# ----------------------------------------------------
# Gradio UI
# ----------------------------------------------------

with gr.Blocks() as interface:
    gr.Markdown("# ID2223 Lab 2: Fine-tune LLM")

    model_selector = gr.Dropdown(
        choices=list(AVAILABLE_MODELS.keys()),
        value=DEFAULT_MODEL,
        label="Select Model",
    )

    model_status = gr.Markdown("")
    chatbot = gr.Chatbot(height=500, type="messages")
    state = gr.State([])

    msg = gr.Textbox(label="Your message")

    msg.submit(
        chat_handler,
        inputs=[msg, state],
        outputs=[chatbot, state, msg],
        queue=True,
    )

    model_selector.change(
        fn=load_model,
        inputs=model_selector,
        outputs=[model_status, state],
    )

if __name__ == "__main__":
    interface.launch()
