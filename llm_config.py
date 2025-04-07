DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

SUPPORTED_LLM_MODELS = {
    "qwen2.5-0.5b-instruct": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "remote_code": False,
    },
    "tiny-llama-1b-chat": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "remote_code": False,
    },
    "DeepSeek-R1-Distill-Qwen-1.5B": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    },
    "DeepSeek-R1-Distill-Qwen-7B": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    },
    "DeepSeek-R1-Distill-Llama-8B": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    },
    "llama-3.2-1b-instruct": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
    },
    "llama-3.2-3b-instruct": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
    },
    "qwen2.5-1.5b-instruct": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "remote_code": False,
    },
    "gemma-2b-it": {
        "model_id": "google/gemma-2b-it",
        "remote_code": False,
    },
    "gemma-2-2b-it": {
        "model_id": "google/gemma-2-2b-it",
        "remote_code": False,
    },
    "red-pajama-3b-chat": {
        "model_id": "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
        "remote_code": False,
    },
    "qwen2.5-3b-instruct": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "remote_code": False,
    },
    "minicpm3-4b": {
        "model_id": "openbmb/MiniCPM3-4B", 
        "remote_code": True, 
    },
    "gemma-7b-it": {
        "model_id": "google/gemma-7b-it",
        "remote_code": False,
    },
    "gemma-2-9b-it": {
        "model_id": "google/gemma-2-9b-it",
        "remote_code": False,
    },
    "llama-2-chat-7b": {
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "remote_code": False,
    },
    "llama-3-8b-instruct": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "remote_code": False,
    },
    "llama-3.1-8b-instruct": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "remote_code": False,
    },
    "mistral-7b-instruct": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.1",
        "remote_code": False,
    },
    "zephyr-7b-beta": {
        "model_id": "HuggingFaceH4/zephyr-7b-beta",
        "remote_code": False,
    },
    "notus-7b-v1": {
        "model_id": "argilla/notus-7b-v1",
        "remote_code": False,
    },
    "neural-chat-7b-v3-3": {
        "model_id": "Intel/neural-chat-7b-v3-3",
        "remote_code": False,
    },
    "phi-3-mini-instruct": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "remote_code": True,
    },
    "phi-3.5-mini-instruct": {
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "remote_code": True,
    },
    "phi-4-mini-instruct": {"model_id": "microsoft/phi-4-mini-instruct", "remote_code": True},
    "phi-4": {"model_id": "microsoft/phi-4", "remote_code": False},
    "qwen2.5-14b-instruct": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "remote_code": False,
    },
}
