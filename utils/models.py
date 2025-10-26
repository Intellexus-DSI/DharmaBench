"""
Wrapper for different LLM APIs.
"""

####################################################################################################
# Imports
import os

from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_together import ChatTogether

# from langchain_cohere import ChatCohere
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.rate_limiters import InMemoryRateLimiter

import sys
sys.path.append("../") 
from utils.pydantic_schemas import PYDANTIC_SCHEMAS
from utils.typed_schemas import TYPED_SCHEMAS

####################################################################################################

####################################################################################################
# Constants
BATCH_SIZE = 16

MODEL_CONFIGS = {
    "gpt-4o": {
        "display_name": "GPT-4o",
        "model_id": "gpt-4o",
        "structured_output": "pydantic",
        "provider": "openai",
    },
    "gpt-4o-mini": {
        "display_name": "GPT-4o mini",
        "model_id": "gpt-4o-mini",
        "structured_output": "pydantic",
        "provider": "openai",
    },
    "scout": {
        "display_name": "Llama-4 Scout",
        "model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "provider": "together",
        "structured_output": "pydantic",
    },
    "qwen-72b": {
        "display_name": "Qwen2.5-72B",
        "model_id": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "structured_output": "pydantic", #"typed_dict",
        "provider": "together",
    },
    "gemma-mitra": {
        "display_name": "Gemma Mitra",
        "model_id": "buddhist-nlp/gemma2-mitra-base",
        "structured_output": None,
        "provider": "local",
    },
    "gemini-2.0-flash": {
        "display_name": "Gemini-2 Flash",
        "model_id": "gemini-2.0-flash",
        "structured_output": "pydantic",
        "provider": "google",
    },
    "gemini-2.5-flash": {
        "display_name": "Gemini-2.5 Flash",
        "model_id": "gemini-2.5-flash",
        "structured_output": "pydantic",
        "provider": "google",
    },
    "gemini-2.5-pro": {
        "display_name": "Gemini-2.5 Pro",
        "model_id": "gemini-2.5-pro",
        "structured_output": "pydantic",
        "provider": "google",
    },
    "claude-3-haiku": {
        "display_name": "Claude-3 Haiku",
        "model_id": "claude-3-haiku-20240307",
        "structured_output": "pydantic",
        "provider": "anthropic",
    },
    "claude-3.7-sonnet": {
        "display_name": "Claude-3.7 Sonnet",
        "model_id": "claude-3-7-sonnet-20250219",
        "structured_output": "pydantic",
        "provider": "anthropic",
    },
    "claude-4-sonnet": {
        "display_name": "Claude-4 Sonnet",
        "model_id": "claude-sonnet-4-20250514",
        "structured_output": "pydantic",
        "provider": "anthropic",
    },
    "deepseek-r1": {
        "display_name": "DeepSeer-R1",
        "model_id": "deepseek-ai/DeepSeek-R1",
        "structured_output": "None",
        "provider": "together",
    },
    "aya-expanse-8b": {
        "display_name": "Aya Expanse 8B",
        "model_id": "c4ai-aya-expanse-8b",
        "structured_output": "pydantic",
        "provider": "cohere",
    },
    "aya-expanse-32b": {
        "display_name": "Aya Expanse 32B",  
        "model_id": "c4ai-aya-expanse-32b",
        "structured_output": "pydantic",
        "provider": "cohere",
    },
    "TowerInstruct-7B": {
        "display_name": "TowerInstruct-7B",
        "model_id": "Unbabel/TowerInstruct-7B-v0.2",
        "structured_output": None,
        "provider": "local",
    },
    "EMMA-500-7B": {
        "display_name": "EMMA-500-7B",
        "model_id": "MaLA-LM/emma-500-llama2-7b",
        "structured_output": None,
        "provider": "local",
    },
    "BayLing-2-8B": {
        "display_name": "BayLing-2-8B",
        "model_id": "ICTNLP/bayling-2-llama-3-8b",
        "structured_output": None,
        "provider": "local",
    },
    "LLaMAX-3-8B": {
        "display_name": "LLaMAX-3-8B",
        "model_id": "LLaMAX/LLaMAX3-8B",
        "structured_output": None,
        "provider": "local",
    },
    "TiLamb-7B": {
        "display_name": "TiLamb-7B",
        "model_id": "YoLo2000/TiLamb-7B",
        "structured_output": None,
        "provider": "local",
    },
}

####################################################################################################

####################################################################################################
# Models

def get_schema(model_name: str, task_type: str):
    print(f"Getting schema for model {model_name} and task type {task_type}")
    structured_output = MODEL_CONFIGS[model_name].get("structured_output", None)
    if structured_output is None:
        return None
    if structured_output == "pydantic":
        return PYDANTIC_SCHEMAS.get(task_type, None)
    elif structured_output == "typed_dict":
        return TYPED_SCHEMAS.get(task_type, None)
    else:
        # Inform the user and return None
        print(f"Model {model_name} does not support structured output.")
        return None



def get_model(model_name: str, temperature: float = 0.0, use_rate_limiter: bool = False, requests_per_second=0.8, check_every_n_seconds=1.5) -> LLMChain:
    """
    Get the model based on the model name.
    :param model_name: The name of the model to get.
    :param temperature: The temperature for the model.
    :param use_rate_limiter: Whether to use a rate limiter for the model.
    :return: The model instance as an LLMChain.
    """
    assert model_name in MODEL_CONFIGS, f"Model {model_name} not found in MODEL_CONFIGS"
    if use_rate_limiter:
        rate_limiter = rate_limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=check_every_n_seconds,  # Wake up every X ms to check whether allowed to make a request,
            max_bucket_size=1,  # Controls the maximum burst size.
        )
    else:
        rate_limiter = None

    model_configs = MODEL_CONFIGS[model_name]
    model_name = model_configs["model_id"]
    provider = model_configs["provider"]

    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=model_name, temperature=temperature, max_retries=3, rate_limiter=rate_limiter
        )

    elif provider == "openai":
        if "o3" in model_name.lower() or "o1" in model_name.lower():
            return ChatOpenAI(model_name=model_name, max_retries=3, rate_limiter=rate_limiter)

        else:
            return ChatOpenAI(
                model_name=model_name, temperature=temperature, max_retries=3, rate_limiter=rate_limiter
            )

    elif provider == "anthropic":
        return ChatAnthropic(model=model_name, temperature=temperature, max_retries=3, rate_limiter=rate_limiter)
    elif provider == "together":
        return ChatTogether(model=model_name, temperature=temperature, max_retries=3, rate_limiter=rate_limiter)
    elif provider == "cohere":
        return ChatCohere(model=model_name, temperature=temperature, rate_limiter=rate_limiter)


    elif provider == "local":
        os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

        llm = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            pipeline_kwargs={"temperature": temperature, "do_sample":True, "max_new_tokens": 4096},
            #model_kwargs={"token": access_token_read},
            batch_size=BATCH_SIZE,
            device_map="auto",
        )
        return llm
    else:
        raise ValueError(f"Unknown model {model_name}")

