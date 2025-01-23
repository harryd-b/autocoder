#!/usr/bin/env python3
"""
api_utils.py

Handles:
1. OpenAI API interactions (with retries and error handling).
2. DeepSeek-Reasoner API interactions (OpenAI-compatible endpoint)
3. Local Triton server calls to Meta-Llama-3-8B on port 8000 (or as configured),
   also with retries and batch prompt support.

Usage:
- For OpenAI calls, use call_openai_chat_completion(messages).
- For DeepSeek calls, use call_deepseek_chat_completion(messages).
- For local Llama calls, use call_local_llama_inference(prompts).
"""

import logging
import os
import yaml
import numpy as np
import tritonclient.http as httpclient
from requests.exceptions import RequestException

import openai
from dotenv import load_dotenv
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type
)

load_dotenv()  # Load environment variables from .env

class OpenAIAPIError(Exception):
    """Custom exception for OpenAI API errors."""
    pass

class DeepSeekAPIError(Exception):
    """Custom exception for DeepSeek API errors."""
    pass

class LocalLLMError(Exception):
    """Custom exception for local Llama Triton server errors."""
    pass

# Load configuration from config.yaml
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Model configurations
GPT_MODEL = config.get("openai_model", "gpt-3.5-turbo")
DEEPSEEK_MODEL = config.get("deepseek_model", "deepseek-reasoner")
DEEPSEEK_BASE_URL = config.get("deepseek_base_url", "https://api.deepseek.com/v1")

# Triton settings
LOCAL_TRITON_URL = os.getenv("LOCAL_TRITON_URL", "localhost:8000")
TRITON_MODEL_NAME = config.get("triton_model_name", "meta-llama_Meta-Llama-3-8B")

# Retry/backoff settings
MAX_RETRIES = config.get("max_retries", 3)
RETRY_BASE_SECONDS = config.get("retry_base_seconds", 1.5)
RETRY_MAX_SECONDS = config.get("retry_max_seconds", 10)

# Initialize API clients
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
deepseek_client = openai.OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
) if DEEPSEEK_API_KEY else None

###############################################################################
# 1. OpenAI ChatCompletion
###############################################################################

@retry(
    reraise=True,
    wait=wait_exponential(multiplier=RETRY_BASE_SECONDS, max=RETRY_MAX_SECONDS),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type(OpenAIAPIError)
)
def call_openai_chat_completion(
    messages: list[dict[str, str]], 
    model: str = GPT_MODEL
) -> openai.types.chat.ChatCompletion:
    """
    Calls the OpenAI ChatCompletion API with retry logic.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: OpenAI model identifier
        
    Returns:
        OpenAI ChatCompletion response
        
    Raises:
        OpenAIAPIError: If API call fails
    """
    if not openai_client:
        raise OpenAIAPIError("OpenAI client not initialized - check API key")

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response
    except Exception as e:
        logging.error("OpenAI API call failed.", exc_info=True)
        raise OpenAIAPIError(str(e)) from e

###############################################################################
# 2. DeepSeek ChatCompletion
###############################################################################

@retry(
    reraise=True,
    wait=wait_exponential(multiplier=RETRY_BASE_SECONDS, max=RETRY_MAX_SECONDS),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type(DeepSeekAPIError)
)
def call_deepseek_chat_completion(messages: list[dict], model: str = DEEPSEEK_MODEL) -> dict:
    """
    Calls the DeepSeek ChatCompletion API with retry logic.
    """
    if not deepseek_client:
        raise DeepSeekAPIError("DeepSeek client not initialized - check API key")

    try:
        response = deepseek_client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response
    except Exception as e:
        logging.error("DeepSeek API call failed.", exc_info=True)
        raise DeepSeekAPIError(str(e)) from e

###############################################################################
# 3. Local Triton Llama inference
###############################################################################

MAX_BATCH_SIZE = config.get('max_batch_size', 32)

@retry(
    reraise=True,
    wait=wait_exponential(multiplier=RETRY_BASE_SECONDS, max=RETRY_MAX_SECONDS),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type(LocalLLMError)
)
def call_local_llama_inference(prompts: list[str]) -> list[str]:
    """
    Calls a local Triton server hosting Meta-Llama-3-8B.
    
    Args:
        prompts: List of input prompts to process
        
    Returns:
        List of generated responses
        
    Raises:
        LocalLLMError: If inference fails
        ValueError: If batch size exceeds limit
    """
    if len(prompts) > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size {len(prompts)} exceeds maximum {MAX_BATCH_SIZE}")
    try:
        client = httpclient.InferenceServerClient(url=LOCAL_TRITON_URL)
    except Exception as e:
        logging.error("Failed to create Triton client.", exc_info=True)
        raise LocalLLMError(str(e)) from e

    batch_size = len(prompts)
    if batch_size == 0:
        return []

    input_data = np.array(prompts, dtype=object).reshape(batch_size, 1)
    input_tensor = httpclient.InferInput("TEXT", [batch_size, 1], "BYTES")
    input_tensor.set_data_from_numpy(input_data)

    output_tensor = httpclient.InferRequestedOutput("GENERATED_TEXT")

    try:
        results = client.infer(
            TRITON_MODEL_NAME,
            inputs=[input_tensor],
            outputs=[output_tensor],
        )
    except RequestException as re:
        logging.error("Triton request failed.", exc_info=True)
        raise LocalLLMError(str(re)) from re
    except Exception as ex:
        logging.error("Inference call failed.", exc_info=True)
        raise LocalLLMError(str(ex)) from ex

    out = results.as_numpy("GENERATED_TEXT")
    return [val.decode("utf-8") if isinstance(val, bytes) else val for val in out[:, 0]] if out is not None else []

###############################################################################
# EXAMPLE USAGE
###############################################################################

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test OpenAI
    try:
        openai_res = call_openai_chat_completion([
            {"role": "user", "content": "Hello from OpenAI!"}
        ])
        logging.info(f"OpenAI Response: {openai_res.choices[0].message.content}")
    except OpenAIAPIError as e:
        logging.error(f"OpenAI Error: {e}")

    # Test DeepSeek
    try:
        deepseek_res = call_deepseek_chat_completion([
            {"role": "user", "content": "Hello from DeepSeek!"}
        ])
        logging.info(f"DeepSeek Response: {deepseek_res.choices[0].message.content}")
    except DeepSeekAPIError as e:
        logging.error(f"DeepSeek Error: {e}")

    # Test Local Llama
    try:
        llama_res = call_local_llama_inference(["Hello from Llama!"])
        logging.info(f"Llama Response: {llama_res[0]}")
    except LocalLLMError as e:
        logging.error(f"Llama Error: {e}")

    logging.info("All tests complete.")

# After loading config
def validate_config(config: dict) -> None:
    """Validates the configuration values."""
    required_fields = ['openai_model', 'deepseek_model', 'deepseek_base_url']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
            
    if config.get('max_retries', 0) < 1:
        raise ValueError("max_retries must be positive")
        
# Add after config loading:
validate_config(config)