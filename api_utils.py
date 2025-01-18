#!/usr/bin/env python3
"""
api_utils.py

Handles:
1. OpenAI API interactions (with retries and error handling).
2. Local Triton server calls to Meta-Llama-3-8B on port 8000 (or as configured),
   also with retries and batch prompt support.

Usage:
- For OpenAI calls, use call_openai_chat_completion(messages).
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

class LocalLLMError(Exception):
    """Custom exception for local Llama Triton server errors."""
    pass

# Load configuration from config.yaml
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("No OPENAI_API_KEY found in .env")

openai.api_key = OPENAI_API_KEY
GPT_MODEL = config.get("model", "gpt-3.5-turbo")

# Triton settings
# Default to "localhost:8000" if not specified
LOCAL_TRITON_URL = os.getenv("LOCAL_TRITON_URL", "localhost:8000")
# Default model name for your Llama on Triton
TRITON_MODEL_NAME = config.get("triton_model_name", "meta-llama_Meta-Llama-3-8B")

# Retry / backoff settings
MAX_RETRIES = config.get("max_retries", 3)
RETRY_BASE_SECONDS = config.get("retry_base_seconds", 1.5)
RETRY_MAX_SECONDS = config.get("retry_max_seconds", 10)

###############################################################################
# 1. OpenAI ChatCompletion with retries
###############################################################################

@retry(
    reraise=True,
    wait=wait_exponential(multiplier=RETRY_BASE_SECONDS, max=RETRY_MAX_SECONDS),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type(OpenAIAPIError)
)
def call_openai_chat_completion(messages: list[dict], model: str = GPT_MODEL) -> dict:
    """
    Calls the OpenAI ChatCompletion API with retry logic.
    Raises OpenAIAPIError on failure.

    :param messages: The conversation history as a list of {"role": str, "content": str}
    :param model: The OpenAI model to use.
    :return: The OpenAI API response (as a dict).
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages
        )
        return response
    except Exception as e:
        logging.error("OpenAI API call failed.", exc_info=True)
        raise OpenAIAPIError(str(e)) from e

###############################################################################
# 2. Local Triton Llama inference with retries
###############################################################################

@retry(
    reraise=True,
    wait=wait_exponential(multiplier=RETRY_BASE_SECONDS, max=RETRY_MAX_SECONDS),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type(LocalLLMError)
)
def call_local_llama_inference(prompts: list[str]) -> list[str]:
    """
    Calls a local Triton server hosting Meta-Llama-3-8B on port 8000 (by default).
    Sends a batch of prompts (shape [batch_size, 1]) and returns a list of generated texts.

    Raises LocalLLMError on failure.

    :param prompts: A list of prompt strings.
    :return: A list of generated text outputs, one for each prompt.
    """
    try:
        client = httpclient.InferenceServerClient(url=LOCAL_TRITON_URL)
    except Exception as e:
        logging.error("Failed to create Triton client.", exc_info=True)
        raise LocalLLMError(str(e)) from e

    batch_size = len(prompts)
    if batch_size == 0:
        return []

    # Numpy array of shape [batch_size, 1], dtype=object
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

    # Parse out the output
    out = results.as_numpy("GENERATED_TEXT")
    if out is None:
        logging.warning("No output from model.")
        return []

    outputs = []
    for i in range(out.shape[0]):
        val = out[i, 0]
        # Might be bytes, so decode if needed
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        outputs.append(val)

    return outputs

###############################################################################
# EXAMPLE USAGE (if run as a script)
###############################################################################

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 1. Test OpenAI call
    messages_openai = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
    try:
        openai_response = call_openai_chat_completion(messages_openai)
        logging.info(f"OpenAI response: {openai_response}")
    except OpenAIAPIError as e:
        logging.error(f"Failed to get response from OpenAI: {e}")

    # 2. Test local Llama inference
    prompts_to_try = [
        "Hello, can you explain what large language models are?",
        "What is concurrency in computer science?",
        "Please summarise the solar system."
    ]
    try:
        local_responses = call_local_llama_inference(prompts_to_try)
        for i, resp in enumerate(local_responses):
            logging.info(f"[Prompt {i}] => {resp}")
    except LocalLLMError as e:
        logging.error(f"Failed to get response from local Llama: {e}")

    logging.info("Done.")
