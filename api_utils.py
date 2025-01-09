#!/usr/bin/env python3
"""
api_utils.py

Handles OpenAI API interactions, including retries and error handling.
"""

import logging
import openai
import yaml
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type
)
from dotenv import load_dotenv

class OpenAIAPIError(Exception):
    """Custom exception for OpenAI API errors."""
    pass

# Load configuration
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("No OPENAI_API_KEY found in .env")

openai.api_key = OPENAI_API_KEY
GPT_MODEL = config.get("model", "gpt-3.5-turbo")

MAX_RETRIES = config.get("max_retries", 3)
RETRY_BASE_SECONDS = config.get("retry_base_seconds", 1.5)
RETRY_MAX_SECONDS = config.get("retry_max_seconds", 10)

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
    :param model: The OpenAI model to use
    :return: The OpenAI API response (as a dict)
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
