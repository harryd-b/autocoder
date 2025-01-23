#!/usr/bin/env python3
"""
verification.py

Verifies code snippets by:
1. Sending them to a verification model ("local" Triton or OpenAI) in "verification mode."
2. Running lint checks (flake8).
3. Running tests (pytest).
"""

import re
import json
import logging
import subprocess
from typing import Any, Dict, Optional
import os

from api_utils import (
    GPT_MODEL,
    config,
    call_openai_chat_completion,
    call_local_llama_inference
)

###############################################################################
# MODEL SELECTION FOR VERIFICATION
###############################################################################
DEFAULT_MODEL_SOURCE = config.get("model_source", "local")  # 'local' or 'openai'

def call_verification_model(code: str, verification_prompt: str) -> str:
    """
    Calls the verification model. If config.yaml sets 'model_source: local',
    we flatten the prompt + code and send to local Triton. Otherwise, we call OpenAI.

    Returns a raw text string response, which is expected to contain JSON.
    """
    # Our "system message" content to prime the verification logic
    system_intro = (
        "You are ChatGPT in verification mode. "
        "You will review the submitted code snippet for completeness, correctness, "
        "and whether it meets typical best practices. Respond in valid JSON."
    )

    # We unify the user prompt
    user_content = f"{verification_prompt}\n\n```python\n{code}\n```"

    if DEFAULT_MODEL_SOURCE.lower() == "openai":
        # Build messages for OpenAI
        system_message = {
            "role": "system",
            "content": system_intro
        }
        user_message = {
            "role": "user",
            "content": user_content
        }
        # Call OpenAI with the typical ChatCompletion format
        response = call_openai_chat_completion(
            messages=[system_message, user_message],
            model=GPT_MODEL
        )
        return response["choices"][0]["message"]["content"].strip()
    else:
        # Local Triton Llama verification
        # Flatten system and user content into a single prompt
        prompt_text = (
            f"SYSTEM:\n{system_intro}\n\n"
            f"USER:\n{user_content}\n"
        )
        local_responses = call_local_llama_inference([prompt_text])
        if local_responses:
            return local_responses[0].strip()
        return ""

###############################################################################
# MAIN VERIFICATION LOGIC
###############################################################################

def verify_code_with_chatgpt(
    code: str,
    verification_prompt: str = None
) -> Optional[Dict[str, Any]]:
    """
    Sends the generated code to a verification model (local or OpenAI).
    Expects a JSON dict with {'complete': bool, 'feedback': str}, or None on error.

    :param code: The code snippet to verify
    :param verification_prompt: An optional custom prompt for verification
    :return: A dict like {"complete": bool, "feedback": str} or None
    """
    if verification_prompt is None:
        verification_prompt = (
            "Review this code snippet. Respond in JSON with 'complete' (true/false) and 'feedback' (string). "
            "Example: {'complete': true, 'feedback': 'Looks good'}"
        )

    try:
        raw_response = call_verification_model(code, verification_prompt)
        if not raw_response:
            logging.warning("Received empty response from verification model.")
            return None

        # Attempt to extract the first JSON object from the raw text
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if not json_match:
            logging.warning("Could not find JSON in the verification response.")
            return None

        json_str = json_match.group(0)
        verification_data = json.loads(json_str)

        return verification_data

    except Exception as e:
        logging.error("Verification failed.", exc_info=True)
        return None

def run_lint_checks(file_path: str) -> bool:
    """
    Runs flake8 on the specified file to check for style/syntax issues.

    :param file_path: Path to the Python file to lint
    :return: True if no lint errors, False otherwise
    """
    full_path = os.path.join(OUTPUT_DIR, file_path)
    logging.info(f"Running flake8 lint checks on {file_path}")
    cmd = ["flake8", full_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logging.warning(f"Lint errors/warnings found:\n{result.stdout}\n{result.stderr}")
        return False
    return True

def run_tests_on_code(file_path: str) -> bool:
    """
    Runs pytest on the file. (For demonstration, we simply call pytest on the entire directory
    or a known test directory.)

    :param file_path: Path to the code file (unused in this simple example).
    :return: True if tests pass, False otherwise.
    """
    full_path = os.path.join(OUTPUT_DIR, file_path)
    logging.info(f"Running tests on {full_path} using pytest (placeholder).")
    # Example: run pytest in current directory
    cmd = ["pytest", full_path, "--maxfail=1", "--disable-warnings"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        logging.info("Tests passed.")
        return True

    logging.warning(f"Tests failed. Output:\n{result.stdout}\n{result.stderr}")
    return False
