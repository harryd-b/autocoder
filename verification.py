#!/usr/bin/env python3
"""
verification.py

Verifies code snippets by:
1. Sending them to ChatGPT in "verification mode."
2. Running lint checks (flake8).
3. Running tests (pytest).
"""

import re
import json
import logging
import subprocess
from typing import Any, Dict, Optional

from api_utils import call_openai_chat_completion, GPT_MODEL
from conversation_manager import ConversationManager

def verify_code_with_chatgpt(code: str, verification_prompt: str = None) -> Optional[Dict[str, Any]]:
    """
    Sends the generated code to a separate ChatGPT conversation for verification.
    Returns a JSON dict with {'complete': bool, 'feedback': str} or None on error.

    :param code: The code snippet to verify
    :param verification_prompt: An optional custom prompt for verification
    :return: A dict like {"complete": bool, "feedback": str} or None
    """
    if verification_prompt is None:
        verification_prompt = (
            "Please verify the following code snippet. "
            "Respond in JSON with fields 'complete' (boolean) and 'feedback' (string)."
        )

    system_message = {
        "role": "system",
        "content": (
            "You are ChatGPT in verification mode. "
            "You will review the submitted code snippet for completeness, correctness, "
            "and whether it meets typical best practices. Respond in valid JSON."
        )
    }
    user_message = {
        "role": "user",
        "content": f"{verification_prompt}\n\n```python\n{code}\n```"
    }

    try:
        response = call_openai_chat_completion([system_message, user_message], model=GPT_MODEL)
        text_content = response["choices"][0]["message"]["content"].strip()

        # Extract the first JSON object
        json_match = re.search(r"\{.*\}", text_content, re.DOTALL)
        if not json_match:
            logging.warning("Could not find JSON in the verification response.")
            return None

        json_str = json_match.group(0)
        verification_data = json.loads(json_str)

        return verification_data
    except Exception as e:
        logging.error("Verification via ChatGPT failed.", exc_info=True)
        return None

def run_lint_checks(file_path: str) -> bool:
    """
    Runs flake8 on the specified file to check for style/syntax issues.

    :param file_path: Path to the Python file to lint
    :return: True if no lint errors, False otherwise
    """
    logging.info(f"Running flake8 lint checks on {file_path}")
    cmd = ["flake8", file_path]
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
    logging.info(f"Running tests on {file_path} using pytest (placeholder).")
    # Example: run pytest in current directory
    cmd = ["pytest", "--maxfail=1", "--disable-warnings"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        logging.info("Tests passed.")
        return True

    logging.warning(f"Tests failed. Output:\n{result.stdout}\n{result.stderr}")
    return False
