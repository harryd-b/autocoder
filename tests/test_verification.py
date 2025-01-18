import pytest
import json
from unittest.mock import patch, MagicMock

from verification import (
    verify_code_with_chatgpt,
    run_lint_checks,
    run_tests_on_code,
    DEFAULT_MODEL_SOURCE,  # Imported to patch the default selection
)
from api_utils import OpenAIAPIError, LocalLLMError

###############################################################################
# Tests for verify_code_with_chatgpt (OpenAI path)
###############################################################################
@patch("verification.call_openai_chat_completion")
def test_verify_code_with_chatgpt_success(mock_api):
    """
    Test a scenario where the verification returns a valid JSON object,
    ensuring we parse it properly.
    """
    # Must be a single valid JSON object, no extra data
    mock_api.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"complete":true,"feedback":"All good"}'
                }
            }
        ]
    }

    code = "print('Hello')"
    result = verify_code_with_chatgpt(code)
    assert result is not None
    assert result["complete"] is True
    assert result["feedback"] == "All good"

@patch("verification.call_openai_chat_completion")
def test_verify_code_with_chatgpt_no_json(mock_api):
    mock_api.return_value = {
        "choices": [
            {"message": {"content": "No JSON here"}}
        ]
    }
    code = "print('Hello')"
    result = verify_code_with_chatgpt(code)
    assert result is None

###############################################################################
# Tests for verify_code_with_chatgpt (local Triton path)
###############################################################################

@patch("verification.call_local_llama_inference")
@patch("verification.DEFAULT_MODEL_SOURCE", new="local")
def test_verify_code_with_chatgpt_local_success(mock_infer):
    """
    Test a successful local verification call, returning valid JSON.
    """
    mock_infer.return_value = ['{"complete": true, "feedback": "Local says all good"}']
    code = "print('Hello from local')"
    result = verify_code_with_chatgpt(code)
    assert result is not None
    assert result["complete"] is True
    assert result["feedback"] == "Local says all good"

@patch("verification.call_local_llama_inference")
@patch("verification.DEFAULT_MODEL_SOURCE", new="local")
def test_verify_code_with_chatgpt_local_no_json(mock_infer):
    """
    Test local verification call returning no JSON object in the text.
    """
    mock_infer.return_value = ["No JSON here at all"]
    code = "print('Hello from local')"
    result = verify_code_with_chatgpt(code)
    assert result is None

###############################################################################
# Tests for lint and test checks
###############################################################################

@patch("subprocess.run")
def test_run_lint_checks_pass(mock_run):
    # Simulate flake8 returning success (returncode=0)
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "All good"
    file_path = "test_file.py"
    assert run_lint_checks(file_path) is True

@patch("subprocess.run")
def test_run_lint_checks_fail(mock_run):
    # Simulate flake8 returning error
    mock_run.return_value.returncode = 1
    mock_run.return_value.stdout = "Lint error"
    file_path = "test_file.py"
    assert run_lint_checks(file_path) is False

@patch("subprocess.run")
def test_run_tests_on_code_pass(mock_run):
    # Simulate pytest passing
    mock_run.return_value.returncode = 0
    file_path = "test_file.py"
    assert run_tests_on_code(file_path) is True

@patch("subprocess.run")
def test_run_tests_on_code_fail(mock_run):
    # Simulate pytest failing
    mock_run.return_value.returncode = 1
    file_path = "test_file.py"
    assert run_tests_on_code(file_path) is False
