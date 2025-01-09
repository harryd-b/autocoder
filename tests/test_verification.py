import pytest
from unittest.mock import patch, MagicMock

from verification import (
    verify_code_with_chatgpt,
    run_lint_checks,
    run_tests_on_code
)

@patch("verification.call_openai_chat_completion")
def test_verify_code_with_chatgpt_success(mock_api):
    mock_api.return_value = {
        "choices": [
            {"message": {"content": '{"complete": true, "feedback": "All good"}'}}
        ]
    }
    code = "print('Hello')"
    result = verify_code_with_chatgpt(code)
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
