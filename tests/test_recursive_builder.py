import pytest
import asyncio
from unittest.mock import patch, MagicMock

from conversation_manager import ConversationManager
from recursive_builder import (
    recursive_prompt,
    extract_questions_and_code,
)

@pytest.mark.asyncio
@patch("recursive_builder.call_model")
@patch("recursive_builder.verify_code_with_chatgpt")
@patch("recursive_builder.run_lint_checks")
@patch("recursive_builder.run_tests_on_code")
async def test_recursive_prompt_basic_flow(
    mock_tests,
    mock_lint,
    mock_verify,
    mock_call_model
):
    """
    Tests a simple scenario where the model returns a single code snippet,
    which is verified as complete.
    """
    # Mock the response from call_model to return a code snippet
    mock_call_model.return_value = {
        "choices": [{
            "message": {
                "content": "Here is some code: ```python\nprint('Hello')\n```"
            }
        }]
    }
    # verification always says "complete"
    mock_verify.return_value = {"complete": True, "feedback": "Looks good"}
    mock_lint.return_value = True
    mock_tests.return_value = True

    cm = ConversationManager()
    branch_name = "test_branch"

    # Add a system message for that branch
    cm.update_conversation(branch_name, "system", "System prompt")

    # Run
    await recursive_prompt(
        conv_manager=cm,
        user_prompt="User's main prompt",
        branch_name=branch_name,
        depth=0,
        max_depth=3
    )

    history = cm.get_conversation(branch_name)
    # Ensure new messages were appended
    assert len(history) >= 3  # system, user, assistant
    # Check that we attempted to verify the code
    mock_verify.assert_called_once()


@pytest.mark.asyncio
@patch("recursive_builder.call_model")
@patch("builtins.input", return_value="Test user answer")
@patch("recursive_builder.verify_code_with_chatgpt")
@patch("recursive_builder.run_lint_checks")
@patch("recursive_builder.run_tests_on_code")
async def test_recursive_prompt_question_flow(
    mock_tests,
    mock_lint,
    mock_verify,
    mock_input,
    mock_call_model
):
    """
    Tests a scenario where the model returns both code and a clarifying question.
    We patch 'input' to simulate real user input. We also verify that the user answer
    is appended to the conversation and triggers another recursion step.
    """
    # We'll simulate the first call to the model returning:
    # 1) A code snippet
    # 2) A clarifying question
    first_mock_response = {
        "choices": [{
            "message": {
                "content": (
                    "Here is some code: ```python\nprint('Hello')\n```\n"
                    "And a question?\n"
                )
            }
        }]
    }

    # For the second call (answering the clarifying question),
    # we'll simulate a simple response with another code snippet
    second_mock_response = {
        "choices": [{
            "message": {
                "content": "Sure, here's another snippet: ```python\nprint('Bye')\n```"
            }
        }]
    }

    # call_model yields two different responses on consecutive calls
    mock_call_model.side_effect = [first_mock_response, second_mock_response]

    # Patch run_lint_checks / run_tests_on_code to return True
    mock_lint.return_value = True
    mock_tests.return_value = True

    # Patch verification so that we always say "complete": True
    mock_verify.return_value = {"complete": True, "feedback": "All good"}

    cm = ConversationManager()
    branch_name = "question_test_branch"

    cm.update_conversation(branch_name, "system", "System prompt")

    await recursive_prompt(
        conv_manager=cm,
        user_prompt="Initial user prompt for question scenario",
        branch_name=branch_name,
        depth=0,
        max_depth=3
    )

    # Let's see what happened in conversation history
    history = cm.get_conversation(branch_name)

    # Expect at least:
    #  - system message
    #  - user message (initial prompt)
    #  - assistant response (with code + question)
    #  - user answer (the "Test user answer" we patched in)
    #  - assistant response #2 (some final snippet)
    # => total 5 or more messages
    assert len(history) >= 5

    # Check that the clarifying question led to user input
    user_entries = [msg for msg in history if msg["role"] == "user"]
    assert any("Test user answer" in msg["content"] for msg in user_entries)

    # call_model was called twice:
    # 1st: for the initial prompt
    # 2nd: after user answered the clarifying question
    assert mock_call_model.call_count == 2

    # verification called for both code snippets
    # => 2 total code blocks
    assert mock_verify.call_count == 2


def test_extract_questions_and_code():
    text = """
    Here is a question?
    And some more text.
    ```python
    print("Hello")
    ```
    Another question?
    ```
    # Another code block
    def foo(): pass
    ```
    """
    result = extract_questions_and_code(text)
    assert len(result["questions"]) == 2
    assert len(result["code_blocks"]) == 2
    assert 'print("Hello")' in result["code_blocks"][0]
