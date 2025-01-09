import pytest
import asyncio
from unittest.mock import patch, MagicMock

from conversation_manager import ConversationManager
from recursive_builder import (
    recursive_prompt,
    extract_questions_and_code,
)

@pytest.mark.asyncio
@patch("recursive_builder.call_openai_chat_completion")
@patch("recursive_builder.verify_code_with_chatgpt")
@patch("recursive_builder.run_lint_checks")
@patch("recursive_builder.run_tests_on_code")
async def test_recursive_prompt_basic_flow(
    mock_tests,
    mock_lint,
    mock_verify,
    mock_api
):
    # Setup
    mock_api.return_value = {
        "choices": [{
            "message": {
                "content": "Here is some code: ```python\nprint('Hello')\n```"
            }
        }]
    }
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
    assert "print(\"Hello\")" in result["code_blocks"][0]
