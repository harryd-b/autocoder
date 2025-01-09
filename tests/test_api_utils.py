import pytest
from unittest.mock import patch, MagicMock

from api_utils import call_openai_chat_completion, OpenAIAPIError

@pytest.fixture
def mock_response():
    return {
        "choices": [
            {
                "message": {
                    "content": "Hello, this is a mocked response."
                }
            }
        ]
    }

@patch("openai.ChatCompletion.create")
def test_call_openai_chat_completion_success(mock_create, mock_response):
    """Test a successful call to the OpenAI ChatCompletion API."""
    mock_create.return_value = mock_response
    messages = [{"role": "user", "content": "Hi"}]
    response = call_openai_chat_completion(messages)
    assert response == mock_response
    mock_create.assert_called_once()

@patch("openai.ChatCompletion.create")
def test_call_openai_chat_completion_failure(mock_create):
    """Test that an OpenAIAPIError is raised upon failure."""
    mock_create.side_effect = Exception("API call failed")
    messages = [{"role": "user", "content": "Hi"}]

    with pytest.raises(OpenAIAPIError):
        call_openai_chat_completion(messages)
