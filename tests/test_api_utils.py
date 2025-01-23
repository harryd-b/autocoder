import pytest
from unittest.mock import patch, MagicMock

import numpy as np
import requests
import openai

from api_utils import (
    call_openai_chat_completion,
    call_local_llama_inference,
    OpenAIAPIError,
    LocalLLMError
)

@pytest.fixture
def mock_openai_response():
    return {
        "choices": [
            {
                "message": {
                    "content": "Hello, this is a mocked OpenAI response."
                }
            }
        ]
    }

@pytest.fixture
def mock_triton_response():
    """
    Example data structure you might receive from Tritonclient after calling `infer()`.
    We'll emulate something that returns a numpy array with shape (batch_size, 1).
    """
    class MockInferResult:
        def as_numpy(self, output_name):
            # Return an array of shape (2,1) for 2 prompts, if desired
            return np.array([[b"Mock Llama Response 1"], [b"Mock Llama Response 2"]], dtype=object)

    return MockInferResult()

###############################################################################
# Tests for call_openai_chat_completion
###############################################################################

@patch("openai.resources.chat.Completions.create")
def test_call_openai_chat_completion_success(mock_create, mock_openai_response):
    """
    Test a successful call to the OpenAI ChatCompletion API.
    """
    mock_create.return_value = openai.types.chat.ChatCompletion(
        id="test_id",
        choices=[{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello, this is a mocked OpenAI response."
            },
            "finish_reason": "stop"
        }],
        created=1234567890,
        model="gpt-3.5-turbo",
        object="chat.completion",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    )
    
    messages = [{"role": "user", "content": "Hi"}]
    response = call_openai_chat_completion(messages)
    assert response.choices[0].message.content == "Hello, this is a mocked OpenAI response."
    mock_create.assert_called_once()

@patch("openai.ChatCompletion.create")
def test_call_openai_chat_completion_failure(mock_create):
    """
    Test that an OpenAIAPIError is raised upon failure.
    """
    mock_create.side_effect = Exception("API call failed")
    messages = [{"role": "user", "content": "Hi"}]

    with pytest.raises(OpenAIAPIError):
        call_openai_chat_completion(messages)

###############################################################################
# Tests for call_local_llama_inference
###############################################################################

@patch("tritonclient.http.InferenceServerClient")
def test_call_local_llama_inference_success(mock_client_class, mock_triton_response):
    """
    Test a successful call to the local Triton server hosting Llama.
    """
    # Mock the client instance returned by InferenceServerClient()
    mock_client_instance = MagicMock()
    mock_client_class.return_value = mock_client_instance

    # We mock the result of client.infer(...)
    mock_client_instance.infer.return_value = mock_triton_response

    # Now call our function
    prompts = ["What is AI?", "Explain concurrency."]
    results = call_local_llama_inference(prompts)

    # We expect 2 items because our mock_triton_response has shape (2,1)
    assert len(results) == 2
    assert "Mock Llama Response 1" in results[0]
    assert "Mock Llama Response 2" in results[1]

    # Ensure client was called as expected
    mock_client_class.assert_called_once()  # We only create 1 client
    mock_client_instance.infer.assert_called_once()

@patch("tritonclient.http.InferenceServerClient")
def test_call_local_llama_inference_failure(mock_client_class):
    """
    Test that a LocalLLMError is raised when something goes wrong with the Triton call.
    """
    # Mock the client and make it raise an exception on infer
    mock_client_instance = MagicMock()
    mock_client_class.return_value = mock_client_instance
    mock_client_instance.infer.side_effect = Exception("Triton call failed")

    prompts = ["Hello local Llama!"]

    with pytest.raises(LocalLLMError):
        call_local_llama_inference(prompts)

###############################################################################
# Optional: Additional scenarios (if you want more coverage)
###############################################################################

@patch("tritonclient.http.InferenceServerClient")
def test_call_local_llama_inference_empty_output(mock_client_class):
    """
    Test scenario where as_numpy(...) returns None or an empty result.
    """
    mock_client_instance = MagicMock()
    mock_client_class.return_value = mock_client_instance

    # Create a mock result that returns None for as_numpy()
    class MockInferResultEmpty:
        def as_numpy(self, output_name):
            return None

    mock_client_instance.infer.return_value = MockInferResultEmpty()

    prompts = ["Test empty output"]
    results = call_local_llama_inference(prompts)
    assert results == [], "Expected an empty list when no output is returned from model."

@patch("tritonclient.http.InferenceServerClient")
def test_call_local_llama_inference_no_prompts(mock_client_class):
    """
    If we pass an empty list of prompts, we expect an empty list back without calling Triton.
    """
    mock_client_instance = MagicMock()
    mock_client_class.return_value = mock_client_instance

    # Provide no prompts
    prompts = []
    results = call_local_llama_inference(prompts)
    assert results == []
    # Because there are no prompts, we do not call client.infer
    mock_client_instance.infer.assert_not_called()
