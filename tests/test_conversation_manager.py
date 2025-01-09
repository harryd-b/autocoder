import pytest
import json
from unittest.mock import patch, mock_open

from conversation_manager import ConversationManager

@pytest.fixture
def sample_convo_data():
    return {
        "root": [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"}
        ]
    }

@patch("builtins.open", new_callable=mock_open, read_data='{"root": []}')
def test_load_conversations(mock_file):
    cm = ConversationManager()
    assert "root" in cm.conversations  # loaded from file

@patch("os.path.exists", return_value=False)
@patch("builtins.open", new_callable=mock_open)
def test_save_conversations(mock_file, mock_exists, sample_convo_data):
    # 1. Initialize conversation manager (will attempt to read, call #1)
    cm = ConversationManager()
    
    # 2. Reset mock so the read call doesn't count in subsequent assertions
    mock_file.reset_mock()

    # 3. Now do the save
    cm.conversations = sample_convo_data
    cm.save_conversations()

    # 4. Assert that we opened the file exactly once in write mode
    mock_file.assert_called_once_with("conversation_state.json", "w", encoding="utf-8")
    
    # 5. Gather all text written to the file across multiple write(...) calls
    handle = mock_file()
    all_write_calls = handle.write.call_args_list  # e.g. [call("{"), call("\n  "), call('"root"'), ...]
    written_pieces = [call_args[0][0] for call_args in all_write_calls]  # extract the text
    full_json_str = "".join(written_pieces)

    # 6. Parse the combined JSON string
    data = json.loads(full_json_str)
    assert data == sample_convo_data

def test_slide_conversation_window():
    cm = ConversationManager()
    cm.conversations["test_branch"] = [
        {"role": "system", "content": "System message"},
    ] + [{"role": "user", "content": f"Message {i}"} for i in range(20)]
    truncated = cm.slide_conversation_window(cm.conversations["test_branch"])
    # Default max convo length is 10 from config
    assert len(truncated) == 10  
    assert truncated[0]["role"] == "system"

def test_update_conversation(sample_convo_data):
    cm = ConversationManager()
    cm.conversations = sample_convo_data
    branch = "root"
    cm.update_conversation(branch, "assistant", "Assistant response")
    assert cm.conversations[branch][-1]["role"] == "assistant"
    assert cm.conversations[branch][-1]["content"] == "Assistant response"
