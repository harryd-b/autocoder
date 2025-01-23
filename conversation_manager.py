#!/usr/bin/env python3
"""
conversation_manager.py

Handles conversation states:
- Storing them in memory
- Persisting them to a local file (JSON)
- Sliding window management
- Optionally providing a "flattened" version for local model usage
"""

import json
import logging
import os
from typing import List, Dict
import yaml
from datetime import datetime

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

CONVERSATION_FILE = config.get("conversation_file", "conversation_state.json")
MAX_CONVERSATION_LENGTH = config.get("max_conversation_length", 10)

class ConversationManager:
    def __init__(self):
        # Dictionary: branch_name -> list of messages (each message is {"role": "...", "content": "..."})
        self.conversations = {}
        # Load existing data if available
        self.load_conversations()

    def load_conversations(self) -> None:
        """
        Loads the conversation file if it exists, otherwise starts fresh.
        """
        if os.path.exists(CONVERSATION_FILE):
            try:
                with open(CONVERSATION_FILE, "r", encoding="utf-8") as f:
                    self.conversations = json.load(f)
                logging.info(f"Loaded existing conversation data from {CONVERSATION_FILE}")
            except Exception as e:
                logging.warning(f"Failed to load conversation file: {e}")
        else:
            logging.info("No existing conversation file found; starting fresh.")

    def save_conversations(self) -> None:
        """
        Persists all conversation data to a local JSON file.
        """
        try:
            with open(CONVERSATION_FILE, "w", encoding="utf-8") as f:
                json.dump(self.conversations, f, indent=2)
            logging.debug(f"Conversation data saved to {CONVERSATION_FILE}")
        except Exception as e:
            logging.error(f"Failed to save conversation data: {e}")

    def get_conversation(self, branch_name: str) -> List[Dict[str, str]]:
        """
        Retrieves the conversation list (messages) for a given branch.
        """
        return self.conversations.get(branch_name, [])

    def validate_message(self, role: str, content: str) -> None:
        """Validates message parameters before adding to conversation."""
        valid_roles = {"system", "user", "assistant"}
        if role not in valid_roles:
            raise ValueError(f"Invalid role: {role}. Must be one of {valid_roles}")
        if not content or not isinstance(content, str):
            raise ValueError("Content must be a non-empty string")

    def update_conversation(self, branch_name: str, role: str, content: str) -> None:
        """Appends a message with metadata to a branch's conversation."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "message_id": len(self.conversations.get(branch_name, [])) + 1
        }
        
        if branch_name not in self.conversations:
            self.conversations[branch_name] = []
        self.conversations[branch_name].append(message)
        self.conversations[branch_name] = self.slide_conversation_window(
            self.conversations[branch_name]
        )
        self.save_conversations()

    def slide_conversation_window(self, conversation_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Retain the system message plus the last (MAX_CONVERSATION_LENGTH - 1) messages.
        """
        if len(conversation_history) > MAX_CONVERSATION_LENGTH:
            system_msg = conversation_history[0]
            truncated = [system_msg] + conversation_history[-(MAX_CONVERSATION_LENGTH - 1):]
            return truncated
        return conversation_history

    def get_flattened_conversation(self, branch_name: str) -> str:
        """
        Returns a single string that concatenates all messages from the conversation in a
        "role: content" format. This can be useful for local Llama usage where you
        want to supply the entire conversation as one prompt.

        Example output:

            SYSTEM:
            You are a helpful AI.

            USER:
            Hello, how are you?

            ASSISTANT:
            I'm doing well, thanks!
        """
        conversation = self.get_conversation(branch_name)
        lines = []
        for msg in conversation:
            role = msg["role"].upper()
            content = msg["content"]
            lines.append(f"{role}:\n{content}\n")
        return "\n".join(lines).strip()

    def delete_branch(self, branch_name: str) -> None:
        """Deletes a conversation branch."""
        if branch_name in self.conversations:
            del self.conversations[branch_name]
            self.save_conversations()
        
    def list_branches(self) -> List[str]:
        """Returns list of all conversation branches."""
        return list(self.conversations.keys())

    def clear_branch(self, branch_name: str) -> None:
        """Clears all messages from a branch except system message."""
        if branch_name in self.conversations:
            system_msgs = [msg for msg in self.conversations[branch_name] 
                          if msg["role"] == "system"]
            self.conversations[branch_name] = system_msgs
            self.save_conversations()

    def get_conversation_size(self, branch_name: str) -> int:
        """Returns the total characters in a conversation branch."""
        conversation = self.get_conversation(branch_name)
        return sum(len(msg["content"]) for msg in conversation)

    def cleanup_old_branches(self, max_branches: int = 100) -> None:
        """Removes oldest branches if total exceeds max_branches."""
        if len(self.conversations) > max_branches:
            # Sort branches by last message timestamp
            sorted_branches = sorted(
                self.conversations.items(),
                key=lambda x: x[1][-1].get("timestamp", "") if x[1] else "",
                reverse=True
            )
            # Keep only max_branches most recent
            self.conversations = dict(sorted_branches[:max_branches])
            self.save_conversations()

    def backup_conversations(self, backup_file: str = None) -> None:
        """Creates a backup of the current conversation state."""
        if backup_file is None:
            backup_file = f"{CONVERSATION_FILE}.backup"
        try:
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(self.conversations, f, indent=2)
            logging.info(f"Backup created at {backup_file}")
        except Exception as e:
            logging.error(f"Failed to create backup: {e}")
