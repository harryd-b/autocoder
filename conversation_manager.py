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

    def update_conversation(self, branch_name: str, role: str, content: str) -> None:
        """
        Appends a message (role + content) to a branch's conversation, applies a sliding window,
        and saves the updated conversation.
        """
        if branch_name not in self.conversations:
            self.conversations[branch_name] = []
        self.conversations[branch_name].append({"role": role, "content": content})
        self.conversations[branch_name] = self.slide_conversation_window(self.conversations[branch_name])
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
