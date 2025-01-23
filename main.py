#!/usr/bin/env python3
"""
main.py

Entry point for the Recursive Code Builder application.
Now requests a root prompt from the user interactively,
then uses real user input to answer clarifying questions.
"""

import asyncio
import logging
import yaml
from conversation_manager import ConversationManager
from recursive_builder import recursive_prompt

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

async def main():
    setup_logging()
    logging.info("=== Recursive Code Builder (Modularised) ===")

    # Load config
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    max_depth = config.get("max_depth", 10)

    # Initialise conversation manager
    conv_manager = ConversationManager()
    branch_name = "root"

    # If this branch doesn't have a conversation yet, set a system message
    if not conv_manager.get_conversation(branch_name):
        # Change this line in the main.py initialization:
        system_message = (
            "You are an analyst programmer. You will interact with the user to clarify "
            "requirements and PRODUCE CODE SNIPPETS IN TRIPLE BACKTICKS. Ask questions only when necessary."
        )
        conv_manager.update_conversation(branch_name, "system", system_message)

    # Prompt the user for the root prompt interactively
    print("\nPlease enter the root prompt describing the software application you want to build.")
    print("For example: 'I want to build a multi-tier web app with Django and React...'")
    user_root_prompt = await asyncio.to_thread(input, "\nRoot Prompt > ")

    # If the user didn't type anything, fall back to a default
    if not user_root_prompt.strip():
        user_root_prompt = (
            "I want to build a complex software application. "
            "Please ask clarifying questions until you have all the details necessary "
            "to produce the code in segments. Each time you have enough details for a "
            "part of the application, produce the code for that part. Then we will verify "
            "it in a separate conversation to ensure completeness. If verified as complete, "
            "that part is finalised. Otherwise, we will refine it further.\n\n"
            "Remember to keep the conversation focused, as we are limiting conversation length. "
            "If you need previous context, let me know."
        )

    # Kick off recursion
    await recursive_prompt(
        conv_manager=conv_manager,
        user_prompt=user_root_prompt,
        branch_name=branch_name,
        depth=0,
        max_depth=max_depth
    )

    logging.info("=== All done! ===\n")
    logging.info(
        "Hint: Check your local directory for any generated .py code files. "
        "See logs above for verification feedback. You can integrate them, run additional tests, "
        "or build Docker images, etc."
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("User interrupted execution.")
    except Exception as e:
        logging.error(f"Unhandled exception: {e}", exc_info=True)
