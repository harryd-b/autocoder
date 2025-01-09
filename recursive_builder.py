#!/usr/bin/env python3
"""
recursive_builder.py

Main logic for the recursive, piecewise code generation process.
"""

import logging
import re
import asyncio
import random
from typing import Dict, List, Any

from conversation_manager import ConversationManager
from api_utils import call_openai_chat_completion, GPT_MODEL
from verification import (
    verify_code_with_chatgpt,
    run_tests_on_code,
    run_lint_checks
)

def extract_questions_and_code(response_text: str) -> Dict[str, List[str]]:
    """
    Parses the response from ChatGPT and attempts to extract clarifying questions and code blocks.
    Returns a dict:
      {
        "questions": [...],
        "code_blocks": [...]
      }
    """
    questions = []
    code_blocks = []

    lines = response_text.split("\n")
    for line in lines:
        if line.strip().endswith("?"):
            questions.append(line.strip())

    code_pattern = r"```(.*?)```"
    matches = re.findall(code_pattern, response_text, re.DOTALL)
    for match in matches:
        code_blocks.append(match.strip())

    return {"questions": questions, "code_blocks": code_blocks}

def save_code_locally(code_str: str, file_path: str) -> None:
    """
    Saves the given code string to a local file.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code_str)
        logging.info(f"Code saved to: {file_path}")
    except Exception as e:
        logging.error(f"Failed to save code to {file_path}: {e}")

async def refine_incomplete_code(
    conv_manager: ConversationManager,
    branch_name: str,
    incomplete_feedback: str,
    existing_code_snippet: str,
    code_index: int
):
    """
    Feed back the verifier's feedback to the main conversation to request improvements.
    """
    logging.info("Refining incomplete code snippet based on verifier feedback.")

    refine_prompt = (
        "We received the following feedback from a verification step:\n\n"
        f"'{incomplete_feedback}'\n\n"
        "Please refine the following code snippet to address these issues. "
        "Only provide the refined snippet in triple backticks:\n\n"
        f"```python\n{existing_code_snippet}\n```"
    )

    conv_manager.update_conversation(branch_name, "user", refine_prompt)

    # Ask ChatGPT for an improved snippet
    conversation_history = conv_manager.get_conversation(branch_name)
    response = call_openai_chat_completion(conversation_history, model=GPT_MODEL)
    improvement_response = response["choices"][0]["message"]["content"]
    conv_manager.update_conversation(branch_name, "assistant", improvement_response)

    # Extract improved code
    parsed = extract_questions_and_code(improvement_response)
    if not parsed["code_blocks"]:
        logging.warning("ChatGPT did not return improved code snippet in refinement step.")
        return

    improved_code = parsed["code_blocks"][0]
    file_name = f"{branch_name}_refined_{code_index}.py"

    # Re-verify improved code
    verification_data = verify_code_with_chatgpt(improved_code)
    if verification_data and verification_data.get("complete") is True:
        # Save code
        save_code_locally(improved_code, file_name)
        # Lint check
        lint_ok = run_lint_checks(file_name)
        # Run tests
        tests_ok = run_tests_on_code(file_name)
        if lint_ok and tests_ok:
            logging.info(f"Refined code verified successfully: {file_name}")
        else:
            logging.warning(f"Refined code has lint/test issues: {file_name}")
    else:
        logging.warning(f"Code snippet {file_name} is still incomplete after refinement.")

async def handle_question(
    conv_manager: ConversationManager,
    question_text: str,
    branch_name: str,
    depth: int,
    max_depth: int
):
    """
    Handles a single clarifying question. 
    In a real scenario, you'd wait for user input. Here, we simulate an 'auto-answer'.
    """
    if depth > max_depth:
        logging.warning("Reached max recursion depth while handling question.")
        return

    # Simulate user input
    user_answer = f"Auto-answer for '{question_text}' (simulated)."
    conv_manager.update_conversation(branch_name, "user", user_answer)

    await recursive_prompt(
        conv_manager=conv_manager,
        user_prompt=user_answer,
        branch_name=branch_name,
        depth=depth,
        max_depth=max_depth
    )

async def recursive_prompt(
    conv_manager: ConversationManager,
    user_prompt: str,
    branch_name: str,
    depth: int,
    max_depth: int
) -> None:
    """
    Recursively gather requirements and generate code from ChatGPT.

    :param conv_manager: A ConversationManager instance
    :param user_prompt: The latest user input to feed ChatGPT
    :param branch_name: A label for the conversation context
    :param depth: Current recursion level
    :param max_depth: Maximum allowed recursion depth
    """
    if depth > max_depth:
        logging.warning(f"Max recursion depth ({max_depth}) reached. Stopping.")
        return

    # Add user prompt to conversation
    conv_manager.update_conversation(branch_name, "user", user_prompt)

    # Call ChatGPT
    conversation_history = conv_manager.get_conversation(branch_name)
    response = call_openai_chat_completion(conversation_history, model=GPT_MODEL)
    assistant_response = response["choices"][0]["message"]["content"].strip()
    if not assistant_response:
        logging.warning("Received empty response from ChatGPT.")
        return

    conv_manager.update_conversation(branch_name, "assistant", assistant_response)

    logging.info(f"[BRANCH={branch_name}] ChatGPT says:\n{assistant_response}\n")

    # Extract questions and code from response
    parsed = extract_questions_and_code(assistant_response)
    questions = parsed["questions"]
    code_blocks = parsed["code_blocks"]

    # Verify code blocks in parallel
    verification_tasks = []
    loop = asyncio.get_event_loop()

    for code_index, code_snippet in enumerate(code_blocks):
        verification_tasks.append(loop.run_in_executor(
            None,  # default executor
            verify_code_with_chatgpt,
            code_snippet
        ))

    verification_results = await asyncio.gather(*verification_tasks)

    # Process verification results
    for i, code_snippet in enumerate(code_blocks):
        verification_data = verification_results[i]
        if not verification_data:
            logging.warning("No verification data received for a code snippet. Skipping save.")
            continue

        complete_flag = verification_data.get("complete", False)
        feedback = verification_data.get("feedback", "No feedback provided.")

        file_name = f"{branch_name}_part{i}.py"

        if complete_flag:
            # Save code
            save_code_locally(code_snippet, file_name)
            # Run lint checks
            lint_ok = run_lint_checks(file_name)
            # Run tests
            tests_ok = run_tests_on_code(file_name)

            if lint_ok and tests_ok:
                logging.info(f"Code snippet verified and tests passed: {file_name}")
            else:
                logging.warning(f"Lint/tests failed for {file_name}.")
        else:
            logging.info(f"Code snippet {file_name} is incomplete or has issues.")
            logging.info(f"Verifier feedback: {feedback}")

            # Refine code
            await refine_incomplete_code(
                conv_manager=conv_manager,
                branch_name=branch_name,
                incomplete_feedback=feedback,
                existing_code_snippet=code_snippet,
                code_index=i
            )

    # Handle clarifying questions with concurrency (example approach)
    # We'll push each question into an async task queue so they can be processed concurrently.
    question_tasks = []
    for i, question in enumerate(questions):
        new_branch = f"{branch_name}_Q{i}"
        question_tasks.append(handle_question(conv_manager, question, new_branch, depth + 1, max_depth))

    if question_tasks:
        await asyncio.gather(*question_tasks)
