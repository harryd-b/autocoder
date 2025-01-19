#!/usr/bin/env python3
"""
recursive_builder.py

Main logic for the recursive, piecewise code generation process.
By default, uses a local Triton server for Meta-Llama-3-8B on port 8000,
but can switch to OpenAI if config.yaml sets 'model_source: openai'.
"""

import logging
import re
import asyncio
import random
from typing import Dict, List

from conversation_manager import ConversationManager
from verification import (
    verify_code_with_chatgpt,
    run_tests_on_code,
    run_lint_checks
)
from api_utils import (
    call_openai_chat_completion,
    call_local_llama_inference,
    GPT_MODEL,
    config
)

###############################################################################
# MODEL-CALLING LOGIC
###############################################################################
DEFAULT_MODEL_SOURCE = config.get("model_source", "local")  # "local" or "openai"

def call_model(conversation_history: List[Dict[str, str]]) -> dict:
    """
    Calls either the local Triton Llama or the OpenAI endpoint, depending on 'model_source' in config.yaml.
    Returns a response dict formatted similarly to the OpenAI ChatCompletion response:
      {
        "choices": [
          {
            "message": {
              "content": "<assistant response text>"
            }
          }
        ]
      }
    """

    if DEFAULT_MODEL_SOURCE.lower() == "openai":
        # Use OpenAI API directly
        response = call_openai_chat_completion(conversation_history, model=GPT_MODEL)
        return response

    else:
        # Default: local Triton server
        # Flatten the conversation into a single prompt string.
        # We combine roles (user/assistant) for context, but only the last user message
        # truly matters for the final request. You can adjust the flattening logic as needed.
        flattened_prompt = []
        for msg in conversation_history:
            role = msg["role"].upper()
            content = msg["content"]
            flattened_prompt.append(f"{role}:\n{content}\n")

        # Combine into a single string to send to local inference
        prompt_text = "\n".join(flattened_prompt).strip()

        # Use the local call, sending a single-element list for a single "chat" prompt
        local_responses = call_local_llama_inference([prompt_text])

        # If we got something back, adapt it into an OpenAI-like structure
        if local_responses:
            assistant_text = local_responses[0]  # For a single prompt, we have one output
        else:
            assistant_text = ""

        # Return a dict that mimics the OpenAI response structure
        return {
            "choices": [
                {
                    "message": {
                        "content": assistant_text
                    }
                }
            ]
        }

###############################################################################
# EXTRACTION & SAVING HELPERS
###############################################################################

def extract_questions_and_code(response_text: str) -> Dict[str, List[str]]:
    """
    Parses the response text and attempts to extract clarifying questions and code blocks.
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
        print(f"DEBUG: line={repr(line)} endswith('?')={line.strip().endswith('?')}")
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

###############################################################################
# REFINEMENT STEPS
###############################################################################

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
    logging.info("Refining incomplete code snippet based on verifier feedback...")

    refine_prompt = (
        "We received the following feedback from a verification step:\n\n"
        f"'{incomplete_feedback}'\n\n"
        "Please refine the following code snippet to address these issues. "
        "Only provide the refined snippet in triple backticks:\n\n"
        f"```python\n{existing_code_snippet}\n```"
    )

    conv_manager.update_conversation(branch_name, "user", refine_prompt)
    conversation_history = conv_manager.get_conversation(branch_name)

    # Call the model (local or OpenAI) for improved snippet
    response = call_model(conversation_history)
    improvement_response = response["choices"][0]["message"]["content"]
    conv_manager.update_conversation(branch_name, "assistant", improvement_response)

    parsed = extract_questions_and_code(improvement_response)
    if not parsed["code_blocks"]:
        logging.warning("No improved code snippet returned in refinement step.")
        return

    improved_code = parsed["code_blocks"][0]
    file_name = f"{branch_name}_refined_{code_index}.py"

    verification_data = verify_code_with_chatgpt(improved_code)
    if verification_data and verification_data.get("complete") is True:
        save_code_locally(improved_code, file_name)
        lint_ok = run_lint_checks(file_name)
        tests_ok = run_tests_on_code(file_name)
        if lint_ok and tests_ok:
            logging.info(f"Refined code verified successfully: {file_name}")
        else:
            logging.warning(f"Refined code has lint/test issues: {file_name}")
    else:
        logging.warning(f"Snippet still incomplete after refinement: {file_name}")

###############################################################################
# QUESTION HANDLING
###############################################################################

async def handle_question(
    conv_manager: ConversationManager,
    question_text: str,
    branch_name: str,
    depth: int,
    max_depth: int
):
    """
    Handles a single clarifying question by prompting the user for input in real time.
    Once the user provides their answer, it is appended to the conversation, and we
    recursively call 'recursive_prompt' to continue code generation or further questions.
    """
    if depth > max_depth:
        logging.warning("Reached max recursion depth while handling question.")
        return

    # Prompt the user interactively. Since we're in an async function, we use asyncio.to_thread
    # so as not to block the event loop.
    user_answer = await asyncio.to_thread(
        input,
        f"\nCHATGPT ASKED: {question_text}\nYour answer: "
    )

    # Update the conversation with the user's real answer
    conv_manager.update_conversation(branch_name, "user", user_answer)

    # Continue the recursion from here
    await recursive_prompt(
        conv_manager=conv_manager,
        user_prompt=user_answer,
        branch_name=branch_name,
        depth=depth,
        max_depth=max_depth
    )


###############################################################################
# RECURSIVE LOGIC
###############################################################################

async def recursive_prompt(
    conv_manager: ConversationManager,
    user_prompt: str,
    branch_name: str,
    depth: int,
    max_depth: int
) -> None:
    """
    Recursively gather requirements and generate code from ChatGPT or local Llama,
    verifying each snippet, refining if incomplete, and following up on questions.
    """
    if depth > max_depth:
        logging.warning(f"Max recursion depth ({max_depth}) reached. Stopping.")
        return

    conv_manager.update_conversation(branch_name, "user", user_prompt)
    conversation_history = conv_manager.get_conversation(branch_name)

    # Call the model (local or OpenAI)
    response = call_model(conversation_history)
    assistant_response = response["choices"][0]["message"]["content"].strip()
    if not assistant_response:
        logging.warning("Received empty response from the model.")
        return

    conv_manager.update_conversation(branch_name, "assistant", assistant_response)
    logging.info(f"[BRANCH={branch_name}] Model says:\n{assistant_response}\n")

    parsed = extract_questions_and_code(assistant_response)
    questions = parsed["questions"]
    code_blocks = parsed["code_blocks"]

    # Verify code blocks in parallel
    verification_tasks = []
    loop = asyncio.get_event_loop()

    for code_index, code_snippet in enumerate(code_blocks):
        # Spin off verification calls in an executor
        verification_tasks.append(loop.run_in_executor(
            None,
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
            save_code_locally(code_snippet, file_name)
            lint_ok = run_lint_checks(file_name)
            tests_ok = run_tests_on_code(file_name)
            if lint_ok and tests_ok:
                logging.info(f"Code snippet verified and tests passed: {file_name}")
            else:
                logging.warning(f"Lint/tests failed for {file_name}.")
        else:
            logging.info(f"Code snippet {file_name} is incomplete or has issues.")
            logging.info(f"Verifier feedback: {feedback}")
            await refine_incomplete_code(
                conv_manager=conv_manager,
                branch_name=branch_name,
                incomplete_feedback=feedback,
                existing_code_snippet=code_snippet,
                code_index=i
            )

    # Handle clarifying questions in parallel
    question_tasks = []
    for i, question in enumerate(questions):
        new_branch = f"{branch_name}_Q{i}"
        question_tasks.append(
            handle_question(conv_manager, question, new_branch, depth + 1, max_depth)
        )

    if question_tasks:
        await asyncio.gather(*question_tasks)
