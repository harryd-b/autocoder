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
    Returns a response dict formatted similarly to the OpenAI ChatCompletion response.
    """

    if DEFAULT_MODEL_SOURCE.lower() == "openai":
        # Use OpenAI API directly
        response = call_openai_chat_completion(conversation_history, model=GPT_MODEL)
        return response
    else:
        # Default: local Triton server
        flattened_prompt = []
        for msg in conversation_history:
            role = msg["role"].upper()
            content = msg["content"]
            flattened_prompt.append(f"{role}:\n{content}\n")

        prompt_text = "\n".join(flattened_prompt).strip()

        local_responses = call_local_llama_inference([prompt_text])
        if local_responses:
            assistant_text = local_responses[0]
        else:
            assistant_text = ""

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
    Improved parsing for code blocks and questions.
    """
    questions = []
    code_blocks = []

    # Extract questions (lines ending with '?')
    questions = [
        line.strip() 
        for line in response_text.split('\n') 
        if line.strip().endswith('?')
    ]

    # Extract code blocks (handle ```python and ```)
    code_pattern = r"```(?:python)?\n(.*?)```"
    matches = re.findall(code_pattern, response_text, re.DOTALL)
    code_blocks = [match.strip() for match in matches]

    return {"questions": questions, "code_blocks": code_blocks}

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
    Once the user provides their answer, it's appended to the SAME conversation branch,
    and we call recursive_prompt again on that same branch to generate the next response.
    """
    if depth > max_depth:
        logging.warning("Reached max recursion depth while handling question.")
        return

    # Prompt the user
    user_answer = await asyncio.to_thread(
        input,
        f"\nCHATGPT ASKED: {question_text}\nYour answer: "
    )

    # Put the user's real answer in the SAME branch
    conv_manager.update_conversation(branch_name, "user", user_answer)

    # Continue recursion from here, same branch
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

    # 1) Append the user's new prompt
    conv_manager.update_conversation(branch_name, "user", user_prompt)

    # 2) Call the model
    conversation_history = conv_manager.get_conversation(branch_name)
    response = call_model(conversation_history)
    assistant_response = response["choices"][0]["message"]["content"].strip()
    if not assistant_response:
        logging.warning("Received empty response from the model.")
        return

    # 3) Add the assistant response
    conv_manager.update_conversation(branch_name, "assistant", assistant_response)
    logging.info(f"[BRANCH={branch_name}] Model says:\n{assistant_response}\n")

    # 4) Extract code & questions
    parsed = extract_questions_and_code(assistant_response)
    questions = parsed["questions"]
    code_blocks = parsed["code_blocks"]

    # 5) Verify code in parallel
    loop = asyncio.get_event_loop()
    verification_futs = [
        loop.run_in_executor(None, verify_code_with_chatgpt, snippet)
        for snippet in code_blocks
    ]
    verification_results = await asyncio.gather(*verification_futs)

    # 6) Process verification results
    for i, snippet in enumerate(code_blocks):
        verification_data = verification_results[i]
        if not verification_data:
            logging.warning("No verification data received for a code snippet. Skipping save.")
            continue

        complete_flag = verification_data.get("complete", False)
        feedback = verification_data.get("feedback", "No feedback provided.")
        file_name = f"{branch_name}_part{i}.py"

        if complete_flag:
            save_code_locally(snippet, file_name)
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
                existing_code_snippet=snippet,
                code_index=i
            )

    # 7) Handle clarifying questions (SEQUENTIALLY in the SAME branch)
    for question_text in questions:
        await handle_question(conv_manager, question_text, branch_name, depth + 1, max_depth)
