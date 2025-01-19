# Recursive Code Builder

A proof-of-concept tool that iteratively generates and verifies code snippets using **ChatGPT** (via OpenAI’s API) **or** a **locally hosted model** on Triton in a step-by-step, conversation-driven workflow. Each piece of code is linted, tested, and re-verified until complete.

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project demonstrates how one might build complex software through multiple interactions with ChatGPT **or** a **local Triton** model, verifying each piece of generated code in a separate “verification” process and refining any incomplete or incorrect snippets. The process is designed to:

1. Ask clarifying questions until enough detail is available for a code snippet.  
2. Generate code in small pieces.  
3. Verify each code snippet for completeness (via either OpenAI or a **locally hosted** model under Triton).  
4. Lint (flake8) and test (pytest) each snippet.  
5. Refine code if incomplete, re-verifying until stable.  
6. Manage conversation history in a sliding window to control token usage.  
7. Persist conversation state in a JSON file for possible resumption.

---

## Project Structure

my_recursive_code_builder/ ├── api_utils.py ├── config.yaml ├── conversation_manager.py ├── main.py ├── recursive_builder.py ├── verification.py ├── requirements.txt └── tests/ ├── test_api_utils.py ├── test_conversation_manager.py ├── test_verification.py ├── test_recursive_builder.py └── test_main.py

yaml
Copy

### Key Modules

- **api_utils.py**  
  Handles either OpenAI calls **or** local Triton calls with tenacity-based retry logic.

- **verification.py**  
  Sends each snippet to a verification model (local Triton or OpenAI), checks for completeness, and runs lint (flake8) plus tests (pytest).

- **conversation_manager.py**  
  Loads and saves conversation states in a JSON file, manages a sliding window to limit token usage.

- **recursive_builder.py**  
  Contains the **recursive** logic for generating code, verifying snippets, refining incomplete ones, **and** handling clarifying questions.  
  *Now keeps clarifying Q&A in the same conversation branch* for coherence.

- **main.py**  
  Entry point. Loads config, sets up logging, and starts the recursive process.

- **tests/**  
  Contains pytest-based tests for each module, using unittest.mock to isolate external calls (OpenAI/local inference).

---

## Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/harryd-b/autocoder.git
   cd my_recursive_code_builder
Create a Virtual Environment (Recommended)

bash
Copy
python3 -m venv venv
source venv/bin/activate    # (Windows: venv\Scripts\activate)
Install Dependencies

bash
Copy
pip install -r requirements.txt
Typically includes openai, tritonclient[http], tenacity, PyYAML, flake8, pytest, etc.

Configure Your Model

If using model_source: openai, sign up for an OpenAI API key.
If using a locally hosted model on Triton, ensure Triton is running on port 8000.
Adjust config.yaml accordingly (e.g., model_source: "local" or "openai").
Configuration
Open config.yaml to customize parameters such as:

yaml
Copy
openai_api_key: "YOUR_OPENAI_API_KEY"
model: "gpt-4"
model_source: "local"          # or "openai"
max_depth: 10
max_conversation_length: 10
max_retries: 3
retry_base_seconds: 1.5
retry_max_seconds: 10
conversation_file: "conversation_state.json"
model_source: "openai" or "local" (Triton).
openai_api_key: Needed if model_source is "openai".
model: If using OpenAI (e.g. gpt-3.5-turbo, gpt-4).
max_depth: Maximum recursion depth.
max_conversation_length: Sliding window size for conversation messages.
max_retries: How many times to retry a failed model call.
conversation_file: Where conversation states are stored.
Usage
Edit config.yaml

Choose model_source: "local" or "openai".
Provide any necessary keys (e.g. openai_api_key).
Run the Main Script

bash
Copy
python main.py
Logs will appear in the console.
Conversation states are stored in conversation_state.json.
Verified code snippets are saved locally as .py files.
Observe Generated Code

Each snippet is linted, tested, and verified.
If declared “complete,” it’s saved (e.g., root_part0.py).
If “incomplete,” the script refines it until stable.
Resume or Restart

Stop and start again; the script picks up conversation_state.json for continuity.
Delete it to start fresh.
How It Works
Root Prompt

A system message sets the scene: “You are ChatGPT. You will interact with the user…”
The user wants to build a complex software application.
ChatGPT (or Local Model) Response

The AI might ask clarifying questions or output code snippets.
Each snippet is verified in a separate step: a “verification mode” prompt checks completeness, plus local lint/tests.
Conversation Manager

Maintains a rolling window to trim older messages, controlling token usage.
Saves data to conversation_state.json.
Refinement

If incomplete, feedback is fed back to the AI for improvement.
Clarifying Questions

Kept in the same conversation branch, prompting user answers (or auto-answers in tests), then continuing the recursion.
Testing
Use pytest:

bash
Copy
python -m pytest tests
Mocking: Network calls to OpenAI/local Triton are mocked out.
Async Tests: Some tests use @pytest.mark.asyncio to handle coroutines properly.
Lint: Run flake8 . to enforce code style.
Troubleshooting
OpenAI Rate Limits: If you get 429 errors, your code will retry. Adjust concurrency or consider an upgraded plan.
Local Triton Issues: Ensure your local server is running on port 8000 and you installed tritonclient[http].
Conversation File: If conversation_state.json gets corrupt or you want a fresh start, remove it.
Incomplete Code: Sometimes AI returns partial answers. The script attempts multiple refinements. Increase max_depth if needed.
Contributing
Fork this repo.
Create a feature branch (git checkout -b feature/new-feature).
Commit changes.
Push to your fork.
Open a Pull Request.
We welcome improvements, bug fixes, or new features—particularly around concurrency, advanced verification strategies, or broader test coverage.

License
MIT License
Feel free to use, modify, and distribute with attribution. For details, see LICENSE.

Thank you for using the Recursive Code Builder! If you have suggestions or questions, open an issue or PR. Happy coding!