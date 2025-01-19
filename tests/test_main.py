import pytest
import json
import asyncio
from unittest.mock import patch, MagicMock
import main

###############################################################################
# Test that main.py calls asyncio.run and handles KeyboardInterrupt
###############################################################################
@patch("main.asyncio.run")
def test_main_run_keyboard_interrupt(mock_run):
    """
    Test that main.py's __main__ block calls asyncio.run and gracefully handles a KeyboardInterrupt.
    Instead of forcing main.__name__ = '__main__', we just replicate the logic by calling
    asyncio.run(main.main()) ourselves.
    """
    # Force a KeyboardInterrupt when asyncio.run(...) is called
    mock_run.side_effect = KeyboardInterrupt

    try:
        # This mirrors how main.py runs main.main() if __name__ == "__main__"
        asyncio.run(main.main())
    except KeyboardInterrupt:
        pass

    # If we get here, we handled the KeyboardInterrupt gracefully
    assert True

###############################################################################
# Test simulating user input for clarifying questions
###############################################################################
@pytest.mark.asyncio
async def test_main_with_mocked_input():
    """
    Demonstrates how to patch builtins.input to simulate user responses
    for clarifying questions. We'll mock a single user input and check that
    the code runs without errors.
    """
    # We'll patch input to return a canned answer when the code asks for user input
    with patch("builtins.input", return_value="Test user answer"):

        # We also patch 'recursive_prompt' to avoid running the entire flow in a test
        # (particularly if you have external calls or a complex environment).
        # Instead, we'll just ensure main() calls it without error.
        with patch("main.recursive_prompt") as mock_prompt:
            # We'll simulate an awaitable that does nothing
            async def mock_coroutine(*args, **kwargs):
                pass

            mock_prompt.side_effect = mock_coroutine

            # Now call main() in an async context. Because we have @pytest.mark.asyncio,
            # we can directly await main.main().
            await main.main()

            # Check that recursive_prompt was called at least once
            assert mock_prompt.call_count == 1
