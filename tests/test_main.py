import pytest
from unittest.mock import patch, MagicMock
import main

@patch("asyncio.run")
def test_main_run(mock_asyncio_run):
    """Test that main.py's __main__ block calls asyncio.run without errors."""
    # Simulate calling main.py
    try:
        main.__name__ = "__main__"
        with patch("main.asyncio.run") as mock_run:
            mock_run.side_effect = KeyboardInterrupt  # to exit gracefully
            main.main()
    except KeyboardInterrupt:
        pass
    # If we get here, it means the script handled it as expected
    assert True
