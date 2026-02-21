import os
import sys
import pytest

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from radiology_assistant.api import app
from radiology_assistant.auth import get_current_user, TokenData, UserRole

# Mock user for all tests
mock_user = TokenData(sub="test_user", role=UserRole.RADIOLOGIST)

@pytest.fixture(autouse=True)
def bypass_auth(monkeypatch):
    """
    Automatically override the get_current_user dependency for all tests.
    """
    app.dependency_overrides[get_current_user] = lambda: mock_user
    yield
    app.dependency_overrides = {}
