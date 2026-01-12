"""
Unit Tests for RVAA

Run with: pytest tests/ -v
"""

import pytest


# Configure pytest-asyncio
@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
