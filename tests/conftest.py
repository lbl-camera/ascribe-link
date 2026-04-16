"""Shared pytest fixtures for ascribe-link tests."""
import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
