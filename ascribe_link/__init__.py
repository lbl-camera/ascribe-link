"""Ascribe-Link: HTTP server for Ascribe-XR processing and curated specimens."""

from ascribe_link.app import create_app
from ascribe_link.federation import FederationClient, FederationHub
from ascribe_link.processing import FunctionRegistry
from ascribe_link.sandbox import (
    SandboxConfig,
    SandboxResult,
    is_firejail_available,
    run_sandboxed,
)
from ascribe_link.specimen_store import SpecimenStore

__all__ = [
    "create_app",
    "FederationClient",
    "FederationHub",
    "FunctionRegistry",
    "SandboxConfig",
    "SandboxResult",
    "SpecimenStore",
    "is_firejail_available",
    "run_sandboxed",
]

# Optional: agent generator (requires claude-agent-sdk)
try:
    from ascribe_link.agent_generator import generate_mesh_with_agent, create_agent_function
    __all__.extend(["generate_mesh_with_agent", "create_agent_function"])
except ImportError:
    pass
