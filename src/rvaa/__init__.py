"""
RVAA: Recursive Vision-Action Agent for Long Video Understanding

This package implements the RLM (Recursive Language Model) paradigm for video
understanding, inspired by the paper "Recursive Language Models" (arXiv:2512.24601).

The core thesis: treat extremely long video as an external environment, not something
to stuff into an LLM context window. The agent interacts with video programmatically
(peek, slice, index, search, verify) and recursively calls sub-models for local 
understanding, then composes a global answer.

Package Structure:
    - agent/: RLM-style orchestrator and sub-agent implementations
    - env/: Video environment abstraction
    - tools/: Video I/O, perception, indexing, and LLM backends
    - protocols/: Typed event schemas for streaming
    - server/: FastAPI server and streaming endpoints
    - eval/: Evaluation scripts and metrics
"""

__version__ = "0.1.0"
__author__ = "RVAA Team"

from rvaa.agent.root_agent import RootAgent
from rvaa.agent.sub_agent import SubAgent
from rvaa.env.video_env import VideoEnv
from rvaa.tools.llm_backends import LLMBackend, get_llm_backend
from rvaa.protocols.events import AgentEvent, EventType

__all__ = [
    "RootAgent",
    "SubAgent",
    "VideoEnv",
    "LLMBackend",
    "get_llm_backend",
    "AgentEvent",
    "EventType",
    "__version__",
]
