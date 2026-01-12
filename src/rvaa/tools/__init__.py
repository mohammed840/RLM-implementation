"""
RVAA Tools Module

This module contains the core tools for video processing, LLM interaction,
perception (vision-to-text), and indexing.
"""

from rvaa.tools.llm_backends import (
    LLMBackend,
    OpenAIBackend,
    QwenBackend,
    ClaudeBackend,
    OpenRouterBackend,
    get_llm_backend,
    BackendConfig,
)
from rvaa.tools.video_io import (
    decode_video,
    extract_keyframes,
    sample_frames,
    FrameCache,
)
from rvaa.tools.perception import (
    PerceptionBackend,
    BLIP2Backend,
    batch_caption,
)
from rvaa.tools.index import (
    SegmentIndex,
    TextEncoder,
    SegmentMetadata,
)

__all__ = [
    # LLM Backends
    "LLMBackend",
    "OpenAIBackend", 
    "QwenBackend",
    "ClaudeBackend",
    "OpenRouterBackend",
    "get_llm_backend",
    "BackendConfig",
    # Video I/O
    "decode_video",
    "extract_keyframes",
    "sample_frames",
    "FrameCache",
    # Perception
    "PerceptionBackend",
    "BLIP2Backend",
    "batch_caption",
    # Indexing
    "SegmentIndex",
    "TextEncoder",
    "SegmentMetadata",
]
