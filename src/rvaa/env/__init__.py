"""
RVAA Environment Module

This module provides the video environment abstraction that represents
the "external context" in the RLM paradigm.
"""

from rvaa.env.video_env import (
    VideoEnv,
    VideoView,
    VideoMetadata,
    FrameData,
)

__all__ = [
    "VideoEnv",
    "VideoView",
    "VideoMetadata",
    "FrameData",
]
