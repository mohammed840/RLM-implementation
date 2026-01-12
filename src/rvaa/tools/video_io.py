"""
Video I/O Utilities

This module provides efficient video decoding, caching, and frame extraction.
It's designed to handle long videos without loading everything into memory.

Key features:
- Memory-mapped video access via PyAV
- Disk-based frame caching with LRU eviction
- Keyframe extraction for efficient navigation
- Batch frame sampling
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

try:
    from diskcache import Cache
    HAS_DISKCACHE = True
except ImportError:
    HAS_DISKCACHE = False

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Basic video file information."""
    path: Path
    duration: float
    fps: float
    total_frames: int
    width: int
    height: int
    codec: str
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "VideoInfo":
        """Extract info from a video file."""
        import av
        
        path = Path(path)
        container = av.open(str(path))
        video_stream = container.streams.video[0]
        
        duration = float(container.duration) / 1_000_000 if container.duration else 0
        fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
        
        info = cls(
            path=path,
            duration=duration,
            fps=fps,
            total_frames=video_stream.frames or int(duration * fps),
            width=video_stream.width,
            height=video_stream.height,
            codec=video_stream.codec.name,
        )
        
        container.close()
        return info


class FrameCache:
    """Disk-based cache for extracted video frames.
    
    Uses diskcache for persistent caching with LRU eviction.
    Falls back to in-memory dict if diskcache is not available.
    
    Attributes:
        cache_dir: Directory for cache storage
        max_size_gb: Maximum cache size in GB
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_size_gb: float = 10.0,
    ):
        """Initialize frame cache.
        
        Args:
            cache_dir: Directory for cache storage
            max_size_gb: Maximum cache size in GB
        """
        self.cache_dir = cache_dir or Path.home() / ".cache" / "rvaa" / "frames"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        
        if HAS_DISKCACHE:
            self._cache: Union[Cache, dict[str, Any]] = Cache(
                str(self.cache_dir),
                size_limit=self.max_size_bytes,
            )
        else:
            logger.warning("diskcache not available, using in-memory cache")
            self._cache = {}
    
    def _make_key(self, video_path: Path, frame_idx: int) -> str:
        """Create a unique cache key for a frame."""
        video_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:12]
        return f"{video_hash}_{frame_idx}"
    
    def get(self, video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
        """Get a cached frame.
        
        Args:
            video_path: Path to the video file
            frame_idx: Frame index
            
        Returns:
            Cached frame as numpy array, or None if not cached
        """
        key = self._make_key(video_path, frame_idx)
        
        if HAS_DISKCACHE:
            return self._cache.get(key)
        else:
            data = self._cache.get(key)
            if data is not None:
                return pickle.loads(data)
            return None
    
    def put(self, video_path: Path, frame_idx: int, frame: np.ndarray) -> None:
        """Cache a frame.
        
        Args:
            video_path: Path to the video file
            frame_idx: Frame index
            frame: Frame as numpy array
        """
        key = self._make_key(video_path, frame_idx)
        
        if HAS_DISKCACHE:
            self._cache.set(key, frame)
        else:
            self._cache[key] = pickle.dumps(frame)
    
    def clear(self) -> None:
        """Clear the cache."""
        if HAS_DISKCACHE:
            self._cache.clear()
        else:
            self._cache.clear()
    
    def __len__(self) -> int:
        """Return number of cached items."""
        return len(self._cache)


def decode_video(
    path: Union[str, Path],
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> list[tuple[float, np.ndarray]]:
    """Decode video frames efficiently.
    
    Args:
        path: Path to video file
        start_time: Start timestamp in seconds
        end_time: End timestamp in seconds (None = end of video)
        max_frames: Maximum number of frames to return
        
    Returns:
        List of (timestamp, frame) tuples
    """
    import av
    
    path = Path(path)
    container = av.open(str(path))
    video_stream = container.streams.video[0]
    
    if end_time is None:
        duration = float(container.duration) / 1_000_000 if container.duration else 0
        end_time = duration
    
    # Seek to start
    target_pts = int(start_time / video_stream.time_base)
    container.seek(target_pts, stream=video_stream)
    
    frames = []
    for frame in container.decode(video_stream):
        current_time = float(frame.pts * video_stream.time_base)
        
        if current_time < start_time:
            continue
        if current_time > end_time:
            break
        if max_frames and len(frames) >= max_frames:
            break
        
        img = frame.to_ndarray(format='rgb24')
        frames.append((current_time, img))
    
    container.close()
    return frames


def extract_keyframes(
    path: Union[str, Path],
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    max_keyframes: Optional[int] = None,
) -> list[tuple[float, np.ndarray]]:
    """Extract only keyframes (I-frames) from a video.
    
    This is much faster than decoding all frames and useful for
    creating video summaries or efficient navigation.
    
    Args:
        path: Path to video file
        start_time: Start timestamp in seconds
        end_time: End timestamp in seconds
        max_keyframes: Maximum number of keyframes
        
    Returns:
        List of (timestamp, frame) tuples for keyframes only
    """
    import av
    
    path = Path(path)
    container = av.open(str(path))
    video_stream = container.streams.video[0]
    
    if end_time is None:
        duration = float(container.duration) / 1_000_000 if container.duration else 0
        end_time = duration
    
    # Skip non-keyframes at decoder level
    video_stream.codec_context.skip_frame = 'NONKEY'
    
    target_pts = int(start_time / video_stream.time_base)
    container.seek(target_pts, stream=video_stream)
    
    keyframes = []
    for frame in container.decode(video_stream):
        current_time = float(frame.pts * video_stream.time_base)
        
        if current_time < start_time:
            continue
        if current_time > end_time:
            break
        if max_keyframes and len(keyframes) >= max_keyframes:
            break
        
        img = frame.to_ndarray(format='rgb24')
        keyframes.append((current_time, img))
    
    container.close()
    return keyframes


def sample_frames(
    path: Union[str, Path],
    start_time: float,
    end_time: float,
    fps: float = 1.0,
    cache: Optional[FrameCache] = None,
) -> list[tuple[float, np.ndarray]]:
    """Sample frames at a specified FPS.
    
    This is the primary method for agents to get frames from specific
    time ranges. A lower FPS (e.g., 1.0) means fewer frames, which is
    more efficient for processing with vision models.
    
    Args:
        path: Path to video file
        start_time: Start timestamp in seconds
        end_time: End timestamp in seconds
        fps: Sampling rate in frames per second
        cache: Optional frame cache
        
    Returns:
        List of (timestamp, frame) tuples
        
    Example:
        >>> frames = sample_frames("video.mp4", 10.0, 20.0, fps=0.5)
        >>> len(frames)  # ~5 frames for 10 second window at 0.5 fps
        5
    """
    import av
    
    path = Path(path)
    
    duration = end_time - start_time
    n_samples = max(1, int(duration * fps))
    
    # Calculate target timestamps
    if n_samples == 1:
        timestamps = [start_time + duration / 2]
    else:
        step = duration / (n_samples - 1)
        timestamps = [start_time + i * step for i in range(n_samples)]
    
    container = av.open(str(path))
    video_stream = container.streams.video[0]
    
    frames = []
    for target_ts in timestamps:
        # Check cache first
        frame_idx = int(target_ts * float(video_stream.average_rate))
        if cache:
            cached = cache.get(path, frame_idx)
            if cached is not None:
                frames.append((target_ts, cached))
                continue
        
        # Seek and decode
        target_pts = int(target_ts / video_stream.time_base)
        container.seek(target_pts, stream=video_stream)
        
        for frame in container.decode(video_stream):
            current_time = float(frame.pts * video_stream.time_base)
            if current_time >= target_ts - 0.1:  # Small tolerance
                img = frame.to_ndarray(format='rgb24')
                frames.append((current_time, img))
                
                # Cache the frame
                if cache:
                    cache.put(path, frame_idx, img)
                break
    
    container.close()
    return frames


def get_frame_at_time(
    path: Union[str, Path],
    timestamp: float,
    cache: Optional[FrameCache] = None,
) -> np.ndarray:
    """Get a single frame at a specific timestamp.
    
    Args:
        path: Path to video file
        timestamp: Target timestamp in seconds
        cache: Optional frame cache
        
    Returns:
        Frame as numpy array (H, W, C) in RGB format
    """
    frames = sample_frames(path, timestamp, timestamp + 0.1, fps=30.0, cache=cache)
    if not frames:
        raise RuntimeError(f"Failed to get frame at {timestamp}s from {path}")
    return frames[0][1]


def create_video_thumbnail(
    path: Union[str, Path],
    output_path: Union[str, Path],
    width: int = 320,
    timestamp: Optional[float] = None,
) -> Path:
    """Create a thumbnail image from a video.
    
    Args:
        path: Path to video file
        output_path: Output image path
        width: Target width (height scales proportionally)
        timestamp: Timestamp to capture (None = 10% into video)
        
    Returns:
        Path to the created thumbnail
    """
    from PIL import Image
    
    path = Path(path)
    output_path = Path(output_path)
    
    # Get video info
    info = VideoInfo.from_file(path)
    
    if timestamp is None:
        timestamp = info.duration * 0.1
    
    # Get frame
    frame = get_frame_at_time(path, timestamp)
    
    # Resize
    img = Image.fromarray(frame)
    aspect = img.height / img.width
    height = int(width * aspect)
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    
    # Save
    img.save(output_path)
    return output_path
