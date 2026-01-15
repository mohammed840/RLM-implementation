"""
Video Environment Abstraction

This module implements the VideoEnv class - the "external environment" that
holds the long video context. Following the RLM paper's paradigm, we treat
the video as something to interact with programmatically rather than stuffing
into an LLM context window.

The VideoEnv is exposed as the `context` variable in the REPL runtime,
analogous to the paper's context string variable. It supports:
- Time-based slicing returning lightweight views
- Metadata inspection
- Frame and clip access
- Integration with perception and indexing tools

Key design principles from the paper:
1. The agent should NOT "read the whole video" at once
2. It should inspect metadata, choose chunking strategies
3. Store intermediate findings in buffers
4. Batch subcalls (not tiny subcalls per frame)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata about a video file.
    
    Attributes:
        path: Path to the video file
        duration_seconds: Total duration in seconds
        fps: Frames per second
        total_frames: Total number of frames
        width: Frame width in pixels
        height: Frame height in pixels
        codec: Video codec name
        file_size_bytes: File size in bytes
    """
    path: Path
    duration_seconds: float
    fps: float
    total_frames: int
    width: int
    height: int
    codec: str = "unknown"
    file_size_bytes: int = 0
    
    def __repr__(self) -> str:
        return (
            f"VideoMetadata(duration={self.duration_seconds:.1f}s, "
            f"fps={self.fps:.1f}, frames={self.total_frames}, "
            f"resolution={self.width}x{self.height})"
        )
    
    def to_context_string(self) -> str:
        """Return a string representation for the REPL context.
        
        This is what the LLM sees when it inspects the context metadata.
        """
        return (
            f"Video file: {self.path.name}\n"
            f"Duration: {self.duration_seconds:.1f} seconds ({self.duration_seconds/60:.1f} minutes)\n"
            f"FPS: {self.fps:.2f}\n"
            f"Total frames: {self.total_frames:,}\n"
            f"Resolution: {self.width}x{self.height}\n"
            f"Codec: {self.codec}\n"
            f"File size: {self.file_size_bytes / (1024*1024):.1f} MB"
        )


@dataclass
class FrameData:
    """Container for a single frame with metadata.
    
    Attributes:
        frame_idx: Frame index (0-based)
        timestamp: Timestamp in seconds
        image: The frame as a numpy array (H, W, C) in RGB format
        is_keyframe: Whether this is a keyframe (I-frame)
    """
    frame_idx: int
    timestamp: float
    image: np.ndarray
    is_keyframe: bool = False
    
    @property
    def shape(self) -> tuple[int, int, int]:
        """Return (height, width, channels)."""
        return self.image.shape  # type: ignore
    
    def __repr__(self) -> str:
        h, w, c = self.shape
        return (
            f"FrameData(idx={self.frame_idx}, t={self.timestamp:.2f}s, "
            f"shape={w}x{h}, keyframe={self.is_keyframe})"
        )


@dataclass
class VideoView:
    """A lightweight view into a portion of a video.
    
    VideoView does NOT hold raw frame data. It's a lazy reference that
    can be used to request frames when needed. This matches the RLM paradigm
    of not loading everything into context.
    
    Attributes:
        source: The parent VideoEnv
        start_time: Start timestamp in seconds
        end_time: End timestamp in seconds
        start_frame: Start frame index
        end_frame: End frame index (exclusive)
    """
    source: "VideoEnv"
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    
    @property
    def duration(self) -> float:
        """Duration of this view in seconds."""
        return self.end_time - self.start_time
    
    @property
    def num_frames(self) -> int:
        """Number of frames in this view."""
        return self.end_frame - self.start_frame
    
    def __repr__(self) -> str:
        return (
            f"VideoView(t={self.start_time:.2f}s-{self.end_time:.2f}s, "
            f"frames={self.start_frame}-{self.end_frame}, "
            f"duration={self.duration:.2f}s)"
        )
    
    def to_context_string(self) -> str:
        """Return a string representation for the REPL context."""
        return (
            f"Video segment: {self.start_time:.2f}s to {self.end_time:.2f}s\n"
            f"Duration: {self.duration:.2f} seconds\n"
            f"Frames: {self.start_frame} to {self.end_frame} ({self.num_frames} frames)"
        )
    
    def get_frames(self, sample_fps: Optional[float] = None) -> list[FrameData]:
        """Get frames from this view.
        
        Args:
            sample_fps: If provided, sample at this fps instead of native.
                       Use this to reduce the number of frames for efficiency.
                       
        Returns:
            List of FrameData objects
        """
        return self.source.get_frames(
            self.start_time, 
            self.end_time, 
            sample_fps=sample_fps
        )
    
    def sample_uniform(self, n_frames: int) -> list[FrameData]:
        """Sample n frames uniformly from this view.
        
        Args:
            n_frames: Number of frames to sample
            
        Returns:
            List of FrameData objects
        """
        return self.source.sample_frames_uniform(
            self.start_time,
            self.end_time,
            n_frames
        )


class VideoEnv:
    """The video environment - external context for the RLM agent.
    
    This class implements the paper's paradigm of treating long context
    as an external environment. The agent interacts with the video through
    programmatic operations rather than seeing all frames at once.
    
    In the REPL runtime, an instance of this class is bound to the `context`
    variable, allowing the agent to:
    - Inspect metadata: `print(context.metadata)`
    - Slice by time: `segment = context[10.0:20.0]`
    - Get specific frames: `frame = context.get_frame(15.5)`
    - Sample frames: `frames = context.sample_frames(0, 60, fps=1)`
    
    Example REPL usage (from agent's perspective):
        ```repl
        # Peek at the video metadata
        print(f"Video duration: {context.duration}s")
        print(f"Total frames: {context.total_frames}")
        
        # Get a segment view
        segment = context[0:30]  # First 30 seconds
        print(segment)
        
        # Sample frames from segment
        frames = segment.sample_uniform(10)  # 10 frames
        for f in frames:
            print(f"Frame at {f.timestamp}s")
        ```
    """
    
    def __init__(
        self,
        video_path: Union[str, Path],
        cache_dir: Optional[Path] = None,
        preload_metadata: bool = True,
    ):
        """Initialize a VideoEnv from a video file.
        
        Args:
            video_path: Path to the video file
            cache_dir: Directory for caching extracted frames
            preload_metadata: Whether to extract metadata immediately
        """
        self.path = Path(video_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")
        
        self.cache_dir = cache_dir or Path.home() / ".cache" / "rvaa" / "frames"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._metadata: Optional[VideoMetadata] = None
        self._decoder: Optional[Any] = None  # Lazy-loaded video decoder
        
        if preload_metadata:
            _ = self.metadata  # Trigger metadata extraction
    
    @property
    def metadata(self) -> VideoMetadata:
        """Get video metadata (lazy-loaded)."""
        if self._metadata is None:
            self._metadata = self._extract_metadata()
        return self._metadata
    
    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return self.metadata.duration_seconds
    
    @property
    def fps(self) -> float:
        """Frames per second."""
        return self.metadata.fps
    
    @property
    def total_frames(self) -> int:
        """Total number of frames."""
        return self.metadata.total_frames
    
    def _extract_metadata(self) -> VideoMetadata:
        """Extract metadata from the video file using PyAV."""
        import av
        
        container = av.open(str(self.path))
        video_stream = container.streams.video[0]
        
        # Calculate duration and frame count
        duration = float(container.duration) / av.time_base if container.duration else 0
        fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
        total_frames = video_stream.frames or int(duration * fps)
        
        metadata = VideoMetadata(
            path=self.path,
            duration_seconds=duration,
            fps=fps,
            total_frames=total_frames,
            width=video_stream.width,
            height=video_stream.height,
            codec=video_stream.codec.name,
            file_size_bytes=self.path.stat().st_size,
        )
        
        container.close()
        return metadata
    
    def __repr__(self) -> str:
        return f"VideoEnv({self.path.name}, {self.duration:.1f}s, {self.total_frames} frames)"
    
    def __str__(self) -> str:
        """String representation for printing in REPL."""
        return self.metadata.to_context_string()
    
    def __len__(self) -> int:
        """Return total number of frames."""
        return self.total_frames
    
    def __getitem__(self, key: Union[slice, float, int]) -> Union[VideoView, FrameData]:
        """Support slicing and indexing.
        
        Slicing by time (floats) returns a VideoView:
            segment = video[10.0:20.0]  # 10s to 20s
            
        Indexing by frame number (int) returns a FrameData:
            frame = video[100]  # Frame 100
            
        Note: The slice notation uses timestamps in seconds, not frame indices.
        """
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0.0
            stop = key.stop if key.stop is not None else self.duration
            
            # Clamp to valid range
            start = max(0.0, float(start))
            stop = min(self.duration, float(stop))
            
            if start >= stop:
                raise ValueError(f"Invalid time range: {start} to {stop}")
            
            return self._create_view(start, stop)
        
        elif isinstance(key, (int, float)):
            if isinstance(key, int):
                # Frame index
                return self.get_frame_by_index(key)
            else:
                # Timestamp
                return self.get_frame(key)
        
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
    
    def _create_view(self, start_time: float, end_time: float) -> VideoView:
        """Create a VideoView for the given time range."""
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        
        return VideoView(
            source=self,
            start_time=start_time,
            end_time=end_time,
            start_frame=start_frame,
            end_frame=end_frame,
        )
    
    def get_frame(self, timestamp: float) -> FrameData:
        """Get a single frame at the given timestamp.
        
        Args:
            timestamp: Time in seconds
            
        Returns:
            FrameData for the frame closest to the timestamp
        """
        frame_idx = int(timestamp * self.fps)
        return self.get_frame_by_index(frame_idx)
    
    def get_frame_by_index(self, frame_idx: int) -> FrameData:
        """Get a frame by its index.
        
        Args:
            frame_idx: Frame index (0-based)
            
        Returns:
            FrameData for the requested frame
        """
        import av
        
        if frame_idx < 0:
            frame_idx = self.total_frames + frame_idx
        
        if not 0 <= frame_idx < self.total_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self.total_frames})")
        
        # Calculate timestamp for seeking
        target_time = frame_idx / self.fps
        
        container = av.open(str(self.path))
        video_stream = container.streams.video[0]
        
        # Seek to the target time
        target_pts = int(target_time / video_stream.time_base)
        container.seek(target_pts, stream=video_stream)
        
        # Decode until we get the target frame
        frame_data = None
        for frame in container.decode(video_stream):
            current_time = float(frame.pts * video_stream.time_base)
            if current_time >= target_time - 0.1:  # Small tolerance
                # Convert to RGB numpy array
                img = frame.to_ndarray(format='rgb24')
                frame_data = FrameData(
                    frame_idx=frame_idx,
                    timestamp=current_time,
                    image=img,
                    is_keyframe=frame.key_frame,
                )
                break
        
        container.close()
        
        if frame_data is None:
            raise RuntimeError(f"Failed to decode frame {frame_idx}")
        
        return frame_data
    
    def get_frames(
        self,
        start_time: float,
        end_time: float,
        sample_fps: Optional[float] = None,
    ) -> list[FrameData]:
        """Get frames in a time range.
        
        Args:
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            sample_fps: Sample rate (None = native fps)
            
        Returns:
            List of FrameData objects
        """
        import av
        
        if sample_fps is None:
            sample_fps = self.fps
        
        # Calculate frame interval for sampling
        frame_interval = self.fps / sample_fps
        
        container = av.open(str(self.path))
        video_stream = container.streams.video[0]
        
        # Seek to start
        target_pts = int(start_time / video_stream.time_base)
        container.seek(target_pts, stream=video_stream)
        
        frames = []
        last_sampled_idx = -frame_interval  # Ensure first frame is sampled
        
        for frame in container.decode(video_stream):
            current_time = float(frame.pts * video_stream.time_base)
            
            if current_time < start_time:
                continue
            if current_time > end_time:
                break
            
            # Sample based on interval
            current_idx = int(current_time * self.fps)
            if current_idx - last_sampled_idx >= frame_interval:
                img = frame.to_ndarray(format='rgb24')
                frames.append(FrameData(
                    frame_idx=current_idx,
                    timestamp=current_time,
                    image=img,
                    is_keyframe=frame.key_frame,
                ))
                last_sampled_idx = current_idx
        
        container.close()
        return frames
    
    def sample_frames_uniform(
        self,
        start_time: float,
        end_time: float,
        n_frames: int,
    ) -> list[FrameData]:
        """Sample n frames uniformly from a time range.
        
        Args:
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            n_frames: Number of frames to sample
            
        Returns:
            List of FrameData objects
        """
        if n_frames <= 0:
            return []
        
        duration = end_time - start_time
        if duration <= 0:
            return []
        
        # Calculate sample timestamps
        if n_frames == 1:
            timestamps = [start_time + duration / 2]
        else:
            step = duration / (n_frames - 1)
            timestamps = [start_time + i * step for i in range(n_frames)]
        
        # Get frame at each timestamp
        frames = []
        for ts in timestamps:
            try:
                frame = self.get_frame(ts)
                frames.append(frame)
            except Exception as e:
                logger.warning(f"Failed to get frame at {ts}s: {e}")
        
        return frames
    
    def iter_segments(
        self,
        segment_duration: float = 10.0,
        overlap: float = 0.0,
    ) -> Iterator[VideoView]:
        """Iterate over video segments.
        
        This is useful for the agent's chunking strategies - it can
        iterate over fixed-length segments rather than loading everything.
        
        Args:
            segment_duration: Duration of each segment in seconds
            overlap: Overlap between segments in seconds
            
        Yields:
            VideoView for each segment
        """
        step = segment_duration - overlap
        current = 0.0
        
        while current < self.duration:
            end = min(current + segment_duration, self.duration)
            yield self._create_view(current, end)
            current += step
    
    def get_keyframes(
        self,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> list[FrameData]:
        """Extract keyframes (I-frames) from the video.
        
        Keyframes are useful for efficient video navigation and
        creating visual summaries.
        
        Args:
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds (None = end of video)
            
        Returns:
            List of FrameData objects for keyframes only
        """
        import av
        
        if end_time is None:
            end_time = self.duration
        
        container = av.open(str(self.path))
        video_stream = container.streams.video[0]
        
        # Only decode keyframes
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
            
            img = frame.to_ndarray(format='rgb24')
            keyframes.append(FrameData(
                frame_idx=int(current_time * self.fps),
                timestamp=current_time,
                image=img,
                is_keyframe=True,
            ))
        
        container.close()
        return keyframes
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "path": str(self.path),
            "duration_seconds": self.duration,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.metadata.width,
            "height": self.metadata.height,
            "codec": self.metadata.codec,
        }


# Type alias for REPL context
VideoContext = Union[VideoEnv, VideoView]


# =============================================================================
# Caption Generation Helper
# =============================================================================

def generate_placeholder_caption(frame: FrameData, video_name: str = "") -> str:
    """Generate a placeholder caption for a frame.
    
    In production, this would use a vision model like GPT-4V, BLIP2, etc.
    For demo purposes, we generate a placeholder based on timestamp.
    """
    t = frame.timestamp
    # Generate a descriptive placeholder
    if video_name:
        return f"[Frame at {t:.1f}s from {video_name}]"
    return f"[Frame at {t:.1f}s]"


# Add caption method to FrameData
def get_frame_caption(self, video_name: str = "") -> str:
    """Get a caption for this frame.
    
    Note: In a full implementation, this would call a vision model.
    """
    return generate_placeholder_caption(self, video_name)

FrameData.get_caption = get_frame_caption


# Add captions property to VideoView
def get_view_captions(self) -> list[tuple[float, str]]:
    """Get captions for this video segment.
    
    Returns list of (timestamp, caption) tuples.
    """
    # Sample a few frames from the segment
    n_samples = min(5, max(1, int(self.duration / 5)))  # 1 sample per 5 seconds
    frames = self.sample_uniform(n_samples)
    video_name = self.source.path.name
    return [(f.timestamp, generate_placeholder_caption(f, video_name)) for f in frames]

VideoView.get_captions = get_view_captions
