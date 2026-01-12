"""
Perception Layer

This module provides the vision-to-text layer for converting video frames
and clips into structured text evidence that the RLM agent can reason over.

The perception layer is pluggable - different backends can be swapped
depending on requirements:
- BLIP2Backend: Open-source, runs locally
- LLaVABackend: More capable, also local
- GPTVisionBackend: Uses GPT-4-Vision API

The key insight from the RLM paper is that the root agent should NOT 
see raw frames. Instead, frames are converted to text descriptions
that can be processed by the language model.
"""

from __future__ import annotations

import base64
import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CaptionResult:
    """Result of captioning a frame or clip.
    
    Attributes:
        timestamp: Timestamp in seconds (mid-point for clips)
        caption: Generated text description
        confidence: Optional confidence score
        objects: Optional list of detected objects
        actions: Optional list of detected actions
        raw_output: Optional raw model output
    """
    timestamp: float
    caption: str
    confidence: Optional[float] = None
    objects: Optional[list[str]] = None
    actions: Optional[list[str]] = None
    raw_output: Optional[Any] = None
    
    def __repr__(self) -> str:
        return f"Caption(t={self.timestamp:.2f}s, '{self.caption[:50]}...')"


class PerceptionBackend(ABC):
    """Abstract base class for perception backends.
    
    All backends must implement caption_frame(). Other methods are optional
    and can return NotImplemented if not supported.
    """
    
    @abstractmethod
    def caption_frame(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
        prompt: Optional[str] = None,
    ) -> CaptionResult:
        """Generate a text caption for a single frame.
        
        Args:
            frame: Frame as numpy array (H, W, C) in RGB format
            timestamp: Timestamp of the frame
            prompt: Optional prompt to guide captioning
            
        Returns:
            CaptionResult with the generated caption
        """
        pass
    
    def caption_clip(
        self,
        frames: Sequence[np.ndarray],
        timestamps: Sequence[float],
        prompt: Optional[str] = None,
    ) -> CaptionResult:
        """Generate a caption for a sequence of frames (video clip).
        
        Default implementation captions each frame and combines.
        Backends with native clip support should override this.
        
        Args:
            frames: List of frames
            timestamps: Corresponding timestamps
            prompt: Optional prompt
            
        Returns:
            CaptionResult for the clip
        """
        if len(frames) == 0:
            return CaptionResult(timestamp=0.0, caption="Empty clip")
        
        # Default: caption each frame and combine
        captions = []
        for frame, ts in zip(frames, timestamps):
            result = self.caption_frame(frame, ts, prompt)
            captions.append(f"[{ts:.1f}s] {result.caption}")
        
        mid_ts = timestamps[len(timestamps) // 2]
        combined = " | ".join(captions)
        
        return CaptionResult(timestamp=mid_ts, caption=combined)
    
    def extract_objects(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Extract objects with bounding boxes from a frame.
        
        Args:
            frame: Frame as numpy array
            timestamp: Timestamp of the frame
            
        Returns:
            List of dicts with 'label', 'bbox', 'confidence' keys
        """
        return []  # Not implemented by default
    
    def answer_question(
        self,
        frame: np.ndarray,
        question: str,
        timestamp: float = 0.0,
    ) -> str:
        """Visual question answering on a frame.
        
        Args:
            frame: Frame as numpy array
            question: Question about the frame
            timestamp: Timestamp of the frame
            
        Returns:
            Answer string
        """
        # Default: use captioning with question as prompt
        result = self.caption_frame(frame, timestamp, prompt=question)
        return result.caption


class BLIP2Backend(PerceptionBackend):
    """BLIP-2 based perception backend.
    
    BLIP-2 is an open-source vision-language model that can run locally.
    It's a good default for captioning when you don't want to use
    cloud APIs.
    
    Requires: transformers, torch
    """
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: str = "auto",
        max_length: int = 100,
    ):
        """Initialize BLIP-2 backend.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ("auto", "cpu", "cuda", "mps")
            max_length: Maximum caption length
        """
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
        self._processor = None
        
        # Determine device
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
    
    def _load_model(self) -> None:
        """Lazy-load the model."""
        if self._model is not None:
            return
        
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import torch
        
        logger.info(f"Loading BLIP-2 model: {self.model_name}")
        
        self._processor = Blip2Processor.from_pretrained(self.model_name)
        
        # Load with appropriate dtype
        if self.device == "cuda":
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
            ).to(self.device)
        else:
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
            ).to(self.device)
        
        logger.info(f"BLIP-2 model loaded on {self.device}")
    
    def caption_frame(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
        prompt: Optional[str] = None,
    ) -> CaptionResult:
        """Generate caption using BLIP-2."""
        from PIL import Image
        import torch
        
        self._load_model()
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(frame)
        
        # Prepare inputs
        if prompt:
            inputs = self._processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            ).to(self.device)
        else:
            inputs = self._processor(
                images=image,
                return_tensors="pt",
            ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=self.max_length,
            )
        
        # Decode
        caption = self._processor.decode(outputs[0], skip_special_tokens=True)
        
        return CaptionResult(
            timestamp=timestamp,
            caption=caption.strip(),
        )
    
    def answer_question(
        self,
        frame: np.ndarray,
        question: str,
        timestamp: float = 0.0,
    ) -> str:
        """Answer a question about the frame using BLIP-2."""
        prompt = f"Question: {question} Answer:"
        result = self.caption_frame(frame, timestamp, prompt=prompt)
        return result.caption


class SimpleCaptionBackend(PerceptionBackend):
    """Simple caption backend using basic image analysis.
    
    This is a fallback that doesn't require heavy ML models.
    It provides basic descriptions based on image statistics.
    Useful for testing or when you don't have GPU access.
    """
    
    def caption_frame(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
        prompt: Optional[str] = None,
    ) -> CaptionResult:
        """Generate a basic caption based on image statistics."""
        h, w, c = frame.shape
        
        # Basic color analysis
        mean_color = frame.mean(axis=(0, 1))
        brightness = mean_color.mean()
        
        # Determine dominant color
        r, g, b = mean_color
        if r > max(g, b) + 20:
            dominant = "red-toned"
        elif g > max(r, b) + 20:
            dominant = "green-toned"
        elif b > max(r, g) + 20:
            dominant = "blue-toned"
        else:
            dominant = "neutral"
        
        # Brightness description
        if brightness > 180:
            bright_desc = "bright"
        elif brightness < 80:
            bright_desc = "dark"
        else:
            bright_desc = "moderately lit"
        
        caption = f"A {bright_desc}, {dominant} frame at {timestamp:.1f}s ({w}x{h})"
        
        return CaptionResult(
            timestamp=timestamp,
            caption=caption,
        )


class GPTVisionBackend(PerceptionBackend):
    """GPT-4 Vision based perception backend.
    
    Uses OpenAI's GPT-4 Vision API for high-quality captioning.
    Requires OPENAI_API_KEY environment variable.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_tokens: int = 300,
    ):
        """Initialize GPT Vision backend.
        
        Args:
            model: OpenAI model to use
            api_key: API key (reads from env if not provided)
            max_tokens: Maximum tokens in response
        """
        import os
        
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.max_tokens = max_tokens
        
        if not self.api_key:
            logger.warning("No OpenAI API key found for GPT Vision backend")
    
    def _encode_image(self, frame: np.ndarray) -> str:
        """Encode frame as base64 JPEG."""
        from PIL import Image
        
        image = Image.fromarray(frame)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def caption_frame(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
        prompt: Optional[str] = None,
    ) -> CaptionResult:
        """Generate caption using GPT-4 Vision."""
        import httpx
        
        image_base64 = self._encode_image(frame)
        
        if prompt is None:
            prompt = "Describe this video frame in detail. Focus on the main subjects, actions, and setting."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ]
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()
        
        caption = data["choices"][0]["message"]["content"]
        
        return CaptionResult(
            timestamp=timestamp,
            caption=caption.strip(),
            raw_output=data,
        )
    
    def answer_question(
        self,
        frame: np.ndarray,
        question: str,
        timestamp: float = 0.0,
    ) -> str:
        """Answer a visual question using GPT-4 Vision."""
        result = self.caption_frame(frame, timestamp, prompt=question)
        return result.caption


def batch_caption(
    frames: list[tuple[float, np.ndarray]],
    backend: PerceptionBackend,
    batch_size: int = 8,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[CaptionResult]:
    """Caption multiple frames in batches.
    
    Args:
        frames: List of (timestamp, frame) tuples
        backend: Perception backend to use
        batch_size: Number of frames to caption in parallel (if supported)
        progress_callback: Optional callback(current, total)
        
    Returns:
        List of CaptionResult objects
    """
    results = []
    total = len(frames)
    
    for i, (timestamp, frame) in enumerate(frames):
        try:
            result = backend.caption_frame(frame, timestamp)
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to caption frame at {timestamp}s: {e}")
            results.append(CaptionResult(
                timestamp=timestamp,
                caption=f"[Caption failed: {e}]",
            ))
        
        if progress_callback:
            progress_callback(i + 1, total)
    
    return results


def get_perception_backend(
    backend_type: str = "simple",
    **kwargs: Any,
) -> PerceptionBackend:
    """Factory function to create perception backends.
    
    Args:
        backend_type: One of "simple", "blip2", "gpt-vision"
        **kwargs: Additional arguments for the backend
        
    Returns:
        PerceptionBackend instance
    """
    if backend_type == "simple":
        return SimpleCaptionBackend()
    elif backend_type == "blip2":
        return BLIP2Backend(**kwargs)
    elif backend_type == "gpt-vision":
        return GPTVisionBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
