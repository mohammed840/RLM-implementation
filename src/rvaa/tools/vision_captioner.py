"""
Vision-based frame captioning using OpenRouter

This module provides real frame captioning using vision models via OpenRouter.
"""

import asyncio
import base64
import os
from io import BytesIO
from typing import Optional
import httpx
from PIL import Image
import numpy as np

from dotenv import load_dotenv
load_dotenv()


class VisionCaptioner:
    """Generate captions for video frames using vision models."""
    
    def __init__(self, model: str = "meta-llama/llama-3.2-11b-vision-instruct"):
        """Initialize the vision captioner.
        
        Args:
            model: The vision model to use via OpenRouter
        """
        self.model = model
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = "https://openrouter.ai/api/v1"
        
    def _encode_frame(self, frame_image: np.ndarray) -> str:
        """Convert numpy frame to base64 encoded image."""
        # Convert to PIL Image
        img = Image.fromarray(frame_image.astype('uint8'))
        
        # Resize to reduce API costs (max 512px on longest side)
        max_size = 512
        ratio = min(max_size / img.width, max_size / img.height)
        if ratio < 1:
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    async def caption_frame(self, frame_image: np.ndarray, context: str = "") -> str:
        """Generate a caption for a single frame.
        
        Args:
            frame_image: The frame as a numpy array (H, W, C) in RGB
            context: Optional context about what we're looking for
            
        Returns:
            Caption describing the frame content
        """
        if not self.api_key:
            return "[No API key - cannot generate caption]"
        
        base64_image = self._encode_frame(frame_image)
        
        prompt = "Describe what you see in this video frame in 1-2 sentences. Focus on: people, actions, text on screen, and setting."
        if context:
            prompt = f"{prompt} Context: {context}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 150,
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    return f"[Caption error: {response.status_code}]"
                    
        except Exception as e:
            return f"[Caption error: {e}]"
    
    def caption_frame_sync(self, frame_image: np.ndarray, context: str = "") -> str:
        """Synchronous wrapper for caption_frame."""
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.caption_frame(frame_image, context))


# Global captioner instance
_captioner: Optional[VisionCaptioner] = None


def get_captioner() -> VisionCaptioner:
    """Get or create the global vision captioner."""
    global _captioner
    if _captioner is None:
        _captioner = VisionCaptioner()
    return _captioner


def caption_video_segment(segment, n_samples: int = 3) -> list[tuple[float, str]]:
    """Generate captions for frames in a video segment.
    
    Args:
        segment: VideoView segment to caption
        n_samples: Number of frames to sample from segment
        
    Returns:
        List of (timestamp, caption) tuples
    """
    captioner = get_captioner()
    captions = []
    
    # Sample frames uniformly
    frames = segment.sample_uniform(n_samples)
    
    for frame in frames:
        caption = captioner.caption_frame_sync(frame.image)
        captions.append((frame.timestamp, caption))
    
    return captions
