"""
Embedding Index for Video Segment Retrieval

This module provides FAISS-based embedding search for efficient retrieval
over long videos. The two-stage search-and-verify pattern:

1. Stage 1 (Coarse): Use embeddings to find candidate segments quickly
2. Stage 2 (Fine): Use subcalls to verify and extract evidence from candidates

This matches the RLM paper's approach of using cheap, scalable retrieval
before expensive LM calls.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SegmentMetadata:
    """Metadata for an indexed video segment.
    
    Attributes:
        segment_id: Unique identifier
        start_time: Start timestamp in seconds
        end_time: End timestamp in seconds
        text: Text content (caption, transcript, etc.)
        embedding: Optional cached embedding vector
        extra: Additional metadata
    """
    segment_id: str
    start_time: float
    end_time: float
    text: str
    embedding: Optional[np.ndarray] = None
    extra: dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def mid_time(self) -> float:
        """Midpoint timestamp."""
        return (self.start_time + self.end_time) / 2
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (without embedding)."""
        return {
            "segment_id": self.segment_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "extra": self.extra,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SegmentMetadata":
        """Create from dictionary."""
        return cls(
            segment_id=data["segment_id"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            text=data["text"],
            extra=data.get("extra", {}),
        )
    
    def __repr__(self) -> str:
        return (
            f"Segment(id={self.segment_id}, "
            f"t={self.start_time:.1f}-{self.end_time:.1f}s, "
            f"text='{self.text[:30]}...')"
        )


@dataclass
class SearchResult:
    """Result from a segment search.
    
    Attributes:
        segment: The matched segment
        score: Similarity score (higher = more similar)
        rank: Rank in results (0-indexed)
    """
    segment: SegmentMetadata
    score: float
    rank: int
    
    def __repr__(self) -> str:
        return f"SearchResult(rank={self.rank}, score={self.score:.3f}, {self.segment})"


class TextEncoder:
    """Text encoder for creating embeddings.
    
    Default implementation uses sentence-transformers.
    Can be swapped for CLIP text encoder or other models.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "auto",
    ):
        """Initialize text encoder.
        
        Args:
            model_name: sentence-transformers model name
            device: Device to run on
        """
        self.model_name = model_name
        self._model = None
        
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
        
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Loading text encoder: {self.model_name}")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(f"Text encoder loaded on {self.device}")
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()
    
    def encode(
        self,
        texts: Union[str, list[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode text(s) to embeddings.
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            Embedding array of shape (n_texts, dim) or (dim,) for single text
        """
        self._load_model()
        
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        
        if single:
            return embeddings[0]
        return embeddings


class SimpleTextEncoder:
    """Simple TF-IDF based text encoder.
    
    Fallback that doesn't require ML models.
    Useful for testing or basic setups.
    """
    
    def __init__(self, dim: int = 128):
        """Initialize with target dimensionality."""
        self.dim = dim
        self._vectorizer = None
    
    @property
    def embedding_dim(self) -> int:
        return self.dim
    
    def encode(
        self,
        texts: Union[str, list[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode using simple hashing."""
        import hashlib
        
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # Simple hash-based embedding
            words = text.lower().split()
            vec = np.zeros(self.dim)
            for word in words:
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                idx = h % self.dim
                vec[idx] += 1.0
            
            if normalize and np.linalg.norm(vec) > 0:
                vec = vec / np.linalg.norm(vec)
            
            embeddings.append(vec)
        
        result = np.array(embeddings, dtype=np.float32)
        
        if single:
            return result[0]
        return result


class SegmentIndex:
    """FAISS-based index for video segment retrieval.
    
    Supports:
    - Building index from segments
    - Approximate nearest neighbor search
    - Save/load for persistence
    - Incremental updates
    
    Example:
        >>> encoder = TextEncoder()
        >>> index = SegmentIndex(encoder)
        >>> segments = [SegmentMetadata(...), ...]
        >>> index.build(segments)
        >>> results = index.search("what happens next?", top_k=5)
    """
    
    def __init__(
        self,
        encoder: Union[TextEncoder, SimpleTextEncoder],
        use_gpu: bool = False,
    ):
        """Initialize segment index.
        
        Args:
            encoder: Text encoder for embeddings
            use_gpu: Whether to use GPU for FAISS (requires faiss-gpu)
        """
        self.encoder = encoder
        self.use_gpu = use_gpu
        
        self._index = None
        self._segments: list[SegmentMetadata] = []
        self._id_to_idx: dict[str, int] = {}
    
    @property
    def size(self) -> int:
        """Number of indexed segments."""
        return len(self._segments)
    
    @property
    def is_built(self) -> bool:
        """Whether the index has been built."""
        return self._index is not None and self.size > 0
    
    def build(
        self,
        segments: Sequence[SegmentMetadata],
        batch_size: int = 32,
    ) -> None:
        """Build the index from segments.
        
        Args:
            segments: List of segments to index
            batch_size: Batch size for encoding
        """
        import faiss
        
        if len(segments) == 0:
            logger.warning("No segments to index")
            return
        
        logger.info(f"Building index from {len(segments)} segments")
        
        # Extract texts and encode
        texts = [s.text for s in segments]
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.encoder.encode(batch)
            embeddings.append(batch_embeddings)
        
        all_embeddings = np.vstack(embeddings).astype(np.float32)
        
        # Create FAISS index
        dim = all_embeddings.shape[1]
        
        if len(segments) < 1000:
            # Small index: use flat search
            self._index = faiss.IndexFlatIP(dim)
        else:
            # Larger index: use IVF for approximate search
            n_clusters = min(int(np.sqrt(len(segments))), 256)
            quantizer = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
            self._index.train(all_embeddings)
        
        # Add embeddings
        self._index.add(all_embeddings)
        
        # Store segments and mapping
        self._segments = list(segments)
        self._id_to_idx = {s.segment_id: i for i, s in enumerate(segments)}
        
        logger.info(f"Index built with {self.size} segments")
    
    def add(self, segment: SegmentMetadata) -> None:
        """Add a single segment to the index.
        
        Args:
            segment: Segment to add
        """
        import faiss
        
        if self._index is None:
            # Initialize index
            dim = self.encoder.embedding_dim
            self._index = faiss.IndexFlatIP(dim)
        
        # Encode and add
        embedding = self.encoder.encode(segment.text)
        embedding = embedding.reshape(1, -1).astype(np.float32)
        self._index.add(embedding)
        
        # Update mapping
        idx = len(self._segments)
        self._segments.append(segment)
        self._id_to_idx[segment.segment_id] = idx
    
    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search for similar segments.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of SearchResult, sorted by score (highest first)
        """
        if not self.is_built:
            logger.warning("Index not built, returning empty results")
            return []
        
        # Encode query
        query_embedding = self.encoder.encode(query)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        k = min(top_k, self.size)
        scores, indices = self._index.search(query_embedding, k)
        
        # Build results
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:  # Invalid index
                continue
            segment = self._segments[idx]
            results.append(SearchResult(
                segment=segment,
                score=float(score),
                rank=rank,
            ))
        
        return results
    
    def search_by_embedding(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search using a pre-computed embedding.
        
        Args:
            embedding: Query embedding (1D array)
            top_k: Number of results to return
            
        Returns:
            List of SearchResult
        """
        if not self.is_built:
            return []
        
        query = embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, self.size)
        scores, indices = self._index.search(query, k)
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            results.append(SearchResult(
                segment=self._segments[idx],
                score=float(score),
                rank=rank,
            ))
        
        return results
    
    def get_segment(self, segment_id: str) -> Optional[SegmentMetadata]:
        """Get a segment by ID."""
        idx = self._id_to_idx.get(segment_id)
        if idx is not None:
            return self._segments[idx]
        return None
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the index to disk.
        
        Args:
            path: Directory to save to
        """
        import faiss
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self._index is not None:
            faiss.write_index(self._index, str(path / "index.faiss"))
        
        # Save segments
        segments_data = [s.to_dict() for s in self._segments]
        with open(path / "segments.json", "w") as f:
            json.dump(segments_data, f)
        
        # Save id mapping
        with open(path / "mapping.pkl", "wb") as f:
            pickle.dump(self._id_to_idx, f)
        
        logger.info(f"Index saved to {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """Load the index from disk.
        
        Args:
            path: Directory to load from
        """
        import faiss
        
        path = Path(path)
        
        # Load FAISS index
        index_path = path / "index.faiss"
        if index_path.exists():
            self._index = faiss.read_index(str(index_path))
        
        # Load segments
        with open(path / "segments.json", "r") as f:
            segments_data = json.load(f)
        self._segments = [SegmentMetadata.from_dict(d) for d in segments_data]
        
        # Load id mapping
        with open(path / "mapping.pkl", "rb") as f:
            self._id_to_idx = pickle.load(f)
        
        logger.info(f"Index loaded from {path} ({self.size} segments)")


def build_video_index(
    captions: list[tuple[float, float, str]],
    encoder: Optional[Union[TextEncoder, SimpleTextEncoder]] = None,
    video_id: str = "video",
) -> SegmentIndex:
    """Build an index from video captions.
    
    Args:
        captions: List of (start_time, end_time, caption_text) tuples
        encoder: Optional text encoder (uses SimpleTextEncoder if None)
        video_id: Video identifier for segment IDs
        
    Returns:
        Built SegmentIndex
    """
    if encoder is None:
        encoder = SimpleTextEncoder()
    
    # Create segments
    segments = []
    for i, (start, end, text) in enumerate(captions):
        segment = SegmentMetadata(
            segment_id=f"{video_id}_seg_{i:04d}",
            start_time=start,
            end_time=end,
            text=text,
        )
        segments.append(segment)
    
    # Build index
    index = SegmentIndex(encoder)
    index.build(segments)
    
    return index
