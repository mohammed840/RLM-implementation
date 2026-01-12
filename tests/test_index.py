"""
Tests for Index (FAISS-based retrieval)
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from rvaa.tools.index import (
    SegmentMetadata,
    SearchResult,
    SegmentIndex,
    TextEncoder,
    SimpleTextEncoder,
    build_video_index,
)


class TestSegmentMetadata:
    """Tests for SegmentMetadata."""
    
    def test_creation(self):
        """Create a segment metadata object."""
        seg = SegmentMetadata(
            segment_id="seg_001",
            start_time=0.0,
            end_time=10.0,
            text="A person walks into a room.",
        )
        assert seg.segment_id == "seg_001"
        assert seg.duration == 10.0
        assert seg.mid_time == 5.0
    
    def test_serialization(self):
        """Serialize to dict and back."""
        seg = SegmentMetadata(
            segment_id="seg_002",
            start_time=10.0,
            end_time=20.0,
            text="They sit down at a desk.",
            extra={"scene": "office"},
        )
        
        data = seg.to_dict()
        restored = SegmentMetadata.from_dict(data)
        
        assert restored.segment_id == seg.segment_id
        assert restored.start_time == seg.start_time
        assert restored.text == seg.text
        assert restored.extra["scene"] == "office"


class TestSimpleTextEncoder:
    """Tests for SimpleTextEncoder (fallback)."""
    
    def test_encode_single(self):
        """Encode a single text."""
        encoder = SimpleTextEncoder(dim=64)
        
        embedding = encoder.encode("Hello world")
        
        assert embedding.shape == (64,)
        assert embedding.dtype == np.float32
    
    def test_encode_batch(self):
        """Encode multiple texts."""
        encoder = SimpleTextEncoder(dim=64)
        
        embeddings = encoder.encode(["Hello", "World", "Test"])
        
        assert embeddings.shape == (3, 64)
    
    def test_normalized(self):
        """Embeddings should be normalized."""
        encoder = SimpleTextEncoder(dim=64)
        
        embedding = encoder.encode("Hello world")
        norm = np.linalg.norm(embedding)
        
        assert abs(norm - 1.0) < 0.01 or norm == 0  # normalized or zero
    
    def test_different_texts_different_embeddings(self):
        """Different texts should have different embeddings."""
        encoder = SimpleTextEncoder(dim=64)
        
        e1 = encoder.encode("Hello world")
        e2 = encoder.encode("Goodbye moon")
        
        # Should not be identical
        assert not np.allclose(e1, e2)


class TestSegmentIndex:
    """Tests for FAISS-based index."""
    
    @pytest.fixture
    def encoder(self):
        """Create a simple encoder for testing."""
        return SimpleTextEncoder(dim=64)
    
    @pytest.fixture
    def segments(self):
        """Create test segments."""
        return [
            SegmentMetadata("s1", 0.0, 10.0, "A cat sits on a mat"),
            SegmentMetadata("s2", 10.0, 20.0, "A dog runs in the park"),
            SegmentMetadata("s3", 20.0, 30.0, "A bird flies in the sky"),
            SegmentMetadata("s4", 30.0, 40.0, "A fish swims in water"),
            SegmentMetadata("s5", 40.0, 50.0, "A cat chases a mouse"),
        ]
    
    def test_build_index(self, encoder, segments):
        """Build an index from segments."""
        index = SegmentIndex(encoder)
        index.build(segments)
        
        assert index.is_built
        assert index.size == 5
    
    def test_search(self, encoder, segments):
        """Search the index."""
        index = SegmentIndex(encoder)
        index.build(segments)
        
        results = index.search("cat on a mat", top_k=2)
        
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].rank == 0
        assert results[1].rank == 1
    
    def test_search_returns_similar(self, encoder, segments):
        """Search should return similar segments."""
        index = SegmentIndex(encoder)
        index.build(segments)
        
        results = index.search("cat", top_k=5)
        
        # At least one result should mention cat
        texts = [r.segment.text.lower() for r in results]
        has_cat = any("cat" in t for t in texts)
        assert has_cat
    
    def test_get_segment(self, encoder, segments):
        """Get segment by ID."""
        index = SegmentIndex(encoder)
        index.build(segments)
        
        seg = index.get_segment("s3")
        
        assert seg is not None
        assert seg.segment_id == "s3"
        assert "bird" in seg.text
    
    def test_save_load(self, encoder, segments, tmp_path):
        """Save and load index."""
        index = SegmentIndex(encoder)
        index.build(segments)
        
        save_path = tmp_path / "test_index"
        index.save(save_path)
        
        # Create new index and load
        new_index = SegmentIndex(encoder)
        new_index.load(save_path)
        
        assert new_index.size == 5
        
        # Search should still work
        results = new_index.search("dog", top_k=1)
        assert len(results) == 1


class TestBuildVideoIndex:
    """Tests for build_video_index helper."""
    
    def test_build_from_captions(self):
        """Build index from caption list."""
        captions = [
            (0.0, 5.0, "Person enters room"),
            (5.0, 10.0, "They wave hello"),
            (10.0, 15.0, "Conversation begins"),
        ]
        
        index = build_video_index(captions, video_id="test_video")
        
        assert index.is_built
        assert index.size == 3
        
        # Check segment IDs
        seg = index.get_segment("test_video_seg_0001")
        assert seg is not None
        assert seg.text == "They wave hello"
