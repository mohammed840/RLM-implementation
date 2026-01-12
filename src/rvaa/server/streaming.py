"""
Streaming Utilities

This module provides SSE (Server-Sent Events) streaming for agent events.
Clients can subscribe to agent runs and receive real-time updates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Optional

from rvaa.protocols.events import AgentEvent, StreamMessage

logger = logging.getLogger(__name__)


@dataclass
class RunState:
    """State of an agent run.
    
    Attributes:
        run_id: Unique run identifier
        query: The query being answered
        video_path: Path to the video
        status: Current status ("pending", "running", "completed", "failed")
        started_at: When the run started
        completed_at: When the run finished
        events: List of events in this run
        result: Final result (if completed)
    """
    run_id: str
    query: str
    video_path: str
    status: str = "pending"
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    events: list[AgentEvent] = field(default_factory=list)
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    
    def add_event(self, event: AgentEvent) -> None:
        """Add an event to this run."""
        self.events.append(event)
    
    def complete(self, result: dict[str, Any]) -> None:
        """Mark run as completed."""
        self.status = "completed"
        self.completed_at = datetime.now()
        self.result = result
    
    def fail(self, error: str) -> None:
        """Mark run as failed."""
        self.status = "failed"
        self.completed_at = datetime.now()
        self.error = error
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "run_id": self.run_id,
            "query": self.query,
            "video_path": self.video_path,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "event_count": len(self.events),
            "result": self.result,
            "error": self.error,
        }


class EventStream:
    """Async event stream for SSE broadcasting.
    
    Allows multiple clients to subscribe to agent events.
    Events are delivered in order with sequence numbers.
    """
    
    def __init__(self, run_id: str):
        """Initialize event stream for a run.
        
        Args:
            run_id: The run ID this stream is for
        """
        self.run_id = run_id
        self._sequence = 0
        self._subscribers: list[asyncio.Queue[StreamMessage]] = []
        self._events: list[StreamMessage] = []
        self._closed = False
    
    async def publish(self, event: AgentEvent) -> None:
        """Publish an event to all subscribers.
        
        Args:
            event: The event to publish
        """
        if self._closed:
            return
        
        self._sequence += 1
        message = StreamMessage(
            event=event,
            run_id=self.run_id,
            sequence=self._sequence,
        )
        
        # Store for late subscribers
        self._events.append(message)
        
        # Broadcast to all subscribers
        for queue in self._subscribers:
            try:
                await queue.put(message)
            except Exception as e:
                logger.warning(f"Failed to publish to subscriber: {e}")
    
    async def subscribe(
        self,
        from_sequence: int = 0,
    ) -> AsyncGenerator[StreamMessage, None]:
        """Subscribe to the event stream.
        
        Args:
            from_sequence: Start from this sequence number (for replay)
            
        Yields:
            StreamMessage events
        """
        queue: asyncio.Queue[StreamMessage] = asyncio.Queue()
        self._subscribers.append(queue)
        
        try:
            # Replay any missed events
            for msg in self._events:
                if msg.sequence > from_sequence:
                    yield msg
            
            # Stream new events
            while not self._closed:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield message
                except asyncio.TimeoutError:
                    # Send keepalive
                    continue
                    
        finally:
            self._subscribers.remove(queue)
    
    def close(self) -> None:
        """Close the stream."""
        self._closed = True


class StreamManager:
    """Manages multiple event streams and run states.
    
    Thread-safe management of agent runs and their event streams.
    """
    
    def __init__(self, max_runs: int = 100):
        """Initialize stream manager.
        
        Args:
            max_runs: Maximum number of runs to keep in memory
        """
        self.max_runs = max_runs
        self._runs: dict[str, RunState] = {}
        self._streams: dict[str, EventStream] = {}
        self._lock = asyncio.Lock()
    
    async def create_run(
        self,
        query: str,
        video_path: str,
    ) -> str:
        """Create a new agent run.
        
        Args:
            query: The query to answer
            video_path: Path to the video
            
        Returns:
            The run ID
        """
        async with self._lock:
            run_id = str(uuid.uuid4())
            
            # Cleanup old runs if needed
            if len(self._runs) >= self.max_runs:
                oldest = min(self._runs.values(), key=lambda r: r.started_at)
                await self.cleanup_run(oldest.run_id)
            
            # Create run state and stream
            self._runs[run_id] = RunState(
                run_id=run_id,
                query=query,
                video_path=video_path,
            )
            self._streams[run_id] = EventStream(run_id)
            
            return run_id
    
    async def get_run(self, run_id: str) -> Optional[RunState]:
        """Get run state by ID."""
        return self._runs.get(run_id)
    
    async def get_stream(self, run_id: str) -> Optional[EventStream]:
        """Get event stream by run ID."""
        return self._streams.get(run_id)
    
    async def publish_event(self, run_id: str, event: AgentEvent) -> None:
        """Publish an event to a run's stream.
        
        Args:
            run_id: The run ID
            event: The event to publish
        """
        run = self._runs.get(run_id)
        stream = self._streams.get(run_id)
        
        if run:
            run.add_event(event)
        if stream:
            await stream.publish(event)
    
    async def complete_run(
        self,
        run_id: str,
        result: dict[str, Any],
    ) -> None:
        """Mark a run as completed.
        
        Args:
            run_id: The run ID
            result: The final result
        """
        run = self._runs.get(run_id)
        stream = self._streams.get(run_id)
        
        if run:
            run.complete(result)
        if stream:
            stream.close()
    
    async def fail_run(self, run_id: str, error: str) -> None:
        """Mark a run as failed.
        
        Args:
            run_id: The run ID
            error: Error message
        """
        run = self._runs.get(run_id)
        stream = self._streams.get(run_id)
        
        if run:
            run.fail(error)
        if stream:
            stream.close()
    
    async def cleanup_run(self, run_id: str) -> None:
        """Remove a run from memory.
        
        Args:
            run_id: The run ID to cleanup
        """
        async with self._lock:
            if run_id in self._runs:
                del self._runs[run_id]
            if run_id in self._streams:
                self._streams[run_id].close()
                del self._streams[run_id]
    
    def list_runs(self) -> list[dict[str, Any]]:
        """List all runs."""
        return [run.to_dict() for run in self._runs.values()]


# Global stream manager instance
_stream_manager: Optional[StreamManager] = None


def get_stream_manager() -> StreamManager:
    """Get the global stream manager instance."""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = StreamManager()
    return _stream_manager
