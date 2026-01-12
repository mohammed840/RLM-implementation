"""
Event Protocols for Agent Streaming

This module defines typed event schemas for streaming agent state to external
interfaces (UI, logging, metrics). Events are designed to be JSON-serializable
for SSE/WebSocket streaming.

Event Types:
- CODE_EXECUTION: Agent executed code in REPL
- LLM_QUERY: Agent made a sub-LM call
- RETRIEVAL: Agent searched the embedding index
- EVIDENCE_UPDATE: Agent updated the evidence table
- FINAL_ANSWER: Agent produced final answer
- THINKING: Agent is reasoning
- ERROR: An error occurred
- METADATA: Session/run metadata
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union


class EventType(str, Enum):
    """Type of agent event."""
    
    # Core agent events
    CODE_EXECUTION = "code_execution"
    LLM_QUERY = "llm_query"
    RETRIEVAL = "retrieval"
    EVIDENCE_UPDATE = "evidence_update"
    FINAL_ANSWER = "final_answer"
    
    # Status events
    THINKING = "thinking"
    ERROR = "error"
    METADATA = "metadata"
    
    # Progress events
    STEP_START = "step_start"
    STEP_END = "step_end"
    RUN_START = "run_start"
    RUN_END = "run_end"


@dataclass
class AgentEvent:
    """Base class for all agent events.
    
    Attributes:
        event_id: Unique identifier
        event_type: Type of event
        timestamp: When event occurred
        step_idx: Step index in trajectory
        data: Event-specific data
    """
    event_type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    step_idx: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "step_idx": self.step_idx,
            "data": self.data,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentEvent":
        """Deserialize from dictionary."""
        return cls(
            event_id=data.get("event_id", ""),
            event_type=EventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            step_idx=data.get("step_idx", 0),
            data=data.get("data", {}),
        )


@dataclass
class CodeExecutionEvent(AgentEvent):
    """Event for REPL code execution.
    
    Includes the code, output, and execution result.
    """
    
    def __init__(
        self,
        code: str,
        output: str,
        success: bool = True,
        error: Optional[str] = None,
        execution_time_ms: float = 0.0,
        step_idx: int = 0,
    ):
        super().__init__(
            event_type=EventType.CODE_EXECUTION,
            step_idx=step_idx,
            data={
                "code": code,
                "output": output,
                "success": success,
                "error": error,
                "execution_time_ms": execution_time_ms,
            },
        )


@dataclass
class LLMQueryEvent(AgentEvent):
    """Event for sub-LM queries (llm_query calls).
    
    Includes query prompt, response, and cost.
    """
    
    def __init__(
        self,
        prompt_preview: str,
        response_preview: str,
        full_prompt_chars: int = 0,
        full_response_chars: int = 0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        model: str = "",
        step_idx: int = 0,
    ):
        super().__init__(
            event_type=EventType.LLM_QUERY,
            step_idx=step_idx,
            data={
                "prompt_preview": prompt_preview,
                "response_preview": response_preview,
                "full_prompt_chars": full_prompt_chars,
                "full_response_chars": full_response_chars,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost_usd,
                "latency_ms": latency_ms,
                "model": model,
            },
        )


@dataclass
class RetrievalEvent(AgentEvent):
    """Event for embedding index search.
    
    Includes query and top-k results with timestamps.
    """
    
    def __init__(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int = 10,
        step_idx: int = 0,
    ):
        """
        Args:
            query: Search query text
            results: List of result dicts with keys:
                - segment_id: str
                - start_time: float
                - end_time: float
                - score: float
                - text_preview: str
            top_k: Number of results requested
        """
        super().__init__(
            event_type=EventType.RETRIEVAL,
            step_idx=step_idx,
            data={
                "query": query,
                "results": results,
                "top_k": top_k,
                "num_results": len(results),
            },
        )


@dataclass
class EvidenceEvent(AgentEvent):
    """Event for evidence table updates.
    
    Tracks accumulated evidence from video segments.
    """
    
    def __init__(
        self,
        action: str,  # "add", "update", "remove"
        evidence: dict[str, Any],
        total_evidence_count: int = 0,
        step_idx: int = 0,
    ):
        """
        Args:
            action: What happened ("add", "update", "remove")
            evidence: The evidence dict with keys:
                - segment_id: str
                - start_time: float
                - end_time: float
                - claim: str
                - confidence: float
            total_evidence_count: Current total evidence items
        """
        super().__init__(
            event_type=EventType.EVIDENCE_UPDATE,
            step_idx=step_idx,
            data={
                "action": action,
                "evidence": evidence,
                "total_evidence_count": total_evidence_count,
            },
        )


@dataclass
class FinalAnswerEvent(AgentEvent):
    """Event for final answer.
    
    Includes answer text and supporting citations.
    """
    
    def __init__(
        self,
        answer: str,
        citations: list[dict[str, Any]],
        success: bool = True,
        total_llm_queries: int = 0,
        total_cost_usd: float = 0.0,
        total_duration_ms: float = 0.0,
        step_idx: int = 0,
    ):
        """
        Args:
            answer: The final answer text
            citations: List of citation dicts with keys:
                - segment_id: str
                - start_time: float
                - end_time: float
                - text: str (supporting evidence)
            success: Whether agent completed successfully
            total_llm_queries: Total number of llm_query calls
            total_cost_usd: Total cost of the run
            total_duration_ms: Total runtime
        """
        super().__init__(
            event_type=EventType.FINAL_ANSWER,
            step_idx=step_idx,
            data={
                "answer": answer,
                "citations": citations,
                "success": success,
                "total_llm_queries": total_llm_queries,
                "total_cost_usd": total_cost_usd,
                "total_duration_ms": total_duration_ms,
            },
        )


@dataclass
class StreamMessage:
    """Wrapper for SSE/WebSocket streaming.
    
    Attributes:
        event: The agent event
        run_id: Unique run identifier
        sequence: Sequence number in stream
    """
    event: AgentEvent
    run_id: str
    sequence: int
    
    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        data = {
            "run_id": self.run_id,
            "sequence": self.sequence,
            **self.event.to_dict(),
        }
        return f"data: {json.dumps(data)}\n\n"
    
    def to_json(self) -> str:
        """Format as JSON (for WebSocket)."""
        return json.dumps({
            "run_id": self.run_id,
            "sequence": self.sequence,
            **self.event.to_dict(),
        })


# =============================================================================
# Factory Functions
# =============================================================================

def create_run_start_event(
    query: str,
    video_path: str,
    video_duration: float,
    config: dict[str, Any],
) -> AgentEvent:
    """Create a run start event."""
    return AgentEvent(
        event_type=EventType.RUN_START,
        data={
            "query": query,
            "video_path": video_path,
            "video_duration": video_duration,
            "config": config,
        },
    )


def create_run_end_event(
    success: bool,
    total_steps: int,
    total_llm_queries: int,
    total_cost_usd: float,
    total_duration_ms: float,
    error: Optional[str] = None,
) -> AgentEvent:
    """Create a run end event."""
    return AgentEvent(
        event_type=EventType.RUN_END,
        data={
            "success": success,
            "total_steps": total_steps,
            "total_llm_queries": total_llm_queries,
            "total_cost_usd": total_cost_usd,
            "total_duration_ms": total_duration_ms,
            "error": error,
        },
    )


def create_thinking_event(
    thought: str,
    step_idx: int = 0,
) -> AgentEvent:
    """Create a thinking event."""
    return AgentEvent(
        event_type=EventType.THINKING,
        step_idx=step_idx,
        data={
            "thought": thought,
        },
    )


def create_error_event(
    error: str,
    error_type: str = "unknown",
    step_idx: int = 0,
) -> AgentEvent:
    """Create an error event."""
    return AgentEvent(
        event_type=EventType.ERROR,
        step_idx=step_idx,
        data={
            "error": error,
            "error_type": error_type,
        },
    )
