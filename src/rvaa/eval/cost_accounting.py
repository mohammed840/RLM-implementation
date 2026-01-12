"""
Cost and Runtime Accounting

This module provides utilities for tracking costs, tokens, and runtime
across agent runs. Essential for academic experiments and ablations.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RunMetrics:
    """Metrics for a single agent run.
    
    Attributes:
        run_id: Unique run identifier
        query: The query
        video_path: Path to video
        backend: LLM backend used
        started_at: Start time
        ended_at: End time
        
        # Token counts
        root_input_tokens: Input tokens to root LM
        root_output_tokens: Output tokens from root LM
        sub_input_tokens: Input tokens to sub LM
        sub_output_tokens: Output tokens from sub LM
        
        # Call counts
        root_calls: Number of root LM calls
        sub_calls: Number of sub LM (llm_query) calls
        code_executions: Number of code executions
        
        # Cost
        root_cost_usd: Cost of root LM calls
        sub_cost_usd: Cost of sub LM calls
        total_cost_usd: Total cost
        
        # Time
        root_latency_ms: Total root LM latency
        sub_latency_ms: Total sub LM latency
        code_execution_ms: Total code execution time
        total_duration_ms: Total run duration
        
        # Result
        success: Whether run succeeded
        answer: Final answer
    """
    run_id: str
    query: str
    video_path: str
    backend: str
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    
    # Token counts
    root_input_tokens: int = 0
    root_output_tokens: int = 0
    sub_input_tokens: int = 0
    sub_output_tokens: int = 0
    
    # Call counts
    root_calls: int = 0
    sub_calls: int = 0
    code_executions: int = 0
    
    # Cost
    root_cost_usd: float = 0.0
    sub_cost_usd: float = 0.0
    
    # Time
    root_latency_ms: float = 0.0
    sub_latency_ms: float = 0.0
    code_execution_ms: float = 0.0
    
    # Result
    success: bool = False
    answer: str = ""
    
    @property
    def total_cost_usd(self) -> float:
        return self.root_cost_usd + self.sub_cost_usd
    
    @property
    def total_tokens(self) -> int:
        return (
            self.root_input_tokens + self.root_output_tokens +
            self.sub_input_tokens + self.sub_output_tokens
        )
    
    @property
    def total_duration_ms(self) -> float:
        if self.ended_at is None:
            return 0.0
        return (self.ended_at - self.started_at).total_seconds() * 1000
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "run_id": self.run_id,
            "query": self.query,
            "video_path": self.video_path,
            "backend": self.backend,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            
            "root_input_tokens": self.root_input_tokens,
            "root_output_tokens": self.root_output_tokens,
            "sub_input_tokens": self.sub_input_tokens,
            "sub_output_tokens": self.sub_output_tokens,
            "total_tokens": self.total_tokens,
            
            "root_calls": self.root_calls,
            "sub_calls": self.sub_calls,
            "code_executions": self.code_executions,
            
            "root_cost_usd": round(self.root_cost_usd, 6),
            "sub_cost_usd": round(self.sub_cost_usd, 6),
            "total_cost_usd": round(self.total_cost_usd, 6),
            
            "root_latency_ms": round(self.root_latency_ms, 2),
            "sub_latency_ms": round(self.sub_latency_ms, 2),
            "code_execution_ms": round(self.code_execution_ms, 2),
            "total_duration_ms": round(self.total_duration_ms, 2),
            
            "success": self.success,
            "answer": self.answer,
        }


class CostTracker:
    """Tracks costs across multiple runs.
    
    Provides aggregated statistics and export functionality
    for academic experiments.
    
    Example:
        >>> tracker = CostTracker("experiment_1")
        >>> metrics = tracker.start_run("run_1", "What happens?", "video.mp4", "openai")
        >>> # ... run agent ...
        >>> tracker.end_run("run_1", success=True, answer="...")
        >>> tracker.save("results/costs.json")
    """
    
    def __init__(self, experiment_id: str):
        """Initialize cost tracker.
        
        Args:
            experiment_id: Identifier for this experiment
        """
        self.experiment_id = experiment_id
        self._runs: dict[str, RunMetrics] = {}
    
    def start_run(
        self,
        run_id: str,
        query: str,
        video_path: str,
        backend: str,
    ) -> RunMetrics:
        """Start tracking a new run.
        
        Args:
            run_id: Unique run identifier
            query: The query
            video_path: Path to video
            backend: LLM backend
            
        Returns:
            RunMetrics object for recording metrics
        """
        metrics = RunMetrics(
            run_id=run_id,
            query=query,
            video_path=video_path,
            backend=backend,
        )
        self._runs[run_id] = metrics
        return metrics
    
    def get_run(self, run_id: str) -> Optional[RunMetrics]:
        """Get metrics for a run."""
        return self._runs.get(run_id)
    
    def end_run(
        self,
        run_id: str,
        success: bool,
        answer: str = "",
    ) -> None:
        """Mark a run as complete.
        
        Args:
            run_id: The run ID
            success: Whether the run succeeded
            answer: Final answer
        """
        metrics = self._runs.get(run_id)
        if metrics:
            metrics.ended_at = datetime.now()
            metrics.success = success
            metrics.answer = answer
    
    def record_root_call(
        self,
        run_id: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float,
    ) -> None:
        """Record a root LM call."""
        metrics = self._runs.get(run_id)
        if metrics:
            metrics.root_calls += 1
            metrics.root_input_tokens += input_tokens
            metrics.root_output_tokens += output_tokens
            metrics.root_cost_usd += cost_usd
            metrics.root_latency_ms += latency_ms
    
    def record_sub_call(
        self,
        run_id: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float,
    ) -> None:
        """Record a sub LM call (llm_query)."""
        metrics = self._runs.get(run_id)
        if metrics:
            metrics.sub_calls += 1
            metrics.sub_input_tokens += input_tokens
            metrics.sub_output_tokens += output_tokens
            metrics.sub_cost_usd += cost_usd
            metrics.sub_latency_ms += latency_ms
    
    def record_code_execution(
        self,
        run_id: str,
        execution_ms: float,
    ) -> None:
        """Record a code execution."""
        metrics = self._runs.get(run_id)
        if metrics:
            metrics.code_executions += 1
            metrics.code_execution_ms += execution_ms
    
    def summary(self) -> dict[str, Any]:
        """Get summary statistics across all runs."""
        if not self._runs:
            return {"experiment_id": self.experiment_id, "num_runs": 0}
        
        runs = list(self._runs.values())
        completed = [r for r in runs if r.ended_at is not None]
        successful = [r for r in completed if r.success]
        
        return {
            "experiment_id": self.experiment_id,
            "num_runs": len(runs),
            "num_completed": len(completed),
            "num_successful": len(successful),
            "success_rate": len(successful) / max(1, len(completed)),
            
            "total_cost_usd": sum(r.total_cost_usd for r in runs),
            "avg_cost_usd": sum(r.total_cost_usd for r in runs) / max(1, len(runs)),
            
            "total_tokens": sum(r.total_tokens for r in runs),
            "avg_tokens": sum(r.total_tokens for r in runs) / max(1, len(runs)),
            
            "total_sub_calls": sum(r.sub_calls for r in runs),
            "avg_sub_calls": sum(r.sub_calls for r in runs) / max(1, len(runs)),
            
            "avg_duration_ms": sum(r.total_duration_ms for r in completed) / max(1, len(completed)),
        }
    
    def to_dataframe(self) -> Any:
        """Convert to pandas DataFrame if pandas is available."""
        try:
            import pandas as pd
            return pd.DataFrame([r.to_dict() for r in self._runs.values()])
        except ImportError:
            logger.warning("pandas not available, returning dict list")
            return [r.to_dict() for r in self._runs.values()]
    
    def save(self, path: Path) -> None:
        """Save results to JSON file.
        
        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "experiment_id": self.experiment_id,
            "summary": self.summary(),
            "runs": [r.to_dict() for r in self._runs.values()],
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved cost report to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "CostTracker":
        """Load from JSON file.
        
        Args:
            path: Input file path
            
        Returns:
            CostTracker with loaded data
        """
        with open(path) as f:
            data = json.load(f)
        
        tracker = cls(data["experiment_id"])
        # Note: This is a simplified load - full implementation would
        # reconstruct RunMetrics objects
        return tracker
