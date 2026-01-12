"""
Sub-Agent for Video Understanding

This module implements the sub-agent that handles recursive llm_query() calls.
The sub-agent processes video segments and extracts structured evidence.

In the RLM paradigm:
- Root agent decides what to look at and how to chunk
- Sub-agent handles local understanding of specific segments
- Results are composed into a global answer

The sub-agent doesn't maintain state - each call is independent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from rvaa.tools.llm_backends import LLMBackend, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    """Structured evidence extracted from a video segment.
    
    Attributes:
        segment_id: Identifier for the source segment
        start_time: Start timestamp
        end_time: End timestamp
        claim: The extracted claim or finding
        confidence: Confidence score (0-1)
        supporting_text: Supporting evidence text
        relevant_to_query: Whether this is relevant to the main query
    """
    segment_id: str
    start_time: float
    end_time: float
    claim: str
    confidence: float = 1.0
    supporting_text: str = ""
    relevant_to_query: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "segment_id": self.segment_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "claim": self.claim,
            "confidence": self.confidence,
            "supporting_text": self.supporting_text,
            "relevant_to_query": self.relevant_to_query,
        }


# =============================================================================
# Sub-Agent Prompt Templates
# =============================================================================

SEGMENT_UNDERSTANDING_TEMPLATE = """You are analyzing a video segment as part of a larger video understanding task.

Main Query: {query}

Segment Information:
- Time Range: {start_time:.1f}s - {end_time:.1f}s
- Duration: {duration:.1f}s

Frame Captions from this segment:
{captions}

Instructions:
1. Analyze the captions to understand what's happening in this segment
2. Identify any information relevant to the main query
3. Note key events, objects, people, or actions
4. Be specific about what you observe vs. what you infer

Provide your analysis in a structured format:
- Key Observations: What is directly visible/described
- Relevant to Query: How this relates to the main question (if at all)
- Confidence: How confident you are in these observations (high/medium/low)
"""

EVIDENCE_EXTRACTION_TEMPLATE = """Extract specific evidence from these video segment captions that relates to the query.

Query: {query}

Segment: {start_time:.1f}s - {end_time:.1f}s

Captions:
{captions}

Extract evidence in the following format:
CLAIM: [Specific claim that can be made from this segment]
CONFIDENCE: [high/medium/low]
SUPPORT: [Which captions support this claim]
RELEVANT: [yes/no - is this relevant to the query]

If no relevant evidence is found, respond with:
NO_RELEVANT_EVIDENCE
"""

VERIFICATION_TEMPLATE = """Verify whether the following claim is supported by the video segment evidence.

Claim to Verify: {claim}

Segment: {start_time:.1f}s - {end_time:.1f}s

Evidence from segment:
{captions}

Determine:
1. Is the claim SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE?
2. What specific evidence supports or contradicts the claim?
3. Confidence level: high/medium/low

Response format:
VERDICT: [SUPPORTED/CONTRADICTED/INSUFFICIENT_EVIDENCE]
CONFIDENCE: [high/medium/low]
REASONING: [Brief explanation]
"""

SUMMARIZE_TEMPLATE = """Summarize the key content and events in this video segment.

Segment: {start_time:.1f}s - {end_time:.1f}s

Frame Captions:
{captions}

Provide a concise summary (2-3 sentences) of:
1. The main scene or setting
2. Key actions or events
3. Any notable objects, people, or changes
"""


class SubAgent:
    """Sub-agent for handling recursive LLM queries.
    
    The sub-agent processes video segments and returns structured
    understanding that the root agent can use to build its answer.
    
    Example:
        >>> sub_agent = SubAgent(llm_backend)
        >>> evidence = await sub_agent.extract_evidence(
        ...     captions=["[0.0s] A person enters the room", "[1.0s] They sit down"],
        ...     query="Who is in the video?",
        ...     start_time=0.0,
        ...     end_time=5.0,
        ... )
    """
    
    def __init__(
        self,
        llm: LLMBackend,
        temperature: float = 0.0,
    ):
        """Initialize sub-agent.
        
        Args:
            llm: LLM backend for queries
            temperature: Sampling temperature
        """
        self.llm = llm
        self.temperature = temperature
    
    async def understand_segment(
        self,
        captions: list[tuple[float, str]],
        query: str,
        start_time: float,
        end_time: float,
    ) -> str:
        """Analyze a video segment in the context of a query.
        
        Args:
            captions: List of (timestamp, caption) tuples
            query: The main query being answered
            start_time: Segment start time
            end_time: Segment end time
            
        Returns:
            Analysis text
        """
        captions_text = "\n".join([f"[{t:.1f}s] {c}" for t, c in captions])
        
        prompt = SEGMENT_UNDERSTANDING_TEMPLATE.format(
            query=query,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            captions=captions_text,
        )
        
        response = await self.llm.query(prompt, temperature=self.temperature)
        return response.content
    
    async def extract_evidence(
        self,
        captions: list[tuple[float, str]],
        query: str,
        start_time: float,
        end_time: float,
        segment_id: Optional[str] = None,
    ) -> Optional[Evidence]:
        """Extract structured evidence from a segment.
        
        Args:
            captions: List of (timestamp, caption) tuples
            query: The query to find evidence for
            start_time: Segment start time
            end_time: Segment end time
            segment_id: Optional segment identifier
            
        Returns:
            Evidence object if found, None otherwise
        """
        captions_text = "\n".join([f"[{t:.1f}s] {c}" for t, c in captions])
        
        prompt = EVIDENCE_EXTRACTION_TEMPLATE.format(
            query=query,
            start_time=start_time,
            end_time=end_time,
            captions=captions_text,
        )
        
        response = await self.llm.query(prompt, temperature=self.temperature)
        content = response.content
        
        # Parse response
        if "NO_RELEVANT_EVIDENCE" in content:
            return None
        
        # Extract fields (simple parsing)
        claim = ""
        confidence = 0.7
        support = ""
        relevant = True
        
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("CLAIM:"):
                claim = line[6:].strip()
            elif line.startswith("CONFIDENCE:"):
                conf_text = line[11:].strip().lower()
                if "high" in conf_text:
                    confidence = 0.9
                elif "low" in conf_text:
                    confidence = 0.5
            elif line.startswith("SUPPORT:"):
                support = line[8:].strip()
            elif line.startswith("RELEVANT:"):
                relevant = "yes" in line.lower()
        
        if not claim:
            return None
        
        return Evidence(
            segment_id=segment_id or f"seg_{start_time:.0f}_{end_time:.0f}",
            start_time=start_time,
            end_time=end_time,
            claim=claim,
            confidence=confidence,
            supporting_text=support,
            relevant_to_query=relevant,
        )
    
    async def verify_claim(
        self,
        claim: str,
        captions: list[tuple[float, str]],
        start_time: float,
        end_time: float,
    ) -> tuple[str, float, str]:
        """Verify a claim against segment evidence.
        
        Args:
            claim: The claim to verify
            captions: List of (timestamp, caption) tuples
            start_time: Segment start time
            end_time: Segment end time
            
        Returns:
            Tuple of (verdict, confidence, reasoning)
            verdict is one of: "SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"
        """
        captions_text = "\n".join([f"[{t:.1f}s] {c}" for t, c in captions])
        
        prompt = VERIFICATION_TEMPLATE.format(
            claim=claim,
            start_time=start_time,
            end_time=end_time,
            captions=captions_text,
        )
        
        response = await self.llm.query(prompt, temperature=self.temperature)
        content = response.content
        
        # Parse response
        verdict = "INSUFFICIENT_EVIDENCE"
        confidence = 0.5
        reasoning = ""
        
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("VERDICT:"):
                verdict_text = line[8:].strip().upper()
                if "SUPPORTED" in verdict_text:
                    verdict = "SUPPORTED"
                elif "CONTRADICTED" in verdict_text:
                    verdict = "CONTRADICTED"
            elif line.startswith("CONFIDENCE:"):
                conf_text = line[11:].strip().lower()
                if "high" in conf_text:
                    confidence = 0.9
                elif "low" in conf_text:
                    confidence = 0.5
                else:
                    confidence = 0.7
            elif line.startswith("REASONING:"):
                reasoning = line[10:].strip()
        
        return verdict, confidence, reasoning
    
    async def summarize_segment(
        self,
        captions: list[tuple[float, str]],
        start_time: float,
        end_time: float,
    ) -> str:
        """Generate a concise summary of a segment.
        
        Args:
            captions: List of (timestamp, caption) tuples
            start_time: Segment start time
            end_time: Segment end time
            
        Returns:
            Summary text
        """
        captions_text = "\n".join([f"[{t:.1f}s] {c}" for t, c in captions])
        
        prompt = SUMMARIZE_TEMPLATE.format(
            start_time=start_time,
            end_time=end_time,
            captions=captions_text,
        )
        
        response = await self.llm.query(prompt, temperature=self.temperature)
        return response.content
    
    async def batch_process(
        self,
        segments: list[dict[str, Any]],
        query: str,
        task: str = "understand",
    ) -> list[str]:
        """Process multiple segments in batch.
        
        This is more efficient than individual calls when possible.
        
        Args:
            segments: List of dicts with 'captions', 'start_time', 'end_time'
            query: The main query
            task: One of "understand", "summarize"
            
        Returns:
            List of results (one per segment)
        """
        import asyncio
        
        if task == "understand":
            coros = [
                self.understand_segment(
                    s["captions"], query, s["start_time"], s["end_time"]
                )
                for s in segments
            ]
        elif task == "summarize":
            coros = [
                self.summarize_segment(
                    s["captions"], s["start_time"], s["end_time"]
                )
                for s in segments
            ]
        else:
            raise ValueError(f"Unknown task: {task}")
        
        results = await asyncio.gather(*coros)
        return list(results)
