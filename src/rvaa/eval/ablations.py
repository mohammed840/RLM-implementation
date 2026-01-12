"""
Ablation Baselines

This module implements ablation baselines for comparison with the full RLM agent:
1. No Sub-Calls: Agent can inspect environment but cannot call sub-LLM
2. Summarize Everything: Naive chunk summarization then answer
3. Retrieval Only: Top-k segment retrieval without recursive refinement

These ablations match the paper's experimental methodology.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from rvaa.env.video_env import VideoEnv, VideoView
from rvaa.tools.llm_backends import LLMBackend
from rvaa.tools.index import SegmentIndex, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Result from an ablation run.
    
    Attributes:
        method: Ablation method name
        answer: Final answer
        evidence: List of evidence segments used
        cost_usd: Total cost
        input_tokens: Total input tokens
        output_tokens: Total output tokens
        duration_ms: Total duration
        success: Whether completed successfully
    """
    method: str
    answer: str
    evidence: list[dict[str, Any]]
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0
    success: bool = True


# =============================================================================
# Ablation 1: No Sub-Calls
# =============================================================================

NO_SUBCALLS_PROMPT = """You are analyzing a video to answer a question. You cannot call any sub-models - you must work only with the information provided.

Video Information:
{video_info}

Frame Captions (sampled throughout the video):
{captions}

Question: {query}

Based ONLY on the captions above, provide your best answer. Be specific about what evidence supports your answer.

Answer:"""


async def run_no_subcalls_ablation(
    query: str,
    video_env: VideoEnv,
    llm: LLMBackend,
    get_captions_fn: Callable[[VideoView], list[tuple[float, str]]],
    num_samples: int = 20,
) -> AblationResult:
    """Run the no-subcalls ablation.
    
    The agent sees sampled captions but cannot make recursive calls.
    This tests whether the RLM's subcall strategy adds value.
    
    Args:
        query: The question
        video_env: Video environment
        llm: LLM backend
        get_captions_fn: Function to get captions for segments
        num_samples: Number of segments to sample
        
    Returns:
        AblationResult
    """
    import time
    
    start_time = time.perf_counter()
    
    # Sample segments uniformly
    segment_duration = video_env.duration / num_samples
    all_captions = []
    evidence = []
    
    for i in range(num_samples):
        start = i * segment_duration
        end = min((i + 1) * segment_duration, video_env.duration)
        view = video_env._create_view(start, end)
        
        captions = get_captions_fn(view)
        for ts, cap in captions:
            all_captions.append(f"[{ts:.1f}s] {cap}")
        
        evidence.append({
            "start_time": start,
            "end_time": end,
            "captions": captions,
        })
    
    # Build prompt
    video_info = f"Duration: {video_env.duration:.1f}s, Frames: {video_env.total_frames}"
    captions_text = "\n".join(all_captions)
    
    prompt = NO_SUBCALLS_PROMPT.format(
        video_info=video_info,
        captions=captions_text,
        query=query,
    )
    
    # Single LLM call
    response = await llm.query(prompt)
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    
    return AblationResult(
        method="no_subcalls",
        answer=response.content,
        evidence=evidence,
        cost_usd=response.cost_usd,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        duration_ms=duration_ms,
    )


# =============================================================================
# Ablation 2: Summarize Everything
# =============================================================================

SUMMARIZE_CHUNK_PROMPT = """Summarize the key events and content in this video segment.

Segment: {start_time:.1f}s - {end_time:.1f}s

Frame Captions:
{captions}

Summary (2-3 sentences):"""

SUMMARIZE_ANSWER_PROMPT = """Based on summaries of video segments, answer the following question.

Question: {query}

Segment Summaries:
{summaries}

Answer based on the summaries above. Cite relevant timestamps.

Answer:"""


async def run_summarize_all_ablation(
    query: str,
    video_env: VideoEnv,
    llm: LLMBackend,
    get_captions_fn: Callable[[VideoView], list[tuple[float, str]]],
    chunk_duration: float = 30.0,
) -> AblationResult:
    """Run the summarize-everything ablation.
    
    Chunks the video, summarizes each chunk, then answers.
    This is a common baseline approach.
    
    Args:
        query: The question
        video_env: Video environment
        llm: LLM backend
        get_captions_fn: Function to get captions
        chunk_duration: Duration of each chunk in seconds
        
    Returns:
        AblationResult
    """
    import time
    
    start_time = time.perf_counter()
    total_cost = 0.0
    total_in = 0
    total_out = 0
    
    # Chunk and summarize
    summaries = []
    evidence = []
    
    for view in video_env.iter_segments(chunk_duration):
        captions = get_captions_fn(view)
        captions_text = "\n".join([f"[{t:.1f}s] {c}" for t, c in captions])
        
        prompt = SUMMARIZE_CHUNK_PROMPT.format(
            start_time=view.start_time,
            end_time=view.end_time,
            captions=captions_text,
        )
        
        response = await llm.query(prompt)
        
        summaries.append(f"[{view.start_time:.1f}s-{view.end_time:.1f}s]: {response.content}")
        evidence.append({
            "start_time": view.start_time,
            "end_time": view.end_time,
            "summary": response.content,
        })
        
        total_cost += response.cost_usd
        total_in += response.input_tokens
        total_out += response.output_tokens
    
    # Answer based on summaries
    summaries_text = "\n\n".join(summaries)
    answer_prompt = SUMMARIZE_ANSWER_PROMPT.format(
        query=query,
        summaries=summaries_text,
    )
    
    response = await llm.query(answer_prompt)
    
    total_cost += response.cost_usd
    total_in += response.input_tokens
    total_out += response.output_tokens
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    
    return AblationResult(
        method="summarize_all",
        answer=response.content,
        evidence=evidence,
        cost_usd=total_cost,
        input_tokens=total_in,
        output_tokens=total_out,
        duration_ms=duration_ms,
    )


# =============================================================================
# Ablation 3: Retrieval Only
# =============================================================================

RETRIEVAL_ANSWER_PROMPT = """Answer the question based on the retrieved video segments.

Question: {query}

Retrieved Segments (ranked by relevance):
{segments}

Answer based on the evidence above. Be specific about which segments support your answer.

Answer:"""


async def run_retrieval_only_ablation(
    query: str,
    index: SegmentIndex,
    llm: LLMBackend,
    top_k: int = 10,
) -> AblationResult:
    """Run the retrieval-only ablation.
    
    Uses embedding-based retrieval without any refinement.
    This tests the value of the recursive verification stage.
    
    Args:
        query: The question
        index: Pre-built segment index
        llm: LLM backend
        top_k: Number of segments to retrieve
        
    Returns:
        AblationResult
    """
    import time
    
    start_time = time.perf_counter()
    
    # Retrieve top-k segments
    results = index.search(query, top_k=top_k)
    
    evidence = []
    segments_text = []
    
    for i, result in enumerate(results):
        seg = result.segment
        segments_text.append(
            f"{i+1}. [{seg.start_time:.1f}s-{seg.end_time:.1f}s] (score: {result.score:.3f})\n"
            f"   {seg.text}"
        )
        evidence.append({
            "rank": i + 1,
            "segment_id": seg.segment_id,
            "start_time": seg.start_time,
            "end_time": seg.end_time,
            "score": result.score,
            "text": seg.text,
        })
    
    # Answer based on retrieved segments
    prompt = RETRIEVAL_ANSWER_PROMPT.format(
        query=query,
        segments="\n\n".join(segments_text),
    )
    
    response = await llm.query(prompt)
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    
    return AblationResult(
        method="retrieval_only",
        answer=response.content,
        evidence=evidence,
        cost_usd=response.cost_usd,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        duration_ms=duration_ms,
    )


# =============================================================================
# Run All Ablations
# =============================================================================

async def run_all_ablations(
    query: str,
    video_env: VideoEnv,
    index: SegmentIndex,
    llm: LLMBackend,
    get_captions_fn: Callable[[VideoView], list[tuple[float, str]]],
) -> dict[str, AblationResult]:
    """Run all ablation baselines.
    
    Args:
        query: The question
        video_env: Video environment
        index: Pre-built segment index
        llm: LLM backend
        get_captions_fn: Function to get captions
        
    Returns:
        Dict mapping method name to result
    """
    results = {}
    
    # No subcalls
    try:
        results["no_subcalls"] = await run_no_subcalls_ablation(
            query, video_env, llm, get_captions_fn
        )
    except Exception as e:
        logger.error(f"No-subcalls ablation failed: {e}")
        results["no_subcalls"] = AblationResult(
            method="no_subcalls",
            answer=f"Error: {e}",
            evidence=[],
            success=False,
        )
    
    # Summarize all
    try:
        results["summarize_all"] = await run_summarize_all_ablation(
            query, video_env, llm, get_captions_fn
        )
    except Exception as e:
        logger.error(f"Summarize-all ablation failed: {e}")
        results["summarize_all"] = AblationResult(
            method="summarize_all",
            answer=f"Error: {e}",
            evidence=[],
            success=False,
        )
    
    # Retrieval only
    try:
        results["retrieval_only"] = await run_retrieval_only_ablation(
            query, index, llm
        )
    except Exception as e:
        logger.error(f"Retrieval-only ablation failed: {e}")
        results["retrieval_only"] = AblationResult(
            method="retrieval_only",
            answer=f"Error: {e}",
            evidence=[],
            success=False,
        )
    
    return results
