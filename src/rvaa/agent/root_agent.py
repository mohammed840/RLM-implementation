"""
Root Agent - RLM-Style Orchestrator

This module implements the core RLM (Recursive Language Model) agent that
orchestrates video understanding through REPL-based interaction.

Key concepts from the paper (arXiv:2512.24601):
1. The video is treated as an EXTERNAL ENVIRONMENT, not stuffed into context
2. The agent writes and executes code in a REPL to interact with the video
3. The agent can call llm_query() for recursive sub-model calls
4. The agent terminates with FINAL() or FINAL_VAR() 

System Prompt Design (from Appendix D.1):
- GPT-5 version: Full REPL access with llm_query() for sub-calls
- Qwen3-Coder version: Same + batching warning about llm_query costs
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Union

from rvaa.env.video_env import VideoEnv, VideoView
from rvaa.tools.llm_backends import LLMBackend, LLMResponse, CostAccounting

logger = logging.getLogger(__name__)


# =============================================================================
# Paper-Faithful System Prompts (Appendix D.1)
# =============================================================================

# (1a) System prompt for RLM with REPL for GPT-5
GPT5_SYSTEM_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

Your context is a {context_type} with {context_total_length} total characters, and is broken up into chunks of char lengths: {context_lengths}.

The REPL environment is initialized with:
1. A 'context' variable that contains extremely important information about your query. This is a VideoEnv object representing a long video. You should check the content of the 'context' variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A 'llm_query' function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
3. The ability to use 'print()' statements to view the output of your REPL code and continue your reasoning.

The VideoEnv context object supports ONLY these attributes and methods (do not use any others):
- context.duration: float - Total video duration in seconds
- context.total_frames: int - Total number of frames
- context.fps: float - Frames per second
- context.metadata: VideoMetadata object (has width, height, fps, duration_seconds, total_frames, codec)
- context.path: Path to the video file
- context[start_time:end_time]: Slice video by time (returns VideoView)
- context.get_frame(timestamp): Get single frame at timestamp, returns FrameData
- context.sample_frames_uniform(start_time, end_time, n_frames): Sample n frames uniformly, returns list[FrameData]
- context.iter_segments(segment_duration): Iterate over fixed-length segments, yields VideoView

FrameData objects have ONLY these attributes:
- frame.timestamp: float - Timestamp in seconds
- frame.frame_idx: int - Frame index
- frame.image: numpy array of shape (height, width, 3)
- frame.shape: tuple (height, width, 3)
- frame.is_keyframe: bool

VideoView objects have:
- view.start_time, view.end_time: floats
- view.duration: float
- view.sample_uniform(n): Sample n frames from this segment

The get_segment_captions(segment) function returns a list of (timestamp, caption_text) tuples.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.

Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed many frame captions per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. 

FIRST, always start by inspecting the video:
```repl
# Step 1: Inspect the video metadata
print(f"Video duration: {{context.duration:.1f}} seconds")
print(f"Total frames: {{context.total_frames}}")
print(f"FPS: {{context.fps:.2f}}")
print(f"Metadata: {{context.metadata}}")
```

Then, sample frames to understand the visual content:
```repl
# Step 2: Sample frames across the video
frames = context.sample_frames_uniform(0, context.duration, 5)
for i, frame in enumerate(frames):
    print(f"Frame {{i}}: timestamp={{frame.timestamp:.1f}}s, shape={{frame.shape}}")
```

To analyze specific segments, use slicing and llm_query for semantic analysis:
```repl
# Step 3: Analyze segments with sub-LLM calls
query = "What happens in this video?"
segment_summaries = []

# Divide video into chunks
n_chunks = 4
chunk_duration = context.duration / n_chunks

for i in range(n_chunks):
    start = i * chunk_duration
    end = (i + 1) * chunk_duration
    
    # Get captions for this segment
    segment = context[start:end]
    captions = get_segment_captions(segment)  # Returns [(timestamp, caption), ...]
    captions_text = "\\n".join([f"[{{t:.1f}}s] {{c}}" for t, c in captions])
    
    # Ask sub-LLM to summarize
    summary = llm_query(f"Given this video segment, answer: {{query}}\\nCaptions:\\n{{captions_text}}")
    segment_summaries.append(f"Segment {{i}} ({{start:.0f}}s-{{end:.0f}}s): {{summary}}")
    print(f"Segment {{i}}: {{summary[:100]}}...")

# Combine summaries for final answer
final = llm_query(f"Based on these segment summaries, provide a complete answer to: {{query}}\\n\\nSummaries:\\n" + "\\n".join(segment_summaries))
print(f"Final answer: {{final}}")
```


IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

CRITICAL INSTRUCTIONS:
- You MUST execute at least one ```repl code block before providing a FINAL answer
- DO NOT describe what the video shows without first executing code to inspect it
- DO NOT hallucinate or guess the video content - you must use the REPL to actually inspect frames/captions
- Your first response MUST contain a ```repl code block to inspect the video
- Only provide FINAL() after you have gathered evidence from the video through code execution

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.

START BY EXECUTING CODE TO INSPECT THE VIDEO. Do not provide an answer without first running code.
"""

# (1b) The diff for Qwen3-Coder - adds batching warning
QWEN_BATCHING_WARNING = """
IMPORTANT: Be very careful about using 'llm_query' as it incurs high runtime costs. Always batch as much information as reasonably possible into each call (aim for around ~200k characters per call). For example, if you have 1000 lines of information to process, it's much better to split into chunks of 5 and call 'llm_query' on each chunk (200 calls total) rather than making 1000 individual calls. Minimize the number of 'llm_query' calls by batching related information together.

"""


def get_system_prompt(
    backend_type: str = "openai",
    context_type: str = "VideoEnv (long video)",
    context_total_length: int = 0,
    context_lengths: str = "N/A (video - use context.duration for length)",
) -> str:
    """Get the appropriate system prompt for the backend.
    
    Args:
        backend_type: "openai", "qwen", or "claude"
        context_type: Description of the context type
        context_total_length: Total context length (for video, use duration * fps)
        context_lengths: Description of context chunk lengths
        
    Returns:
        Formatted system prompt
    """
    base_prompt = GPT5_SYSTEM_PROMPT.format(
        context_type=context_type,
        context_total_length=context_total_length,
        context_lengths=context_lengths,
    )
    
    if backend_type == "qwen":
        # Insert batching warning after the first few paragraphs
        lines = base_prompt.split("\n")
        # Insert after the "REPL environment is initialized with" section
        insert_idx = 15
        lines.insert(insert_idx, QWEN_BATCHING_WARNING)
        return "\n".join(lines)
    
    return base_prompt


# =============================================================================
# REPL Runtime
# =============================================================================

class REPLExecutionError(Exception):
    """Error during REPL code execution."""
    pass


@dataclass
class REPLOutput:
    """Output from a REPL execution.
    
    Attributes:
        stdout: Captured stdout output
        return_value: Return value (if any)
        error: Error message (if execution failed)
        execution_time_ms: Execution time in milliseconds
    """
    stdout: str
    return_value: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    
    @property
    def success(self) -> bool:
        return self.error is None
    
    def truncated_output(self, max_chars: int = 5000) -> str:
        """Get truncated output for feeding back to LLM."""
        output = self.stdout
        if len(output) > max_chars:
            output = output[:max_chars] + f"\n... [truncated, {len(self.stdout) - max_chars} more chars]"
        return output


class REPLRuntime:
    """Python REPL runtime for agent code execution.
    
    This implements the sandboxed environment where the agent executes code.
    It provides:
    - A 'context' variable bound to the VideoEnv
    - A 'llm_query' function for recursive sub-calls
    - Standard library access
    - Output truncation to avoid context overflow
    """
    
    def __init__(
        self,
        context: VideoEnv,
        llm_query_fn: Callable[[str], str],
        get_segment_captions_fn: Optional[Callable[[VideoView], list[tuple[float, str]]]] = None,
        max_output_chars: int = 10000,
    ):
        """Initialize REPL runtime.
        
        Args:
            context: VideoEnv to bind as 'context' variable
            llm_query_fn: Function to call for llm_query()
            get_segment_captions_fn: Function to get captions for a segment
            max_output_chars: Maximum output characters before truncation
        """
        self.context = context
        self.llm_query_fn = llm_query_fn
        self.get_segment_captions_fn = get_segment_captions_fn or self._default_get_captions
        self.max_output_chars = max_output_chars
        
        # Persistent namespace across executions
        self._namespace: dict[str, Any] = {}
        self._setup_namespace()
    
    def _default_get_captions(self, segment: VideoView) -> list[tuple[float, str]]:
        """Generate captions for video segment using vision model."""
        try:
            from rvaa.tools.vision_captioner import get_captioner
            
            captioner = get_captioner()
            captions = []
            
            # Sample 2-3 frames from the segment
            n_samples = min(3, max(1, int(segment.duration / 60)))  # 1 per minute
            frames = segment.sample_uniform(n_samples)
            
            for frame in frames:
                try:
                    caption = captioner.caption_frame_sync(frame.image)
                    captions.append((frame.timestamp, caption))
                except Exception as e:
                    captions.append((frame.timestamp, f"[Frame at {frame.timestamp:.1f}s]"))
            
            return captions if captions else [(segment.start_time, f"[Segment {segment.start_time:.0f}s-{segment.end_time:.0f}s]")]
        except Exception as e:
            # Fallback to placeholder
            mid_time = (segment.start_time + segment.end_time) / 2
            return [(mid_time, f"[Segment at {segment.start_time:.1f}s-{segment.end_time:.1f}s]")]
    
    def _setup_namespace(self) -> None:
        """Set up the REPL namespace with builtins and context."""
        import builtins
        import math
        import re as re_module
        import json
        import collections
        
        # Safe builtins
        safe_builtins = {
            'print': print,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'reversed': reversed,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'min': min,
            'max': max,
            'sum': sum,
            'abs': abs,
            'round': round,
            'any': any,
            'all': all,
            'isinstance': isinstance,
            'type': type,
            'hasattr': hasattr,
            'getattr': getattr,
            'setattr': setattr,
        }
        
        self._namespace = {
            '__builtins__': safe_builtins,
            'context': self.context,
            'llm_query': self.llm_query_fn,
            'get_segment_captions': self.get_segment_captions_fn,
            # Common imports
            'math': math,
            're': re_module,
            'json': json,
            'collections': collections,
        }
    
    def execute(self, code: str) -> REPLOutput:
        """Execute code in the REPL environment.
        
        Args:
            code: Python code to execute
            
        Returns:
            REPLOutput with stdout and any errors
        """
        import io
        import sys
        
        start_time = time.perf_counter()
        
        # Capture stdout
        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        
        try:
            sys.stdout = stdout_capture
            
            # Execute the code
            exec(code, self._namespace)
            
            # Get output
            stdout = stdout_capture.getvalue()
            
            # Truncate if needed
            if len(stdout) > self.max_output_chars:
                stdout = stdout[:self.max_output_chars] + f"\n[Output truncated at {self.max_output_chars} chars]"
            
            exec_time = (time.perf_counter() - start_time) * 1000
            
            return REPLOutput(
                stdout=stdout,
                execution_time_ms=exec_time,
            )
            
        except Exception as e:
            exec_time = (time.perf_counter() - start_time) * 1000
            return REPLOutput(
                stdout=stdout_capture.getvalue(),
                error=f"{type(e).__name__}: {e}",
                execution_time_ms=exec_time,
            )
        finally:
            sys.stdout = old_stdout
    
    def get_variable(self, name: str) -> Any:
        """Get a variable from the namespace."""
        return self._namespace.get(name)
    
    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the namespace."""
        self._namespace[name] = value


# =============================================================================
# Trajectory Tracking
# =============================================================================

class StepType(str, Enum):
    """Type of trajectory step."""
    THINK = "think"
    CODE = "code"
    LLM_QUERY = "llm_query"
    FINAL = "final"
    ERROR = "error"


@dataclass
class TrajectoryStep:
    """A single step in the agent's trajectory.
    
    Attributes:
        step_idx: Step index (0-based)
        step_type: Type of step
        input_text: Input (code, query, etc.)
        output_text: Output (execution result, LLM response, etc.)
        timestamp: When this step occurred
        duration_ms: How long this step took
        tokens_in: Input tokens (for LLM calls)
        tokens_out: Output tokens (for LLM calls)
        cost_usd: Cost of this step
    """
    step_idx: int
    step_type: StepType
    input_text: str
    output_text: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "step_idx": self.step_idx,
            "step_type": self.step_type.value,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
        }


@dataclass
class Trajectory:
    """Complete trajectory of an agent run.
    
    Attributes:
        query: The original query
        steps: List of trajectory steps
        final_answer: The final answer (if reached)
        total_cost_usd: Total cost of the trajectory
        total_duration_ms: Total duration in milliseconds
        success: Whether the agent completed successfully
    """
    query: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    total_cost_usd: float = 0.0
    total_duration_ms: float = 0.0
    success: bool = False
    
    def add_step(self, step: TrajectoryStep) -> None:
        """Add a step to the trajectory."""
        self.steps.append(step)
        self.total_cost_usd += step.cost_usd
        self.total_duration_ms += step.duration_ms
    
    @property
    def num_llm_queries(self) -> int:
        """Count of llm_query calls."""
        return sum(1 for s in self.steps if s.step_type == StepType.LLM_QUERY)
    
    @property
    def num_code_executions(self) -> int:
        """Count of code execution steps."""
        return sum(1 for s in self.steps if s.step_type == StepType.CODE)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "query": self.query,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "total_cost_usd": self.total_cost_usd,
            "total_duration_ms": self.total_duration_ms,
            "success": self.success,
            "num_llm_queries": self.num_llm_queries,
            "num_code_executions": self.num_code_executions,
        }


# =============================================================================
# Root Agent
# =============================================================================

class RootAgent:
    """RLM-style root agent for video understanding.
    
    The root agent:
    1. Receives a query about a video
    2. Generates and executes code in a REPL to explore the video
    3. Uses llm_query() for recursive sub-calls to understand segments
    4. Terminates with FINAL() or FINAL_VAR() when ready to answer
    
    Paper-faithful implementation following Appendix D.1.
    
    Example:
        >>> agent = RootAgent(root_llm, sub_llm)
        >>> trajectory = await agent.run("What events happen in this video?", video_env)
        >>> print(trajectory.final_answer)
    """
    
    # Regex patterns for parsing agent output
    CODE_PATTERN = re.compile(r"```repl\n(.*?)```", re.DOTALL)
    FINAL_PATTERN = re.compile(r"FINAL\((.*)\)", re.DOTALL)
    FINAL_VAR_PATTERN = re.compile(r"FINAL_VAR\((\w+)\)")
    
    def __init__(
        self,
        root_llm: LLMBackend,
        sub_llm: LLMBackend,
        backend_type: str = "openai",
        max_steps: int = 20,
        max_output_chars: int = 10000,
    ):
        """Initialize the root agent.
        
        Args:
            root_llm: LLM backend for the root agent
            sub_llm: LLM backend for sub-calls (llm_query)
            backend_type: "openai", "qwen", or "claude" (affects system prompt)
            max_steps: Maximum number of REPL steps before forcing termination
            max_output_chars: Maximum output chars per code execution
        """
        self.root_llm = root_llm
        self.sub_llm = sub_llm
        self.backend_type = backend_type
        self.max_steps = max_steps
        self.max_output_chars = max_output_chars
        
        self._sub_call_count = 0
    
    async def run(
        self,
        query: str,
        video_env: VideoEnv,
        get_captions_fn: Optional[Callable[[VideoView], list[tuple[float, str]]]] = None,
        on_step: Optional[Callable[[TrajectoryStep], None]] = None,
    ) -> Trajectory:
        """Run the agent on a query.
        
        Args:
            query: The question about the video
            video_env: The video environment
            get_captions_fn: Optional function to get captions for video segments
            on_step: Optional callback for each step (for streaming)
            
        Returns:
            Complete trajectory with final answer
        """
        trajectory = Trajectory(query=query)
        self._sub_call_count = 0
        
        # Create llm_query function that tracks calls
        async def llm_query_async(prompt: str) -> str:
            self._sub_call_count += 1
            response = await self.sub_llm.query(prompt)
            
            # Track the sub-call
            step = TrajectoryStep(
                step_idx=len(trajectory.steps),
                step_type=StepType.LLM_QUERY,
                input_text=prompt[:500] + "..." if len(prompt) > 500 else prompt,
                output_text=response.content,
                tokens_in=response.input_tokens,
                tokens_out=response.output_tokens,
                cost_usd=response.cost_usd,
                duration_ms=response.latency_ms,
            )
            trajectory.add_step(step)
            if on_step:
                on_step(step)
            
            return response.content
        
        # Sync wrapper for REPL - handle nested async with new event loop
        def llm_query_sync(prompt: str) -> str:
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(llm_query_async(prompt))
        
        # Set up REPL
        repl = REPLRuntime(
            context=video_env,
            llm_query_fn=llm_query_sync,
            get_segment_captions_fn=get_captions_fn,
            max_output_chars=self.max_output_chars,
        )
        
        # Get system prompt
        system_prompt = get_system_prompt(
            backend_type=self.backend_type,
            context_type=f"VideoEnv ({video_env.path.name})",
            context_total_length=video_env.total_frames,
        )
        
        # Build initial prompt with query
        messages = [
            f"Query: {query}\n\n"
            f"Video context info: {video_env}"
        ]
        
        # Main loop
        for step_num in range(self.max_steps):
            # Build the conversation so far
            prompt = "\n\n".join(messages)
            
            # Query root LLM
            start_time = time.perf_counter()
            response = await self.root_llm.query(
                prompt=prompt,
                system_prompt=system_prompt,
            )
            latency = (time.perf_counter() - start_time) * 1000
            
            llm_output = response.content
            
            # Check for FINAL
            final_match = self.FINAL_PATTERN.search(llm_output)
            final_var_match = self.FINAL_VAR_PATTERN.search(llm_output)
            
            # IMPORTANT: Reject FINAL if no code has been executed yet
            # This prevents the model from hallucinating without exploration
            has_executed_code = trajectory.num_code_executions > 0
            
            if (final_match or final_var_match) and not has_executed_code:
                # Force the model to execute code first
                messages.append(
                    "ERROR: You cannot provide a FINAL answer without first executing code to inspect the video. "
                    "You MUST run at least one ```repl code block to examine the video before answering. "
                    "Please write code to inspect the video content, then provide your answer based on what you find."
                )
                continue
            
            if final_match:
                # Direct answer
                answer = final_match.group(1).strip()
                trajectory.final_answer = answer
                trajectory.success = True
                
                step = TrajectoryStep(
                    step_idx=len(trajectory.steps),
                    step_type=StepType.FINAL,
                    input_text="",
                    output_text=answer,
                    tokens_in=response.input_tokens,
                    tokens_out=response.output_tokens,
                    cost_usd=response.cost_usd,
                    duration_ms=latency,
                )
                trajectory.add_step(step)
                if on_step:
                    on_step(step)
                break
            
            elif final_var_match:
                # Variable reference
                var_name = final_var_match.group(1)
                var_value = repl.get_variable(var_name)
                answer = str(var_value) if var_value is not None else f"[Variable {var_name} not found]"
                trajectory.final_answer = answer
                trajectory.success = var_value is not None
                
                step = TrajectoryStep(
                    step_idx=len(trajectory.steps),
                    step_type=StepType.FINAL,
                    input_text=var_name,
                    output_text=answer,
                    tokens_in=response.input_tokens,
                    tokens_out=response.output_tokens,
                    cost_usd=response.cost_usd,
                    duration_ms=latency,
                )
                trajectory.add_step(step)
                if on_step:
                    on_step(step)
                break
            
            # Check for code blocks
            code_matches = self.CODE_PATTERN.findall(llm_output)
            
            if code_matches:
                # Execute each code block
                for code in code_matches:
                    repl_output = repl.execute(code)
                    
                    step = TrajectoryStep(
                        step_idx=len(trajectory.steps),
                        step_type=StepType.CODE,
                        input_text=code,
                        output_text=repl_output.stdout if repl_output.success else f"Error: {repl_output.error}",
                        duration_ms=repl_output.execution_time_ms,
                    )
                    trajectory.add_step(step)
                    if on_step:
                        on_step(step)
                    
                    # Add output to messages for next iteration
                    if repl_output.success:
                        messages.append(f"Code executed:\n```repl\n{code}\n```\n\nOutput:\n{repl_output.truncated_output(2000)}")
                    else:
                        messages.append(f"Code execution failed:\n```repl\n{code}\n```\n\nError: {repl_output.error}")
            else:
                # No code, just thinking
                step = TrajectoryStep(
                    step_idx=len(trajectory.steps),
                    step_type=StepType.THINK,
                    input_text="",
                    output_text=llm_output[:1000],
                    tokens_in=response.input_tokens,
                    tokens_out=response.output_tokens,
                    cost_usd=response.cost_usd,
                    duration_ms=latency,
                )
                trajectory.add_step(step)
                if on_step:
                    on_step(step)
                
                messages.append(f"Agent: {llm_output}")
        
        else:
            # Hit max steps without FINAL
            trajectory.final_answer = "[Max steps reached without final answer]"
            trajectory.success = False
        
        return trajectory
    
    def get_cost_summary(self) -> dict[str, Any]:
        """Get cost summary from both LLMs."""
        return {
            "root_llm": self.root_llm.accounting.summary(),
            "sub_llm": self.sub_llm.accounting.summary(),
            "total_sub_calls": self._sub_call_count,
        }
