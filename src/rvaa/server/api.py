"""
FastAPI Server for RVAA

This module provides the REST API for the Recursive Vision-Action Agent.

Endpoints:
- POST /query: Submit a query about a video
- GET /stream/{run_id}: SSE stream of agent events
- GET /status/{run_id}: Check run status
- POST /upload: Upload a video file
- GET /health: Health check
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

# Load .env file
from dotenv import load_dotenv
load_dotenv()  # Loads from current working directory

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from rvaa.agent.root_agent import RootAgent, TrajectoryStep
from rvaa.env.video_env import VideoEnv
from rvaa.tools.llm_backends import get_llm_backend, ModelType
from rvaa.tools.perception import get_perception_backend
from rvaa.protocols.events import (
    CodeExecutionEvent,
    LLMQueryEvent,
    FinalAnswerEvent,
    create_run_start_event,
    create_run_end_event,
    create_error_event,
)
from rvaa.server.streaming import get_stream_manager, StreamManager

logger = logging.getLogger(__name__)

# Get the static files directory
STATIC_DIR = Path(__file__).parent / "static"


# =============================================================================
# Configuration
# =============================================================================

class Settings(BaseSettings):
    """Application settings."""
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # LLM Backend
    llm_backend: str = "openrouter"  # openrouter, openai, qwen, claude
    
    # Video storage
    upload_dir: str = "/tmp/rvaa/uploads"
    max_upload_size_mb: int = 500
    
    # Agent
    max_steps: int = 20
    max_output_chars: int = 10000
    
    class Config:
        env_prefix = "RVAA_"


settings = Settings()


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="RVAA - Recursive Vision-Action Agent",
    description="API for long video understanding using the RLM paradigm",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# =============================================================================
# Web Interface
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the web interface."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        return HTMLResponse("""
        <html>
        <head><title>RVAA</title></head>
        <body style="font-family: sans-serif; padding: 40px; background: #0f0f14; color: #f0f0f5;">
            <h1>🎬 RVAA - Recursive Vision-Action Agent</h1>
            <p>API is running. Use the endpoints:</p>
            <ul>
                <li>POST /query - Submit a query</li>
                <li>GET /stream/{run_id} - Stream events</li>
                <li>GET /health - Health check</li>
            </ul>
            <p><a href="/docs" style="color: #4d9fff;">API Documentation</a></p>
        </body>
        </html>
        """)


# =============================================================================
# Request/Response Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request to query a video."""
    
    query: str = Field(..., description="Question about the video")
    video_path: str = Field(..., description="Path to video file")
    backend: str = Field(default="openrouter", description="LLM backend: openrouter, openai, qwen, claude")
    max_steps: int = Field(default=20, description="Maximum agent steps")


class QueryResponse(BaseModel):
    """Response to query request."""
    
    run_id: str = Field(..., description="Run ID for tracking")
    status: str = Field(..., description="Initial status")
    stream_url: str = Field(..., description="URL for SSE streaming")


class RunStatusResponse(BaseModel):
    """Run status response."""
    
    run_id: str
    status: str
    query: str
    video_path: str
    started_at: str
    completed_at: Optional[str] = None
    event_count: int
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class UploadResponse(BaseModel):
    """Video upload response."""
    
    video_id: str
    path: str
    size_mb: float
    duration_seconds: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str
    backend: str


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        backend=settings.llm_backend,
    )


@app.get("/video")
async def stream_video(path: str):
    """Stream a video file for preview."""
    video_path = Path(path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    from starlette.responses import FileResponse
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=video_path.name,
    )


@app.post("/query", response_model=QueryResponse)
async def submit_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
) -> QueryResponse:
    """Submit a query about a video.
    
    The query runs asynchronously. Use the returned stream_url to
    receive real-time updates via SSE.
    """
    # Validate video exists
    video_path = Path(request.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
    
    # Create run
    manager = get_stream_manager()
    run_id = await manager.create_run(
        query=request.query,
        video_path=str(video_path),
    )
    
    # Start agent in background
    background_tasks.add_task(
        run_agent,
        run_id=run_id,
        query=request.query,
        video_path=video_path,
        backend=request.backend,
        max_steps=request.max_steps,
    )
    
    return QueryResponse(
        run_id=run_id,
        status="pending",
        stream_url=f"/stream/{run_id}",
    )


@app.get("/stream/{run_id}")
async def stream_events(run_id: str, from_seq: int = 0):
    """Stream agent events via SSE.
    
    Connect to this endpoint to receive real-time updates about
    the agent's progress. Events are sent as Server-Sent Events.
    """
    manager = get_stream_manager()
    stream = await manager.get_stream(run_id)
    
    if not stream:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    
    async def event_generator():
        async for message in stream.subscribe(from_sequence=from_seq):
            yield message.to_sse()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/status/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str) -> RunStatusResponse:
    """Get the status of an agent run."""
    manager = get_stream_manager()
    run = await manager.get_run(run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    
    return RunStatusResponse(**run.to_dict())


@app.post("/upload", response_model=UploadResponse)
async def upload_video(
    file: UploadFile = File(...),
) -> UploadResponse:
    """Upload a video file.
    
    Videos are stored temporarily. The returned path can be used
    for subsequent queries.
    """
    # Check size
    file.file.seek(0, 2)  # Seek to end
    size = file.file.tell()
    file.file.seek(0)  # Seek back
    
    max_size = settings.max_upload_size_mb * 1024 * 1024
    if size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_upload_size_mb}MB"
        )
    
    # Save file
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    video_id = str(uuid.uuid4())
    ext = Path(file.filename or "video").suffix or ".mp4"
    save_path = upload_dir / f"{video_id}{ext}"
    
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Get video info
    duration = None
    try:
        video_env = VideoEnv(save_path, preload_metadata=True)
        duration = video_env.duration
    except Exception as e:
        logger.warning(f"Could not read video metadata: {e}")
    
    return UploadResponse(
        video_id=video_id,
        path=str(save_path),
        size_mb=size / (1024 * 1024),
        duration_seconds=duration,
    )


@app.get("/runs")
async def list_runs() -> list[dict[str, Any]]:
    """List all agent runs."""
    manager = get_stream_manager()
    return manager.list_runs()


# =============================================================================
# Background Task
# =============================================================================

async def run_agent(
    run_id: str,
    query: str,
    video_path: Path,
    backend: str,
    max_steps: int,
) -> None:
    """Run the agent as a background task.
    
    This handles all the agent execution and event publishing.
    """
    manager = get_stream_manager()
    run = await manager.get_run(run_id)
    
    if not run:
        logger.error(f"Run not found: {run_id}")
        return
    
    try:
        # Update status
        run.status = "running"
        
        # Initialize components
        video_env = VideoEnv(video_path)
        
        # Get LLM backends
        root_llm = get_llm_backend(backend, ModelType.ROOT)
        sub_llm = get_llm_backend(backend, ModelType.SUB)
        
        # Create agent
        agent = RootAgent(
            root_llm=root_llm,
            sub_llm=sub_llm,
            backend_type=backend,
            max_steps=max_steps,
        )
        
        # Publish run start
        start_event = create_run_start_event(
            query=query,
            video_path=str(video_path),
            video_duration=video_env.duration,
            config={"backend": backend, "max_steps": max_steps},
        )
        await manager.publish_event(run_id, start_event)
        
        # Define step callback for streaming
        def on_step(step: TrajectoryStep) -> None:
            # Convert step to event
            if step.step_type.value == "code":
                event = CodeExecutionEvent(
                    code=step.input_text,
                    output=step.output_text,
                    success="Error" not in step.output_text,
                    execution_time_ms=step.duration_ms,
                    step_idx=step.step_idx,
                )
            elif step.step_type.value == "llm_query":
                event = LLMQueryEvent(
                    prompt_preview=step.input_text,  # Send full prompt
                    response_preview=step.output_text,  # Send full response
                    full_prompt_chars=len(step.input_text),
                    full_response_chars=len(step.output_text),
                    tokens_in=step.tokens_in,
                    tokens_out=step.tokens_out,
                    cost_usd=step.cost_usd,
                    latency_ms=step.duration_ms,
                    step_idx=step.step_idx,
                )
            else:
                return  # Skip other step types for now
            
            # Run in event loop (we're in a callback)
            asyncio.create_task(manager.publish_event(run_id, event))
        
        # Run agent
        trajectory = await agent.run(
            query=query,
            video_env=video_env,
            on_step=on_step,
        )
        
        # Publish final answer
        final_event = FinalAnswerEvent(
            answer=trajectory.final_answer or "",
            citations=[],  # TODO: Extract citations from trajectory
            success=trajectory.success,
            total_llm_queries=trajectory.num_llm_queries,
            total_cost_usd=trajectory.total_cost_usd,
            total_duration_ms=trajectory.total_duration_ms,
        )
        await manager.publish_event(run_id, final_event)
        
        # Publish run end
        end_event = create_run_end_event(
            success=trajectory.success,
            total_steps=len(trajectory.steps),
            total_llm_queries=trajectory.num_llm_queries,
            total_cost_usd=trajectory.total_cost_usd,
            total_duration_ms=trajectory.total_duration_ms,
        )
        await manager.publish_event(run_id, end_event)
        
        # Mark complete
        await manager.complete_run(run_id, trajectory.to_dict())
        
    except Exception as e:
        logger.exception(f"Agent run failed: {e}")
        
        # Publish error
        error_event = create_error_event(
            error=str(e),
            error_type=type(e).__name__,
        )
        await manager.publish_event(run_id, error_event)
        
        # Mark failed
        await manager.fail_run(run_id, str(e))


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Run the server."""
    import uvicorn
    
    uvicorn.run(
        "rvaa.server.api:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
