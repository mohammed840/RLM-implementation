#!/usr/bin/env python3
"""
Simple RVAA Server

A lightweight server to demonstrate the RVAA web interface.
Simulates agent behavior for testing the UI.

Run with: python run_server.py
"""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# App setup
app = FastAPI(title="RVAA Demo Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR = Path(__file__).parent / "src" / "rvaa" / "server" / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory storage
runs: Dict[str, dict] = {}
events: Dict[str, List[dict]] = {}


class QueryRequest(BaseModel):
    query: str
    video_path: str


class QueryResponse(BaseModel):
    run_id: str
    status: str
    stream_url: str


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the web interface."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>RVAA Server Running</h1><p><a href='/docs'>API Docs</a></p>")


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "0.1.0"}


@app.post("/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest):
    """Submit a query and start the agent."""
    run_id = str(uuid.uuid4())[:8]
    
    runs[run_id] = {
        "query": request.query,
        "video_path": request.video_path,
        "status": "running",
        "started_at": time.time(),
    }
    events[run_id] = []
    
    # Start background task to simulate agent
    asyncio.create_task(simulate_agent(run_id, request.query, request.video_path))
    
    return QueryResponse(
        run_id=run_id,
        status="pending",
        stream_url=f"/stream/{run_id}"
    )


async def simulate_agent(run_id: str, query: str, video_path: str):
    """Simulate agent execution with events."""
    
    # Step 1: Code execution - inspect video
    await asyncio.sleep(1)
    events[run_id].append({
        "event_type": "code_execution",
        "data": {
            "code": f'''# Inspect video metadata
video = VideoEnv("{video_path}")
print(f"Duration: {{video.duration}}s")
print(f"Frames: {{video.total_frames}}")''',
            "output": "Duration: 120.5s\nFrames: 2892"
        }
    })
    
    # Step 2: Code - sample segments
    await asyncio.sleep(1.5)
    events[run_id].append({
        "event_type": "code_execution",
        "data": {
            "code": '''# Sample key segments
segments = [context[i*30:(i+1)*30] for i in range(4)]
for i, seg in enumerate(segments):
    print(f"Segment {i}: {seg.start:.1f}s - {seg.end:.1f}s")''',
            "output": "Segment 0: 0.0s - 30.0s\nSegment 1: 30.0s - 60.0s\nSegment 2: 60.0s - 90.0s\nSegment 3: 90.0s - 120.0s"
        }
    })
    
    # Step 3: LLM sub-call
    await asyncio.sleep(2)
    events[run_id].append({
        "event_type": "llm_query",
        "data": {
            "prompt_preview": f"Analyze this segment and answer: {query[:50]}...",
            "response_preview": "In this segment, I observe a person entering a room and sitting at a desk. The scene appears to be an office environment...",
            "tokens_in": 450,
            "tokens_out": 180,
            "cost_usd": 0.0025,
            "latency_ms": 1200
        }
    })
    
    # Step 4: Another code execution
    await asyncio.sleep(1)
    events[run_id].append({
        "event_type": "code_execution",
        "data": {
            "code": '''# Search for relevant segments
results = index.search(query, top_k=5)
for r in results:
    print(f"[{r.score:.2f}] {r.start:.1f}s: {r.text[:50]}...")''',
            "output": "[0.92] 45.0s: Person working at computer...\n[0.87] 78.0s: Document being reviewed...\n[0.81] 102.0s: Meeting discussion..."
        }
    })
    
    # Step 5: Another LLM call
    await asyncio.sleep(1.5)
    events[run_id].append({
        "event_type": "llm_query",
        "data": {
            "prompt_preview": "Synthesize evidence from segments to answer the query...",
            "response_preview": "Based on the analysis of multiple segments, the video shows a typical workday scenario...",
            "tokens_in": 820,
            "tokens_out": 290,
            "cost_usd": 0.0045,
            "latency_ms": 1800
        }
    })
    
    # Step 6: Final answer
    await asyncio.sleep(1)
    events[run_id].append({
        "event_type": "final_answer",
        "data": {
            "answer": f"Based on my analysis of the video at {video_path}, the answer to '{query}' is: The video depicts a professional office environment where a person is engaged in various work activities including computer work, document review, and collaborative discussions. Key events occur at timestamps 45s (focused work session), 78s (document analysis), and 102s (team meeting).",
            "total_cost_usd": 0.0070,
            "success": True
        }
    })
    
    runs[run_id]["status"] = "completed"


@app.get("/stream/{run_id}")
async def stream_events(run_id: str):
    """SSE stream of agent events."""
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    async def event_generator():
        last_idx = 0
        while True:
            # Send new events
            while last_idx < len(events[run_id]):
                event = events[run_id][last_idx]
                yield f"data: {json.dumps(event)}\n\n"
                last_idx += 1
                
                # If final answer, we're done
                if event.get("event_type") == "final_answer":
                    yield f"data: {json.dumps({'event_type': 'run_end'})}\n\n"
                    return
            
            await asyncio.sleep(0.1)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.get("/status/{run_id}")
async def get_status(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")
    return runs[run_id]


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎬 RVAA - Recursive Vision-Action Agent")
    print("="*60)
    print(f"\n📍 Open in browser: http://localhost:8000")
    print(f"📚 API docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
