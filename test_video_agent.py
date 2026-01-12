#!/usr/bin/env python3
"""Test the full video agent"""
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Use Qwen (cheaper, good at code)
# os.environ["RVAA_ROOT_MODEL"] = "qwen/qwen-2.5-coder-32b-instruct"
# os.environ["RVAA_SUB_MODEL"] = "qwen/qwen-2.5-coder-32b-instruct"

from rvaa.agent.root_agent import RootAgent, TrajectoryStep
from rvaa.env.video_env import VideoEnv
from rvaa.tools.llm_backends import get_llm_backend, ModelType

async def test_video_agent():
    """Test the agent with a real video."""
    video_path = Path("/Users/mohammedalshehri/RLM-implementation/Full speech_ President Trump addresses nation after U.S. captures Venezuela's Maduro.mp4")
    
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return
    
    print(f"Loading video: {video_path.name}")
    video_env = VideoEnv(video_path)
    print(f"Video loaded: duration={video_env.duration:.1f}s, frames={video_env.total_frames}")
    
    # Get LLM backends
    print("\nInitializing LLMs...")
    root_llm = get_llm_backend("openrouter", ModelType.ROOT)
    sub_llm = get_llm_backend("openrouter", ModelType.SUB)
    print(f"Root LLM: {root_llm.config.model_name}")
    print(f"Sub LLM: {sub_llm.config.model_name}")
    
    # Create agent
    agent = RootAgent(
        root_llm=root_llm,
        sub_llm=sub_llm,
        backend_type="openrouter",
        max_steps=10,  # More steps
    )
    
    def on_step(step: TrajectoryStep):
        print(f"\n{'='*60}")
        print(f"STEP {step.step_idx} - {step.step_type.value.upper()}")
        print(f"{'='*60}")
        if step.step_type.value == "code":
            print(f"CODE:\n{step.input_text[:800]}")
            print(f"\nOUTPUT:\n{step.output_text[:800]}")
        elif step.step_type.value == "llm_query":
            print(f"SUB-CALL PROMPT: {step.input_text[:300]}...")
            print(f"SUB-CALL RESPONSE: {step.output_text[:300]}...")
            print(f"Cost: ${step.cost_usd:.4f}")
        elif step.step_type.value == "final":
            print(f"ANSWER: {step.output_text}")
        else:
            print(f"THINKING: {step.output_text[:500]}...")
    
    print("\nRunning agent...")
    query = "What is the main topic of this video? Describe what you see."
    trajectory = await agent.run(
        query=query,
        video_env=video_env,
        on_step=on_step,
    )
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Query: {trajectory.query}")
    print(f"Total Steps: {len(trajectory.steps)}")
    print(f"Code executions: {trajectory.num_code_executions}")
    print(f"LLM sub-queries: {trajectory.num_llm_queries}")
    print(f"Total cost: ${trajectory.total_cost_usd:.4f}")
    print(f"Success: {trajectory.success}")
    print(f"\nFINAL ANSWER:\n{trajectory.final_answer}")

if __name__ == "__main__":
    asyncio.run(test_video_agent())
