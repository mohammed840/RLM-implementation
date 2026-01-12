"""
RVAA Demo Script

Demonstrates the Recursive Vision-Action Agent on a sample video.
Run with: python -m rvaa.demo --video path/to/video.mp4 --question "What happens?"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run_demo(
    video_path: Path,
    question: str,
    backend: str = "openai",
    max_steps: int = 15,
    use_simple_perception: bool = True,
) -> None:
    """Run the RVAA demo.
    
    Args:
        video_path: Path to video file
        question: Question about the video
        backend: LLM backend to use
        max_steps: Maximum agent steps
        use_simple_perception: Use simple perception (no ML models)
    """
    from rvaa.agent.root_agent import RootAgent, TrajectoryStep, StepType
    from rvaa.env.video_env import VideoEnv, VideoView
    from rvaa.tools.llm_backends import get_llm_backend, ModelType
    from rvaa.tools.perception import get_perception_backend
    
    print("\n" + "="*60)
    print("🎬 RVAA: Recursive Vision-Action Agent Demo")
    print("="*60 + "\n")
    
    # Load video
    print(f"📹 Loading video: {video_path}")
    video_env = VideoEnv(video_path)
    print(f"   Duration: {video_env.duration:.1f}s")
    print(f"   Frames: {video_env.total_frames:,}")
    print(f"   Resolution: {video_env.metadata.width}x{video_env.metadata.height}")
    
    # Set up perception
    print(f"\n👁️ Setting up perception layer...")
    perception = get_perception_backend(
        "simple" if use_simple_perception else "blip2"
    )
    
    # Caption function for segments
    def get_segment_captions(segment: VideoView) -> list[tuple[float, str]]:
        """Get captions for a segment (simplified for demo)."""
        # Sample a few frames
        frames = segment.sample_uniform(3)
        captions = []
        for frame in frames:
            result = perception.caption_frame(frame.image, frame.timestamp)
            captions.append((frame.timestamp, result.caption))
        return captions
    
    # Set up LLM backends
    print(f"\n🤖 Setting up LLM backends ({backend})...")
    try:
        root_llm = get_llm_backend(backend, ModelType.ROOT)
        sub_llm = get_llm_backend(backend, ModelType.SUB)
        print(f"   Root LLM: {root_llm.config.model_name}")
        print(f"   Sub LLM: {sub_llm.config.model_name}")
    except Exception as e:
        print(f"❌ Failed to initialize LLM backends: {e}")
        print("   Make sure you have set the appropriate API key:")
        print("   - OPENAI_API_KEY for OpenAI")
        print("   - FIREWORKS_API_KEY for Qwen")
        print("   - ANTHROPIC_API_KEY for Claude")
        return
    
    # Create agent
    print(f"\n🧠 Creating RLM agent...")
    agent = RootAgent(
        root_llm=root_llm,
        sub_llm=sub_llm,
        backend_type=backend,
        max_steps=max_steps,
    )
    
    # Step callback for live display
    def on_step(step: TrajectoryStep) -> None:
        """Display step in real-time."""
        icon = {
            StepType.THINK: "💭",
            StepType.CODE: "📝",
            StepType.LLM_QUERY: "🔍",
            StepType.FINAL: "✅",
            StepType.ERROR: "❌",
        }.get(step.step_type, "➡️")
        
        print(f"\n{icon} Step {step.step_idx}: {step.step_type.value}")
        
        if step.step_type == StepType.CODE:
            print(f"   Code: {step.input_text[:100]}...")
            if step.output_text:
                output = step.output_text[:200]
                print(f"   Output: {output}...")
        elif step.step_type == StepType.LLM_QUERY:
            print(f"   Query: {step.input_text[:100]}...")
            print(f"   Response: {step.output_text[:100]}...")
            print(f"   Cost: ${step.cost_usd:.4f}")
        elif step.step_type == StepType.FINAL:
            print(f"   Answer: {step.output_text}")
    
    # Run agent
    print(f"\n❓ Question: {question}")
    print("\n" + "-"*60)
    print("Starting agent execution...")
    print("-"*60)
    
    try:
        trajectory = await agent.run(
            query=question,
            video_env=video_env,
            get_captions_fn=get_segment_captions,
            on_step=on_step,
        )
    except Exception as e:
        print(f"\n❌ Agent execution failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print results
    print("\n" + "="*60)
    print("📊 Results")
    print("="*60)
    
    print(f"\n✅ Success: {trajectory.success}")
    print(f"📝 Final Answer: {trajectory.final_answer}")
    print(f"\n📈 Statistics:")
    print(f"   Total steps: {len(trajectory.steps)}")
    print(f"   Code executions: {trajectory.num_code_executions}")
    print(f"   LLM sub-calls: {trajectory.num_llm_queries}")
    print(f"   Total cost: ${trajectory.total_cost_usd:.4f}")
    print(f"   Total time: {trajectory.total_duration_ms/1000:.1f}s")
    
    # Cost breakdown
    cost_summary = agent.get_cost_summary()
    print(f"\n💰 Cost Breakdown:")
    print(f"   Root LLM: ${cost_summary['root_llm']['total_cost_usd']:.4f}")
    print(f"   Sub LLM: ${cost_summary['sub_llm']['total_cost_usd']:.4f}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RVAA Demo - Recursive Vision-Action Agent for Video Understanding"
    )
    parser.add_argument(
        "--video", "-v",
        type=Path,
        required=True,
        help="Path to video file",
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        required=True,
        help="Question about the video",
    )
    parser.add_argument(
        "--backend", "-b",
        type=str,
        default="openai",
        choices=["openai", "qwen", "claude"],
        help="LLM backend to use",
    )
    parser.add_argument(
        "--max-steps", "-s",
        type=int,
        default=15,
        help="Maximum agent steps",
    )
    parser.add_argument(
        "--use-blip2",
        action="store_true",
        help="Use BLIP-2 for perception (requires GPU)",
    )
    
    args = parser.parse_args()
    
    # Validate video exists
    if not args.video.exists():
        print(f"❌ Video not found: {args.video}")
        sys.exit(1)
    
    # Run demo
    asyncio.run(run_demo(
        video_path=args.video,
        question=args.question,
        backend=args.backend,
        max_steps=args.max_steps,
        use_simple_perception=not args.use_blip2,
    ))


if __name__ == "__main__":
    main()
