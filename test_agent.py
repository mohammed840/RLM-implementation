#!/usr/bin/env python3
"""Quick test of the agent with OpenRouter"""
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Check API key
api_key = os.environ.get("OPENROUTER_API_KEY", "")
print(f"API Key found: {api_key[:20]}..." if api_key else "NO API KEY!")

from rvaa.tools.llm_backends import get_llm_backend, ModelType

async def test_simple_query():
    """Test a simple query to see what the model returns."""
    root_llm = get_llm_backend("openrouter", ModelType.ROOT)
    
    prompt = """Query: What is 2+2?

You have access to a REPL environment. To answer this, write Python code in ```repl blocks.
When you have the answer, use FINAL(your answer here).

Example:
```repl
result = 2 + 2
print(f"The answer is: {result}")
```

Then after seeing the output, say: FINAL(4)"""
    
    print("Sending test query to GPT-5...")
    response = await root_llm.query(prompt)
    print(f"\n=== LLM Response ===\n{response.content}\n")
    print(f"\nTokens: {response.input_tokens} in, {response.output_tokens} out")
    print(f"Cost: ${response.cost_usd:.4f}")
    print(f"Model: {response.model}")

if __name__ == "__main__":
    asyncio.run(test_simple_query())
