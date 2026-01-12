"""
Tests for Root Agent

Tests the RLM-style agent including:
- System prompt generation
- REPL runtime
- Code parsing
- FINAL/FINAL_VAR detection
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from rvaa.agent.root_agent import (
    RootAgent,
    REPLRuntime,
    TrajectoryStep,
    StepType,
    GPT5_SYSTEM_PROMPT,
    QWEN_BATCHING_WARNING,
    get_system_prompt,
)


class TestSystemPrompt:
    """Tests for system prompt generation."""
    
    def test_gpt5_prompt_contains_context_variable(self):
        """Prompt should mention the context variable."""
        assert "context" in GPT5_SYSTEM_PROMPT
        assert "VideoEnv" in GPT5_SYSTEM_PROMPT
    
    def test_gpt5_prompt_contains_llm_query(self):
        """Prompt should mention llm_query function."""
        assert "llm_query" in GPT5_SYSTEM_PROMPT
    
    def test_gpt5_prompt_contains_final(self):
        """Prompt should mention FINAL termination."""
        assert "FINAL(" in GPT5_SYSTEM_PROMPT
        assert "FINAL_VAR(" in GPT5_SYSTEM_PROMPT
    
    def test_qwen_batching_warning(self):
        """Qwen prompt should include batching warning."""
        assert "200k characters" in QWEN_BATCHING_WARNING
        assert "batch" in QWEN_BATCHING_WARNING.lower()
    
    def test_get_system_prompt_openai(self):
        """OpenAI prompt should not have batching warning."""
        prompt = get_system_prompt("openai")
        assert "200k characters" not in prompt
    
    def test_get_system_prompt_qwen(self):
        """Qwen prompt should have batching warning."""
        prompt = get_system_prompt("qwen")
        assert "200k characters" in prompt


class TestREPLRuntime:
    """Tests for REPL runtime environment."""
    
    @pytest.fixture
    def mock_video_env(self):
        """Create a mock VideoEnv."""
        env = MagicMock()
        env.duration = 120.0
        env.total_frames = 3600
        env.metadata = MagicMock()
        return env
    
    @pytest.fixture
    def mock_llm_query(self):
        """Create a mock llm_query function."""
        return MagicMock(return_value="Test response")
    
    @pytest.fixture
    def repl(self, mock_video_env, mock_llm_query):
        """Create a REPL runtime instance."""
        return REPLRuntime(
            context=mock_video_env,
            llm_query_fn=mock_llm_query,
        )
    
    def test_context_is_bound(self, repl, mock_video_env):
        """Context variable should be bound to VideoEnv."""
        assert repl.get_variable("context") is mock_video_env
    
    def test_llm_query_is_bound(self, repl, mock_llm_query):
        """llm_query function should be bound."""
        assert repl.get_variable("llm_query") is mock_llm_query
    
    def test_execute_simple_code(self, repl):
        """Execute simple code and capture output."""
        output = repl.execute("print('hello world')")
        assert output.success
        assert "hello world" in output.stdout
    
    def test_execute_with_context(self, repl):
        """Execute code that accesses context."""
        output = repl.execute("print(f'Duration: {context.duration}')")
        assert output.success
        assert "120.0" in output.stdout
    
    def test_execute_with_error(self, repl):
        """Execute code that raises an error."""
        output = repl.execute("raise ValueError('test error')")
        assert not output.success
        assert "ValueError" in output.error
    
    def test_variable_persistence(self, repl):
        """Variables should persist across executions."""
        repl.execute("x = 42")
        output = repl.execute("print(x)")
        assert output.success
        assert "42" in output.stdout
    
    def test_output_truncation(self, mock_video_env, mock_llm_query):
        """Long output should be truncated."""
        repl = REPLRuntime(
            context=mock_video_env,
            llm_query_fn=mock_llm_query,
            max_output_chars=100,
        )
        output = repl.execute("print('x' * 500)")
        assert len(output.stdout) <= 200  # Some overhead for truncation message


class TestCodeParsing:
    """Tests for parsing agent responses."""
    
    @pytest.fixture
    def agent(self):
        """Create agent for testing patterns."""
        mock_llm = MagicMock()
        return RootAgent(
            root_llm=mock_llm,
            sub_llm=mock_llm,
        )
    
    def test_parse_code_block(self, agent):
        """Parse a REPL code block from response."""
        response = """
Let me analyze the video.

```repl
print(context.duration)
```

This will show the duration.
"""
        matches = agent.CODE_PATTERN.findall(response)
        assert len(matches) == 1
        assert "print(context.duration)" in matches[0]
    
    def test_parse_multiple_code_blocks(self, agent):
        """Parse multiple code blocks."""
        response = """
```repl
x = 1
```

Then:

```repl
y = 2
```
"""
        matches = agent.CODE_PATTERN.findall(response)
        assert len(matches) == 2
    
    def test_parse_final_answer(self, agent):
        """Parse FINAL() termination."""
        response = "Based on my analysis, FINAL(The answer is 42)"
        match = agent.FINAL_PATTERN.search(response)
        assert match is not None
        assert match.group(1).strip() == "The answer is 42"
    
    def test_parse_final_var(self, agent):
        """Parse FINAL_VAR() termination."""
        response = "FINAL_VAR(result)"
        match = agent.FINAL_VAR_PATTERN.search(response)
        assert match is not None
        assert match.group(1) == "result"
    
    def test_no_final_in_code(self, agent):
        """Code blocks should not be parsed for FINAL."""
        response = """
```repl
answer = "test"
print("FINAL(not this)")
```
"""
        # Code should be found
        code_matches = agent.CODE_PATTERN.findall(response)
        assert len(code_matches) == 1


class TestTrajectoryStep:
    """Tests for trajectory tracking."""
    
    def test_step_creation(self):
        """Create a trajectory step."""
        step = TrajectoryStep(
            step_idx=0,
            step_type=StepType.CODE,
            input_text="print('hello')",
            output_text="hello",
        )
        assert step.step_idx == 0
        assert step.step_type == StepType.CODE
    
    def test_step_serialization(self):
        """Serialize step to dict."""
        step = TrajectoryStep(
            step_idx=1,
            step_type=StepType.LLM_QUERY,
            input_text="What is this?",
            output_text="A video",
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.01,
        )
        data = step.to_dict()
        assert data["step_idx"] == 1
        assert data["step_type"] == "llm_query"
        assert data["tokens_in"] == 100
        assert data["cost_usd"] == 0.01
