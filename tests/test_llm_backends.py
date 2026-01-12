"""
Tests for LLM Backends

Tests the unified LLM interface and cost tracking.
Uses mocks to avoid actual API calls.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from rvaa.tools.llm_backends import (
    LLMBackend,
    OpenAIBackend,
    QwenBackend,
    ClaudeBackend,
    BackendConfig,
    LLMResponse,
    CostAccounting,
    get_llm_backend,
    ModelType,
)


class TestBackendConfig:
    """Tests for BackendConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BackendConfig(
            model_name="test-model",
            api_base="https://api.test.com/v1",
            api_key_env="TEST_API_KEY",
        )
        assert config.max_tokens == 4096
        assert config.max_context == 128000
        assert config.input_price_per_1k == 0.0


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""
    
    def test_response_creation(self):
        """Create an LLM response."""
        response = LLMResponse(
            content="Hello world",
            input_tokens=10,
            output_tokens=5,
            latency_ms=100.0,
            cost_usd=0.001,
            model="test-model",
        )
        assert response.content == "Hello world"
        assert response.input_tokens == 10


class TestCostAccounting:
    """Tests for cost accounting."""
    
    def test_record_response(self):
        """Record a response and track costs."""
        accounting = CostAccounting()
        
        response = LLMResponse(
            content="Test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200.0,
            cost_usd=0.01,
            model="test-model",
        )
        
        accounting.record(response)
        
        assert accounting.total_input_tokens == 100
        assert accounting.total_output_tokens == 50
        assert accounting.total_cost_usd == 0.01
        assert accounting.total_queries == 1
    
    def test_multiple_records(self):
        """Record multiple responses."""
        accounting = CostAccounting()
        
        for i in range(3):
            response = LLMResponse(
                content="Test",
                input_tokens=100,
                output_tokens=50,
                latency_ms=100.0,
                cost_usd=0.01,
                model="test-model",
            )
            accounting.record(response)
        
        assert accounting.total_queries == 3
        assert accounting.total_cost_usd == 0.03
    
    def test_summary(self):
        """Get accounting summary."""
        accounting = CostAccounting()
        
        response = LLMResponse(
            content="Test",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=500.0,
            cost_usd=0.05,
            model="gpt-4",
        )
        accounting.record(response)
        
        summary = accounting.summary()
        
        assert summary["total_input_tokens"] == 1000
        assert summary["total_queries"] == 1
        assert "gpt-4" in summary["queries_by_model"]


class TestOpenAIBackend:
    """Tests for OpenAI backend."""
    
    def test_config_gpt5(self):
        """Check GPT-5 configuration."""
        config = OpenAIBackend.GPT5_CONFIG
        assert config.model_name == "gpt-5"
        assert config.max_context == 272000
    
    def test_config_gpt5_mini(self):
        """Check GPT-5-mini configuration."""
        config = OpenAIBackend.GPT5_MINI_CONFIG
        assert config.model_name == "gpt-5-mini"
    
    def test_calculate_cost(self):
        """Test cost calculation."""
        backend = OpenAIBackend(api_key="test")
        cost = backend.calculate_cost(1000, 500)
        
        # Cost = (1000/1000) * input_price + (500/1000) * output_price
        expected = (1.0 * backend.config.input_price_per_1k + 
                   0.5 * backend.config.output_price_per_1k)
        assert cost == expected
    
    def test_estimate_tokens(self):
        """Test token estimation."""
        backend = OpenAIBackend(api_key="test")
        
        text = "Hello world, this is a test."
        tokens = backend.estimate_tokens(text)
        
        assert tokens > 0
        assert tokens < 20  # Should be reasonable for short text


class TestQwenBackend:
    """Tests for Qwen backend."""
    
    def test_batching_warning(self):
        """Check batching warning content."""
        assert "200k characters" in QwenBackend.BATCHING_WARNING
        assert "llm_query" in QwenBackend.BATCHING_WARNING
    
    def test_add_batching_warning(self):
        """Test adding batching warning to prompt."""
        backend = QwenBackend(api_key="test")
        
        base_prompt = "Line 1\nLine 2\nLine 3\n" * 10
        modified = backend.get_system_prompt_with_warning(base_prompt)
        
        assert "200k characters" in modified
        assert len(modified) > len(base_prompt)


class TestClaudeBackend:
    """Tests for Claude backend."""
    
    def test_config_opus(self):
        """Check Claude Opus configuration."""
        config = ClaudeBackend.CLAUDE_OPUS_CONFIG
        assert config.model_name == "claude-3-opus-20240229"
        assert config.max_context == 200000


class TestGetLLMBackend:
    """Tests for backend factory function."""
    
    def test_get_openai_root(self):
        """Get OpenAI backend for root agent."""
        backend = get_llm_backend("openai", ModelType.ROOT)
        assert isinstance(backend, OpenAIBackend)
    
    def test_get_openai_sub(self):
        """Get OpenAI backend for sub agent."""
        backend = get_llm_backend("openai", ModelType.SUB)
        assert isinstance(backend, OpenAIBackend)
    
    def test_get_qwen(self):
        """Get Qwen backend."""
        backend = get_llm_backend("qwen", ModelType.ROOT)
        assert isinstance(backend, QwenBackend)
    
    def test_get_claude(self):
        """Get Claude backend."""
        backend = get_llm_backend("claude", ModelType.ROOT)
        assert isinstance(backend, ClaudeBackend)
    
    def test_invalid_backend(self):
        """Invalid backend type raises error."""
        with pytest.raises(ValueError):
            get_llm_backend("invalid", ModelType.ROOT)
