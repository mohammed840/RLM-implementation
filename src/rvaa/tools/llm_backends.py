"""
LLM Backend Implementations

This module provides unified interfaces for different LLM providers,
matching the paper's model configurations:
- GPT-5 (root) + GPT-5-mini (sub) for closed-model replication
- Qwen3-Coder-480B-A35B for open-model replication
- Optional Claude Opus backend

Each backend tracks costs and token usage for reproducible experiments.
"""

from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

import httpx
import tiktoken

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Model type enumeration for configuration."""
    ROOT = "root"
    SUB = "sub"


@dataclass
class BackendConfig:
    """Configuration for an LLM backend.
    
    Attributes:
        model_name: The model identifier (e.g., "gpt-5", "gpt-5-mini")
        api_base: Base URL for the API
        api_key_env: Environment variable name for API key
        max_tokens: Maximum output tokens
        max_context: Maximum context window in tokens
        input_price_per_1k: Price per 1K input tokens in USD
        output_price_per_1k: Price per 1K output tokens in USD
        supports_vision: Whether this model supports vision inputs
    """
    model_name: str
    api_base: str
    api_key_env: str
    max_tokens: int = 4096
    max_context: int = 128000
    input_price_per_1k: float = 0.0
    output_price_per_1k: float = 0.0
    supports_vision: bool = False


@dataclass
class LLMResponse:
    """Response from an LLM query.
    
    Attributes:
        content: The text response
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        latency_ms: Response latency in milliseconds
        cost_usd: Estimated cost in USD
        model: Model used for the query
        raw_response: Raw API response for debugging
    """
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    model: str
    raw_response: Optional[dict[str, Any]] = None


@dataclass
class CostAccounting:
    """Tracks cumulative costs and usage across queries.
    
    Attributes:
        total_input_tokens: Total input tokens across all queries
        total_output_tokens: Total output tokens across all queries
        total_cost_usd: Total cost in USD
        total_queries: Number of queries made
        total_latency_ms: Total latency in milliseconds
        queries_by_model: Number of queries per model
    """
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_queries: int = 0
    total_latency_ms: float = 0.0
    queries_by_model: dict[str, int] = field(default_factory=dict)
    
    def record(self, response: LLMResponse) -> None:
        """Record a query response for accounting."""
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_cost_usd += response.cost_usd
        self.total_queries += 1
        self.total_latency_ms += response.latency_ms
        self.queries_by_model[response.model] = (
            self.queries_by_model.get(response.model, 0) + 1
        )
    
    def summary(self) -> dict[str, Any]:
        """Return a summary of the accounting."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_queries": self.total_queries,
            "avg_latency_ms": round(
                self.total_latency_ms / max(1, self.total_queries), 2
            ),
            "queries_by_model": self.queries_by_model,
        }


class LLMBackend(ABC):
    """Abstract base class for LLM backends.
    
    All backends must implement the query method and provide
    cost tracking through the accounting property.
    """
    
    def __init__(self, config: BackendConfig):
        self.config = config
        self._accounting = CostAccounting()
    
    @property
    def accounting(self) -> CostAccounting:
        """Get the cost accounting tracker."""
        return self._accounting
    
    @abstractmethod
    async def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        """Query the LLM with a prompt.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum output tokens (uses config default if None)
            stop: Optional stop sequences
            
        Returns:
            LLMResponse with content and metadata
        """
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string.
        
        Uses tiktoken for OpenAI models, approximation for others.
        """
        try:
            # Try to use the exact encoder for the model
            encoding = tiktoken.encoding_for_model(self.config.model_name)
            return len(encoding.encode(text))
        except KeyError:
            # Fall back to cl100k_base (GPT-4 encoding) as approximation
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception:
                # Rough approximation: ~4 chars per token
                return len(text) // 4
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost in USD for a query."""
        input_cost = (input_tokens / 1000) * self.config.input_price_per_1k
        output_cost = (output_tokens / 1000) * self.config.output_price_per_1k
        return input_cost + output_cost


class OpenAIBackend(LLMBackend):
    """OpenAI API backend supporting GPT-5/GPT-4 models.
    
    Paper configuration:
    - Root LM: GPT-5 (or GPT-4-turbo as fallback)
    - Sub LM: GPT-5-mini (or GPT-4o-mini as fallback)
    """
    
    # Default configurations matching the paper
    GPT5_CONFIG = BackendConfig(
        model_name="gpt-5",
        api_base="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        max_tokens=4096,
        max_context=272000,  # 272K context window per paper
        input_price_per_1k=0.01,  # Estimated
        output_price_per_1k=0.03,
        supports_vision=True,
    )
    
    GPT5_MINI_CONFIG = BackendConfig(
        model_name="gpt-5-mini",
        api_base="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        max_tokens=4096,
        max_context=128000,
        input_price_per_1k=0.00025,
        output_price_per_1k=0.001,
        supports_vision=False,
    )
    
    # Fallback configurations for current models
    GPT4_TURBO_CONFIG = BackendConfig(
        model_name="gpt-4-turbo",
        api_base="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        max_tokens=4096,
        max_context=128000,
        input_price_per_1k=0.01,
        output_price_per_1k=0.03,
        supports_vision=True,
    )
    
    GPT4O_MINI_CONFIG = BackendConfig(
        model_name="gpt-4o-mini",
        api_base="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        max_tokens=4096,
        max_context=128000,
        input_price_per_1k=0.00015,
        output_price_per_1k=0.0006,
        supports_vision=True,
    )
    
    def __init__(
        self,
        config: Optional[BackendConfig] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize OpenAI backend.
        
        Args:
            config: Backend configuration (defaults to GPT-5 config)
            api_key: API key (reads from env if not provided)
        """
        import os
        
        config = config or self.GPT4_TURBO_CONFIG  # Use available model as default
        super().__init__(config)
        
        self.api_key = api_key or os.environ.get(config.api_key_env, "")
        if not self.api_key:
            logger.warning(
                f"No API key found for {config.model_name}. "
                f"Set {config.api_key_env} environment variable."
            )
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.api_base,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
        return self._client
    
    async def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        """Query the OpenAI API."""
        client = await self._get_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        request_body = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if stop:
            request_body["stop"] = stop
        
        start_time = time.perf_counter()
        
        try:
            response = await client.post("/chat/completions", json=request_body)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", self.estimate_tokens(prompt))
        output_tokens = usage.get("completion_tokens", self.estimate_tokens(content))
        
        cost = self.calculate_cost(input_tokens, output_tokens)
        
        llm_response = LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            model=self.config.model_name,
            raw_response=data,
        )
        
        self._accounting.record(llm_response)
        
        logger.debug(
            f"OpenAI query: {input_tokens} in, {output_tokens} out, "
            f"${cost:.4f}, {latency_ms:.0f}ms"
        )
        
        return llm_response
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()


class QwenBackend(LLMBackend):
    """Qwen3-Coder backend for open-model replication.
    
    Uses Fireworks.ai as the provider (matching the paper).
    
    Paper configuration:
    - Qwen3-Coder-480B-A35B with batching warning
    - Add extra line to system prompt about llm_query batching
    """
    
    QWEN3_CODER_CONFIG = BackendConfig(
        model_name="accounts/fireworks/models/qwen3-coder-480b-a35b-instruct",
        api_base="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_API_KEY",
        max_tokens=4096,
        max_context=128000,
        # Fireworks pricing for Qwen3-Coder (estimated)
        input_price_per_1k=0.0009,
        output_price_per_1k=0.0009,
        supports_vision=False,
    )
    
    # The critical batching warning from Appendix D.1 (1b)
    BATCHING_WARNING = """
IMPORTANT: Be very careful about using 'llm_query' as it incurs high runtime costs. Always batch as much information as reasonably possible into each call (aim for around ~200k characters per call). For example, if you have 1000 lines of information to process, it's much better to split into chunks of 5 and call 'llm_query' on each chunk (200 calls total) rather than making 1000 individual calls. Minimize the number of 'llm_query' calls by batching related information together.
"""
    
    def __init__(
        self,
        config: Optional[BackendConfig] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize Qwen backend.
        
        Args:
            config: Backend configuration (defaults to Qwen3-Coder config)
            api_key: Fireworks API key (reads from env if not provided)
        """
        import os
        
        config = config or self.QWEN3_CODER_CONFIG
        super().__init__(config)
        
        self.api_key = api_key or os.environ.get(config.api_key_env, "")
        if not self.api_key:
            logger.warning(
                f"No API key found for Qwen. "
                f"Set {config.api_key_env} environment variable."
            )
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.api_base,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=180.0,  # Longer timeout for large model
            )
        return self._client
    
    def get_system_prompt_with_warning(self, base_prompt: str) -> str:
        """Add the batching warning to a system prompt.
        
        This implements the Appendix D.1 (1b) diff for Qwen3-Coder.
        """
        # Insert warning after the first few lines, matching paper structure
        lines = base_prompt.split("\n")
        insert_idx = min(15, len(lines))  # Insert around line 15
        lines.insert(insert_idx, self.BATCHING_WARNING)
        return "\n".join(lines)
    
    async def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        """Query the Qwen API via Fireworks."""
        client = await self._get_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        request_body = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if stop:
            request_body["stop"] = stop
        
        start_time = time.perf_counter()
        
        try:
            response = await client.post("/chat/completions", json=request_body)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            logger.error(f"Qwen/Fireworks API error: {e}")
            raise
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", self.estimate_tokens(prompt))
        output_tokens = usage.get("completion_tokens", self.estimate_tokens(content))
        
        cost = self.calculate_cost(input_tokens, output_tokens)
        
        llm_response = LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            model=self.config.model_name,
            raw_response=data,
        )
        
        self._accounting.record(llm_response)
        
        logger.debug(
            f"Qwen query: {input_tokens} in, {output_tokens} out, "
            f"${cost:.4f}, {latency_ms:.0f}ms"
        )
        
        return llm_response
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()


class ClaudeBackend(LLMBackend):
    """Anthropic Claude backend (optional).
    
    Provides Claude Opus as an alternative to the paper's configurations.
    """
    
    CLAUDE_OPUS_CONFIG = BackendConfig(
        model_name="claude-3-opus-20240229",
        api_base="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        max_tokens=4096,
        max_context=200000,
        input_price_per_1k=0.015,
        output_price_per_1k=0.075,
        supports_vision=True,
    )
    
    CLAUDE_SONNET_CONFIG = BackendConfig(
        model_name="claude-3-5-sonnet-20241022",
        api_base="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        max_tokens=4096,
        max_context=200000,
        input_price_per_1k=0.003,
        output_price_per_1k=0.015,
        supports_vision=True,
    )
    
    def __init__(
        self,
        config: Optional[BackendConfig] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize Claude backend.
        
        Args:
            config: Backend configuration (defaults to Claude Opus config)
            api_key: Anthropic API key (reads from env if not provided)
        """
        import os
        
        config = config or self.CLAUDE_SONNET_CONFIG  # Use Sonnet as practical default
        super().__init__(config)
        
        self.api_key = api_key or os.environ.get(config.api_key_env, "")
        if not self.api_key:
            logger.warning(
                f"No API key found for Claude. "
                f"Set {config.api_key_env} environment variable."
            )
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.api_base,
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                timeout=120.0,
            )
        return self._client
    
    async def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        """Query the Claude API."""
        client = await self._get_client()
        
        request_body: dict[str, Any] = {
            "model": self.config.model_name,
            "max_tokens": max_tokens or self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        if system_prompt:
            request_body["system"] = system_prompt
        if temperature > 0:
            request_body["temperature"] = temperature
        if stop:
            request_body["stop_sequences"] = stop
        
        start_time = time.perf_counter()
        
        try:
            response = await client.post("/messages", json=request_body)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            logger.error(f"Claude API error: {e}")
            raise
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        content = data["content"][0]["text"]
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", self.estimate_tokens(prompt))
        output_tokens = usage.get("output_tokens", self.estimate_tokens(content))
        
        cost = self.calculate_cost(input_tokens, output_tokens)
        
        llm_response = LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            model=self.config.model_name,
            raw_response=data,
        )
        
        self._accounting.record(llm_response)
        
        logger.debug(
            f"Claude query: {input_tokens} in, {output_tokens} out, "
            f"${cost:.4f}, {latency_ms:.0f}ms"
        )
        
        return llm_response
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()


class OpenRouterBackend(LLMBackend):
    """OpenRouter backend - unified access to multiple LLM providers.
    
    OpenRouter provides a single API endpoint for accessing models from
    OpenAI, Anthropic, Google, Meta, Mistral, and more.
    
    Benefits:
    - Single API key for all providers
    - Automatic fallbacks
    - Unified pricing
    - No need to manage multiple API keys
    
    Get your API key at: https://openrouter.ai/keys
    """
    
    # Common model configurations via OpenRouter
    # Pricing from https://openrouter.ai/models
    
    # ==========================================================================
    # PAPER-EXACT MODELS (arXiv:2512.24601)
    # These are the EXACT models used in the RLM paper!
    # ==========================================================================
    
    # GPT-5 - Root LM from the paper (400K context)
    GPT5_CONFIG = BackendConfig(
        model_name="openai/gpt-5",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=4096,
        max_context=400000,  # 400K context!
        input_price_per_1k=1.25,
        output_price_per_1k=10.0,
        supports_vision=True,
    )
    
    # GPT-5-mini - Sub LM from the paper (400K context, cheaper)
    GPT5_MINI_CONFIG = BackendConfig(
        model_name="openai/gpt-5-mini",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=4096,
        max_context=400000,  # 400K context!
        input_price_per_1k=0.25,
        output_price_per_1k=2.0,
        supports_vision=True,
    )
    
    # GPT-5-nano - Even cheaper option for sub-calls
    GPT5_NANO_CONFIG = BackendConfig(
        model_name="openai/gpt-5-nano",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=4096,
        max_context=400000,
        input_price_per_1k=0.05,
        output_price_per_1k=0.40,
        supports_vision=True,
    )
    
    # ==========================================================================
    # Alternative Models
    # ==========================================================================
    
    GPT4_TURBO_CONFIG = BackendConfig(
        model_name="openai/gpt-4-turbo",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=4096,
        max_context=128000,
        input_price_per_1k=10.0,
        output_price_per_1k=30.0,
        supports_vision=True,
    )
    
    GPT4O_CONFIG = BackendConfig(
        model_name="openai/gpt-4o",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=4096,
        max_context=128000,
        input_price_per_1k=2.50,
        output_price_per_1k=10.0,
        supports_vision=True,
    )
    
    GPT4O_MINI_CONFIG = BackendConfig(
        model_name="openai/gpt-4o-mini",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=4096,
        max_context=128000,
        input_price_per_1k=0.15,
        output_price_per_1k=0.60,
        supports_vision=True,
    )
    
    CLAUDE_SONNET_CONFIG = BackendConfig(
        model_name="anthropic/claude-3.5-sonnet",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=4096,
        max_context=200000,
        input_price_per_1k=0.003,
        output_price_per_1k=0.015,
        supports_vision=True,
    )
    
    CLAUDE_HAIKU_CONFIG = BackendConfig(
        model_name="anthropic/claude-3-haiku",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=4096,
        max_context=200000,
        input_price_per_1k=0.00025,
        output_price_per_1k=0.00125,
        supports_vision=True,
    )
    
    GEMINI_PRO_CONFIG = BackendConfig(
        model_name="google/gemini-pro-1.5",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=8192,
        max_context=1000000,  # 1M context!
        input_price_per_1k=0.00125,
        output_price_per_1k=0.005,
        supports_vision=True,
    )
    
    LLAMA_70B_CONFIG = BackendConfig(
        model_name="meta-llama/llama-3.1-70b-instruct",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=4096,
        max_context=131072,
        input_price_per_1k=0.00035,
        output_price_per_1k=0.0004,
        supports_vision=False,
    )
    
    # ==========================================================================
    # Paper-Faithful Model Configurations (arXiv:2512.24601)
    # ==========================================================================
    # The RLM paper uses:
    # - Closed-model: GPT-5 (root) + GPT-5-mini (sub)
    # - Open-model: Qwen3-Coder-480B-A35B
    #
    # Best available equivalents on OpenRouter:
    # - GPT-4o → closest to GPT-5 (128K context, vision capable)
    # - GPT-4o-mini → closest to GPT-5-mini (fast, cheap)
    # - Qwen 2.5 Coder 32B → closest to Qwen3-Coder
    # ==========================================================================
    
    # Qwen Coder 32B - closest to paper's Qwen3-Coder-480B-A35B
    QWEN_CODER_32B_CONFIG = BackendConfig(
        model_name="qwen/qwen-2.5-coder-32b-instruct",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=4096,
        max_context=131072,  # 128K context
        input_price_per_1k=0.00018,
        output_price_per_1k=0.00018,
        supports_vision=False,
    )
    
    # For backward compatibility
    QWEN_72B_CONFIG = BackendConfig(
        model_name="qwen/qwen-2.5-72b-instruct",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        max_tokens=4096,
        max_context=32768,
        input_price_per_1k=0.00035,
        output_price_per_1k=0.0004,
        supports_vision=False,
    )
    
    def __init__(
        self,
        config: Optional[BackendConfig] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        site_name: Optional[str] = None,
        site_url: Optional[str] = None,
    ):
        """Initialize OpenRouter backend.
        
        Args:
            config: Backend configuration
            model_name: Override model name (e.g., "openai/gpt-4-turbo")
            api_key: OpenRouter API key (reads from OPENROUTER_API_KEY if not provided)
            site_name: Optional site name for rankings
            site_url: Optional site URL for rankings
        """
        import os
        
        # Default to GPT-4-turbo config
        config = config or self.GPT4_TURBO_CONFIG
        
        # Allow overriding model name
        if model_name:
            config = BackendConfig(
                model_name=model_name,
                api_base=config.api_base,
                api_key_env=config.api_key_env,
                max_tokens=config.max_tokens,
                max_context=config.max_context,
                input_price_per_1k=config.input_price_per_1k,
                output_price_per_1k=config.output_price_per_1k,
                supports_vision=config.supports_vision,
            )
        
        super().__init__(config)
        
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.site_name = site_name or os.environ.get("OPENROUTER_SITE_NAME", "RVAA")
        self.site_url = site_url or os.environ.get("OPENROUTER_SITE_URL", "")
        
        if not self.api_key:
            logger.warning(
                "No OpenRouter API key found. "
                "Set OPENROUTER_API_KEY environment variable. "
                "Get your key at: https://openrouter.ai/keys"
            )
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name,
            }
            
            self._client = httpx.AsyncClient(
                base_url=self.config.api_base,
                headers=headers,
                timeout=120.0,
            )
        return self._client
    
    async def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        """Query via OpenRouter API."""
        client = await self._get_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        request_body: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if stop:
            request_body["stop"] = stop
        
        start_time = time.perf_counter()
        
        try:
            response = await client.post("/chat/completions", json=request_body)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            logger.error(f"OpenRouter API error: {e}")
            raise
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", self.estimate_tokens(prompt))
        output_tokens = usage.get("completion_tokens", self.estimate_tokens(content))
        
        cost = self.calculate_cost(input_tokens, output_tokens)
        
        llm_response = LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            model=self.config.model_name,
            raw_response=data,
        )
        
        self._accounting.record(llm_response)
        
        logger.debug(
            f"OpenRouter query ({self.config.model_name}): {input_tokens} in, "
            f"{output_tokens} out, ${cost:.4f}, {latency_ms:.0f}ms"
        )
        
        return llm_response
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
    
    @classmethod
    def get_available_models(cls) -> dict[str, BackendConfig]:
        """Return a dict of available model configurations.
        
        Paper-exact models (arXiv:2512.24601):
        - openai/gpt-5 → Root LM (400K context)
        - openai/gpt-5-mini → Sub LM (400K context, cheaper)
        - openai/gpt-5-nano → Budget option for sub-calls
        """
        return {
            # PAPER-EXACT MODELS (use these for faithful reproduction!)
            "openai/gpt-5": cls.GPT5_CONFIG,
            "openai/gpt-5-mini": cls.GPT5_MINI_CONFIG,
            "openai/gpt-5-nano": cls.GPT5_NANO_CONFIG,
            # GPT-4 series
            "openai/gpt-4o": cls.GPT4O_CONFIG,
            "openai/gpt-4o-mini": cls.GPT4O_MINI_CONFIG,
            "openai/gpt-4-turbo": cls.GPT4_TURBO_CONFIG,
            # Qwen (open-model from paper)
            "qwen/qwen-2.5-coder-32b-instruct": cls.QWEN_CODER_32B_CONFIG,
            "qwen/qwen-2.5-72b-instruct": cls.QWEN_72B_CONFIG,
            # Other models
            "anthropic/claude-3.5-sonnet": cls.CLAUDE_SONNET_CONFIG,
            "anthropic/claude-3-haiku": cls.CLAUDE_HAIKU_CONFIG,
            "google/gemini-pro-1.5": cls.GEMINI_PRO_CONFIG,
            "meta-llama/llama-3.1-70b-instruct": cls.LLAMA_70B_CONFIG,
        }


def get_llm_backend(
    backend_type: Literal["openai", "qwen", "claude", "openrouter"] = "openrouter",
    model_type: ModelType = ModelType.ROOT,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> LLMBackend:
    """Factory function to create LLM backends.
    
    Args:
        backend_type: The backend provider to use (default: openrouter)
        model_type: ROOT for root agent, SUB for sub-agent calls
        api_key: Optional API key override
        model_name: Optional model name override (for openrouter)
        
    Returns:
        Configured LLM backend instance
        
    Example:
        >>> # Using OpenRouter (recommended - single API key for all models)
        >>> root_llm = get_llm_backend("openrouter", ModelType.ROOT)
        >>> sub_llm = get_llm_backend("openrouter", ModelType.SUB)
        >>>
        >>> # Or use a specific model via OpenRouter
        >>> root_llm = get_llm_backend("openrouter", model_name="anthropic/claude-3.5-sonnet")
        >>>
        >>> # Direct API access
        >>> root_llm = get_llm_backend("openai", ModelType.ROOT)
    """
    import os
    
    if backend_type == "openrouter":
        # Read model from environment or use PAPER-EXACT defaults
        # GPT-5 and GPT-5-mini are now available on OpenRouter!
        if model_name is None:
            if model_type == ModelType.ROOT:
                # Paper uses: GPT-5 for root LM
                model_name = os.environ.get("RVAA_ROOT_MODEL", "openai/gpt-5")
            else:
                # Paper uses: GPT-5-mini for sub LM
                model_name = os.environ.get("RVAA_SUB_MODEL", "openai/gpt-5-mini")
        
        # Get config from available models or use default
        available = OpenRouterBackend.get_available_models()
        config = available.get(model_name, OpenRouterBackend.GPT5_CONFIG)
        
        return OpenRouterBackend(config=config, model_name=model_name, api_key=api_key)
    
    elif backend_type == "openai":
        if model_type == ModelType.ROOT:
            config = OpenAIBackend.GPT4_TURBO_CONFIG
        else:
            config = OpenAIBackend.GPT4O_MINI_CONFIG
        return OpenAIBackend(config=config, api_key=api_key)
    
    elif backend_type == "qwen":
        # Qwen uses same model for root and sub (paper configuration)
        return QwenBackend(api_key=api_key)
    
    elif backend_type == "claude":
        if model_type == ModelType.ROOT:
            config = ClaudeBackend.CLAUDE_OPUS_CONFIG
        else:
            config = ClaudeBackend.CLAUDE_SONNET_CONFIG
        return ClaudeBackend(config=config, api_key=api_key)
    
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

