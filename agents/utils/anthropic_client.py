"""
Anthropic Claude Client Utility
Handles all Claude API interactions for Q-Aero agents
"""
import os
import json
from typing import List, Dict, Any, AsyncGenerator, Optional
from anthropic import AsyncAnthropic, APIError
import asyncio
from agents.config.config import ANTHROPIC_CONFIG


class ClaudeClient:
    """Singleton Claude API client for all agents"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.client = AsyncAnthropic(
            api_key=ANTHROPIC_CONFIG["api_key"]
        )
        self.default_model = ANTHROPIC_CONFIG["default_model"]
        self._initialized = True

    async def create_message(
        self,
        system: str,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Create a single message (non-streaming)

        Args:
            system: System prompt
            messages: List of messages [{"role": "user", "content": "..."}]
            model: Claude model to use (default: sonnet-4.5)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Optional tool definitions

        Returns:
            Response dictionary
        """
        try:
            kwargs = {
                "model": model or self.default_model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": messages,
            }

            if tools:
                kwargs["tools"] = tools

            response = await self.client.messages.create(**kwargs)

            return {
                "id": response.id,
                "role": response.role,
                "content": response.content,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "stop_reason": response.stop_reason,
            }

        except APIError as e:
            print(f"Claude API Error: {e}")
            raise

    async def stream_message(
        self,
        system: str,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a message response

        Args:
            system: System prompt
            messages: List of messages
            model: Claude model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Text chunks as they arrive
        """
        try:
            async with self.client.messages.stream(
                model=model or self.default_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except APIError as e:
            print(f"Claude API Error during streaming: {e}")
            raise

    async def create_message_with_retry(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create message with automatic retry on failure

        Args:
            system: System prompt
            messages: List of messages
            max_retries: Maximum retry attempts
            **kwargs: Additional arguments for create_message

        Returns:
            Response dictionary
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return await self.create_message(system, messages, **kwargs)
            except APIError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"All {max_retries} attempts failed")
                    raise last_error

        raise last_error

    async def extract_json_from_response(
        self,
        response: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from Claude response

        Args:
            response: Claude API response

        Returns:
            Parsed JSON dictionary or None
        """
        try:
            content = response["content"][0]
            if content["type"] == "text":
                text = content["text"]
                # Try to find JSON in code blocks or raw text
                if "```json" in text:
                    json_str = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    json_str = text.split("```")[1].split("```")[0].strip()
                else:
                    json_str = text.strip()

                return json.loads(json_str)
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"Failed to extract JSON: {e}")
            return None


# Singleton instance
claude_client = ClaudeClient()
